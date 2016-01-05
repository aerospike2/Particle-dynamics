# --------------------------------------------------------------------------------------- #
# ----------------------- DYNAMICS OF PROTOCELLS LIKE IDENTITIES ------------------------ #
# --------------------------------------------------------------------------------------- #

# Abhishek Sharma
# BioMIP, Ruhr University, Bochum, 


import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

ratio = 0.5

class MinimalCellBox:

    """
    Init_state is an [N x 4] array, where N is the number of cells:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    
    Here, I define two different types of cells, one consists of chemical substrate for self-replication and enzyme and other
    consists of primers and building blocks for the reaction.
    
    ratio : describes the ratio of number of cells of type "1" to type "2".
    
    """
    
    # Setting up initial system #
    
    def __init__(self,
                 init_state = [ 1, 0, 0, -1],
                 bounds = [-2, 2, -2, 2],
                 size = 0.04,
                 M = 0.05,
                 G = 0.0,
                 pType = []):
                                  
        # Setting initial state          
        self.init_state = np.asarray(init_state, dtype=float)
        
        # Setting mass to the particles 
        self.M = M * np.ones(self.init_state.shape[0])
        
        # Setting cell size
        self.size = size
        
        # Setting initial conditions
        self.state = self.init_state.copy()
        
        # Setting different particle types
        self.pType = np.asarray(pType, dtype=float)
        
        # Setting up time 
        self.time_elapsed = 0
        
        # Setting boundaries
        self.bounds = bounds
        
        # Setting random force
        self.G = G
    
    
    # Function for creating single step of simulation  
    
    def step(self, dt):
    
        """step once by dt seconds"""
    
        self.time_elapsed += dt
        
        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)
                        
            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
                        
            if int(pType[i1]) == 1 and int(pType[i2]) == 2:
                self.state[i1, 2:] = v_cm 
                self.state[i2, 2:] = v_cm 
            
            else:
                self.state[i1, 2:] = (v_cm + v_rel * m1 / (m1 + m2))
                self.state[i2, 2:] = (v_cm - v_rel * m2 / (m1 + m2))
                                          
             
        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1

        # Add external force, like gravity
        self.state[:, 3] -= self.M * self.G * dt


#------------------------------------------------------------
# set up initial state

np.random.seed(0)

init_pos = 3.9*(-0.5 + np.random.random((500, 2)))
init_vel = 0.1*(-0.5 + np.random.random((500, 2)))

init_state = np.concatenate([init_pos, init_vel], axis=1)
pType = np.concatenate([np.ones(init_state.shape[0]*ratio), 2*np.ones(init_state.shape[0]*(1-ratio))])
box = MinimalCellBox(init_state, size=0.04)
dt = 1. / 30 # 30fps


#------------------------------------------------------------
# set up figure and animation

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

# cells holds the locations of the cells

cells1, = ax.plot([], [], color = 'c', marker = 'o', ls = '', ms=6)
cells2, = ax.plot([], [], color = 'r', marker = 'o', ls = '', ms=6) 

# rect is the box edge

rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)

def init():
    """initialize animation"""
    global box, rect
    cells1.set_data([], [])
    cells2.set_data([], [])        
    rect.set_edgecolor('none')
    return cells1, cells2, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    # update pieces of the animation
    rect.set_edgecolor('k')
    cells1.set_data(box.state[250:, 0], box.state[250:, 1])
    cells1.set_markersize(ms)
    
    cells2.set_data(box.state[:250, 0], box.state[:250, 1])
    cells2.set_markersize(ms)
    
    return cells1, cells2 , rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True, init_func=init)
plt.show()
 