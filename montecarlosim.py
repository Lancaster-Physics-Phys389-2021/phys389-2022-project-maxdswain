import numpy as np
import pandas as pd
from scipy.stats import maxwell
from scipy import constants

#maybe split into simulation, walls and particles classes. maybe using pandas is more efficient than various numpy arrays - remember to pickle with pandas?
class Simulation:

    def __init__(self):
        self.N=50
        self.dT=1
        self.T=50
        self.k=constants.k
        self.m=32*constants.atomic_mass

    def randomGeneration(self):
        rng=np.random.default_rng(seed=11)
        self.positions=rng.integers(low=0, high=100, size=(self.N, 3))
        self.velocities=maxwell.rvs(loc=10, size=(self.N, 3), random_state=11)

    #can improve beyond using basic Euler method
    def update(self):
        self.positions+=np.dot(self.velocities, self.dT)

    def wall_collision_detection(self):
        pass

    def specular_surface(self, index1, index2):
        self.velocities[index1][index2]-=self.velocities[index1][index2]

    #this is just the probability distribution so far, not calculating actual velocities yet
    def thermal_wall(self, index1, index2):
        self.velocities[index1][index2]*=self.m*np.exp(-self.m*self.velocities[index1][index2]**2/(2*self.k*self.T))/(self.k*self.T)
        for i in [0, 1, 2]:
            if i != index2:
                self.velocities[index1][i]=np.sqrt(self.m/(2*constants.pi*self.k*self.T))*np.exp(-self.m*self.velocities[index1][i]**2/(2*self.k*self.T))

    def periodic_boundary(self):
        pass

test=Simulation()
test.randomGeneration()
print(test.velocities[1])
