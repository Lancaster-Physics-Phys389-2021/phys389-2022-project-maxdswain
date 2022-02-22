import numpy as np
import pandas as pd
from scipy.stats import maxwell
from scipy import constants

#maybe split into simulation, walls and particles classes. maybe using pandas is more efficient than various numpy arrays - remember to pickle with pandas?
class Simulation:

    #research range of values acceptable based on mean path length and contraints for a dilute gas
    def __init__(self):
        self.N=50 #number of particles
        self.dT=1 #time step
        self.T=50 #temperature
        self.k=constants.k #Boltzmann constant
        self.m=32*constants.atomic_mass #mass of one molecular of oxygen
        self.effectiveDiamter=2 #temp number, need to research effective diameter of oxygen
        self.numberDensity=2 #temp number, need to research number density for oxygen

    def randomGeneration(self):
        rng=np.random.default_rng(seed=11)
        self.positions=rng.integers(low=0, high=100, size=(self.N, 3))
        self.velocities=maxwell.rvs(loc=10, size=(self.N, 3), random_state=11)

    def meanPathLength(self):
        return 1/(np.sqrt(2)*constants.pi*(self.effectiveDiamter**2)*self.numberDensity)

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

    def linearMomentum(self):
        return self.m*np.linalg.norm(np.sum(self.velocities, axis=0))

test=Simulation()
test.randomGeneration()
print(test.velocities[1])
