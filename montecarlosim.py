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
        self.T=50 #temperature in Kelvin
        self.length=1000 #length of the sides of the box in pm
        self.k=constants.k #Boltzmann constant
        self.m=32*constants.atomic_mass #mass of one molecule of oxygen in kg
        self.effectiveDiamter=346 #effective diameter of oxygen in pm
        self.numberDensity=self.N/(self.length)**3

    def randomGeneration(self):
        rng=np.random.default_rng(seed=11)
        self.positions=rng.integers(low=0, high=self.length+1, size=(self.N, 3)) #randomly generated positions of N particles in pm
        self.velocities=maxwell.rvs(size=(self.N, 3), random_state=11) #velocities randomly generated using Maxwell distribution - need to add negative velocities

    def meanPathLength(self):
        return 1/(np.sqrt(2)*constants.pi*(self.effectiveDiamter**2)*self.numberDensity)

    #can improve beyond using basic Euler method - Runge Kutta method
    def update(self):
        self.positions+=np.dot(self.velocities, self.dT)

    def wall_collision_detection(self):
        pass

    def particle_collision_detection(self):
        pass

    def specular_surface(self, index1, index2):
        self.velocities[index1][index2]-=self.velocities[index1][index2]

    #this is just the probability distribution so far, not calculating actual velocities yet - ask about in coding session Monday
    def thermal_wall(self, index1, index2):
        self.velocities[index1][index2]*=self.m*np.exp(-self.m*self.velocities[index1][index2]**2/(2*self.k*self.T))/(self.k*self.T)
        for i in [x for x in range(3) if x!=index2]:
            self.velocities[index1][i]=np.sqrt(self.m/(2*constants.pi*self.k*self.T))*np.exp(-self.m*self.velocities[index1][i]**2/(2*self.k*self.T))

    def periodic_boundary(self, index1, index2):
        if self.velocities[index1][index2]<0: self.velocities[index1][index2]+=self.length
        else:
            self.velocities[index1][index2]-=self.length

    def linearMomentum(self):
        return self.m*np.sum(self.velocities, axis=0)

    def angularMomentum(self):
        return np.sum(np.cross(self.positions, self.m*self.velocities), axis=0)

test=Simulation()
test.randomGeneration()
print(test.velocities[1])
