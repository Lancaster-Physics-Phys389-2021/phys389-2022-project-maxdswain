import numpy as np
import pandas as pd
from scipy.stats import maxwell
from scipy import constants

#maybe split into simulation, walls and particles classes. maybe using pandas is more efficient than various numpy arrays - remember to pickle with pandas?
#type of walls can be randomly generated, add negatives to self.velocities in randomGeneration, improve wall collision/running methods, maybe take into account particle radius when generating positions
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
        self.walls=[0, 0, 0, 0, 0, 0] #list of 6 walls 0 - periodic, 1 - specular, 2 - thermal; check folder for cube with labelled faces, list is in ascending order of index.

    def randomGeneration(self):
        rng=np.random.default_rng(seed=11)
        self.positions=rng.integers(low=0, high=self.length+1, size=(self.N, 3)) #randomly generated positions of N particles in pm
        self.velocities=maxwell.rvs(size=(self.N, 3), random_state=11) #velocities randomly generated using Maxwell distribution

    def meanPathLength(self):
        return 1/(np.sqrt(2)*constants.pi*(self.effectiveDiamter**2)*self.numberDensity)

    #can improve beyond using basic Euler method - Runge Kutta method
    def update(self):
        self.positions+=np.dot(self.velocities, self.dT)

    #searches through positions finding any values out of the cube then runs wall collision methpds appropriate to the designated wall
    def wall_collision_detection(self):
        indicies1, indicies2=np.where(self.positions<=0), np.where(self.positions >= self.length)
        for i, j in enumerate(indicies1[1]):
            if j==0:
                self.run_wall_collision(self.walls[0], [indicies1[0][i], indicies1[1][i]])
            if j==1:
                self.run_wall_collision(self.walls[2], [indicies1[0][i], indicies1[1][i]])
            if j==2:
                self.run_wall_collision(self.walls[4], [indicies1[0][i], indicies1[1][i]])
        for i, j in enumerate(indicies2[1]):
            if j==0:
                self.run_wall_collision(self.walls[1], [indicies2[0][i], indicies2[1][i]])
            if j==1:
                self.run_wall_collision(self.walls[3], [indicies2[0][i], indicies2[1][i]])
            if j==2:
                self.run_wall_collision(self.walls[5], [indicies2[0][i], indicies2[1][i]])

    def run_wall_collision(self, wall, indicies):
        if wall==0:
            self.periodic_boundary(indicies)
        if wall==1:
            self.specular_surface(indicies)
        if wall==2:
            self.thermal_wall(indicies)

    def particle_collision_detection(self):
        pass

    def specular_surface(self, indicies):
        self.velocities[indicies[0]][indicies[1]]-=self.velocities[indicies[0]][indicies[1]]

    #this is just the probability distribution so far, not calculating actual velocities yet - ask about in coding session Thursday
    def thermal_wall(self, indicies):
        self.velocities[indicies[0]][indicies[1]]*=self.m*np.exp(-self.m*self.velocities[indicies[0]][indicies[1]]**2/(2*self.k*self.T))/(self.k*self.T)
        for i in [x for x in range(3) if x!=indicies[1]]:
            self.velocities[indicies[0]][i]=np.sqrt(self.m/(2*constants.pi*self.k*self.T))*np.exp(-self.m*self.velocities[indicies[0]][i]**2/(2*self.k*self.T))

    def periodic_boundary(self, indicies):
        if self.velocities[indicies[0]][indicies[1]]<0: self.velocities[indicies[0]][indicies[1]]+=self.length
        else:
            self.velocities[indicies[0]][indicies[1]]-=self.length

    def linearMomentum(self):
        return self.m*np.sum(self.velocities, axis=0)

    def angularMomentum(self):
        return np.sum(np.cross(self.positions, self.m*self.velocities), axis=0)

test=Simulation()
test.randomGeneration()
print(test.velocities[1])
