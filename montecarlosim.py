import numpy as np
import pandas as pd
from scipy.stats import maxwell
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from copy import deepcopy

#mean vx, vy, vyz, mean kinetic energy vs time. 2D plots xy, yz etc, test initialisation of histogram, vx, vy, vz vs maxwell - animate these, units test with pytest, how to implement runge kutta method?
class Simulation:

    #research range of values acceptable based on mean path length and contraints for a dilute gas
    def __init__(self):
        self.N=50 #number of particles - this can represent Ne effective particles in a physical system, currently because N is small in testing Ne=N but when simulating a gas with large Ne, N would be a fraction of Ne - write about the fraction in the report
        self.Ne=1*self.N #number of effective particles
        self.dT=0.1 #time step
        self.timeIntervals=500
        self.T=50 #temperature in Kelvin
        self.length=1000 #length of the sides of the box in pm
        self.k=constants.k #Boltzmann constant
        self.m=32*constants.atomic_mass #mass of one molecule of oxygen in kg
        self.effectiveDiamter=346 #effective diameter of oxygen in pm
        self.numberDensity=self.N/(self.length)**3
        self.rng=np.random.default_rng(seed=11)
        deltaZ=np.max([num for num in range(1, self.length) if self.length%num==0 and num<self.meanPathLength()])
        self.cells=[[deltaZ*i, (i+1)*deltaZ] for i in range(int(self.length/deltaZ))]
        self.cellVolume=deltaZ*self.length**2

    def uniformAngleGeneration(self):
        azimuthal=2*constants.pi*self.rng.random(None)
        q=2*self.rng.random(None)-1
        cosTheta, sinTheta=q, np.sqrt(1-q**2)
        return np.array([sinTheta*np.cos(azimuthal), sinTheta*np.sin(azimuthal), cosTheta])

    def randomGeneration(self):
        self.positions=self.rng.integers(low=0, high=self.length+1, size=(self.N, 3)).astype(float) #randomly generated positions of N particles in pm
        self.speeds=maxwell.rvs(scale=5, size=(self.N, 1), random_state=11) #velocities randomly generated using Maxwell distribution - adjust scale as appropriate to adjust speeds
        self.velocities=np.array([self.speeds[i][0]*self.uniformAngleGeneration() for i in range(self.N)]).reshape(self.N, 3)
        self.walls=[0, 2, 0, 1, 0, 1] #list of 6 walls 0 - periodic, 1 - specular, 2 - thermal; check folder for cube with labelled faces, list is in ascending order of index.

    def meanPathLength(self):
        return 1/(np.sqrt(2)*constants.pi*(self.effectiveDiamter**2)*self.numberDensity)

    #can improve beyond basic Euler method using the Runge Kutta method
    def update(self):
        self.positions+=np.dot(self.velocities, self.dT)

    #searches through positions finding any values out of the cube then runs wall collision method appropriate to the designated wall
    def wall_collision_detection(self):
        indicies1, indicies2=np.where(self.positions<=0), np.where(self.positions >= self.length)
        walls=[self.periodic_boundary, self.specular_surface, self.thermal_wall]
        for i, j in enumerate(indicies1[1]):
            walls[self.walls[j*2]]([indicies1[0][i], indicies1[1][i]])
        for i, j in enumerate(indicies2[1]):
            walls[self.walls[j*2+1]]([indicies2[0][i], indicies2[1][i]])

    def particle_collision_detection(self):
        for cell in self.cells:
            particlesInCell=np.argwhere((self.positions>=cell[0]) & (self.positions<cell[1]))
            numberOfParticlesInCell, n, stuck=len(particlesInCell), 0, 0
            if numberOfParticlesInCell<2: continue
            velDiff=[np.linalg.norm(self.velocities[array[0]][array[1]]-self.velocities[particle[0]][particle[1]]) for particle in particlesInCell for array in particlesInCell if (array == particle).all()==False]
            avgRvel=np.mean(velDiff) #average difference in speed between all particles in the cell
            numberOfCollisions=np.rint(numberOfParticlesInCell**2*constants.pi*self.effectiveDiamter**2*avgRvel*self.Ne*self.dT/(2*self.cellVolume)).astype(int)
            while n<numberOfCollisions:
                stuck+=1
                if stuck>75: break
                randomParticles=[particlesInCell[self.rng.integers(numberOfParticlesInCell)], particlesInCell[self.rng.integers(numberOfParticlesInCell)]] #need to prevent it from randomly selecting the same particle (chance of happening in cells with low number of particles)
                condition=np.linalg.norm(self.velocities[randomParticles[0][0]]-self.velocities[randomParticles[1][0]])/(np.max(velDiff))
                if condition>self.rng.random(1):
                    n+=1
                    velCM=0.5*np.array(self.velocities[randomParticles[0][0]]+self.velocities[randomParticles[1][0]])
                    velR=np.linalg.norm(self.velocities[randomParticles[0][0]]-self.velocities[randomParticles[1][0]])*self.uniformAngleGeneration()
                    self.velocities[randomParticles[0][0]]=velCM+0.5*velR
                    self.velocities[randomParticles[1][0]]=velCM-0.5*velR

    #what order should I run different parts of my code in?
    def run(self):
        self.randomGeneration()
        tempPos, tempVel=[deepcopy(self.positions)], [deepcopy(self.velocities)]
        for x in range(self.timeIntervals):
            self.update()
            self.wall_collision_detection()
            self.particle_collision_detection()
            tempPos.append(deepcopy(self.positions))
            tempVel.append(deepcopy(self.velocities))
        self.time=[i*self.dT for i in range(self.timeIntervals+1)]
        df=pd.DataFrame(data={"Time": self.time, "Position": tempPos, "Velocity": tempVel})
        df.to_pickle("Simulation_Data.csv")

    def specular_surface(self, indicies):
        self.velocities[indicies[0]][indicies[1]]=-self.velocities[indicies[0]][indicies[1]]

    def thermal_wall(self, indicies):
        self.velocities[indicies[0]][indicies[1]]=-np.sign(self.velocities[indicies[0]][indicies[1]])*abs(np.sqrt(self.T)*self.rng.normal(0, 1))
        for i in [x for x in range(3) if x!=indicies[1]]:
            self.velocities[indicies[0]][i]=np.sqrt(-2*self.T*np.log(self.rng.random(None)))

    def periodic_boundary(self, indicies):
        if self.velocities[indicies[0]][indicies[1]]<0: self.positions[indicies[0]][indicies[1]]+=self.length
        else:
            self.positions[indicies[0]][indicies[1]]-=self.length

    def linearMomentum(self):
        return self.m*np.sum(self.velocities, axis=0)

    def angularMomentum(self):
        return np.sum(np.cross(self.positions, self.m*self.velocities), axis=0)

    def plot(self):
        fig=plt.figure()
        ax=fig.add_subplot(111, projection="3d")
        for n in range(self.N):
            ax.scatter(self.positions[n][0], self.positions[n][1], self.positions[n][2], c="black")
        r=[0, self.length]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s, e), color="red")
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_zlabel("z position")
        plt.show()

test=Simulation()
test.run()
test.plot()
