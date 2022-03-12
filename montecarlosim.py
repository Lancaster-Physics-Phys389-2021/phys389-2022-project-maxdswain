import numpy as np
import pandas as pd
from scipy.stats import maxwell
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from copy import deepcopy
import json

#animate histogram vx, vy, vz vs maxwell plots; check conserved quantities are conserved; detailed quality comments and docstrings
class Simulation:

    def __init__(self):
        with open("config.json", "r") as f:
            config=json.load(f)
        self.N=config["N"] #number of particles - this can represent Ne effective particles in a physical system, currently because N is small in testing Ne=N but when simulating a gas with large Ne, N would be a fraction of Ne - write about the fraction in the report
        self.dT=config["Time Step"] #time step
        self.timeIntervals=config["Time Intervals"]
        self.T=config["Temperature (K)"] #temperature in Kelvin
        self.length=config["Length of Box"] #length of the sides of the box in pm
        self.k=config["Boltzmann Constant"] #Boltzmann constant
        self.m=config["Mass"] #mass of one molecule of oxygen in kg
        self.effectiveDiamter=config["Effective Diameter"] #effective diameter of oxygen in pm
        self.walls=config["Walls"] #list of 6 walls 0 - periodic, 1 - specular, 2 - thermal; check folder for cube with labelled faces, list is in ascending order of index.
        self.Ne=1*self.N #number of effective particles
        self.numberDensity=self.N/(self.length)**3
        self.rng=np.random.default_rng(seed=11)
        deltaZ=np.max([num for num in range(1, self.length) if self.length%num==0 and num<self.meanPathLength()])
        self.cells=[[deltaZ*i, (i+1)*deltaZ] for i in range(int(self.length/deltaZ))]
        self.cellVolume=deltaZ*self.length**2

    def uniformAngleGeneration(self):
        azimuthal=2*np.pi*self.rng.random(None)
        q=2*self.rng.random(None)-1
        cosTheta, sinTheta=q, np.sqrt(1-q**2)
        return np.array([sinTheta*np.cos(azimuthal), sinTheta*np.sin(azimuthal), cosTheta])

    def randomGeneration(self):
        self.positions=self.rng.integers(low=0, high=self.length+1, size=(self.N, 3)).astype(float) #randomly generated positions of N particles in pm
        self.speeds=maxwell.rvs(scale=5, size=(self.N, 1), random_state=11) #velocities randomly generated using Maxwell distribution - adjust scale as appropriate to adjust speeds
        self.velocities=np.array([self.speeds[i][0]*self.uniformAngleGeneration() for i in range(self.N)]).reshape(self.N, 3)

    def meanPathLength(self):
        return 1/(np.sqrt(2)*np.pi*(self.effectiveDiamter**2)*self.numberDensity)

    def update(self):
        self.positions+=np.dot(self.velocities, self.dT)

    #searches through positions finding any values out of the cube then runs wall collision method appropriate to the designated wall
    def wall_collision_detection(self):
        indices1, indices2=np.where(self.positions<=0), np.where(self.positions >= self.length)
        walls=[self.periodic_boundary, self.specular_surface, self.thermal_wall]
        for i, j in enumerate(indices1[1]):
            walls[self.walls[j*2]]([indices1[0][i], indices1[1][i]])
        for i, j in enumerate(indices2[1]):
            walls[self.walls[j*2+1]]([indices2[0][i], indices2[1][i]])

    def particle_collision_detection(self):
        for cell in self.cells:
            posZ=self.positions[:, 2] #taking only z components as cells are divided in z axis
            particlesInCell=np.argwhere((posZ>=cell[0]) & (posZ<cell[1]))
            numberOfParticlesInCell=len(particlesInCell)
            if numberOfParticlesInCell<2: continue
            velMax=25 #chosen by calculating the velocity difference between particles then looking at the maxes of that over numerous iterations - is an overestimate
            numberOfCollisions=np.rint(numberOfParticlesInCell**2*np.pi*self.effectiveDiamter**2*velMax*self.Ne*self.dT/(2*self.cellVolume)).astype(int)
            for x in range(numberOfCollisions):
                randomParticles=[particlesInCell[self.rng.integers(numberOfParticlesInCell)], particlesInCell[self.rng.integers(numberOfParticlesInCell)]] #need to prevent it from randomly selecting the same particle (chance of happening in cells with low number of particles)
                norm=np.linalg.norm(self.velocities[randomParticles[0][0]]-self.velocities[randomParticles[1][0]])
                condition=norm/velMax
                if condition>self.rng.random(1):
                    velCM=0.5*np.array(self.velocities[randomParticles[0][0]]+self.velocities[randomParticles[1][0]])
                    velR=norm*self.uniformAngleGeneration()
                    self.velocities[randomParticles[0][0]]=velCM+0.5*velR
                    self.velocities[randomParticles[1][0]]=velCM-0.5*velR

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
        df.to_pickle("Simulation_Data.pkl")

    def specular_surface(self, indices):
        self.velocities[indices[0]][indices[1]]=-self.velocities[indices[0]][indices[1]]

    def thermal_wall(self, indices):
        self.velocities[indices[0]][indices[1]]=-np.sign(self.velocities[indices[0]][indices[1]])*abs(np.sqrt(self.T)*self.rng.normal(0, 1))
        for i in [x for x in range(3) if x!=indices[1]]:
            self.velocities[indices[0]][i]=np.sqrt(-2*self.T*np.log(self.rng.random(None)))

    def periodic_boundary(self, indices):
        if self.velocities[indices[0]][indices[1]]<0: self.positions[indices[0]][indices[1]]+=self.length
        else:
            self.positions[indices[0]][indices[1]]-=self.length

    def linearMomentum(self):
        return self.m*np.sum(self.velocities, axis=0)

    def angularMomentum(self):
        return np.sum(np.cross(self.positions, self.m*self.velocities), axis=0)

    def meanKineticEnergy(self):
        return 0.5*self.m*np.mean(np.linalg.norm(self.velocities, axis=0))**2

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
