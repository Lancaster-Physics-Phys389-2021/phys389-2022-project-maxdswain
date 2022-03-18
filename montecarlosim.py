import numpy as np
from numba import njit
from itertools import product, combinations
import pandas as pd
from copy import deepcopy
from scipy.stats import maxwell
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

#set suitable initial conditions (paper + gh) and start getting results

#Function to seed NumPy random number generation for Numba
@njit
def set_seed(value):
    np.random.seed(value)

class Simulation:
    """
    Class of intialising and running the 3D particle in a box Monte Carlo simulation as well as outputting data.
    
    Parameters
    ----------
    N: int
        Number of particles in the simulation, this can represent Ne effective particles in a physical system.
    dT: float
        The value of the time step between time intervals in seconds.
    timeIntervals: int
        The number of time steps that will be ran if the simulation is ran.
    Tw: float
        Temperature of the thermal wall in Kelvin.
    length: float
        Length of the sides of the box in pm.
    k: float
        Boltzmann constant in standard SI units.
    m: float
        Mass of the particles in the simulation in kg.
    effectiveDiameter: float
        Effective diameter of the particles in the simulation in pm.
    walls: list
        List of ints representing the 6 walls of the cube, 0 - periodic, 1 - specular, 2 - thermal.
        The list is arranged in the following way [-x, +x, -y, +y, -z, +z]
    """
    def __init__(self):
        with open("config.json", "r") as f:
            config=json.load(f)
        self.N=config["N"]
        self.dT=config["Time Step"]
        self.timeIntervals=config["Time Intervals"]
        self.Tw=config["Wall Temperature (K)"]
        self.length=config["Length of Box"]
        self.k=config["Boltzmann Constant"]
        self.m=config["Mass"]
        self.effectiveDiameter=config["Effective Diameter"]
        self.walls=config["Walls"]
        self.Ne=1*self.N #number of effective particles
        self.numberDensity=self.N/(self.length)**3
        self.rng=np.random.default_rng(seed=11) #seeding rng in general python code
        set_seed(11) #seeding rng for Numba
        #calculating the largest possible value for the width of the cells given that it has to be smaller than the mean free path length
        deltaZ=np.max([num for num in range(1, self.length) if self.length%num==0 and num<self.meanPathLength()])
        self.cells=np.array([[deltaZ*i, (i+1)*deltaZ] for i in range(int(self.length/deltaZ))])
        self.cellVolume=deltaZ*self.length**2

    #Generates an NumPy array from spherical coordinates that when multiplied by a magnitude generates 3D Cartesian coordinates at uniformly distribute angles
    @staticmethod
    @njit
    def uniformAngleGeneration():
        azimuthal=2*np.pi*np.random.random()
        q=2*np.random.random()-1
        cosTheta, sinTheta=q, np.sqrt(1-q**2)
        return np.array([sinTheta*np.cos(azimuthal), sinTheta*np.sin(azimuthal), cosTheta])

    #Initialise positions using a random distribution inside the cube and velocities using a Maxwell distribution
    def randomGeneration(self):
        self.positions=self.rng.integers(low=0, high=self.length+1, size=(self.N, 3)).astype(float) #randomly generated positions of N particles in pm
        self.speeds=maxwell.rvs(scale=5, size=self.N, random_state=11)
        self.velocities=np.array([self.speeds[i]*self.uniformAngleGeneration() for i in range(self.N)]).reshape(self.N, 3)

    def meanPathLength(self):
        return 1/(np.sqrt(2)*np.pi*(self.effectiveDiameter**2)*self.numberDensity)

    #Updates positions of all the particles in the simulation using the Euler method
    def update(self):
        self.positions+=np.dot(self.velocities, self.dT)

    #Searches through positions finding any values out of the cube then runs wall collision method appropriate to the designated wall
    def wall_collision_detection(self):
        callWallCollision=[self.periodic_boundary, self.specular_surface, self.thermal_wall]
        for i, j in np.argwhere(self.positions<=0):
            callWallCollision[self.walls[j*2]]([i, j])
        for i, j in np.argwhere(self.positions >= self.length):
            callWallCollision[self.walls[j*2+1]]([i, j])

    #Runs Monte-Carlo determining the collisions that take place
    @staticmethod
    @njit
    def particle_collision_detection(cells, positions, effectiveDiameter, Ne, dT, cellVolume, velocities, uniformAngleGeneration):
        for cell in cells:
            posZ=positions[:, 2] #taking only z components as cells are divided in z axis
            particlesInCell=np.argwhere((posZ>=cell[0]) & (posZ<cell[1]))
            numberOfParticlesInCell=len(particlesInCell)
            if numberOfParticlesInCell<2: continue
            velMax=25 #chosen by calculating the velocity difference between particles then looking at the maxes of that over numerous iterations - is an overestimate
            numberOfCollisions=int(np.rint(numberOfParticlesInCell**2*np.pi*effectiveDiameter**2*velMax*Ne*dT/(2*cellVolume)))
            for x in range(numberOfCollisions):
                randomParticlesOne, randomParticleTwo=particlesInCell[np.random.randint(numberOfParticlesInCell)], particlesInCell[np.random.randint(numberOfParticlesInCell)]
                norm=np.linalg.norm(velocities[randomParticlesOne[0]]-velocities[randomParticleTwo[0]])
                if norm/velMax>np.random.random():
                    velCM=0.5*(velocities[randomParticlesOne[0]]+velocities[randomParticleTwo[0]])
                    velR=norm*uniformAngleGeneration()
                    velocities[randomParticlesOne[0]]=velCM+0.5*velR
                    velocities[randomParticleTwo[0]]=velCM-0.5*velR
        return velocities

    #Runs the whole simulation for chosen time intervals, stores data after every time step  and ouputs this into a pandas DataFrame
    def run(self):
        self.randomGeneration()
        tempPos, tempVel=[deepcopy(self.positions)], [deepcopy(self.velocities)]
        for x in range(self.timeIntervals):
            self.update()
            self.wall_collision_detection()
            self.velocities=self.particle_collision_detection(self.cells, 
                self.positions, 
                self.effectiveDiameter,
                self.Ne, 
                self.dT, 
                self.cellVolume, 
                self.velocities, 
                self.uniformAngleGeneration
                )
            tempPos.append(deepcopy(self.positions))
            tempVel.append(deepcopy(self.velocities))
        self.time=[i*self.dT for i in range(self.timeIntervals+1)]
        df=pd.DataFrame(data={"Time": self.time, "Position": tempPos, "Velocity": tempVel})
        df.to_pickle("Simulation_Data.pkl")

    def specular_surface(self, indices):
        self.velocities[indices[0]][indices[1]]=-self.velocities[indices[0]][indices[1]]

    def thermal_wall(self, indices):
        self.velocities[indices[0]][indices[1]]=-np.sign(self.velocities[indices[0]][indices[1]])*abs(np.sqrt(self.Tw)*self.rng.normal(0, 1))
        for i in [x for x in range(3) if x!=indices[1]]:
            self.velocities[indices[0]][i]=np.sqrt(-2*self.Tw*np.log(self.rng.random(None)))

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

    #3D scatter plot of the positions of every particle in the simulation at the current point in time with a red outline of the cube
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
