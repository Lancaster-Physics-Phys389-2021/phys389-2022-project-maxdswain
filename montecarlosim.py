import json
from collections.abc import Callable
from copy import deepcopy
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit


# Function to seed NumPy random number generation for Numba
@njit
def set_seed(value: int) -> None:
    np.random.seed(value)


class Simulation:
    """
    Class intialising and running the 3D particle in a box Monte Carlo simulation as well as outputting data.
    
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
    T: float
        Temperature of the gas in Kelvin.
    length: float
        Length of the sides of the box in pm.
    k: float
        Boltzmann constant in standard SI units.
    mass: float
        Mass of the particles in the simulation in kg.
    effectiveDiameter: float
        Effective diameter of the particles in the simulation in pm.
    walls: list
        List of ints representing the 6 walls of the cube, 0 - periodic, 1 - specular, 2 - thermal.
        The list is arranged in the following way [-x, +x, -y, +y, -z, +z]
    """

    __slots__ = ("N", "dT", "timeIntervals", "Tw", "T", "length", "k", "mass", "effectiveDiameter",  
                 "walls", "Ne", "numberDensity", "rng", "cells", "cellVolume", "positions", "velocities")

    def __init__(self) -> None:
        with open("config.json", "r") as f:
            config = json.load(f)
        self.N = config["N"]
        self.dT = config["Time Step"]
        self.timeIntervals = config["Time Intervals"]
        self.Tw = config["Wall Temperature (K)"]
        self.T = config["Gas Temperature (K)"]
        self.length = config["Length of Box"]
        self.k = config["Boltzmann Constant"]
        self.mass = config["Mass"]
        self.effectiveDiameter = config["Effective Diameter"]
        self.walls = config["Walls"]
        self.Ne = 1 * self.N  # Number of effective particles
        self.numberDensity = self.N / self.length**3
        self.rng = np.random.default_rng(seed=11)  # Seeding rng in general python code
        set_seed(11)  # Seeding rng for Numba
        # Calculating the largest possible value for the width of the cells given that it has to be smaller than the mean free path length
        deltaZ = np.max([num for num in range(1, self.length) if self.length % num == 0 and num < self.mean_path_length()])
        self.cells = np.array([[deltaZ * i, (i + 1) * deltaZ] for i in range(int(self.length / deltaZ))])
        self.cellVolume = deltaZ * self.length**2

    # Generates an NumPy array from spherical coordinates that when multiplied by a magnitude generates 3D Cartesian coordinates at uniformly distribute angles
    @staticmethod
    @njit
    def uniform_angle_generation() -> npt.NDArray[np.float64]:
        azimuthal = 2 * np.pi * np.random.random()
        q = 2 * np.random.random() - 1
        cosTheta, sinTheta = q, np.sqrt(1 - q**2)
        return np.array([sinTheta * np.cos(azimuthal), sinTheta * np.sin(azimuthal), cosTheta])

    # Initialise positions using a random distribution inside the cube and velocities using a Maxwell distribution
    def random_generation(self) -> None:
        self.positions = self.rng.integers(0, self.length + 1, (self.N, 3)).astype(float)
        speeds = self.rng.normal(0.0, np.sqrt(self.k * self.T / self.mass) / 3, (self.N,))
        self.velocities = np.array([speeds[i] * self.uniform_angle_generation() for i in range(self.N)])

    def mean_path_length(self) -> float:
        return 1 / (np.sqrt(2) * np.pi * (self.effectiveDiameter**2) * self.numberDensity)

    # Updates positions of all the particles in the simulation using the Euler method
    def update(self) -> None:
        self.positions += np.dot(self.velocities, self.dT)

    # Searches through positions finding any values out of the cube then runs wall collision method appropriate to the designated wall
    def wall_collision_detection(self) -> None:
        callWallCollision = [self.periodic_boundary, self.specular_surface, self.thermal_wall]
        for i, j in np.argwhere(self.positions <= 0):
            callWallCollision[self.walls[j * 2]]((i, j))
        for i, j in np.argwhere(self.positions >= self.length):
            callWallCollision[self.walls[j * 2 + 1]]((i, j))

    # Runs Monte-Carlo determining the collisions that take place
    @staticmethod
    @njit
    def particle_collision_detection(
            cells: npt.NDArray[np.int32], positions: npt.NDArray[np.float64],
            effectiveDiameter: float, Ne: int,
            dT: float, cellVolume: float,
            velocities: npt.NDArray[np.float64], uniform_angle_generation: Callable[[], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        for cell in cells:
            posZ = positions[:, 2]  # Taking only z components as cells are divided in z axis
            particlesInCell = np.argwhere((posZ >= cell[0]) & (posZ < cell[1]))
            numberOfParticlesInCell = len(particlesInCell)
            if numberOfParticlesInCell < 2: continue
            velMax = 25  # Chosen by calculating the velocity difference between particles then looking at the maxes of that over numerous iterations - is an overestimate
            numberOfCollisions = int(np.rint(numberOfParticlesInCell**2 * np.pi * effectiveDiameter**2 * velMax * Ne * dT / (2 * cellVolume)))
            for x in range(numberOfCollisions):
                randomParticlesOne, randomParticleTwo = particlesInCell[np.random.randint(numberOfParticlesInCell)], particlesInCell[np.random.randint(numberOfParticlesInCell)]
                norm = np.linalg.norm(velocities[randomParticlesOne[0]] - velocities[randomParticleTwo[0]])
                if norm / velMax > np.random.random():
                    velCM = 0.5 * (velocities[randomParticlesOne[0]] + velocities[randomParticleTwo[0]])
                    velR = norm * uniform_angle_generation()
                    velocities[randomParticlesOne[0]] = velCM + 0.5 * velR
                    velocities[randomParticleTwo[0]] = velCM - 0.5 * velR
        return velocities

    # Runs the whole simulation for chosen time intervals, stores data after every time step  and ouputs this into a pandas DataFrame
    def run(self) -> None:
        self.random_generation()
        tempPos, tempVel = [deepcopy(self.positions)], [deepcopy(self.velocities)]
        self.progress_bar(0, self.timeIntervals)
        for x in range(self.timeIntervals):
            self.update()
            self.wall_collision_detection()
            self.velocities = self.particle_collision_detection(
                self.cells, self.positions, 
                self.effectiveDiameter, self.Ne, 
                self.dT, self.cellVolume, 
                self.velocities, self.uniform_angle_generation)
            tempPos.append(deepcopy(self.positions))
            tempVel.append(deepcopy(self.velocities))
            self.progress_bar(x + 1, self.timeIntervals)
        time = [i * self.dT for i in range(self.timeIntervals + 1)]
        df = pd.DataFrame(data={"Time": time, "Position": tempPos, "Velocity": tempVel})
        df.to_pickle("Simulation_Data.pkl")

    def specular_surface(self, indices: tuple[int, int]) -> None:
        self.velocities[indices] = -self.velocities[indices]

    def thermal_wall(self, indices: tuple[int, int]) -> None:
        self.velocities[indices] = -np.sign(self.velocities[indices]) * abs(np.sqrt(self.Tw) * self.rng.normal(0, 1))
        for i in [x for x in range(3) if x != indices[1]]:
            self.velocities[indices[0]][i] = np.sqrt(-2 * self.Tw * np.log(self.rng.random(None)))

    def periodic_boundary(self, indices: tuple[int, int]) -> None:
        if self.velocities[indices] < 0:
            self.positions[indices] += self.length
        else:
            self.positions[indices] -= self.length

    def linear_momentum(self) -> npt.NDArray[np.float64]:
        return self.mass * np.sum(self.velocities, axis=0)

    def angular_momentum(self) -> npt.NDArray[np.float64]:
        return np.sum(np.cross(self.positions, self.mass * self.velocities), axis=0)

    def mean_kinetic_energy(self) -> float:
        return 0.5 * self.mass * np.mean(np.linalg.norm(self.velocities, axis=0))**2

    # 3D scatter plot of the positions of every particle in the simulation at the current point in time with a red outline of the cube
    def plot(self) -> None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], c="black", depthshade=False)
        r = [0, self.length]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                ax.plot3D(*zip(s, e), color="red")
        ax.set(xlabel="x position (pm)", ylabel="y position (pm)", zlabel="z position (pm)")
        plt.savefig("visuals/3D_scatter_plot.png")
        plt.show()
    
    @staticmethod
    def progress_bar(progress: int, total: int) -> None:
        percentage = 100 * progress / total
        bar = f"{'???' * int(percentage)}{'-' * (100 - int(percentage))}"
        print(f"\r|{bar}| {percentage:.2f}%", end="\r")


# Example code that could be used to run a simulation and plot the result of it
if __name__ == "__main__":
    example = Simulation()
    example.run()
    example.plot()
