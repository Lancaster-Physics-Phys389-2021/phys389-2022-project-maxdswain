from itertools import product, combinations
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import maxwell


class Analysis:
    """
    Class of various functions to analyse and visualise the data produced by running the simulation.

    Parameters
    ----------
    df: DataFrame
        pandas data frame read from data frame in a pickle file produced from running the simulation.
    N: int
        Number of particles in the simulation, this can represent Ne effective particles in a physical system.
    size: int
        Number of data points in the data frame.
    mass: float
        Mass of the particles in the simulation in kg.
    k: float
        Boltzmann constant in standard SI units.
    """
    def __init__(self):
        self.df = pd.read_pickle("Simulation_Data.pkl")
        self.N = self.df["Position"][0].shape[0]
        self.size = self.df.shape[0]
        with open("config.json", "r") as f:
            config = json.load(f)
        self.mass = config["Mass"]
        self.k = config["Boltzmann Constant"]
        self.length = config["Length of Box"]

    # Function to setup 3D scatter plot animation
    def setup_animation(self):
        self.scatters = self.ax.scatter(self.df["Position"][0][:, 0], self.df["Position"][0][:, 1], self.df["Position"][0][:, 2], c="black", alpha=1)
        return self.scatters,

    # Function used for producing a frame in the 3D scatter plot animation of the simulation
    def animation_frame(self, iteration):
        self.scatters.set_offsets(self.df["Position"][iteration][:, :2])
        self.scatters.set_3d_properties(self.df["Position"][iteration][:, 2], "z")
        return self.scatters,

    # 3D scatter plot animation of the simulation
    def animate(self):
        fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
        r = [0, self.length]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                self.ax.plot3D(*zip(s, e), color="red")
        self.ax.set(xlabel="x position", ylabel="y position", zlabel="z position")
        self.ax.view_init(30, 50)
        ani = animation.FuncAnimation(fig, self.animation_frame, int(self.size), init_func=self.setup_animation, blit=True, repeat=True)
        writer = animation.FFMpegWriter(fps=30)
        ani.save("visuals/animation.mp4", writer=writer)

    # Plots the mean velocity of one component (same as fluid velocity as mass of all particles is the same) vs time
    def plot_mean_vel(self):
        component = 0 #0, 1, 2 (x, y, z)
        meanVel = [np.mean(self.df["Velocity"][i][:, component]) for i in range(self.size)]
        plt.plot(self.df["Time"], meanVel)
        plt.show()

    # Plots the mean kinetic energy of the system vs time
    def plot_meanKE(self):
        meanKE = [0.5 * self.mass * np.mean(np.linalg.norm(self.df["Velocity"][i], axis=0))**2 for i in range(self.size)]
        plt.plot(self.df["Time"], meanKE)
        plt.show()

    # Plots the temperature of the system vs time using equipartition theorem
    def plot_temp(self):
        fig, ax = plt.subplots()
        meanVel = np.array([np.mean(self.df["Velocity"][i][:]) for i in range(self.size)])
        meanKEOverM = np.array([0.5 * np.mean(np.linalg.norm(self.df["Velocity"][i], axis=0))**2 for i in range(self.size)])
        T = 2/3 * self.mass / self.k * (meanKEOverM - 1/2 * np.linalg.norm(meanVel, axis=0))
        ax.plot(self.df["Time"], T)
        plt.show()

    # Function to setup 2D scatter animation
    def setup_2D(self):
        self.scatters_2D = self.ax.scatter(self.df["Position"][0][:, 0], self.df["Position"][0][:, 1], c="black")  # Change to 1:3 for yz ani
        self.ax.set(xlabel="x position", ylabel="y position")
        return self.scatters_2D,

    # Function used for producing a frame in the 2D scatter plot animation of the simulation
    def animation_frame2D(self, iteration):
        self.scatters_2D.set_offsets(self.df["Position"][iteration][:, :2])  # Change to 1:3 for yz ani
        return self.scatters_2D,

    # 2D scatter plot animation of the simulation
    def animate2D(self):
        fig, self.ax = plt.subplots()
        ani = animation.FuncAnimation(fig, self.animation_frame2D, int(self.size), init_func=self.setup_2D, blit=True, repeat=True)
        writer = animation.FFMpegWriter(fps=30)
        ani.save("visuals/animation2D.mp4", writer=writer)
        plt.show()

    # Plots a histogram of the speed in the simulation against a Maxwell pdf plot for a given time in the simulation
    def maxwell_hist(self):
        x = np.linspace(-100, 250, 100)
        r = np.linalg.norm(self.df["Velocity"][0], axis=1)
        params = maxwell.fit(r, floc=0)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, maxwell.pdf(x, *params), "r-", lw=3, alpha=0.6, label="maxwell pdf")
        ax.hist(r, density=True, histtype="stepfilled", alpha=0.3)
        ax.legend(loc="best", frameon=False)
        plt.show()

    # Function used for animating histogram, uses blitting
    def prepare_animation_hist(self, bar_container):
        def animation_hist(iteration):
            r = np.linalg.norm(self.df["Velocity"][iteration], axis=1)
            n, _ = np.histogram(r)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            return bar_container.patches
        return animation_hist

    # Animated histogram of speeds over time
    def animate_hist(self):
        fig, ax = plt.subplots()
        r = np.linalg.norm(self.df["Velocity"][0], axis=1)
        _, _, bar_container=ax.hist(r, alpha=0.3)
        ani = animation.FuncAnimation(fig, self.prepare_animation_hist(bar_container), int(self.size), repeat=True, blit=True)
        writer = animation.FFMpegWriter(fps=30)
        ani.save("visuals/animationHist.mp4", writer=writer)
        plt.ylim(ymax=185)
        plt.show()
    
    # Plots a multiplot of all components of the mean velocity as well as a histogram of the initial distribution of speeds
    def vel_multiplot(self):
        fig, axs = plt.subplots(2, 2)
        components=["x", "y", "z"]
        for j in range(3):
            meanVel = [np.mean(self.df["Velocity"][i][:, j]) for i in range(self.size)]
            axs.flat[j].plot(self.df["Time"], meanVel)
            axs.flat[j].set(xlabel="Time (s)", ylabel=f"Mean of the {components[j]} Component\n of Velocity (pm/s)")
        x = np.linspace(-3, 100, 100)
        r = np.linalg.norm(self.df["Velocity"][0], axis=1)
        params = maxwell.fit(r, floc=0)
        axs[1, 1].plot(x, maxwell.pdf(x, *params), "r-", lw=3, alpha=0.6, label="Maxwell pdf")
        axs[1, 1].hist(r, density=True, histtype="stepfilled", alpha=0.3)
        axs[1, 1].legend(loc="best", frameon=False)
        axs[1, 1].set(xlabel="Speed (pm/s)", ylabel="Percentage")
        plt.tight_layout()
        plt.savefig("visuals/vel_multiplot.png")
        plt.show()

    # Plots a multiplot of kinetic energy and temperature
    def KE_temp_multiplot(self):
        fig, axs = plt.subplots(2, sharex=True)
        meanKE = [0.5 * self.mass * np.mean(np.linalg.norm(self.df["Velocity"][i], axis=0))**2 for i in range(self.size)]
        axs[0].set(ylabel="Mean Kinetic Energy (J)")
        axs[0].plot(self.df["Time"], meanKE)
        meanVel = np.array([np.mean(self.df["Velocity"][i][:]) for i in range(self.size)])
        meanKEOverM = np.array([0.5 * np.mean(np.linalg.norm(self.df["Velocity"][i], axis=0))**2 for i in range(self.size)])
        T = 2/3 * self.mass / self.k * (meanKEOverM - 1/2 * np.linalg.norm(meanVel, axis=0))
        axs[1].set(xlabel="Time (s)", ylabel="Temperature (K)")
        axs[1].plot(self.df["Time"], T)
        plt.savefig("visuals/KE_temp_multiplot.png")
        plt.show()

# Example code that could be used to run one of the analysis plots
if __name__ == "__main__":
    test = Analysis()
    test.animate()
