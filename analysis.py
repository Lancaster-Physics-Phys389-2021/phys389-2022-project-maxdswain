import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from itertools import product, combinations
from scipy.stats import maxwell
import json

df=pd.read_pickle("Simulation_Data.pkl")

#Function used for producing a frame in the 3D scatter plot animation of the simulation
def animation_frame(iteration, df, scatters):
    for i in range(df["Position"][0].shape[0]):
        scatters[i]._offsets3d=(df["Position"][iteration][i, 0:1], df["Position"][iteration][i, 1:2], df["Position"][iteration][i, 2:])
    return scatters

#Function used for producing a frame in the 2D scatter plot animation of the simulation
def animation_frame2D(iteration, df, scatters):
    for i in range(df["Position"][0].shape[0]):
        scatters[i]._offsets=([[df["Position"][iteration][i, 0], df["Position"][iteration][i, 1]]])
    return scatters

#Function used for animating histogram, uses blitting
def prepare_animation_hist(bar_container):
    def animation_hist(iteration):
        r=np.linalg.norm(df["Velocity"][iteration], axis=1)
        n, _=np.histogram(r)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)
        return bar_container.patches
    return animation_hist

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
        self.df=df
        self.N=self.df["Position"][0].shape[0]
        self.size=self.df.shape[0]
        with open("config.json", "r") as f:
            config=json.load(f)
        self.mass=config["Mass"]
        self.k=config["Boltzmann Constant"]

    #3D scatter plot animation of the simulation 
    def animate(self):
        iterations=int(self.size)
        fig=plt.figure()
        ax=fig.add_subplot(111, projection="3d")
        scatters=[ax.scatter(self.df["Position"][0][i][0], self.df["Position"][0][i][1], self.df["Position"][0][i][2], c="black") for i in range(self.N)]
        r=[0, 1000]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s, e), color="red")
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_zlabel("z position")
        ax.view_init(25, 10)
        ani=animation.FuncAnimation(fig, animation_frame, iterations, fargs=(self.df, scatters), blit=False, repeat=True)
        writer=animation.FFMpegWriter(fps=30)
        ani.save("visuals/animation.mp4", writer=writer)
        plt.show()

    #Plots the mean velocity of one component (same as fluid velocity as mass of all particles is the same) vs time
    def plotMeanVel(self):
        component=0 #0, 1, 2 (x, y, z)
        meanVel=[np.mean(self.df["Velocity"][i][:, component]) for i in range(self.size)]
        plt.plot(self.df["Time"], meanVel)
        plt.show()

    #Plots the mean kinetic energy of the system vs time
    def plotMeanKE(self):
        meanKE=[0.5*self.mass*np.mean(np.linalg.norm(self.df["Velocity"][i], axis=0))**2 for i in range(self.size)]
        plt.plot(self.df["Time"], meanKE)
        plt.show()

    #Plots the temperature of the system vs time using equipartition theorem
    def plotTemp(self):
        fig, ax=plt.subplots()
        meanVel=np.array([np.mean(self.df["Velocity"][i][:]) for i in range(self.size)])
        meanKEOverM=np.array([0.5*np.mean(np.linalg.norm(self.df["Velocity"][i], axis=0))**2 for i in range(self.size)])
        T=2/3*self.mass/self.k*(meanKEOverM-1/2*np.linalg.norm(meanVel, axis=0))
        ax.plot(self.df["Time"], T)
        ax.legend()
        plt.show()

    #2D scatter plot animation of the simulation
    def animate2D(self):
        fig, ax=plt.subplots()
        scatters=[ax.scatter(self.df["Position"][0][i][0], self.df["Position"][0][i][1], c="black") for i in range(self.N)]
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ani=animation.FuncAnimation(fig, animation_frame2D, int(self.size), fargs=(self.df, scatters), blit=False, repeat=True)
        writer=animation.FFMpegWriter(fps=30)
        ani.save("visuals/animation2D.mp4", writer=writer)
        plt.show()

    #Plots a histogram of the speed in the simulation against a Maxwell pdf plot for a given time in the simulation
    def maxwellHist(self):
        x=np.linspace(0, 25, 100)
        r=np.linalg.norm(self.df["Velocity"][0], axis=1)
        params=maxwell.fit(r, floc=0)
        fig, ax=plt.subplots(1, 1)
        ax.plot(x, maxwell.pdf(x, *params), 'r-', lw=3, alpha=0.6, label="maxwell pdf")
        ax.hist(r, density=True, histtype="stepfilled", alpha=0.3)
        ax.legend(loc="best", frameon=False)
        plt.show()

    #Animated histogram of speeds over time
    def animateHist(self):
        fig, ax=plt.subplots()
        r=np.linalg.norm(self.df["Velocity"][0], axis=1)
        _, _, bar_container=ax.hist(r, alpha=0.3)
        ani=animation.FuncAnimation(fig, prepare_animation_hist(bar_container), int(self.size), repeat=True, blit=True)
        writer=animation.FFMpegWriter(fps=30)
        ani.save("visuals/animationHist.mp4", writer=writer)
        plt.show()

if __name__=="__main__":
    test=Analysis()
    test.plotTemp()
