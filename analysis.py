import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from itertools import product, combinations
from scipy.stats import maxwell

def animation_frame(iteration, df, scatters):
    for i in range(df["Position"][0].shape[0]):
        scatters[i]._offsets3d=(df["Position"][iteration][i, 0:1], df["Position"][iteration][i, 1:2], df["Position"][iteration][i, 2:])
    return scatters

def animation_frame2D(iteration, df, scatters):
    for i in range(df["Position"][0].shape[0]):
        scatters[i]._offsets=([[df["Position"][iteration][i, 0], df["Position"][iteration][i, 1]]])
    return scatters

def animation_hist(iteration, df, scatters):
    pass

class Analysis:

    def __init__(self):
        self.df=pd.read_pickle("Simulation_Data.pkl")
        self.N=self.df["Position"][0].shape[0]
        self.size=self.df.shape[0]

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
        ani.save("animation.mp4", writer=writer)
        plt.show()

    def plotMeanVel(self):
        component=0 #0, 1, 2 (x, y, z)
        meanVel=[np.mean(self.df["Velocity"][i][:, component]) for i in range(self.size)]
        plt.plot(self.df["Time"], meanVel)
        plt.show()

    def plotMeanKE(self):
        mass=5.31372501312e-26 #from config
        meanKE=[0.5*mass*np.mean(np.linalg.norm(self.df["Velocity"][i], axis=0))**2 for i in range(self.size)]
        plt.plot(self.df["Time"], meanKE)
        plt.show()

    def animate2D(self):
        fig, ax=plt.subplots()
        scatters=[ax.scatter(self.df["Position"][0][i][0], self.df["Position"][0][i][1], c="black") for i in range(self.N)]
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ani=animation.FuncAnimation(fig, animation_frame2D, int(self.size), fargs=(self.df, scatters), blit=False, repeat=True)
        writer=animation.FFMpegWriter(fps=30)
        ani.save("animation2D.mp4", writer=writer)
        plt.show()

    def maxwellHist(self):
        x=np.linspace(0, 25, 100)
        r=np.linalg.norm(self.df["Velocity"][0], axis=1)
        params=maxwell.fit(r, floc=0)
        fig, ax=plt.subplots(1, 1)
        ax.plot(x, maxwell.pdf(x, *params), 'r-', lw=3, alpha=0.6, label="maxwell pdf")
        ax.hist(r, density=True, histtype="stepfilled", alpha=0.3)
        ax.legend(loc="best", frameon=False)
        plt.show()

    def animateHist(self):
        pass

if __name__=="__main__":
    test=Analysis()
    test.animate2D()
