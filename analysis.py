import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from itertools import product, combinations

def animation_frame(iteration, df, scatters, k):
    for i in range(df["Position"][0].shape[0]):
        scatters[i]._offsets3d=(df["Position"][iteration*k][i, 0:1], df["Position"][iteration*k][i, 1:2], df["Position"][iteration*k][i, 2:])
    return scatters

class Analysis:

    def __init__(self):
        self.df=pd.read_pickle("Simulation_Data.csv")

    def animate(self):
        iterations=int(self.df.shape[0]/100)
        fig=plt.figure()
        ax=p3.Axes3D(fig)
        scatters=[ax.scatter(self.df["Position"][0][i][0], self.df["Position"][0][i][1], self.df["Position"][0][i][2], c="black") for i in range(self.df["Position"][0].shape[0])]
        r=[0, 1000]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s, e), color="red")
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_zlabel("z position")
        ax.view_init(25, 10)
        ani=animation.FuncAnimation(fig, animation_frame, iterations, fargs=(self.df, scatters, 100), blit=False, repeat=True)
        #writer=animation.FFMpegWriter(fps=60)
        writer=animation.PillowWriter(fps=60) 
        ani.save("animation.gif", writer=writer)
        plt.show()

test=Analysis()
test.animate()
