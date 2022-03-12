import pytest
import numpy as np
import pandas as pd
from montecarlosim import Simulation

testSimulation=Simulation()

class TestIntialisation:
    def test_randomGeneration(self):
        testSimulation.length=5000
        testSimulation.N=1500
        testSimulation.randomGeneration()
        assert testSimulation.positions.shape[0]==testSimulation.N and testSimulation.positions.shape[1]==3, "Shape of particles positions is incorrect"
        assert len(np.argwhere((testSimulation.positions>=testSimulation.length) & (testSimulation.positions<0)))==0, "Particles have been initialised outside of the box"
        assert testSimulation.velocities.shape[0]==testSimulation.N and testSimulation.velocities.shape[1]==3, "Shape of particles velocities is incorrect"

    #length of array is 3, values fall between -1 and 1
    def test_uniformAngleGeneration(self):
        angleArray=testSimulation.uniformAngleGeneration()
        assert angleArray.shape[0]==3, "Shape of the angle array is incorrect"
        assert (angleArray >= -1).all() and (angleArray <=1).all(), "Angles are not being generated correctly"

class TestDynamics:
    def test_eulerMethod(self):
        testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        testSimulation.positions=np.array([[4, 2, 0], [3, 3, 4], [0, 2, 5]])
        testSimulation.dT=2
        testSimulation.update()
        assert (testSimulation.positions==np.array([[8, 10, 12], [9, 5, 10], [10, 12, 7]])).all(), "Positions have been updated incorrectly using the Euler method"

    #can make more comprehensive
    def test_wallCollisionDetection(self):
        testSimulation.length=15
        testSimulation.positions=np.array([[4, 17, 0], [3, 3, 4], [0, 2, 6]])
        testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        testSimulation.wall_collision_detection()
        assert (testSimulation.velocities==np.array([[2, -4, 6], [3, 1, 3], [5, 5, 1]])).all(), "Particles outside of the box have not been identified and dealt with correctly"

    def test_periodicWall(self):
        testSimulation.positions=np.array([[4, 2, 0], [3, 3, 4], [0, 2, 6]])
        testSimulation.length=5
        testSimulation.periodic_boundary([2, 2])
        assert (testSimulation.positions==np.array([[4, 2, 0], [3, 3, 4], [0, 2, 1]])).all(), "Periodic wall is not working correctly"

    def test_specularSurface(self):
        testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        testSimulation.specular_surface([2, 2])
        assert (testSimulation.velocities==np.array([[2, 4, 6], [3, 1, 3], [5, 5, -1]])).all(), "Specular wall is not working correctly"

    def test_thermalWall(self):
        testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        testSimulation.thermal_wall([0, 1])
        assert (testSimulation.velocities==np.array([[8, -1, 4], [3, 1, 3], [5, 5, 1]])).all(), "Thermal wall is not working correctly"

    def test_particleCollision(self):
        testSimulation.positions=np.array([[4, 2, 0], [3, 3, 4], [0, 2, 4]])
        testSimulation.velocities=np.array([[3, 4, 2], [3, 1, 3], [5, 5, 1]])
        testSimulation.length=5
        testSimulation.dT=0.1
        testSimulation.N=3
        testSimulation.particle_collision_detection()
        assert (testSimulation.velocities==np.array([[4, 1, 3], [1, 3, 1], [5, 5, 1]])).all(), "Particle collision detection is not working correctly"

class TestGeneral:
    def test_meanFreePathLength(self):
        testSimulation.numberDensity=10
        testSimulation.effectiveDiamter=25
        assert np.around(testSimulation.meanPathLength(), decimals=8)==3.601*10**-5, "Mean free path length is not being calculated correctly"

    def test_linearMomentum(self):
        testSimulation.m=10
        testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        assert (testSimulation.linearMomentum()==np.array([100, 100, 100])).all(), "Linear momentum is not being calculated correctly"

    def test_angularMomentum(self):
        testSimulation.m=10
        testSimulation.velocities=np.array([[2, 4, 6]])
        testSimulation.positions=np.array([[3, 1, 3]])
        assert (testSimulation.angularMomentum()==np.array([-60, -120, 100])).all(), "Angular momentum is not being calculated correctly"

    #Test if particles have left the box, test if energy is conserved to an appropriate degree 
    #random wall assignment with no thermal walls as KE is not conserved then is [1, 0, 1, 1, 0, 1], ran with length 1000 over 500 iterations
    def test_run(self):
        df=pd.read_pickle("Simulation_Data.pkl")
        testSimulation.m=5.31372501312e-26
        assert len(np.argwhere((df["Position"][500]>=1000) & (df["Position"][500]<0)))==0, "Particles have escaped the box during the simulation"
        meanKineticEnergyBefore=0.5*testSimulation.m*np.mean(np.linalg.norm(df["Velocity"][0], axis=0))**2
        meanKineticEnergyAfter=0.5*testSimulation.m*np.mean(np.linalg.norm(df["Velocity"][500], axis=0))**2
        assert meanKineticEnergyBefore - meanKineticEnergyAfter < 3*10**(-25), "Kinetic energy is not conserved in the simulation"
