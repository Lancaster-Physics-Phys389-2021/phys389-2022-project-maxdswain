import pytest
import numpy as np
import pandas as pd
from montecarlosim import Simulation

#Group testing into classes, add notes to why each test fails
testSimulation=Simulation()

def test_randomGeneration():
    testSimulation.length=5000
    testSimulation.N=1500
    testSimulation.randomGeneration()
    assert testSimulation.positions.shape[0]==testSimulation.N and testSimulation.positions.shape[1]==3
    assert len(np.argwhere((testSimulation.positions>=testSimulation.length) & (testSimulation.positions<0)))==0
    assert testSimulation.velocities.shape[0]==testSimulation.N and testSimulation.velocities.shape[1]==3

#length of array is 3, values fall between -1 and 1
def test_uniformAngleGeneration():
    angleArray=testSimulation.uniformAngleGeneration()
    assert angleArray.shape[0]==3
    assert (angleArray >= -1).all() and (angleArray <=1).all()

def test_eulerMethod():
    testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
    testSimulation.positions=np.array([[4, 2, 0], [3, 3, 4], [0, 2, 5]])
    testSimulation.dT=2
    testSimulation.update()
    assert (testSimulation.positions==np.array([[8, 10, 12], [9, 5, 10], [10, 12, 7]])).all()

#can make more comprehensive
def test_wallCollisionDetection():
    testSimulation.length=15
    testSimulation.positions=np.array([[4, 17, 0], [3, 3, 4], [0, 2, 6]])
    testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
    testSimulation.wall_collision_detection()
    assert (testSimulation.velocities==np.array([[2, -4, 6], [3, 1, 3], [5, 5, 1]])).all()

def test_periodicWall():
    testSimulation.positions=np.array([[4, 2, 0], [3, 3, 4], [0, 2, 6]])
    testSimulation.length=5
    testSimulation.periodic_boundary([2, 2])
    assert (testSimulation.positions==np.array([[4, 2, 0], [3, 3, 4], [0, 2, 1]])).all()

def test_specularSurface():
    testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
    testSimulation.specular_surface([2, 2])
    assert (testSimulation.velocities==np.array([[2, 4, 6], [3, 1, 3], [5, 5, -1]])).all()

def test_thermalWall():
    pass

def test_particleCollision():
    pass

def test_meanFreePathLength():
    testSimulation.numberDensity=10
    testSimulation.effectiveDiamter=25
    assert np.around(testSimulation.meanPathLength(), decimals=8)==3.601*10**-5

def test_linearMomentum():
    testSimulation.m=10
    testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
    assert (testSimulation.linearMomentum()==np.array([100, 100, 100])).all()

def test_angularMomentum():
    testSimulation.m=10
    testSimulation.velocities=np.array([[2, 4, 6]])
    testSimulation.positions=np.array([[3, 1, 3]])
    assert (testSimulation.angularMomentum()==np.array([-60, -120, 100])).all()

#Test if particles have left the box, test if energy is conserved to an appropriate degree 
#random wall assignment with no thermal walls as KE is not conserved then is [1, 0, 1, 1, 0, 1], ran with length 1000 over 500 iterations
def test_run():
    df=pd.read_pickle("Simulation_Data.csv")
    testSimulation.m=5.31372501312e-26
    assert len(np.argwhere((df["Position"][500]>=1000) & (df["Position"][500]<0)))==0
    meanKineticEnergyBefore=0.5*testSimulation.m*np.mean(np.linalg.norm(df["Velocity"][0], axis=0))**2
    meanKineticEnergyAfter=0.5*testSimulation.m*np.mean(np.linalg.norm(df["Velocity"][500], axis=0))**2
    assert meanKineticEnergyBefore - meanKineticEnergyAfter < 3*10**(-25)
