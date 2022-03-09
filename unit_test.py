import pytest
import numpy as np
from montecarlosim import Simulation

#Group testing into classes, look up 281 test files for more examples
testSimulation=Simulation()
def func(x):
    return x+1

def test_func():
    assert func(3)==5

def test_randomGeneration():
    pass

def test_uniformAngleGeneration():
    pass

def test_eulerMethod():
    pass

def test_wallCollisionDetection():
    pass

def test_periodicWall():
    pass

def test_specularSurface():
    pass

def test_thermalWall():
    pass

def test_particleCollision():
    pass

def test_meanFreePathLength():
    pass

def test_linearMomentum():
    testSimulation.m=10
    testSimulation.velocities=np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
    assert (testSimulation.linearMomentum()==np.array([100, 100, 100])).all()

def test_angularMomentum():
    pass
