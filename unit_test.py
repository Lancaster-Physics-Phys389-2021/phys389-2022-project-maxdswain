import numpy as np
import pandas as pd

from montecarlosim import Simulation

# Creating a test instance of the simulation
testSimulation = Simulation()


class TestIntialisation:
    # Tests that the position and velocities of N particles in 3D have been initialised correctly and also generated inside the box
    def test_random_generation(self):
        testSimulation.length = 5000
        testSimulation.N = 1500
        testSimulation.random_generation()
        assert testSimulation.positions.shape[0] == testSimulation.N and testSimulation.positions.shape[1] == 3, "Shape of particles positions is incorrect"
        assert len(np.argwhere((testSimulation.positions >= testSimulation.length) & (testSimulation.positions < 0))) == 0, "Particles have been initialised outside of the box"
        assert testSimulation.velocities.shape[0] == testSimulation.N and testSimulation.velocities.shape[1] == 3, "Shape of particles velocities is incorrect"

    # Tests that an array of length 3 is being generated correctly and that the angles are within the appropriate range
    def test_uniform_angle_generation(self):
        angleArray=testSimulation.uniform_angle_generation()
        assert angleArray.shape[0] == 3, "Shape of the angle array is incorrect"
        assert (angleArray >= -1).all() and (angleArray <= 1).all(), "Angles are not being generated correctly"


class TestDynamics:
    # Test of the Euler method using an analytical solution I calculated
    def test_euler_method(self):
        testSimulation.velocities = np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        testSimulation.positions = np.array([[4, 2, 0], [3, 3, 4], [0, 2, 5]])
        testSimulation.dT = 2
        testSimulation.update()
        assert (testSimulation.positions == np.array([[8, 10, 12], [9, 5, 10], [10, 12, 7]])).all(), "Positions have been updated incorrectly using the Euler method"

    # Tests if the correct wall has been detected as having a collision and that the applicable method is ran using an analytical solution
    def test_wall_collision_detection(self):
        testSimulation.length = 15
        testSimulation.positions = np.array([[4, 17, 0], [3, 3, 4], [0, 2, 6]])
        testSimulation.velocities = np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        testSimulation.wall_collision_detection()
        assert (testSimulation.velocities == np.array([[2, -4, 6], [3, 1, 3], [-5, 5, 1]])).all(), "Particles outside of the box have not been identified and dealt with correctly"

    # Test of the periodic wall method using an analytical solution I calculated
    def test_periodic_boundary(self):
        testSimulation.positions = np.array([[4, 2, 0], [3, 3, 4], [0, 2, 6]])
        testSimulation.length = 5
        testSimulation.periodic_boundary((2, 2))
        assert (testSimulation.positions == np.array([[4, 2, 0], [3, 3, 4], [0, 2, 1]])).all(), "Periodic wall is not working correctly"

    # Test of the specular wall method using an analytical solution I calculated
    def test_specular_surface(self):
        testSimulation.velocities = np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        testSimulation.specular_surface((2, 2))
        assert (testSimulation.velocities == np.array([[2, 4, 6], [3, 1, 3], [5, 5, -1]])).all(), "Specular wall is not working correctly"

    # Test of the thermal wall method using an analytical solution I calculated
    def test_thermal_wall(self):
        testSimulation.velocities = np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        testSimulation.thermal_wall((0, 1))
        assert (testSimulation.velocities == np.array([[12, -4, 6], [3, 1, 3], [5, 5, 1]])).all(), "Thermal wall is not working correctly"

    # Tests that particles collide and that the velocity is correct after the collision using an analytical solution I calculated
    def test_particle_collision(self):
        testSimulation.positions = np.array([[4, 2, 0], [3, 3, 4], [0, 2, 4]]).astype(float)
        testSimulation.velocities = np.array([[3, 4, 2], [3, 1, 3], [5, 5, 1]]).astype(float)
        testSimulation.length = 5
        testSimulation.dT = 0.1
        testSimulation.N = 3
        testSimulation.velocities = testSimulation.particle_collision_detection(
            testSimulation.cells, testSimulation.positions, 
            testSimulation.effectiveDiameter, testSimulation.Ne, 
            testSimulation.dT, testSimulation.cellVolume, 
            testSimulation.velocities, testSimulation.uniform_angle_generation)
        assert (testSimulation.velocities == np.array([[3., 4., 2.], [3., 1., 3.], [5., 5., 1.]])).all(), "Particle collision detection is not working correctly"


class TestGeneral:
    # Test of calculating the mean free path length using an analytical solution I calculated
    def test_mean_free_path_length(self):
        testSimulation.numberDensity = 10
        testSimulation.effectiveDiameter = 25
        assert np.around(testSimulation.mean_path_length(), decimals=8) == 3.601 * 10**-5, "Mean free path length is not being calculated correctly"

    # Test of calculating the linear momentum using an analytical solution I calculated
    def test_linearMomentum(self):
        testSimulation.m = 10
        testSimulation.velocities = np.array([[2, 4, 6], [3, 1, 3], [5, 5, 1]])
        assert (testSimulation.linear_momentum() == np.array([100, 100, 100])).all(), "Linear momentum is not being calculated correctly"

    # Test of calculating the angular momentum using an analytical solution I calculated
    def test_angularMomentum(self):
        testSimulation.m = 10
        testSimulation.velocities = np.array([[2, 4, 6]])
        testSimulation.positions = np.array([[3, 1, 3]])
        assert (testSimulation.angular_momentum() == np.array([-60, -120, 100])).all(), "Angular momentum is not being calculated correctly"

    # Test if particles have left the box, test if energy is conserved to an appropriate degree 
    # Walls are assigned randomly with no thermal walls (walls=[1, 0, 1, 1, 0, 1]) as KE is not conserved then, ran with length 100000 over 500 iterations with time step of 10s
    def test_run(self):
        df=pd.read_pickle("Simulation_Data.pkl")
        testSimulation.m = 5.31372501312e-26
        assert len(np.argwhere((df["Position"][500] >= 1000) & (df["Position"][500] < 0))) == 0, "Particles have escaped the box during the simulation"
        meanKineticEnergyBefore = 0.5 * testSimulation.m * np.mean(np.linalg.norm(df["Velocity"][0], axis = 0))**2
        meanKineticEnergyAfter = 0.5 * testSimulation.m * np.mean(np.linalg.norm(df["Velocity"][500], axis=0))**2
        assert meanKineticEnergyBefore - meanKineticEnergyAfter < 3 * 10**-23, "Kinetic energy is not conserved in the simulation"
