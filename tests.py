import unittest
import plotting
import slam
import numpy as np


class UtilitiesTests(unittest.TestCase):
  
  def test_scan(self):
    y_polar, J_y = slam.scan(np.array([1, 1]))
    np.testing.assert_approx_equal(y_polar[0], 1.41421356)
    np.testing.assert_approx_equal(y_polar[1], 0.78539816)


class SlamProcessTests(unittest.TestCase):

  def test_slam_sim_200_steps(self):
    plotting.plots(*slam.run(200), title="200 steps")

  def test_slam_sim_5_steps(self):
    plotting.plots(*slam.run(5), title="5 steps")


if __name__ == '__main__':
    unittest.main()
