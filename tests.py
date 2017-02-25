import unittest
import plotting
import slam
import numpy as np


class UtilitiesTests(unittest.TestCase):
  
  def setUp(self):
    #                  R      0    1    2    3
    self.x = np.array([1,1,1, 2,3, 4,5, 6,7, 8,9])
    #                  0 1 2  3 4  5 6  7 8  9 10

  def test_landmark_find(self):
    l = slam.find_landmark(1)
    self.assertTrue(all(l==[5, 6]), l)

  def test_find_all_landmarks(self):
    L = slam.find_all_landmarks(self.x)
    self.assertTrue(all(L[0]==[3, 4]), L[0])
    self.assertTrue(all(L[2]==[7, 8]), L[2])

  def test_scan(self):
    y_polar, J_y = slam.scan(np.array([1, 1]))
    np.testing.assert_approx_equal(y_polar[0], 1.41421356)
    np.testing.assert_approx_equal(y_polar[1], 0.78539816)


class SlamProcessTests(unittest.TestCase):

  def test_slam_sim(self):
    steps = 200
    plotting.plots(*slam.run(steps), title="{} steps".format(steps))


if __name__ == '__main__':
    unittest.main()
