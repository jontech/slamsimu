import unittest
import plotting
import slam
import numpy as np
from worlds import cloister
from sim import move_simulation
from plotting import make_ellip
from matplotlib import pyplot as plt


class StateTests(unittest.TestCase):
  
  def setUp(self):
    self.state = slam.State()
    #                        R      0    1    2    3
    self.state.x = np.array([1,1,1, 2,3, 4,5, 6,7, 0,0])
    #                        0 1 2  3 4  5 6  7 8  9 10

  def test_landmark_exist_by_index(self):
    self.assertFalse(self.state.landmark_exist(3))
    self.assertTrue(self.state.landmark_exist(1))
    self.assertTrue(self.state.landmark_exist(2))

  def test_landmark_find_by_index(self):
    l = self.state.landmark(1)
    self.assertTrue(all(l==[5, 6]), l)

  def test_find_all_landmarks(self):
    L = self.state.all_landmarks
    self.assertTrue(all(L[0]==[3, 4]), L[0])
    self.assertTrue(all(L[2]==[7, 8]), L[2])

  def test_scan(self):
    y_polar, J_y = slam.scan(np.array([1, 1]))
    np.testing.assert_approx_equal(y_polar[0], 1.41421356)
    np.testing.assert_approx_equal(y_polar[1], 0.78539816)


class SlamProcessTests(unittest.TestCase):

  def setUp(self):
    self.W = cloister.T

  def test_slam_simulation(self):
    res = list(slam.run(self.W, steps=30, u=np.array([5, 0])))
    R_sim = np.array(list(map(lambda r: r[0], res)))
    states = list(map(lambda r: r[1], res))

    plotting.plots(R_sim, states, self.W)


class PlottingTests(unittest.TestCase):

  def test_ellipse_on_landmark(self):
    l = np.array([1, 2])
    P_l = np.array([[0.21, 0],
                    [0, 2]])

    x, y = make_ellip(l, P_l)

    # plt.plot(x, y)
    # plt.show()


if __name__ == '__main__':
    unittest.main()
