import unittest
import plotting
import slam
from math import pi
import numpy as np
from worlds import cloister
from plotting import make_ellip
from matplotlib import pyplot as plt


class StateTests(unittest.TestCase):
  
  def setUp(self):
    W = np.array([[2,3],
                  [4,5],
                  [6,7],
                  [0,0]])

    self.state = slam.State(np.array([1,1,1]))
    for i, w in enumerate(W):
      l = self.state.new_slot(i)
      self.state.x[l] = w

  def test_landmark_find_by_index(self):
    l = self.state.landmark(1)
    self.assertTrue(all(l==[5, 6]), l)

  def test_scan(self):
    y_polar, J_y = slam.scan(np.array([1, 1]))
    np.testing.assert_approx_equal(y_polar[0], 1.41421356)
    np.testing.assert_approx_equal(y_polar[1], 0.78539816)

  def test_dynamic_landmark_init(self):
    slot = self.state.new_slot(4)
    self.assertEqual(slot[0], 11)
    self.assertEqual(slot[1], 12)
    self.assertTrue(all(self.state.x[slot]==0))


class SlamProcessTests(unittest.TestCase):

  def setUp(self):
    self.W = cloister.T

  def test_slam_simulation(self):
    res = list(slam.run(
      self.W,
      steps=20,
      u=np.array([20, 0]),
      R=np.array([30, 40, 0]),
      q=np.array([.01, .01]),
      s=np.array([.25, 1*pi/180])))
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
