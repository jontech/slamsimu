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
    l = self.state.landmarks(1)[0]
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
    params = dict(
      steps=2,
      u=np.array([20, 0]),
      R=np.array([30, 40, 0]),
      q=np.array([.01, 1*pi/180]),
      s=np.array([.01, 1*pi/180]))

    res = list(slam.run(self.W, **params))

    plotting.sim_plots(res, self.W, params)


class RegistrationTests(unittest.TestCase):
  def setUp(self):
    self.state = slam.State(np.array([-6.242e-03, 3.000e+01, 1.308e-02]))
    l = self.state.new_slot(0)
    self.state.x[l] = np.array([9.724e+00, 4.026e+01])
    self.state.P = np.array([
      [ 0.,     0.,     0.   ,  0.   ,  0.   ],
      [ 0.,     0.,     0.   ,  0.   ,  0.   ],
      [ 0.,     0.,     0.   , -0.001,  0.001],
      [ 0.,     0.,    -0.001,  0.043, -0.04 ],
      [ 0.,     0.,     0.001, -0.04 ,  0.038]])

  def test_registration_registeret_landmark(self):
    y, _, _ = slam.observe(self.state.R, np.array([9.724e+00, 4.026e+01]))
    L = list(slam.registration_existing(self.state, y))
    self.assertEqual(len(L), 1)
    

class PlottingTests(unittest.TestCase):

  def test_ellipse_on_landmark(self):
    L = np.array([3, 3])

    P = np.array([[ 2,  1],
                  [ 1,  2]])

    X, Y = make_ellip(L, P)
    
    # assert ellipse centre
    self.assertTrue(np.isclose(np.mean(X), 2.90598272437), np.mean(X))
    self.assertTrue(np.isclose(np.mean(Y), 2.9468896756), np.mean(Y))


class ObservationTests(unittest.TestCase):

  def test_scan_point(self):
    p = np.array([2, 3])
    y, J = slam.scan(p)

    self.assertTrue(
      np.allclose(y, np.array([ 3.606,  0.983]), atol=0.001), y)

    self.assertTrue(
      np.allclose(J, np.array([[ 0.555,  0.832],
                               [-0.231,  0.154]]), atol=0.001), J)


if __name__ == '__main__':
    unittest.main()
