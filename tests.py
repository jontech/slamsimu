import unittest
from slam import slam
import plotting

class MainSlamTesting(unittest.TestCase):

  def test_slam_with_100_steps(self):
    plotting.plots(*slam(100))

  def test_slam_with_10_steps(self):
    plotting.plots(*slam(10))

  def test_slam_with_5_steps(self):
    plotting.plots(*slam(5))


if __name__ == '__main__':
    unittest.main()
