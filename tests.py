import unittest
import plotting
import slam

class MainSlamTesting(unittest.TestCase):

  def test_slam_with_alot_steps(self):
    plotting.plots(*slam.run(100), title="100 steps")

  def test_slam_with_little_steps(self):
    plotting.plots(*slam.run(5), title="5 steps")


if __name__ == '__main__':
    unittest.main()
