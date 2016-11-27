import numpy as np
from math import pi

def sensor_sim(R, W, N, v_m):
    # measurements of all landmarks in W at time
    Y = np.zeros([2, N])      
    for i in range(N):
        # +/-2pi wide measurements with noise
        y, _, _ = observe(R, W[:, i], v_m)
        # simulate sensor range and angle
        p, fi = y
        if p < 100 and fi > -pi/4 and fi < pi/4:
            Y[:, i] = y     # measurements at time t
    return Y

