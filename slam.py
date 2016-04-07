from math import atan2, pi, sqrt, sin, cos, atan2
import numpy as np
from matplotlib import pyplot as plt


# SIMULATOR

R = np.array([4, 3, -pi/2])  # robot pose (x, y, th) (makes map anchor point)
u = np.array([0, 0])         # control signal (forward velocity, rotation)

W = np.array([[-18, -80],
              [-69,  36],
              [-26,  41],
              [-81,  47],
              [  5, -58],
              [ 19, -77],
              [  3, -90],
              [-35, -80],
              [-57, -91],
              [-73,  57]]).T

N = W.shape[1]

Y = np.zeros([2, N])             # measurements of all landmarks in W at time


# ESTIMATOR

# n perturbation

x = np.zeros([R.size + W.size])       # state vector as map means
P = np.zeros([x.size]*2)              # state covariance

# system noise: Gaussian {0, Q}
q = np.array([.01, .01])
Q = np.diag(q**2)                     # noise covarinace

# measurement noise: Gaussian {0, S}
s = np.array([.1, 1*pi/180])          # standart deviation
S = np.diag(s**2)                     # covarince

# Map index management
mapspace = np.zeros([x.size], dtype=bool)   # fill with false

# Landmarks index management
landmarks = np.zeros([2, N], dtype=int)

# Place robot in map
r = np.where(mapspace==False)[0][0:R.size]        # set robot pointer
mapspace[r] = True                                  # block map positions
x[r] = R

# initialize robot states
P[r,r] = 0    # initialize robot covariance


def transform_global(F, p):
    """ Tronsform point by reference frame.
    F = [x, y, theta], p = [x, y]
    """
    a = F[2]
    t = F[0:2]

    R = np.array([[cos(a), -sin(a)],     # rotation matrix
                  [sin(a), cos(a)]])

    pw = R.dot(p) + t  

    x, y = p
    J_f = np.array([[1, 0, -y * cos(a) - x * sin(a)],
                    [0, 1, x * cos(a) - y * sin(a)]])
        
    J_p = R
    return pw, J_f, J_p


def transform_local(F, p):
    """ Tronsform point by reference frame.
    F = [x, y, theta], p = [x, y]
    """
    a = F[2]
    t = F[0:2]

    R = np.array([[cos(a), -sin(a)],     # rotation matrix
                  [sin(a), cos(a)]])

    pf = R.T.dot(p - t)

    px, py = p
    x, y = t
    
    J_f = np.array([[-cos(a), -sin(a), cos(a) * (py - y) - sin(a) * (px - x)],
                    [sin(a), -cos(a), -cos(a) * (px - x) - sin(a) * (py - y)]])
    J_p = R.T

    return pf, J_f, J_p


def move(r, u, n):
    """ Move robot using control.
    Returns updated robot position applying control u = [vel, angle] to robot
    pose r = [x, y, th] with noise vector n for u.
    """
    a = r[2]
    u = u + n

    # change robot rotation a
    a = a + u[1]

    # negative angles?
    a = a - 2*pi if a > pi else a
    a = a + 2*pi if a < -pi else a

    # position increment from control signal in global frame,
    # we transform only x coordinate as vel of u = [vel, angle]
    t, J_f, J_p = transform_global(r, np.array([u[0], 0]))

    J_r = np.vstack([J_f, [0, 0, 1]])
    J_n = np.vstack([np.vstack([J_p[:, 0], np.zeros([1, 2])]).T, [0, 1]])

    return np.r_[t, a], J_r, J_n


def inv_scan(w):
    p, fi = w
    return np.array([p * cos(fi), p * sin(fi)])


def scan(w):
    x, y = w
    return np.array([sqrt(x**2 + y**2), atan2(y, x)])


def observe(r, w):
    """Returns measurement! in polar coordinates [p, fi] to point w"""
    return scan(transform_local(r, w)[0])


def inv_observe(r, w):
    return transform_global(r, inv_scan(w))[0]

    
R_res = []
Y_polar = []

for t in np.arange(1, 10):

    # Robot motion

    # robot motion control sim
    n = q * np.random.random(2)         # motion control noise
    R, _, _ = move(R, u, np.zeros(2))   # move robot with no perturbation    
    R_res.append(R) 

    # Prediction (EKF)

    # takes all landmark-robot variances (suboptimal)
    # use m instead : where m = landmarks[landmarks!=0)].T

    x[r], J_r, J_n = move(x[r], u, np.zeros(2))
    P[r, :] = J_r.dot(P[r, :])
    P[:, r] = P[r, :].T
    P[r, r] = (J_r.dot(P[r, r]).dot(J_r.T) + J_n.dot(Q).dot(J_n.T))[r, r]

    
    # known landmark correction


    # observe landmarks
    for i in range(N):
        v = np.random.random(2) * 0.05
        y = observe(R, W[:, i]) + v    # +/-2pi wide measurements with noise
        # simulate sensor range and angle
        p, fi = y
        if p < 100 and fi > -pi/4 and fi < pi/4:
            Y_polar.append(y)  # measurements
            Y[:, i] = inv_observe(R, y)
    
    # landmark initialization (registrtion)
    lids = np.nonzero(landmarks[1, :]==0)[0] # find 
    i = lids[np.random.randint(lids.size)]   # takes one random landmark index
    l = np.nonzero(mapspace==False)[0][:2]   # find first 2 free indeces in mapspace
    mapspace[l] = True    # mark landmarks l in mapspace
    landmarks[:, i] = l   # register landmark indexes from mapsapce

    x[l] = Y[:, i]  # invObserve(x[r], Yi)
    # P[l, rm] = ...    
    # P[rm, l] = P[l, rm].T
    # P[l, l] = ...


R_res = np.array(R_res)
Y_polar = np.array(Y_polar)

# Plots
plt.figure(1)

plt.subplot(2, 1, 1)
plt.grid(True)
plt.xlim([-120, 120])
plt.ylim([-120, 120])
plt.plot(R_res[:, 0], R_res[:, 1], 'o',
         W[0, :], W[1, :], '*',
         Y[0, :], Y[1, :], 'r+')  # also plots Y zeros

plt.subplot(2, 1, 2, projection='polar')
plt.grid(True)
plt.polar(Y_polar[:, 1], Y_polar[:, 0], 'g.',
          [0, -pi/4], [0, 100], 'r-',
          [0, pi/4], [0, 100], 'r-')

plt.figure(2)
plt.pcolor(P)
plt.colorbar()

plt.show()
