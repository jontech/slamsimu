from math import atan2, pi, sqrt, sin, cos, atan2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


# SIMULATOR

R = np.array([4, 3, -pi/2])     # robot pose (x, y, th) (makes map anchor point)
u = np.array([10, 0])         # control signal (distance, rotation)

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
q = np.array([.01, .01])              # aplitute or standart deviation
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
    """ transform point by reference frame.
    F = [x, y, theta], p = [x, y]
    """
    a_F = F[2]
    t_F = F[0:2]

    R = np.array([[cos(a_F), -sin(a_F)],     # rotation matrix
                  [sin(a_F), cos(a_F)]])

    p_w = R.dot(p) + t_F  

    x_p, y_p = p
    J_f = np.array([[1, 0, -y_p * cos(a_F) - x_p * sin(a_F)],
                    [0, 1, x_p * cos(a_F) - y_p * sin(a_F)]])
        
    J_p = R
    return p_w, J_f, J_p


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

    1) apply control noise
    2) update robot angle by u 
    3) update robot position in global frame by u

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
    print t

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

for t in np.arange(1, 3):

    # simulate robot move
    n = q * np.random.random(2)         # motion control noise

    R, _, _ = move(R, u, np.zeros(2))   # simulate move robot with no perturbation    
    R_res.append(R) 


    # Prediction (EKF)

    # takes all landmark-robot variances (suboptimal)
    # use m instead : where m = landmarks[landmarks!=0)].T

    x[r], J_r, J_n = move(x[r], u, np.zeros(2))   # plan robot position given control u
    P[r, :] = J_r.dot(P[r, :])
    P[:, r] = P[r, :].T             # arba J_r.dot(P[:, r])
    P[r, r] = (J_r.dot(P[r, r]).dot(J_r.T) + J_n.dot(Q).dot(J_n.T))[r, r]

    
    # known landmark correction


    # landmarks observation
    Y_polar = []   # reset to see last observation
    for i in range(N):
        v = np.random.random(2) * 0.05
        y = observe(R, W[:, i]) + v    # +/-2pi wide measurements with noise
        # simulate sensor range and angle
        p, fi = y
        if p < 100 and fi > -pi/4 and fi < pi/4:
            Y_polar.append(y)                  # measurements
            Y[:, i] = inv_observe(R, y)
    print(Y)

    # landmark initialization (registrtion)
    lids = np.nonzero(landmarks[1, :]==0)[0] # find free positions for new landmarks
    i = lids[np.random.randint(lids.size)]   # takes one random landmark index
    l = np.nonzero(mapspace==False)[0][:2]   # find first 2 free indeces in mapspace
    mapspace[l] = True    # mark landmarks l in mapspace
    landmarks[:, i] = l   # register landmark indexes from mapsapce
    print(i)

    # x[l] = Y[:, i]  # invObserve(x[r], Yi)
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


#Animations

# fig, ax = plt.subplots()
# plt.grid(True)
# plt.xlim([-120, 120])
# plt.ylim([-120, 120])
# line, = ax.plot(0, 0, 'o')

# def animate(p):
#     line.set_ydata(p[0])
#     line.set_xdata(p[1])
#     return line,

# # Init only required for blitting to give a clean slate.
# def anim_init():
#     line.set_ydata(0)
#     line.set_xdata(0)
#     return line,

# ani = animation.FuncAnimation(fig, animate, R_res, init_func=anim_init,
#                               interval=100, blit=True)
# # ani.save('/tmp/Animation.gif',writer='imagemagick',fps=5);
# plt.show()
