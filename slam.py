"""
Data management:

r         ------v
mapspace  ----> x
landmarks ------^
"""
from math import atan2, pi, sqrt, sin, cos, atan2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


# SIMULATOR

R = np.array([4, 3, -pi/2]) # robot pose (x, y, th) (makes map anchor point)
u = np.array([5, 0])        # control signal (distance, rotation)

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

Y = np.zeros([2, N])      # measurements of all landmarks in W at time


# ESTIMATOR

# n perturbation

x = np.zeros([R.size + W.size]) # state vector as map means
P = np.zeros([x.size]*2)        # state covariance

# system noise: Gaussian {0, Q}
q = np.array([.01, .01])        # aplitute or standart deviation
Q = np.diag(q**2)               # noise covarinace

# measurement noise: Gaussian {0, S}
s = np.array([.1, 1*pi/180])    # standart deviation
S = np.diag(s**2)               # covarince

# State index management
mapspace = np.zeros([x.size], dtype=bool) # fill with false

# Observed landmarks pointers to mapspace
landmarks = np.zeros([2, N], dtype=int)

# Place robot in map
r = np.where(mapspace==False)[0][0:R.size] # set robot pointer
mapspace[r] = True                         # block map positions
x[r] = R

# initialize robot states
P[r,r] = 0                      # initialize robot covariance


def transform_global(F, p):
    """ transform point by reference frame.
    F = [x, y, theta], p = [x, y]
    """
    a_F = F[2]
    t_F = F[0:2]

    R = np.array([
        [cos(a_F), -sin(a_F)], # rotation matrix
        [sin(a_F), cos(a_F)]
    ])

    p_w = R.dot(p) + t_F  

    x_p, y_p = p
    J_f = np.array([
        [1, 0, -y_p * cos(a_F) - x_p * sin(a_F)],
        [0, 1, x_p * cos(a_F) - y_p * sin(a_F)]
    ])
        
    J_p = R
    return p_w, J_f, J_p


def transform_local(F, p):
    """ Tronsform point by reference frame.
    F = [x, y, theta], p = [x, y]
    """
    a = F[2]
    t = F[0:2]

    R = np.array([
        [cos(a), -sin(a)], # rotation matrix
        [sin(a), cos(a)]
    ])

    pf = R.T.dot(p - t)

    px, py = p
    x, y = t
    
    J_f = np.array([
        [-cos(a), -sin(a), cos(a) * (py - y) - sin(a) * (px - x)],
        [sin(a), -cos(a), -cos(a) * (px - x) - sin(a) * (py - y)]
    ])
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

    J_r = np.vstack([J_f, [0, 0, 1]])
    J_n = np.vstack([np.vstack([J_p[:, 0], np.zeros([1, 2])]).T, [0, 1]])

    return np.r_[t, a], J_r, J_n


def scan(w):
    """ 2D point to polar measure with jacobian to w
    """
    x, y = w

    p = sqrt(x**2 + y**2)
    fi = atan2(y, x)
    
    J_w = np.array([
        [x / sqrt(x**2 + y**2), y / sqrt(x**2 + y**2)],
        [-y / (x**2*(y**2 / x**2 + 1)), 1 / (x*(y**2 / x**2 + 1))]
    ])
    return np.array([p, fi]), J_w


def observe(r, w, v=np.zeros(2)):
    """ Returns measurement! in polar coordinates [p, fi] to point w
    """
    y, J_w = scan(transform_local(r, w)[0])
    return y + v, J_w


def inv_scan(w):
    """ polar measure w to 2D point [x, y] with jacobian to w
    """
    p, fi = w

    J_y = np.array([
        [cos(fi), -p*sin(fi)],
        [sin(fi), p*cos(fi)]
    ])

    return np.array([p*cos(fi), p*sin(fi)]), J_y


def inv_observe(r, w):
    y, J_y = inv_scan(w)
    p, J_f, J_p = transform_global(r, y)

    J_chained = J_p.dot(J_y)
    
    return p, J_f, J_chained, 
    
R_res = []
Y_polar = []

for t in np.arange(1, 2):

    # simulate robot move
    n = q * np.random.random(2) # motion control noise

    R, _, _ = move(R, u, np.zeros(2)) # simulate move robot with no perturbation    
    R_res.append(R) 


    # b. observation (with sensor from -pi/4 to pi/4 angle)

    Y_polar = []                # reset to see last observation
    for i in range(N):
        v_m = np.random.random(2) * 0.05 # measurement noise
        y_m, _ = observe(R, W[:, i], v_m) # +/-2pi wide measurements with noise

        # simulate sensor range and angle
        p, fi = y_m
        if p < 100 and fi > -pi/4 and fi < pi/4:
            Y_polar.append(y_m)           # measurements


    # Estimator (EKF)


    # b) Robot motion prediction

    # takes all landmark-robot variances (suboptimal)
    # use m instead : where m = landmarks[landmarks!=0)].T

    x[r], J_r, J_n = move(x[r], u, np.zeros(2)) # plan robot position given control u
    P[r, :] = J_r.dot(P[r, :])
    P[:, r] = P[r, :].T         # arba J_r.dot(P[:, r])
    P[r, r] = (J_r.dot(P[r, r]).dot(J_r.T) + J_n.dot(Q).dot(J_n.T))[r, r]
    
    # c) known landmark correction

    # lids = np.where(landmarks[0, :])[0] # find all registered landmark indices
    # for l in lids:
    #     l_i = np.where(landmarks[0, :]==0)[0] # find free landmark slot indices
    #     m_i = np.where(mapspace==False)[0][:2] # find first 2 free indeces in mapspace
    #     mapspace[m_i] = True    # mark landmarks occupied for x
    #     landmarks[:, i] = l     # landmark indexes from mapsapce

    #     x[l] = Y[:, i]          # invObserve(x[r], Yi)
    #     print l


    # d) new landmark initialization

    for y_m in Y_polar:

        lids = np.where(landmarks[0, :]==0)[0] # find free slots for landmark

        if all(y_m!=0) and any(lids):
            i = lids[np.random.randint(lids.size)] # find random slot for landmark
            l = np.where(mapspace==False)[0][:2] # find 

            mapspace[l] = True          # reserve landmark positions in mapspace
            landmarks[:, i] = l         # store landmark pointers to x
        
            x[l], J_r, J_y = inv_observe(R, y_m) # global landmark pose
            P[l, :] = J_r.dot(P[r, :])
            P[:, l] = P[l, :].T
            #P[l, l] = J_r.dot(P[r,r]).dot(J_r.T) + J_y.dot(S).dot(J_y.T)

            Y[:, i] = x[l]

            # update covariances P

    print "AFTER INIT"
    print landmarks
    print mapspace
    print x[np.where(mapspace==True)]

R_res = np.array(R_res)
Y_polar = np.array(Y_polar)


def plots():
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.grid(True)
    plt.xlim([-120, 120])
    plt.ylim([-120, 120])
    plt.plot(R_res[:, 0], R_res[:, 1], 'o',
             W[0, :], W[1, :], '*',
             Y[0, :], Y[1, :], 'r+') # also plots Y zeros

    plt.subplot(2, 1, 2, projection='polar')
    plt.grid(True)
    plt.polar(Y_polar[:, 1], Y_polar[:, 0], 'g.',
              [0, -pi/4], [0, 100], 'r-',
              [0, pi/4], [0, 100], 'r-')

    plt.figure(2)
    plt.pcolor(P)
    plt.colorbar()

    plt.show()
plots()


def animations():
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.xlim([-120, 120])
    plt.ylim([-120, 120])
    line, = ax.plot(0, 0, 'o')
    
    def animate(p):
        line.set_ydata(p[0])
        line.set_xdata(p[1])
        return line,

    # Init only required for blitting to give a clean slate.
    def anim_init():
        line.set_ydata(0)
        line.set_xdata(0)
        return line,

    ani = animation.FuncAnimation(fig, animate, R_res, init_func=anim_init,
                                  interval=100, blit=True)
    # ani.save('/tmp/Animation.gif',writer='imagemagick',fps=5);
    plt.show()
