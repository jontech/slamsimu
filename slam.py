from math import atan2, pi, sqrt, sin, cos, atan2
import numpy as np
from worlds import cloister
import plotting
from time import sleep
from importlib import reload

reload(plotting)

def transform_global(F, p):
    """ transform point from frame F to global frame.
    F = [x, y, theta], p = [x, y]
    J_f jacobian respect to frame F
    J_p jacobian respect to point p
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
        [1, 0, -y_p*cos(a_F) - x_p*sin(a_F)],
        [0, 1, x_p*cos(a_F) - y_p*sin(a_F)]
    ])
        
    J_p = R

    return p_w, J_f, J_p


def transform_local(F, p):
    """ transforms point [x, y] to frame [x, y, theta] with
    jacobians with respect to frame and point.
    """
    a = F[2]
    t = F[0:2]

    R = np.array([
        [cos(a), -sin(a)],                             # rotation matrix
        [sin(a), cos(a)]
    ])

    p = R.T.dot(p - t)

    p_x, p_y = p
    x, y = t
    
    J_f = np.array([
        [-cos(a), -sin(a), cos(a) * (p_y - y) - sin(a) * (p_x - x)],
        [sin(a), -cos(a), -cos(a) * (p_x - x) - sin(a) * (p_y - y)]
    ])

    J_p = R.T

    return p, J_f, J_p


def move(r, u, n):
    """ Move robot using control.

    Returns updated robot position applying control
    u = [vel, angle] to robot
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

    J_r = np.vstack([
        J_f,
        [0, 0, 1]
    ])
    J_n = np.vstack([
        np.vstack([J_p[:, 0], np.zeros([1, 2])]).T,
        [0, 1]
    ])

    return np.hstack([t, a]), J_r, J_n


def scan(p):
    """ 2D point to polar measure with jacobian to p
    """
    x, y = p

    d = sqrt(x**2 + y**2)
    fi = atan2(y, x)
    
    J_p = np.array([
        [x / sqrt(x**2 + y**2),          y / sqrt(x**2 + y**2)],
        [-y / (x**2*(y**2 / x**2 + 1)),  1 / (x*(y**2 / x**2 + 1))]
    ])

    return np.array([d, fi]), J_p


def inv_scan(y):
    """ polar measure p to 2D point [x, y] with jacobian to p
    """
    d, fi = y

    p = np.array([d*cos(fi), d*sin(fi)])

    J_y = np.array([
        [cos(fi), -d*sin(fi)],
        [sin(fi), d*cos(fi)]
    ])

    return p, J_y


def observe(r, p, v=np.zeros(2)):
    """ Returns `measurement` in polar coordinates [d, fi] to point p
    """
    p, J_r, J_p = transform_local(r, p)
    y, J_y = scan(p)           # take measurement to robot-local point

    J_yr = J_y.dot(J_r)         # chain with robot frame
    J_yp = J_y.dot(J_p)         # chain with point

    return y + v, J_yr, J_yp


def inv_observe(r, y):
    """ local measurement to global point
    """
    p_local, J_y = inv_scan(y)
    p, J_r, J_p = transform_global(r, p_local)

    J_y = J_p.dot(J_y)          # chain rule

    return p, J_r, J_y, 


# MAIN

R = np.array([100, 30, 0])      # robot initial pose (x, y, th)
u = np.array([4, 0])          # control signal (step, rotation)
W = cloister.T
N = W.shape[1]
x = np.zeros([R.size + W.size]) # state vector as map means
P = np.zeros([x.size]*2)        # state covariance
Y = np.zeros([2, N])            # observation measurements 
# system noise: Gaussian {0, Q}
q = np.array([.01, .01])        # noise standart deviation
Q = np.diag(q**2)               # noise covarinace ??
# measurement noise: Gaussian {0, S}
s = np.array([.1, 1*pi/180])              # noise standart deviation
S = np.diag(s**2)                         # noise covarince
# State index management
mapspace = np.zeros([x.size], dtype=bool) # fill with false
# Observed landmarks pointers to mapspace
landmarks = np.zeros([2, N], dtype=int)
# Place robot in map
r = np.where(mapspace==False)[0][0:R.size] # takes first 3 positions
mapspace[r] = True                         # block map positions
# initialize robot states
x[r] = R
P[r,r] = 0                      # initialize robot covariance
R_res = []                      # robot positions for plots

# main loop
for t in np.arange(1, 10):

    # simulate robot move
    n = q * np.random.random(2) # motion control noise
    R, _, _ = move(R, u, n)
    R_res.append(R)

    # observation ALL landmarks in world
    # TODO landmark asociation (same landmark?)
    for i in range(N):
        v_m = s * np.random.random(2) # measurement noise
        Y[:, i], _, _ = observe(R, W[:, i], v_m)

    # Estimator (EKF)

    # SLAM update robot position
    # takes all landmark-robot variances (suboptimal)
    x[r], J_r, J_n = move(x[r], u, np.zeros(2))
    P[r, :] = J_r.dot(P[r, :])
    P[:, r] = P[r, :].T
    P[r[:, np.newaxis], r] = J_r.dot(P[r, r]).dot(J_r.T) + J_n.dot(
        Q).dot(J_n.T)

    # SLAM landmark correction

    # find all landmarks indices in landmarks
    # lids -> landmarks -> x
    lids = np.where(landmarks[0, :])[0]

    for i in lids:
        l = landmarks[:, i]      # landmark pointer to x
        rl = np.hstack([r, l])   # robot and landmark pointer in x

        y, J_r, J_y = observe(x[r], x[l]) # measurement y = h(x)
        J_ry = np.hstack([J_r, J_y])      # expectation jacobian
            
        # meassurement inovation z with covariance Z
        z = Y[:, i] - y
        # angle correction
        z[1] = z[1] - 2*pi if z[1] > pi else z[1]
        z[1] = z[1] + 2*pi if z[1] < -pi else z[1]
        # inovation covariance with sensor noise
        Z = J_ry.dot(P[rl[:, np.newaxis], rl]).dot(J_ry.T) + S         
            
        # Kalman gain P*H'*Z^-1
        # when P (variability) large (low confidence) K also large
        # when robot static state P and K should go to 0
        try:
            K = P[rl[:, np.newaxis], rl].dot(J_ry.T).dot(np.linalg.inv(Z))
        except np.linalg.linalg.LinAlgError as e:
            print("landmarks K update fail: {}, Z={}".format(e, Z))
            continue
        else:
            # posteriori update
            x[rl] = x[rl] + K.dot(z)
            P[rl[:, np.newaxis], rl] = P[rl[:, np.newaxis], rl] - K.dot(
                Z).dot(K.T)


    # SLAM new landmarks
    for y in Y.T:
        # find free slots for landmark
        lids = np.where(landmarks[0, :]==0)[0] 
        if all(y!=0) and any(lids):
            # find random slot for landmark
            i = lids[np.random.randint(lids.size)] 
            l = np.where(mapspace==False)[0][:2]

            mapspace[l] = True  # reserve landmark in mapspace
            landmarks[:, i] = l # store landmark pointers to x

            x[l], J_r, J_y = inv_observe(R, y) # global landmark pose
            P[l, :] = J_r.dot(P[r, :])
            P[:, l] = P[l, :].T
            P[l[:, np.newaxis], l] = J_r.dot(
                P[r[:, np.newaxis], r]).dot(
                    J_r.T) + J_y.dot(S).dot(J_y.T)

# landmark value array
#import pdb;pdb.set_trace()
x_lms = x[landmarks[:, np.where(landmarks[0, :]!=0)[0]]].T 
plotting.plots(np.array(R_res), W, x_lms, Y)
