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
    """ transforms point [x, y] to frame [x, y, theta] with jacobians with respect
    to frame and point.
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
    """ Returns measurement! in polar coordinates [d, fi] to point p
    """
    p, J_r, J_p = transform_local(r, p)
    y, J_y = scan(p)            # take measurement to robot-local point

    J_yr = J_y.dot(J_r)            # chain with robot frame
    J_yp = J_y.dot(J_p)            # chain with point

    return y + v, J_yr, J_yp


def inv_observe(r, y):
    """ local measurement to global point
    """
    p_local, J_y = inv_scan(y)
    p, J_r, J_p = transform_global(r, p_local)

    J_y = J_p.dot(J_y)         # chain rule

    return p, J_r, J_y, 


def correction(i, r, x, P, Y, S, landmarks):
    """ SLAM
    """
    l = landmarks[:, i]               # landmark pointer to x
    rl = np.hstack([r, l])                     # robot and known landmark in x

    y, J_r, J_y = observe(x[r], x[l]) # get measurement y = h(x)
    E_rl = np.hstack([J_r, J_y])               # expectation jacobian
    # import pdb;pdb.set_trace()
    E = E_rl.dot(P[rl[:, None], rl]).dot(E_rl.T)

    y_i = Y[:, i]               # take one measurement from set
    
    # make inovation
    z = y_i - y
    # negative angles?, angle correction
    z[1] = z[1] - 2*pi if z[1] > pi else z[1]
    z[1] = z[1] + 2*pi if z[1] < -pi else z[1]
    Z = S + E                   # inovation covariance with noise

    # always correct
    # TODO mahalanobis distance (3 sigma) check
    # FIXME
    K = P[rl[:, None], rl].dot(E_rl.T).dot(np.linalg.inv(Z)) # Kalman gain P*H'*Z^-1

    # update
    # x = x + K * z;
    # P = P - K * Z * K';

    return x, P


def initialization(y, x, R, S, r, P, landmarks, mapspace):
    """ SLAM new landmark initialization
    """
    lids = np.where(landmarks[0, :]==0)[0]       # find free slots for landmark

    if all(y!=0) and any(lids):
        i = lids[np.random.randint(lids.size)]  # find random slot for landmark
        l = np.where(mapspace==False)[0][:2]    # find 

        mapspace[l] = True             # reserve landmark positions in mapspace
        landmarks[:, i] = l            # store landmark pointers to x

        x[l], J_r, J_y = inv_observe(R, y)         # global landmark pose

        P[l, :] = J_r.dot(P[r, :])
        P[:, l] = P[l, :].T
        P[l[:, None], l] = J_r.dot(P[r[:, None], r]).dot(J_r.T)# + J_y.dot(S).dot(J_y.T)

    return x, P, landmarks, mapspace



def simulation():
    R = np.array([4, 3, -pi/2]) # robot pose (x, y, th) (makes map anchor point)
    u = np.array([0, 0])        # control signal (distance, rotation)

    W = np.array([
        [-18, -69, -26, -81,   5,  19,   3, -35, -57, -73],
        [-80,  36,  41,  47, -58, -77, -90, -80, -91,  57]
    ])

    N = W.shape[1]

    Y = np.zeros([2, N])      # measurements of all landmarks in W at time
    
    x = np.zeros([R.size + W.size]) # state vector as map means
    P = np.zeros([x.size]*2)        # state covariance

    # system noise: Gaussian {0, Q}
    q = np.array([.01, .01])    # aplitute or standart deviation
    Q = np.diag(q**2)           # noise covarinace

    # measurement noise: Gaussian {0, S}
    s = np.array([.1, 1*pi/180]) # noise standart deviation
    S = np.diag(s**2)            # noise covarince

    # State index management
    mapspace = np.zeros([x.size], dtype=bool) # fill with false

    # Observed landmarks pointers to mapspace
    landmarks = np.zeros([2, N], dtype=int)

    # Place robot in map
    r = np.where(mapspace==False)[0][0:R.size] # takes first 3 positions
    mapspace[r] = True                         # block map positions
 
    # initialize robot states
    x[r] = R
    P[r,r] = 0                                            # initialize robot covariance
    R_res = []

    for t in np.arange(1, 100):   # main loop
        # simulate robot move
        n = q * np.random.random(2) # motion control noise

        R, _, _ = move(R, u, np.zeros(2)) # simulate move (no perturbation)   
        R_res.append(R) 

        # b. observation (with sensor from -pi/4 to pi/4 angle)
        for i in range(N):
            v_m = np.random.random(2) * 0.05      # measurement noise
            y, _, _ = observe(R, W[:, i], v_m)    # +/-2pi wide measurements with noise
            # simulate sensor range and angle
            p, fi = y
            if p < 100 and fi > -pi/4 and fi < pi/4:
                Y[:, i] = y     # measurements at time t

        # Estimator (EKF)

        # takes all landmark-robot variances (suboptimal)
        # use m instead : where m = landmarks[landmarks!=0)].T
        x[r], J_r, J_n = move(x[r], u, np.zeros(2))
        P[r, :] = J_r.dot(P[r, :])
        P[:, r] = P[r, :].T     # arba J_r.dot(P[:, r])
        P[r[:, None], r] = J_r.dot(P[r, r]).dot(J_r.T) + J_n.dot(Q).dot(J_n.T)

        # find all landmarks indices in landmarks
        # lids -> landmarks -> x
        for lid in np.where(landmarks[0, :])[0]:
            correction(lid, r, x, P, Y, S, landmarks)

        for y in Y.T:
            x, P, landmarks, mapspace = initialization(y, x, R, S, r, P, landmarks, mapspace)

        # print ("AFTER INIT")
        # print (landmarks)
        # print (mapspace)
        # print (x[np.where(mapspace==True)])
        
    plots(R_res, W, Y, P)

def plots(R_res, W, Y, P):
    R_res = np.array(R_res)
    plt.figure(1)
    
    plt.subplot(2, 1, 1)
    plt.grid(True)
    plt.xlim([-120, 120])
    plt.ylim([-120, 120])
    plt.plot(R_res[:, 0], R_res[:, 1], 'o',
             W[0, :], W[1, :], '*')

    plt.subplot(2, 1, 2, projection='polar')
    plt.grid(True)
    plt.polar(Y[1, :], Y[0, :], 'g.',
              [0, -pi/4], [0, 100], 'r-',
              [0, pi/4], [0, 100], 'r-')
    
    plt.figure(2)
    plt.pcolor(P)
    plt.colorbar()
    
    plt.show()

    
def animations(R_res):
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.xlim([-120, 120])
    plt.ylim([-120, 120])

    def animate(line, p):
        line.set_ydata(p[0])
        line.set_xdata(p[1])
        return line,

    # Init only required for blitting to give a clean slate.
    def anim_init(line):
        line.set_ydata(0)
        line.set_xdata(0)
        return line,

    line, = ax.plot(0, 0, 'o')
    ani = animation.FuncAnimation(fig,
                                  lambda p: animate(line, p),
                                  R_res,
                                  init_func=lambda: anim_init(line),
                                  interval=100,
                                  blit=True)

    ani.save('/tmp/Animation.gif',writer='imagemagick',fps=5);


simulation()
