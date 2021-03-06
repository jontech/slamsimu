from math import atan2, pi, sqrt, sin, cos, atan2
import numpy as np
from time import sleep
from copy import deepcopy


np.set_printoptions(precision=3)
np.seterr(all='raise')



class State:
    r = np.array([0, 1, 2])     # robot pose index in x

    def __init__(self, R):
        self.r_n = self.r.size
        self.x = np.zeros([self.r_n])  # state vector as map means
        self.P = np.zeros([self.r_n]*2) # state covariance
        self.x[self.r] = R               # add robot pose
        self.slots = []                    # world-landmark mapping

    def landmarks(self, i):
        slots = filter(lambda s: s[0] == i, self.slots)
        return np.array(list(map(lambda s: s[1], slots)))

    def landmark(self, i):
        return self.landmarks(i)[0, :]

    def P_l(self, i):
        """get landmark cov matrix by landmark index"""
        l = self.landmark(i)
        return self.P[np.ix_(l, l)]

    @property
    def P_r(self):
        return self.P[np.ix_(self.r, self.r)]

    @property
    def P_r_pos(self):
        r_pos = self.r[0:2]
        return self.P[np.ix_(r_pos, r_pos)]

    def new_slot(self, i):
        self.x = np.pad(self.x, (0, 2), 'constant')
        self.P = np.pad(self.P, ((0, 2), (0, 2)), 'constant')
        n = self.x.shape[0]
        l = np.array([n-2, n-1])
        self.slots.append((i, l))
        return l

    @property
    def R(self):
        return self.x[self.r]


def zero_angles(a):
    """ Make angle ``a'' around zero
    """
    a = a - 2*pi if a > pi else a
    a = a + 2*pi if a < -pi else a
    return a


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
    x, y = p
    J_f = np.array([
        [1, 0, -y*cos(a_F) - x*sin(a_F)],
        [0, 1, x*cos(a_F) - y*sin(a_F)]
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


def move(r, u=np.array([0, 0])):
    """ Robot motion simulation and odometry
    r - current position
    u - control (distance, angle in radians), default not moving
    n - motion noise, default no noise
    returns new robot odometry-like pose
    """
    # change robot rotation a
    a = r[2] + u[1]
    a = zero_angles(a)

    # position increment from control signal in global frame,
    # we transform only x coordinate as vel of u = [vel, angle]
    t, J_f, J_p = transform_global(r, np.array([u[0], 0]))

    J_r = np.vstack([
        J_f,
        [0, 0, 1]
    ])

    J_n = np.vstack([
        np.vstack([J_p[:, 0], [0, 0]]),
        [0, 1]
    ])

    return np.hstack([t, a]), J_r, J_n


def scan(p):
    """ 2D point to polar measure with jacobian to p
    returns (distance, angle), jacobian
    """
    x, y = p
    d = sqrt(x**2 + y**2)
    fi = atan2(y, x)
    J_p = np.array([
        [x / sqrt(x**2 + y**2), y / sqrt(x**2 + y**2)],
        [-y / (x**2 + y**2),    x / (y**2 + x**2)]
    ])
    return np.array([d, fi]), J_p


def inv_scan(y):
    """ polar meassurement y to 2D point [x, y] with jacobian to p
    """
    d, fi = y
    p = np.array([d*cos(fi), d*sin(fi)])
    J_y = np.array([
        [cos(fi), -d*sin(fi)],
        [sin(fi), d*cos(fi)]
    ])
    return p, J_y


def observe(r, p):
    """ h(x) Returns measurement in polar coordinates [d, fi] to point p.
    v - meassurement noise
    """
    p, J_r, J_p = transform_local(r, p)
    y, J_y = scan(p)           # take measurement to robot-local point
    J_yr = J_y.dot(J_r)         # chain with robot frame
    J_yp = J_y.dot(J_p)         # chain with point
    return y, J_yr, J_yp


def inv_observe(r, y):
    """ local measurement to global point
    """
    p_local, J_y = inv_scan(y)
    p, J_r, J_p = transform_global(r, p_local)
    J_y = J_p.dot(J_y)          # chain rule
    return p, J_r, J_y, 


def observe_landmarks(W, R, s=np.array([0, 0])):
    """observation simulation, return messurements Y.
    """
    N = W.shape[1]               # world size
    Y = np.zeros([2, N])         # init observation measurements
    for i in range(N):
        y, _, _ = observe(R, W[:, i])
        v = s*np.random.randn(1, 2)[0]
        y = y + v
        # simulate sensor with angle and range
        p, fi = y
        Y[:, i] = y if p < 100 and fi > -pi/2 and fi < pi/2 else np.inf
    return Y


def landmark_correction(y, l, state, S):
    """ x, P correction by observed landmarks.
    """
    r = state.r
    x = state.x
    P = state.P
    rl = np.hstack([r, l])   # robot and landmark pointer in x
    
    x_y, J_r, J_y = observe(x[r], x[l]) # expectation measurement y = h(x)
    J_ry = np.hstack([J_r, J_y])      # expectation jacobian

    # meassurement inovation z with covariance Z
    z = y - x_y
    z[1] = zero_angles(z[1])
    # inovation expectation covariance with sensor noise covariance
    Z = J_ry.dot(P[np.ix_(rl, rl)]).dot(J_ry.T) + S

    # Kalman gain P*H'*Z^-1
    K = P[:, rl].dot(J_ry.T).dot(np.linalg.inv(Z))

    # posteriori update
    state.x = x + K.dot(z)
    state.P = P - K.dot(Z).dot(K.T)

    return state


def landmark_creation(y, i_y, state, S):
    """Add new landmarks to x 
    y - landmark measurement (d, phi)
    S - noise system covariance
    """
    l = state.new_slot(i_y)
    R = state.R
    P = state.P
    x = state.x
    r = state.r

    x[l], J_r, J_y = inv_observe(R, y) # global landmark pose

    P_rr = P[np.ix_(r, r)]
    P_lm = J_r.dot(P[r, :])
    P[l, :] = P_lm
    P[:, l] = P_lm.T
    P[np.ix_(l, l)] = J_r.dot(P_rr).dot(J_r.T) + J_y.dot(S).dot(J_y.T)

    state.x = x
    state.P = P

    return state


def update_robot(state, Q, x_r, J_r, J_n):
    r = state.r
    x = state.x
    P = state.P

    x[r] = x_r
    P_rr = P[np.ix_(r, r)]
    P_rm = J_r.dot(P[r, :])
    P[r, :] = P_rm
    P[:, r] = P_rm.T
    P[np.ix_(r, r)] = J_r.dot(P_rr).dot(J_r.T) + J_n.dot(Q).dot(J_n.T)

    state.x = x
    state.P = P

    return state


def registration_existing(state, y):
    for w_i, l in state.slots:
        y_x = observe(state.R, state.x[l])[0]
        z = y - y_x
        z[1] = zero_angles(z[1])
        P_l = state.P[np.ix_(l, l)]
        md = z.dot(np.linalg.inv(P_l)).dot(z)
        if md < 9:
            yield l
            

def run(W,
        steps=10,
        R=np.array([0, 0, 0]),
        u=np.array([0, 0]),
        q=np.array([.01, .01]),
        s=np.array([.05, 1*pi/180])):
    """ Runs SLAM simulation
    steps - simulation time steps number
    W - world as 2xn array
    R - initial robot pose array (x, y, th)
    u - robot control array (len/step th)
    q - move noise standart deviation as Gaussian {q, Q}
    s - landmark noise [dist noise cm, angle noise]
    yields - tuple of robot steps and estimator state
    """
    Q = np.diag(q**2)           # noise system cov
    S = np.diag(s**2)           # noise landmark cov

    state = State(R=R)

    # run simulation
    for t in np.arange(0, steps):
     
        # Simulation actual robot move and observe

        R, _, _ = move(R, u=u)
        Y = observe_landmarks(W, R, s=s)

        # Estimator (EKF)

        # robot move prediction
        n = q*np.random.randn(1, 2)[0]
        x_r, J_r, J_n = move(state.R, u=u+n)  # u odometry vector

        state = update_robot(deepcopy(state), Q, x_r, J_r, J_n)

        for i_lw, y in enumerate(Y.T):
            if all(y!=np.inf):
                landmarks = state.landmarks(i_lw)
                if len(landmarks) > 0:
                    for l in landmarks: 
                        # correct all similar landmarks to messurement y
                        state = landmark_correction(y, l, deepcopy(state), S)
                else:
                    # new landmarks integration
                    state = landmark_creation(y, i_lw, deepcopy(state), S)

        yield (R, state)
