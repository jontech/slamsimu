from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from math import pi
from scipy.spatial import distance


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
    ani = animation.FuncAnimation(
        fig,
        lambda p: animate(line, p),
        R_res,
        init_func=lambda: anim_init(line),
        interval=100,
        blit=True)
    
    ani.save('/tmp/Animation.gif',writer='imagemagick',fps=5);


def make_ellip(l, P, n=16, sigma=1):
    """
    
    """
    alpha = 2*pi/(n-2)*np.arange(1, n)
    circle = np.array([np.cos(alpha), np.sin(alpha)])

    # svd decompose covariance
    U, s, Vh = np.linalg.svd(P, full_matrices=True)
    V = Vh.T
    d = np.sqrt(s)
    b = np.identity(2) * d

    # build ellipse: n-sigma, rotate, align circle
    ellip = np.dot(sigma, V).dot(b.T).dot(circle)

    # center to point l
    X = l[0] + ellip[0,:]
    Y = l[1] + ellip[1,:]

    return X, Y


def sim_plots(res, W, params):
    states = list(map(lambda r: r[1], res))
    state = states[-1:][0]

    R = np.array(list(map(lambda r: r[0], res)))
    L = np.array(list(map(lambda i_L: state.x[i_L[1]], state.slots)))
    R_ekf = np.array(list(map(lambda s: s.R, states)))
    L_dist = np.array(list(map(lambda s: (s[0], distance.euclidean(
        state.x[s[1]], W[:, s[0]])), state.slots)))

    gs = GridSpec(3, 3)

    fig1 = plt.figure(1, figsize=(10, 10),)
    fig1.suptitle("EKF-SLAM simulation steps={steps}".format(**params), fontsize=20)
    fig1.hspace = 30

    ax = fig1.add_subplot(gs[:-1, :])
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("q={q}, s={s}".format(**params))

    ax.plot(
        R[:, 0], R[:, 1], 'o',
        W[0, :], W[1, :], '.',
        L[:, 0], L[:, 1], '+',
        R_ekf[:, 0], R_ekf[:, 1], 'o',
    )

    def annotate_landmarks():
        for i, w in enumerate(W.T):
            ax.annotate(i, xy=w)

        for i, l in state.slots:
            ax.annotate(i, xy=state.x[l])
            L = state.x[l]
            X, Y = make_ellip(L, state.P_l(i), sigma=3)
            ax.plot(X, Y, 'b')
        return ax
    ax = annotate_landmarks()

    def annotate_robot():
        for state in states:
            X, Y = make_ellip(state.R, state.P_r_pos, sigma=3)
            ax.plot(X, Y, 'r')
        return ax
    ax = annotate_robot()

    plt.xlim((-100, 600))
    plt.ylim((-100, 600))

    def plot_covariance():
        plt.subplot(gs[-1, :-1])
        plt.pcolor(state.P)
        plt.colorbar()
        plt.grid(True)
        plt.title("P")
        plt.xlabel("[R M]")
        plt.ylabel("[R M]")

    def plot_landmark_dist():
        plt.subplot(gs[-1, -1])
        markerline, stemlines, baseline = plt.stem(L_dist[:, 0], L_dist[:, 1])
        plt.setp(baseline, 'color', 'r', 'linewidth', 2)
        plt.grid(True)
        plt.title("dist(L, w)")
        plt.ylabel("dist")
        plt.xlabel("index")

    plot_covariance()
    plot_landmark_dist()

    plt.subplots_adjust(hspace=0.3)
    plt.show()
