from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from math import pi


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


def make_ellip(l, P_l, N=16, n=1):
    alpha = 2*pi/N*np.arange(0, N)
    circle = np.array([np.cos(alpha), np.sin(alpha)])
    U, S, Vh = np.linalg.svd(P_l)
    V = Vh.T
    d = np.sqrt(U)
    ellip = np.dot(n, V).dot(d).dot(circle)
    x = l[0] + ellip[0,:]
    y = l[1] + ellip[1,:]
    return x, y


def plots(R_res, state, W):
    
    L = state.x[state.all_landmarks]

    fig1 = plt.figure(1, figsize=(10, 4),)
    ax = fig1.add_subplot(1, 2, 1)
    ax.grid(True)

    ax.plot(
        R_res[:, 0], R_res[:, 1], 'o',
        W[0, :], W[1, :], '*',
        L[:, 0], L[:, 1], '.'
    )

    for i, w in enumerate(W.T):
        ax.annotate(i, xy=w)

    for i, l in enumerate(L):
        ax.annotate(i, xy=l)
        #ax.plot(make_ellip(l, state.P_l(i)))

    ax.set_title("EKF SLAM simulation. Steps={}".format(R_res.shape[0]))

    plt.xlim((-100, 600))
    plt.ylim((-100, 600))

    # covariance
    plt.subplot(1, 2, 2)
    plt.pcolor(state.P)
    plt.colorbar()

    plt.show()
