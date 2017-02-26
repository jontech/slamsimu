from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import numpy as np

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

def plot_covariance(self, P):
    plt.figure(2)
    plt.pcolor(P)
    plt.colorbar()
    plt.show()

def plots(R_res, W, Y, state, title="N/A"):
    
    L = state.x[state.all_landmarks]

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1, 1, 1)
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

    ax.set_title(title)

    plt.xlim((-100, 600))
    plt.ylim((-100, 600))
    plt.show()
