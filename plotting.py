from matplotlib import pyplot as plt
import matplotlib.animation as animation

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


def plots(R_res, W, L, Y, P):
    fig1 = plt.figure(1)
    
    ax = fig1.add_subplot(1, 1, 1, autoscale_on=True)
    ax.grid(True)
    ax.plot(R_res[:, 0], R_res[:, 1], 'o',
             W[0, :], W[1, :], '*',
             L[:, 0], L[:, 1], '.')

    for i, l in enumerate(L):
        ax.annotate(i, xy=l)

    # ax = fig1.add_subplot(2, 1, 2, projection='polar')
    # ax.grid(True)
    # ax.plot(Y[1, :], Y[0, :], 'g.',
    #         [0, -pi/4], [0, 300], 'r-',
    #         [0, pi/4], [0, 300], 'r-')
    
    plt.figure(2)
    plt.pcolor(P)
    plt.colorbar()
    
    plt.show()
