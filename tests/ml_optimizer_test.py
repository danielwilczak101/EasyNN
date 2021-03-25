import sys
sys.path.append("../")

from typing import Optional, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation, rc, cm
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
from EasyNN.ml_data_structure.Point import Point
from EasyNN.ml_data_structure.MachineLearning import MachineLearning as ML
from EasyNN.ml_data_structure.optimizers.GradientDescent import GradientDescent as GD

# matplotlib.use('Qt4Agg')

rc('animation', html='html5')

def plot_3D_test(
        *,
        file_name: str,
        func: Callable[[float, float], float],
        derivative: Callable[[float, float], Tuple[float, float]],
        iterations: int = 10000,
        learning_rate: float = 3e-3,
        bounds: Tuple[float, float] = (-10, +10),
        initial_point: Optional[Tuple[float, float]] = None,
        frames: int = 25,
    ):
    """
    Runs and plots the result of a machine learning algorithm in 3D.

    @params:
        file_name: name of file to be saved (without the .gif at the end).
        frames: the number of frames used for the gif (approximately).
        func: function to be optimized, taking 2 inputs and giving one output.
        derivative: the gradient of the func, taking 2 inputs and giving two outputs.
        iterations: the number of iterations run.
        learning_rate: the learning rate used in gradient descent.
        bounds: bounds used to pick a randomized initial point.
        initial_point: an initial point to be used instead of the bounds.
    """
    # initialize the first point if not given
    if initial_point is None:
        initial_point = np.random.uniform(bounds, (2,))

    # construct values/derivatives tensor
    tensor = np.zeros([2, 2])
    tensor[0] = initial_point

    # save to ML object
    ml = ML(Point(tensor), GD(learning_rate))

    # store the XYZ points
    X = [initial_point[0]]
    Y = [initial_point[1]]
    Z = [func(X[0], Y[0])]

    # update and save points over the given amount of iterations
    for _ in range(iterations):
        ml.derivatives = derivative(*ml.values)
        ml.optimize()
        X.append(ml.values[0])
        Y.append(ml.values[1])
        Z.append(func(*ml.values))

    # convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # get bounds
    dX = abs(X[0] - X[-1])
    dY = abs(Y[0] - Y[-1])
    Xbounds = X[-1] - 2*dX, X[-1] + 2*dX
    Ybounds = Y[-1] - 2*dY, Y[-1] + 2*dY

    # compute points
    funcX = np.linspace(*Xbounds, 100)
    funcY = np.linspace(*Ybounds, 100)
    funcZ = np.array([
        func(x, y)
        for y in funcY
        for x in funcX
    ])

    # only keep points where funZ < max(Z) + margin
    valid_points = funcZ < Z.max() + 0.1*Z.max()**2

    # convert X and Y to meshgrid
    funcX, funcY = np.meshgrid(funcX, funcY)
    funcX = funcX.flatten()[valid_points]
    funcY = funcY.flatten()[valid_points]
    funcZ = funcZ[valid_points]

    # make figure, plot, and line
    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca(projection='3d')
    ax.set_xlim(Xbounds)
    ax.set_ylim(Ybounds)
    ax.set_zlim(min(funcZ), max(Z))

    # set labels and viewing angle
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(60, 35)

    # use trisurf plot with custom transparency
    theCM = cm.get_cmap()
    theCM._init()
    theCM._lut[:-3,-1] = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    ax.plot_trisurf(funcX, funcY, funcZ, cmap=theCM, linewidth=0.1)

    line, = ax.plot3D(np.array([]), [], [], 'k-')

    # print the results as a table
    print(tabulate(zip(X, Y, Z), headers='XYZ', showindex='always'))

    # only use some of the iterations for the animation
    frames = len(X) // frames
    X, Y, Z = X[::frames], Y[::frames], Z[::frames]

    # animation functions
    def init():
        line.set_xdata(np.array([]))
        line.set_ydata([])
        line.set_3d_properties([])
        return line,
    def animate(i):
        line.set_xdata(X[:i+1])
        line.set_ydata(Y[:i+1])
        line.set_3d_properties(Z[:i+1])
        plt.title(f"Iteration: {i*frames}\nfunc({X[i]:.3}, {Y[i]:.3}) = {Z[i]:.3}")
        return line,

    # save the animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(X), interval=20, blit=True)
    anim.save(f'{file_name}.gif', writer=animation.ImageMagickFileWriter())
