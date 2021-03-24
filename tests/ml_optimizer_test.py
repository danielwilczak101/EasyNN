import sys
sys.path.append("../")
from typing import Optional, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tabulate import tabulate
from EasyNN.ml_data_structure.Point import Point
from EasyNN.ml_data_structure.MachineLearning import MachineLearning as ML
from EasyNN.ml_data_structure.optimizers.GradientDescent import GradientDescent as GD

rc('animation', html='html5')

def plot_3D_test(
        *,
        file_name: str,
        func: Callable[[float, float], float],
        derivative: Callable[[float, float], Tuple[float, float]],
        iterations: int = 1000,
        learning_rate: float = 0.1,
        bounds: Tuple[float, float] = (-10, +10),
        initial_point: Optional[Tuple[float, float]] = None,
    ):
    """
    Runs and plots the result of a machine learning algorithm in 3D.

    @params:
        file_name: name of file to be saved (without the .gif at the end).
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

    # make figure, plot, and line
    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(projection = '3d')
    line, = ax.plot([], [], 'bo-')

    # animation functions
    def init():
        line.set_xdata(np.array([]))
        line.set_ydata([])
        line.set_3d_properties([])
        return line,
    def animate(i):
        line.set_xdata(X[i])
        line.set_ydata(Y[i][1])
        line.set_3d_properties(Z[i])
        plt.title(f"Iteration: {i}\nfunc({X[i]}, {Y[i]}) = {Z[i]}")
        return line,

    # update and save points over the given amount of iterations
    for _ in range(iterations):
        ml.derivatives = derivative(*ml.values)
        ml.optimize()
        X.append(ml.values[0])
        Y.append(ml.values[1])
        Z.append(func(*ml.values))

    # print the results as a table
    print(tabulate(zip(X, Y, Z), headers='XYZ', showindex='always'))

    # save the animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=iterations, interval=20, blit=True)
    anim.save(f'{file_name}.gif', writer=animation.ImageMagickFileWriter())

# show usage when importing file
if __name__ == '__main__':
    help(plot_3D_test)
