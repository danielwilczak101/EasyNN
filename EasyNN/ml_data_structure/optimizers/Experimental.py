from typing import Optional
from numpy import sqrt
from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.Point import Point
from EasyNN.ml_data_structure.Tensor import TensorLike
from EasyNN.ml_data_structure.Momentum import Momentum

def norm(vector: TensorLike) -> float:
    return sqrt(sum(x*x for x in vector))

def normalize(vector: TensorLike, epsilon: float) -> TensorLike:
    return vector / (norm(vector) + epsilon)

def dot(v1: TensorLike, v2: TensorLike) -> float:
    return sum(x1 * x2 for x1, x2 in zip(v1, v2))

def project(v1: TensorLike, v2: TensorLike) -> TensorLike:
    return dot(v1, v2) * v2


class Experimental(Optimizer):
    """
    Experimental optimizer by using a modification of momentum
    by increasing the component which aligns with the gradient.
    """
    learning_rate: float
    momentum: Momentum
    alignment_rate: float
    epsilon: float

    def __init__(
            self,
            learning_rate: float = 0.1,
            momentum_rate: float = 0.99,
            alignment_rate: float = 3,
            epsilon: float = 1e-8,
        ):
        self.learning_rate = learning_rate
        self.momentum = Momentum(rate=momentum_rate)
        self.alignment_rate = alignment_rate - 1
        self.epsilon = epsilon

    def update(self, iteration: int, items: Point):
        gradient = items.derivatives
        momentum = self.momentum(gradient)
        normalized_momentum = normalize(momentum, self.epsilon)
        normalized_gradient = normalize(gradient, self.epsilon)
        alignment = normalize(normalized_momentum + normalized_gradient, self.epsilon)
        items.values -= self.learning_rate * (momentum + self.alignment_rate * project(momentum, alignment))
