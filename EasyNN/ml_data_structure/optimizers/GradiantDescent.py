import Optimizer

class GradiantDescent(Optimizer):
    """ Just your super simple GO STRAIGHT DOWN optimizer """
    
    def __init__(self, learningRate):
        """ requires a learning rate"""
        self.learningRate = learningRate

        
    def update(self, iteration: int, values: TensorLikem derivatives: TensorLike):

        # Im not sure if this is necessary
        
        for (value, derivative) in zip(values, derivatives):
            value += learningRate * derivative
