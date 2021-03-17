from Optimizer import Optimizer


class Optimizer:

    def __init__(self, learningRate=1.0, decay=0.0, epsilon=1e-7):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.epsilon = epsilon


    def preUpdateParams(self, iteration):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update(self, iteration: int, items: Point):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * iteration))
        if not hasattr(items, 'cache'):
            items.cache = np.zeros_like(items.values)

        items.cache += items.derivatives ** 2

        items.values += -self.current_learning_rate * items.derivatives / (np.sqrt(items.cache) + self.epsilon)
        
