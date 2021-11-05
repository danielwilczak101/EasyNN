from typing import Protocol


class Callback(Protocol):

    def on_optimization_start(self, model):
        pass

    def on_optimization_end(self, model):
        pass

    def on_training_start(self, model):
        pass

    def on_training_end(self, model):
        pass

    def on_validation_start(self, model):
        pass

    def on_validation_end(self, model):
        pass

    def on_testing_start(self, model):
        pass

    def on_testing_end(self, model):
        pass

    def on_epoch_start(self, model):
        pass

    def on_epoch_end(self, model):
        pass