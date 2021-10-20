from dataclasses import dataclass

@dataclass
class printer:
    iteration: bool = False
    training_loss: bool = False
    validation_loss: bool = False
    testing_loss: bool = False
    training_accuracy: bool = False
    validation_accuracy: bool = False
    testing_accuracy: bool = False
    frequency: int = 1
    i: int = -1

    def __call__(self, model) -> None:
        # Increment the counter and if it doesn't match the frequency,
        # then just skip this callback.
        self.i += 1
        if self.i % self.frequency != 0:
            return
        # For the various variables, print the results.
        if self.iteration:
            print(f"{  model.training.iteration = }")
        if self.training_loss:
            print(f"{    model.loss(*model.training.sample)   = }")
        if self.validation_loss:
            print(f"{    model.loss(*model.validation.sample) = }")
        if self.testing_loss:
            print(f"{    model.loss(*model.testing.sample)    = }")
        if self.training_accuracy:
            print(f"{      model.accuracy(*model.training.sample)   = }")
        if self.validation_accuracy:
            print(f"{      model.accuracy(*model.validation.sample) = }")
        if self.testing_accuracy:
            print(f"{      model.accuracy(*model.testing.sample)    = }")