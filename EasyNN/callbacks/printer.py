from dataclasses import dataclass


@dataclass
class Printer:
    iteration: bool = False
    training_loss: bool = False
    validation_loss: bool = False
    testing_loss: bool = False
    training_accuracy: bool = False
    validation_accuracy: bool = False
    testing_accuracy: bool = False
    learning_rate: bool = False
    frequency: int = 1
    i: int = -1
    indent: int = 2
    start: str = ""
    end: str = "\n"

    def __call__(self, model) -> None:
        # Increment the counter and if it doesn't match the frequency,
        # then just skip this callback.
        self.i += 1
        if self.i % self.frequency != 0:
            return
        # For the various variables, print the results.
        if self.iteration:
            self.formatted_print(f"Iteration: {model.training.iteration}")
        if self.training_loss:
            self.formatted_print(f"Training Loss: {model.loss(*model.training.sample)}")
        if self.validation_loss:
            self.formatted_print(f"Validation Loss: {model.loss(*model.validation.sample)}")
        if self.testing_loss:
            self.formatted_print(f"Testing Loss: {model.loss(*model.testing.sample)}")
        if self.training_accuracy:
            self.formatted_print(f"Training Accuracy: {model.accuracy(*model.training.sample)}")
        if self.validation_accuracy:
           self.formatted_print(f"Validation Accuracy: {model.accuracy(*model.validation.sample)}")
        if self.testing_accuracy:
            self.formatted_print(f"Testing Accuracy: {model.accuracy(*model.testing.sample)}")
        if self.learning_rate:
            self.formatted_print(f"Training Rate: {model._optimizer_lr}")

    def formatted_print(self, message: str) -> None:
        print(" " * self.indent + self.start + message, end=self.end)