from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any


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


@dataclass
class Print:
    settings: dict[str, list[list]] = field(default_factory=lambda: defaultdict(list))

    def get_attribute(self, model, attribute: str) -> str:
        if attribute == "iteration":
            return f"Iteration: {model.training.iteration}"
        elif attribute == "validation_accuracy":
            return f"Validation Accuracy: {model.accuracy(*model.validation.sample)}"
        else:
            raise NotImplementedError
    
    def callback(self, model, setting, translation, frequency: int=1, **kwargs) -> None:
        if model is None:
            setting.append([0, frequency, [translation.get(attribute, attribute) for attribute in kwargs]])
            return
        for line in setting:
            if line[0] == 0:
                print(*[self.get_attribute(model, attribute) for attribute in line[2]], sep=", ")
            line[0] += 1
            line[0] %= line[1]

    # -------------- Training -------------- #

    def on_training_start(self, model=None, **kwargs) -> None:
        # For `model.print.on_optimization_start(validation_accuracy=True)`.
        self.callback(
            model,
            self.settings["on_training_start"],
            {"accuracy": "training_accuracy", "loss": "training_loss"},
            **kwargs
        )


    def on_training_end(self, model=None, **kwargs) -> None:
        # For `model.print.on_optimization_start(validation_accuracy=True)`.
        self.callback(
            model,
            self.settings["on_training_end"],
            {"accuracy": "training_accuracy", "loss": "training_loss"},
            **kwargs
        )
    # -------------- Validation -------------- #
    
    def on_validation_start(self, model=None, **kwargs) -> None:
        self.callback(
            model,
            self.settings["on_validation_start"],
            {"accuracy": "validation_accuracy", "loss": "validation_loss"},
            **kwargs
        )

    def on_validation_end(self, model=None, **kwargs) -> None:
        self.callback(
            model,
            self.settings["on_validation_end"],
            {"accuracy": "validation_accuracy", "loss": "validation_loss"},
            **kwargs
        )
    # -------------- Optimization -------------- #

    def on_optimization_start(self, model=None, **kwargs) -> None:
        # Doesnt print anything right now
        self.callback(
            model,
            self.settings["on_optimization_start"],
            {},
            **kwargs
        )

    def on_optimization_end(self, model=None, **kwargs) -> None:
        # Doesnt print anything right now
        self.callback(
            model,
            self.settings["on_optimization_end"],
            {},
            **kwargs
        )