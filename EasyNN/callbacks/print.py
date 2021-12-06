from dataclasses import dataclass, field
from collections import defaultdict


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