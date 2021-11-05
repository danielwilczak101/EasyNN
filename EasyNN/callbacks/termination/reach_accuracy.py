from dataclasses import dataclass, field
from EasyNN.model.abc import Model


@dataclass
class ReachValidationAccuracy:
    """
    ReachValidationAccuracy(limit: float = 0.90, patience: int = 1)

    Callback used to terminate when the validation accuracy reaches a specified limit.

    Parameters:
        limit:
            The accuracy limit desired. By default, aims for 90% validation accuracy.
        patience:
            The number of times the accuracy limit must be reached in a row.
            Prevents "lucky" termination.

    Callbacks:
        on_validation_start:
            Checks the validation accuracy on this callback step.

    Example:
        Terminate when 80% validation accuracy is reached.
        >>> from EasyNN.examples.mnist.numbers.untrained import model
        >>> model.callback(ReachValidationAccuracy(limit=0.80))
        >>> model.train()
    """

    limit: float = 0.90
    patience: int = 1
    successes: int = field(default=0, init=False)

    def on_validation_start(self, model: Model) -> None:
        if model.accuracy(*model.validation.sample) > self.limit:
            self.successes += 1
        else:
            self.successes = 0
        model.stop_training |= self.successes >= self.patience
