from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np

from EasyNN.model.abc import Model
from EasyNN.utilities.momentum import Momentum


@dataclass
class Plotter:
    lr: float = 0.3
    name: str = "data"
    data: list = field(default_factory=list, init=False)

    @abstractmethod
    def x_values(self, model: Model) -> np.ndarray:
        """Rescales the iteration into another format (e.g. validation iteration, training iteration, or epoch)."""

    def on_testing_start(self, model: Model) -> None:
        x = self.x_values(model)
        y = self.data
        plt.plot(x, y, label=f"raw {self.name}")
        smoothener = Momentum(self.lr)
        y_smooth = [smoothener.update(data) for data in y]
        plt.plot(x, y_smooth, label=f"smooth {self.name}")
        plt.title(self.name.title())
        plt.legend(loc="lower left")

    def on_testing_end(self, model: Model) -> None:
        plt.show()
