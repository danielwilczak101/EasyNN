"""Interface for Optimizer class."""

from Point import Point


class Optimizer:
    """Base Optimizer class for machine learning algorithms."""

    def update(self, iteration: int, items: Point):
        """Updates items.values using other parameters."""
        raise NotImplemented("Requires Optimizer.update(...) to be overriden.")
