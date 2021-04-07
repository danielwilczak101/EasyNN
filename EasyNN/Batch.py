import numpy as np
from numpy.typing import ArrayLike
from typing import Iterable


class Batch:
    """Base batch class, uses the entire batch at a time."""

    def __call__(self, size: int) -> Iterable[ArrayLike]:
        """
        Generates the original batch as the only batch.

        Parameters
        ----------
        size : int
            The amount inputs and outputs.

        Returns
        -------
        Iterable[ArrayLike]
            Yields all of the indexes.
        """
        yield np.arange(size)


class MiniBatch(Batch):
    """Mini-batch class, use chunks of the batch at a time."""
    batch_size: int

    def __init__(self, batch_size: int = 8) -> int:
        """Initialize the batch size."""
        self.batch_size = batch_size

    def __call__(self, size: int, remainder: bool = True) -> Iterable[ArrayLike]:
        """
        Generates mini-batches at a time.

        Parameters
        ----------
        size : int
            The amount inputs and outputs.
        remainder : bool = True
            Determines if the remaining data (not forming a full batch) is also used at the end.

        Returns
        -------
        Iterable[ArrayLike]
            Yields the indexes of each mini-batch.
        """
        indexes = np.arange(size)
        np.random.shuffle(indexes)

        if remainder:
            last_index = size
        else:
            last_index = size - size % self.batch_size

        for i in range(0, last_index, self.batch_size):
            yield indexes[i:i+self.batch_size]


class StochasticBatch(Batch):
    """Stochastic gradient descent only uses one data element at a time."""

    def __call__(self, size: int) -> Iterable[ArrayLike]:
        """
        Generates batches of one item at a time in a random order.

        Parameters
        ----------
        size : int
            The amount inputs and outputs.

        Returns
        -------
        Iterable[ArrayLike]
            Yields the indexes individually.
        """
        indexes = np.arange(size)
        np.random.shuffle(indexes)
        for i in indexes:
            yield [i]
