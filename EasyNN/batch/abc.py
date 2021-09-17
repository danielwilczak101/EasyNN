from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Iterator
from EasyNN.typing import ArrayND


class Dataset:
    """Class for storing datasets and looping over multiple arrays together."""
    _indexes: Iterator[Union[int, slice, tuple[int]]]
    data: tuple[ArrayND, ...]

    def __init__(self: Dataset, *data: ArrayND, indexes: Iterator[Union[int, slice, tuple[int, ...]]]) -> None:
        self.data = data
        self._indexes = indexes

    def __iter__(self: Dataset) -> Iterator[tuple[ArrayND, ...]]:
        """Loops over the data in accordance to the indexes."""
        for i in self._indexes:
            yield tuple([x[i] for i in self.data])


class Batch(ABC):
    """
    Abstract Base Class for storing and iterating over batches.

    Datasets are also automatically split up into training, testing, and validation.
    """
    training: Dataset
    testing: Dataset
    validation: Dataset

    def set_data(
        self: Batch,
        training: tuple[ArrayND, ...],
        testing: tuple[ArrayND, ...] = None,
        validation: tuple[ArrayND, ...] = None,
        test_percent: float = 0.15,
        validate_percent: float = 0.15,
    ) -> None:
        # Type-check the training data.
        if not isinstance(training, tuple):
            raise TypeError(f"training must be a tuple, not {type(training).__name__}")
        elif not all(isinstance(x, ArrayND) for x in training):
            raise TypeError("training must be a tuple of numpy arrays")
        # Shuffle the training data.
        elif len(training) > 0:
            shuffle_indexes = np.random.permutation(len(training[0]))
            training = tuple([x[shuffle_indexes] for x in training])
        # Extract testing data from the training data if necessary.
        if testing is None:
            validate_percent /= 1 - test_percent
            testing = tuple([x[:int(test_percent * len(training))] for x in training])
            training = tuple([x[int(test_percent * len(training)):] for x in training])
        # Type-check the testing data.
        elif not isinstance(testing, tuple):
            raise TypeError(f"testing must be a tuple, not {type(testing).__name__}")
        elif not all(isinstance(x, ArrayND) for x in testing):
            raise TypeError("testing must be a tuple of numpy arrays")
        # Extract validation data from the training data if necessary.
        if validation is None:
            validation = tuple([x[:int(validate_percent * len(training))] for x in training])
            training = tuple([x[int(validate_percent * len(training)):] for x in training])
        # Type-check the validation data.
        elif not isinstance(validation, tuple):
            raise TypeError(f"validation must be a tuple, not {type(validation).__name__}")
        elif not all(isinstance(x, ArrayND) for x in validation):
            raise TypeError("testing must be a tuple of numpy arrays")
        # Convert the data into datasets.
        self.training = Dataset(*training, indexes=self.__indexes__(training))
        self.testing = Dataset(*testing, indexes=self.__indexes__(testing))
        self.validation = Dataset(*validation, indexes=self.__indexes__(validation))

    def __iter__(self: Batch) -> Iterator[tuple[ArrayND, ...]]:
        """Loop through the training dataset repeatedly."""
        return iter(self.training)

    @abstractmethod
    def __indexes__(self: Batch, data: ArrayND) -> Union[int, slice, tuple[int, ...]]:
        """Compute indexes to loop over."""
        raise NotImplementedError
