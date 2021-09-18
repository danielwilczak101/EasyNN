from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import typing import Iterable, Iterator
from EasyNN.typing import ArrayND, Sample

if sys.version_info < (3, 9):
    class Iterator(Generic

def validate(data: Sample) -> None:
    """Type-check the training/testing/validation data."""
    if not isinstance(training, tuple):
        raise TypeError(f"training must be a tuple, not {type(training).__name__}")
    elif not all(isinstance(x, ArrayND) for x in training):
        raise TypeError("training must be a tuple of ND-Arrays")

def split(data: Sample, percent: float) -> tuple[Sample, Sample]:
    """Split the training data up to fill in for testing/validation data."""
    return = (
        tuple([x[:int(test_percent * len(training))] for x in training]),
        tuple([x[int(test_percent * len(training)):] for x in training]),
    )


class Dataset:
    """Class for storing datasets and looping over multiple arrays together."""
    _indexes: Iterator[Union[int, slice, tuple[int]]]
    data: Sample
    sample: Sample

    def __init__(self: Dataset, *data: ArrayND, indexes: Iterable[Union[int, slice, tuple[int, ...]]]) -> None:
        self.data = data
        self._indexes = iter(indexes)
        next(self)

    def __iter__(self: Dataset) -> Dataset:
        """Loops over the data in accordance to the indexes."""
        return self

    def __next__(self: Dataset) -> Sample:
        """
        Update and return the next sample.

        Example
        -------
        >>> sample = next(batch.testing)
        >>> assert sample == batch.testing.sample
        """
        i = next(self._indexes)
        self.sample = tuple([x[i] for i in self.data])
        return self.sample


class Batch(ABC):
    """
    Abstract Base Class for storing and iterating over batches.

    Datasets are also automatically split up into training, testing, and validation.
    They can be looped over, and the current sample 
    """
    training: Dataset
    testing: Dataset
    validation: Dataset

    def __init__(
        self: Batch,
        training: Sample,
        testing: Sample = None,
        validation: Sample = None,
        /,
        *,
        test_percent: float = 0.15,
        validate_percent: float = 0.15,
    ) -> None:
        # Type-check the training data.
        validate(training)
        # Shuffle the training data.
        if len(training) > 0:
            shuffle_indexes = np.random.permutation(len(training[0]))
            training = tuple([x[shuffle_indexes] for x in training])
        # Extract testing data from the training data if necessary.
        if testing is None:
            validate_percent /= 1 - test_percent
            testing, training = split(training, test_percent)
        # Type-check the testing data.
        validate(testing)
        # Extract validation data from the training data if necessary.
        if validation is None:
            validation, training = split(training, validate_percent)
        # Type-check the validation data.
        validate(validation)
        # Convert the data into datasets.
        self.training = Dataset(*training, indexes=self.__indexes__(training))
        self.testing = Dataset(*testing, indexes=self.__indexes__(testing))
        self.validation = Dataset(*validation, indexes=self.__indexes__(validation))

    def __iter__(self: Batch) -> Batch:
        """
        Loop through the training dataset.

        Example
        -------
        >>> for sample in batch:
        ...     print(sample)
        """
        return self

    def __next__(self: Batch) -> Sample:
        """
        Update the training sample.

        Example
        -------
        >>> sample = next(batch)
        >>> assert sample == batch.training.sample
        """
        return next(self.training)

    @abstractmethod
    def __indexes__(self: Batch, data: Sample) -> Union[int, slice, tuple[int, ...]]:
        """Compute indexes to loop over."""
        raise NotImplementedError
