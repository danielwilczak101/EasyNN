from __future__ import annotations
from typing import Generic, IO, Iterator, NoReturn, TypeVar, Union
from EasyNN.typing import ArrayND

ArrayIn = TypeVar("ArrayIn", bound=ArrayND)
ArrayOut = TypeVar("ArrayOut", bound=ArrayND)


class Dataset(Generic[ArrayIn, ArrayOut]):
    """
    Class for storing a dataset and looping over multiple arrays together.

    Example:
        >>> x = np.array([...])
        >>> y = np.array([...])
        >>> training = Dataset()
        >>> training.data = (x, y)
        >>> assert training[0] == (x[0], y[0])
    """
    _batch_size: int
    _data: tuple[ArrayIn, ArrayOut]
    file: Union[str, IO]
    url: str
    sample: tuple[ArrayIn, ArrayOut]
    samples: Iterator[tuple[ArrayIn, ArrayOut]]
    iteration: int = -1

    def __getitem__(self: Dataset[ArrayIn, ArrayOut], indexes: Union[int, tuple[int, ...]]) -> tuple[ArrayIn, ArrayOut]:
        """Get a sample from the data."""
        return tuple([x[indexes] for x in self.data])

    def __iter__(self: Dataset[ArrayIn, ArrayOut]) -> Dataset[ArrayIn, ArrayOut]:
        """Loops over the data in accordance to the indexes."""
        return self

    def __len__(self: Dataset[ArrayIn, ArrayOut]) -> int:
        """Returns the size of the dataset."""
        return len(self.data[0])

    def __next__(self: Dataset[ArrayIn, ArrayOut]) -> tuple[ArrayIn, ArrayOut]:
        """Update and return the next sample."""
        # Each sample corresponds to 1 iteration.
        self.iteration += 1
        self.sample = next(self.samples)
        return self.sample

    @property
    def epoch(self: Dataset[ArrayIn, ArrayOut]) -> int:
        """The epoch is how many times the whole dataset has been looped through."""
        return self.iteration * self._batch_size // len(self)

    @property
    def batch(self: Dataset[ArrayIn, ArrayOut]) -> NoReturn:
        """When setting the batch, get the samples using the batch. Don't store the batch."""
        raise NotImplementedError

    @batch.setter
    def batch(self: Dataset[ArrayIn, ArrayOut], batch: Batch) -> None:
        self.samples = batch.generate_samples(self)

    @property
    def data(self: Dataset[ArrayIn, ArrayOut]) -> tuple[ArrayIn, ArrayOut]:
        """
        The dataset's data. If the data is not found, this first checks
        for a file storing the data, then a url to load the data from.
        """
        # TODO: Dan implementation for using the file and url.
        return self._data

    @data.setter
    def data(self: Dataset[ArrayIn, ArrayOut], data: tuple[ArrayIn, ArrayOut]) -> None:
        # TODO: Maybe some input checking?
        self._data = data
