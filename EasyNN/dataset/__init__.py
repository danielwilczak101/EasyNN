from __future__ import annotations
from typing import Any, Generic, IO, Iterator, TypeVar, Union
from EasyNN.utilities.download import download
from EasyNN.utilities.data.load import load
from EasyNN.typing import ArrayND
import EasyNN.batch.abc

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
    _batch: EasyNN.batch.abc.Batch
    _batch_size: int
    _data: tuple[ArrayIn, ArrayOut]  # Don't save
    percent: float = 0.05
    file: Union[str, IO]
    url: str
    sample: tuple[ArrayIn, ArrayOut]
    samples: Iterator[tuple[ArrayIn, ArrayOut]]  # Don't save
    iteration: int = -1

    def __getstate__(self) -> dict[str, Any]:
        return {
            name.removeprefix("_") if name != "_batch_size" else "_batch_size": attribute
            for name, attribute in vars(self).items()
            if name not in ("_data", "sample", "samples")
        }

    def __getitem__(self, indexes: Union[int, tuple[int, ...]]) -> tuple[ArrayIn, ArrayOut]:
        """Get a sample from the data."""
        return tuple([x[indexes] for x in self.data])

    def __iter__(self):
        """Loops over the data in accordance to the indexes."""
        return self

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.data[0])

    def __next__(self) -> tuple[ArrayIn, ArrayOut]:
        """Update and return the next sample."""
        # Each sample corresponds to 1 iteration.
        self.iteration += 1
        self.sample = next(self.samples)
        return self.sample

    @property
    def epoch(self) -> int:
        """The epoch is how many times the whole dataset has been looped through."""
        return self.iteration * self._batch_size // len(self)

    @property
    def batch(self) -> EasyNN.batch.abc.Batch:
        """When setting the batch, get the samples using the batch."""
        return self._batch

    @batch.setter
    def batch(self, batch: EasyNN.batch.abc.Batch) -> None:
        self._batch = batch
        self._batch_size = batch.size
        self.samples = batch.generate_samples(self)

    @property
    def data(self) -> tuple[ArrayIn, ArrayOut]:
        """
        The dataset's data. If the data is not found, this first checks
        for a file storing the data, then a url to load the data from.
        """
        return self._data

    @data.setter
    def data(self, data: tuple[ArrayIn, ArrayOut]) -> None:
        # TODO: Maybe some input checking?
        self._data = data
