from __future__ import annotations
from itertools import chain
import numpy as np
from typing import Iterator
from EasyNN.batch.abc import Batch
from EasyNN.dataset import Dataset
from EasyNN.typing import Sample


class MiniBatch(Batch):
    """Produces mini-batches, which are multiple data points at a time."""
    size: int

    def __init__(self: MiniBatch, size: int = 32) -> None:
        self.size = size

    def generate_samples(self: MiniBatch, dataset: Dataset) -> Iterator[Sample]:
        # Loop over random indexes grouped together as the batch size.
        dataset._batch_size = self.size
        random_indexes = chain.from_iterable(np.random.permutation(len(dataset)) for _ in iter(int, None))
        grouped_indexes = zip(*[random_indexes] * self.size)
        return (dataset[i,] for i in grouped_indexes)
