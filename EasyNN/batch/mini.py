from __future__ import annotations
from itertools import chain
import numpy as np
from typing import Iterator
from EasyNN.batch.abc import Batch
from EasyNN.typing import ArrayND, Sample


class MiniBatch(Batch):
    """Produces mini-batches, which are multiple data points at a time."""
    size: int

    def __init__(self: MiniBatch, *args: Sample, size: int = 16, **kwargs: float) -> None:
        super().__init__(*args, **kwargs)
        self.size = size

    def __indexes__(self: MiniBatch, data: Sample) -> Iterator[tuple[int, ...]]:
        if len(data) == 0:
            return iter(tuple, None)
        indexes = chain.from_iterable(np.random.permutation(len(data[0])) for _ in iter(int, None))
        return zip(*[indexes] * self.size)
