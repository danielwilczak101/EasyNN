from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator
from EasyNN._abc import AutoDocumentation
from EasyNN.dataset import Dataset
from EasyNN.typing import Sample


class Batch(AutoDocumentation, ABC):
    """
    Abstract Base Class for generating samples to loop over from the dataset.
    """
    size: int

    @abstractmethod
    def generate_samples(self: Batch, dataset: Dataset) -> Iterator[Sample]:
        """Generates samples to loop over."""
        raise NotImplementedError
