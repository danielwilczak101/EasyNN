from __future__ import annotations
from abc import ABC, abstractmethod
from EasyNN.model.abc import Model
from EasyNN.typing import ArrayND


class Loss(ABC):
    """
    TODO: documentation.
    """
    model: Model
    
