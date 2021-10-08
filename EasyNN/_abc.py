"""
TODO: documentation.
"""
from __future__ import annotations
from abc import ABC
from typing import Any, Type


class AutoDocumentation(ABC):
    """
    Inherited by EasyNN.abc classes to automatically document implementation methods
    with the doc-string from the parent abstract method.
    """

    def __init_subclass__(cls: Type[AutoDocumentation], **kwargs: Any) -> None:
        """
        Initiallize subclasses by setting up the documentation for dunder methods
        which don't have any documentation by using the default documentation.
        """
        super().__init_subclass__(**kwargs)
        # Search for abstract methods.
        for super_cls in cls.mro():
            for name, attr in vars(super_cls).items():
                if getattr(attr, "__isabstractmethod__", False) and hasattr(cls, name) and getattr(cls, name).__doc__ is None:
                    getattr(cls, name).__doc__ = attr.__doc__
