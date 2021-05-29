from __future__ import annotations
from typing import Any, Dict
from dataclasses import dataclass


#=====================================#
# !! See bottom for some examples. !! #
#=====================================#


class Handler:
    """
    Creates a handle object which can be used a decorator to collect
    objects and allows indexed referencing by names and default options.

    Note
    ----
    When the default is used, it is called with no arguments and the
    returned value is not saved.

    Usage:
    - Use the @handler decorator on functions and classes.
      - Use @handler(name=...) to assign a specific name.
        - By default, the name is the object's .name attribute (if it has one) or .__name__ attribute.
          - Some name must be specified.
    - Set a default case by using name='_default'.
      - The default must be callable.
    - Use handler['name'] to retrieve an object.
    """
    _objs: Dict[str, Any]

    def __init__(self: Handler) -> None:
        """Initialize Handler with no objects."""
        self._objs = dict()

    def __call__(self: Handler, *obj: Any, name: Optional[str] = None) -> Any:
        """
        Decorator for adding a function to the functions.

        Parameters
        ----------
        obj : Any
            An object being added to the handler.
        name : str = obj.__name__
            The name of the object (checks __name__ by default).

        Returns
        -------
        obj : Any
            The input is returned unchanged.

        Raises
        ------
        ValueError("Handler() takes 1 string kwarg 'name' but 'name' was given")
            Non-string name given.
        ValueError("no name given and no attributes 'name' or '__name__' found")
            No name given and no name found on the object.
        ValueError(f"duplicate name 'name' was given, use another name instead")
            An already used name is being assigned.
        ValueError("The '_default' object must be callable with no args")
            Trying to assign to '_default' a non-callable.
        """
        # require valid name type
        if name is not None and not isinstance(name, str):
            raise ValueError(f"Handler() takes 1 string kwarg 'name' but {name} was given")
        # using handler(name=...) without the object yet
        elif len(obj) == 0:
            return lambda obj: self(obj, name=name)
        # passing in an object
        elif len(obj) == 1:
            # using default name
            if name is None:
                # try to use an assigned name from the object
                if hasattr(obj[0], "handler_name"):
                    name = obj[0].handler_name
                # try to use the given __name__ from functions and classes
                elif hasattr(obj[0], "__name__"):
                    name = obj[0].__name__
                # no name found
                else:
                    raise ValueError("no name given and no attributes 'name' or '__name__' found")
            # name isn't a string
            if not isinstance(name, str):
                raise ValueError(f"Handler() takes 1 string kwarg 'name' but {name} was given")
            # name is already taken
            elif name in self._objs:
                raise ValueError(f"duplicate name '{name}' was given, use another name instead")
            # name is the default, check if callable
            elif name == "_default" and not callable(obj[0]):
                raise ValueError("The '_default' object must be callable with no args")
            # try to assign the new name onto the object for future consistent reference
            try:
                obj.handler_name = name
            except Exception:
                pass
            # store and return the object
            self._objs[name] = obj[0]
            return obj[0]
        # passing in too many objects
        else:
            raise TypeError(f"Handler() takes 0 or 1 positional arguments but {len(obj)} was given")

    def __getitem__(self: Handler, name: str) -> Any:
        """
        Call function by name by indexing.

        Parameters
        ----------
        name : str
            The name of the requested function.

        Returns
        -------
        obj : Any
            The named object is returned, or the '_default' if none found.
            Note : If the default is used, it is called with no arguments,
                   and the result is not saved.

        Raises
        ------
        KeyError(name)
            Neither the name nor a "_default" was found.
        """
        # the given name is found
        if name in self._objs:
            return self._objs[name]
        # default case
        elif "_default" in self._objs:
            return self._objs["_default"]()
        # no object found
        else:
            raise KeyError(name)
