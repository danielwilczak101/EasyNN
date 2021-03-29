from EasyNN.ml_data_structure.Tensor import TensorLike


class Momentum:
    """
    Stores a momentum approximation of a tensor.

    >>> import numpy as np
    >>> m = Momentum(rate=0.99, biased=False)
    >>> m(np.array([1, 2]))
    array([1., 2.])
    >>> m(np.array([2, 1]))
    array([1.50251256, 1.49748744])
    >>> m(np.array([2, 1]))
    array([1.67001111, 1.32998889])
    >>> m(np.array([2, 1]))
    array([1.75375616, 1.24624384])
    """
    _rate: float
    _value: TensorLike
    _biased: bool
    _bias: float

    def __init__(self, *, rate: float = 0.99, biased: bool = False):
        """
        Initialize the momentum with a rate and biased flag.

        rate: Determines how much momentum is kept.
              By default, 99% of the momentum is preserved.
        biased: Determines if the momentum should be biased towards 0.
                By default, this is disabled.

        Note: If a biased momentum is used, the returned value is the internal value,
              meaning it can be mutated, potentially resulting in unexpected behavior.
        """

        try:
            rate = float(rate)
        except Exception as e:
            raise TypeError("rate has to be a float")

        if 0 < rate < 1:
            self._rate = rate
            self._value = 0.0
            self._biased = biased
            self._bias = 0.0

        else:
            raise ValueError("rate has to be between 0 and 1")

    def __call__(self, value: TensorLike) -> TensorLike:
        """
        Update the momentum and return it.

        Use m = Momentum(...) to initialize the momentum.
        Use momentum = m(value) to update and get the momentum.
        """

        # update momentum
        factor = 1 - self._rate
        self._value += factor * (value - self._value)

        # return biased value
        if self._biased:
            return self._value

        # return unbiased value
        else:
            self._bias += factor * (1 - self._bias)
            return self._value / self._bias
