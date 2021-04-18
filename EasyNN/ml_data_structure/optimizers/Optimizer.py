"""Interface for Optimizer class."""


class Optimizer:
    """Base Optimizer class for machine learning algorithms."""

    def update(self, model: "Model") -> None:
        """
        Updates items.values using other parameters.

        Parameters
        ----------
        model : Model
            The model to be optimized.
        model.values : np.ndarray
            The parameter values being optimized.
        model.derivatives : np.ndarray
            The parameter derivatives being optimized.

        Raises
        ------
        NotImplementedError : This method needs to be implemented by subclasses.
        """
        raise NotImplementedError("Requires Optimizer.update(...) to be implemented by a subclass.")
