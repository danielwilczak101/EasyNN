import numpy as np
from typing import TypeVar
from EasyNN.typing import ArrayND

ArrayOut = TypeVar("ArrayOut")
LabelType = TypeVar("LabelType")


class Classifier:
    """Used for giving users the ability to tell model accuracy using
     validation data from the dataset. It also and classify images using
      the predictions and labels."""

    labels: list[LabelType]
    
    def accuracy(self, y_pred: ArrayOut, y: ArrayOut)-> float:
        """
        Used to tell the user how accurate the model is using a portion
        of the dataset that the model has never seen.

        Args:
            validation_data: A batch of validation_data (data,label) used to check the model.
        Return:
            Return a float that is used to establish how good the model is.
        Example:
            >>> model.accuracy(dataset)
            0.92
        """

        correct = 0

        for predictions,label in zip(y_pred,y):
            if np.argmax(predictions) == label:
                correct += 1
        
        return correct / len(y_pred)

        

    def classify(self, y_pred: ArrayOut, labels) -> LabelType:
        """
        Takes the labels and prediction from the model and 
        
        """
        return labels[np.argmax(y_pred)]

        