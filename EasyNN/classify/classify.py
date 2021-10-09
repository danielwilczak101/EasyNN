import numpy as np
from typing import TypeVar
from EasyNN.typing import ArrayND

ArrayOut = TypeVar("ArrayOut")
LabelType = TypeVar("LabelType")


class Classify:
    """Used for giving users the ability to tell accuracy and classify images."""
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

        # Take in validation dataset -> x,y

            # For loop throw validation data
                # probabilities = model(x)
                # if argmax(probabilities) == y
                    # +1 correct
        # (correct / len(validation data)) 

        # sparse y = [3, 6, 0, 2, 4]
        # dense y  = [[0, 0, 0, 1, 0, 0, 0], ...]


        #if y_pred.ndim == y.ndim:
        pass

        

    def classify(self, y_pred: ArrayOut) -> LabelType:
        """
        Takes the labels and prediction from the model and 
        
        """
        pass

        