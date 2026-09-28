import abc
import numpy as np


class Evaluator(abc.ABC):
    r"""
    Abstract class for evaluating the model.
    """

    @abc.abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray) -> dict:
        """
        Evaluate the model.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted target values.

        Returns:
            Dict: The evaluation metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def print_metrics(self):
        """
        Print the calculated metrics.
        """
        raise NotImplementedError
