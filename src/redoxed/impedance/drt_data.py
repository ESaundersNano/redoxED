# from redoxed.impedance.eis_data import EISData

import pandas as pd

import numpy as np

from nanodrt.dataobjects.dataobject import EISDataObject
from nanodrt.calculators import ImpedanceCalculator
from nanodrt.radialbasisfunctions import GaussianRBF
from nanodrt.optimizers import ImpedanceOptimizer
import jax.numpy as jnp


class DRTData:
    """
    A class to store and process Distribution of Relaxation Times (DRT) data.

    Attributes:
        tau (np.ndarray): Array of relaxation times.
        gamma (np.ndarray): Array of gamma values corresponding to tau.
        label (Optional[str]): An optional label for the dataset.

    Notes:
    Unless otherwise stated. It is always assumed that gamma refers to the gamma in log-tau space.
    """

    tau: np.ndarray
    gamma: np.ndarray
    label: str | None

    def __init__(self, tau: np.ndarray, gamma: np.ndarray, label=None) -> None:
        self.tau = tau
        self.gamma = gamma
        self.label = label
        self._validate()

    def _validate(self) -> None:
        """
        Validates the input data to ensure consistency and correctness.

        Raises:
            ValueError: If tau and gamma do not have the same shape.
            ValueError: If tau and gamma are not 1D arrays.
            ValueError: If tau or gamma arrays are empty.
        """
        if self.tau.shape != self.gamma.shape:
            raise ValueError("tau and gamma must have the same shape.")
        if len(self.tau.shape) != 1:
            raise ValueError("tau and gamma must be 1D arrays.")
        if self.tau.size == 0 or self.gamma.size == 0:
            raise ValueError("tau and gamma arrays must not be empty.")
