"""Model evaluation: general regression metrics and dataset-specific evaluators."""

from .base import Evaluator
from .regression import RegressionEvaluator
from .airfoil_unsteady import AirfoilUnsteadyEvaluator, AirfoilUnsteadyFFTEvaluator
from .fluidbench import FluidBenchWrapper

__all__ = [
    "Evaluator",
    "RegressionEvaluator",
    "AirfoilUnsteadyEvaluator",
    "AirfoilUnsteadyFFTEvaluator",
    "FluidBenchWrapper",
]
