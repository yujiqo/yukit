"""Step by step dataset preparation: missing values, noise removal, scaling, and encoding.

Start with manual managemnet and then call switch_phase() to proceed with automatic cleaning pipeline. the first step is to do split_columns().
"""

from .dataset import Dataset
from .exceptions import (
    ManualOperationError,
    MismatchError,
    PhaseMismatchError,
    StepMismatchError,
)

__all__ = [
    "Dataset",
    "ManualOperationError",
    "MismatchError",
    "PhaseMismatchError",
    "StepMismatchError",
]
