class MismatchError(ValueError):
    """Raised on values mismatch."""

    pass


class ManualOperationError(Exception):
    """Raised on manual operation call during automatic phase of the Dataset pipeline."""

    pass


class PhaseMismatchError(MismatchError):
    """Raised when the operation does not align with the current phase."""

    pass


class StepMismatchError(MismatchError):
    """Raised on step mismatch during automatic phase of the Dataset pipeline."""

    pass
