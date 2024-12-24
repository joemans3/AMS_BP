# error base classes for SMS_BP


class HurstValueError(Exception):
    """Raised when the Hurst value is not within the range (0, 1)"""

    pass


class SpaceLimitError(Exception):
    """Raised when the space limit is not within the range (-inf, inf)"""

    pass


class DiffusionHighError(Exception):
    """Raised when the diffusion value is too high for the space limit"""

    pass


class HurstHighError(Exception):
    """Raised when the Hurst value is too high for the space limit"""

    pass


class ConfigValidationError(Exception):
    """Exception raised for errors in the configuration validation."""

    pass


class ConfigConversionError(Exception):
    """Exception raised for errors in the configuration conversion process."""

    pass
