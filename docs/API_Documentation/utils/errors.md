# `errors.py`

### Exceptions

#### `HurstValueError`

```python
class HurstValueError(Exception):
    """Raised when the Hurst value is not within the range (0, 1)"""
    pass
```

- **Description**: This exception is raised when the Hurst value provided is not within the valid range of 0 to 1.

#### `SpaceLimitError`

```python
class SpaceLimitError(Exception):
    """Raised when the space limit is not within the range (-inf, inf)"""
    pass
```

- **Description**: This exception is raised when the space limit provided is not within the valid range of negative infinity to positive infinity.

#### `DiffusionHighError`

```python
class DiffusionHighError(Exception):
    """Raised when the diffusion value is too high for the space limit"""
    pass
```

- **Description**: This exception is raised when the diffusion value provided is too high relative to the space limit.

#### `HurstHighError`

```python
class HurstHighError(Exception):
    """Raised when the Hurst value is too high for the space limit"""
    pass
```

- **Description**: This exception is raised when the Hurst value provided is too high relative to the space limit.

#### `ConfigValidationError`

```python
class ConfigValidationError(Exception):
    """Exception raised for errors in the configuration validation."""
    pass
```

- **Description**: This exception is raised when there is an error in the validation of the configuration settings.

#### `ConfigConversionError`

```python
class ConfigConversionError(Exception):
    """Exception raised for errors in the configuration conversion process."""
    pass
```

- **Description**: This exception is raised when there is an error in the conversion process of the configuration settings.

---