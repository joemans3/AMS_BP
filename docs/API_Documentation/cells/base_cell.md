# `base_cell.py` Documentation

## Module Overview

The `base_cell.py` module defines an abstract base class `BaseCell` for representing different types of cells in a 2D plane. This class provides a common interface and structure for subclasses to implement specific cell behaviors, such as validation, volume calculation, and point containment checks.

## Classes

### `BaseCell`

```python
@dataclass
class BaseCell(ABC):
```

Abstract base class for all cell types.

#### Attributes

- **`origin`** (`np.ndarray`): The (x, y) coordinates of the cell's origin in the XY plane.

#### Methods

##### `__post_init__`

```python
def __post_init__(self):
```

Validates inputs and converts them to numpy arrays if needed.

- **Behavior**:
  - Converts the `origin` attribute to a numpy array with `dtype=float`.
  - Ensures the `origin` is a 2D point (x, y). If not, raises a `ValueError`.
  - Calls the abstract method `_validate_specific()` to validate cell-specific parameters.
  - Calculates the volume of the cell using the abstract method `_calculate_volume()` and stores it in the `_volume` attribute.

##### `_validate_specific`

```python
@abstractmethod
def _validate_specific(self) -> None:
```

Abstract method to validate cell-specific parameters.

- **Purpose**: Subclasses must implement this method to ensure that their specific parameters are valid.

##### `_calculate_volume`

```python
@abstractmethod
def _calculate_volume(self) -> float:
```

Abstract method to calculate the volume of the cell.

- **Purpose**: Subclasses must implement this method to compute the volume of the cell.

##### `center`

```python
@property
@abstractmethod
def center(self) -> np.ndarray:
```

Abstract property to get the center point of the cell.

- **Purpose**: Subclasses must implement this property to return the center coordinates of the cell.

##### `get_bounds`

```python
@abstractmethod
def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
```

Abstract method to get the minimum and maximum points that define the cell's bounds.

- **Returns**:
  - A tuple containing two `np.ndarray` objects:
    - The first array represents the minimum point (bottom-left corner).
    - The second array represents the maximum point (top-right corner).

##### `contains_point`

```python
@abstractmethod
def contains_point(self, point: np.ndarray) -> bool:
```

Abstract method to check if a point lies within the cell.

- **Parameters**:
  - **`point`** (`np.ndarray`): The (x, y) coordinates of the point to check.
- **Returns**:
  - `True` if the point is inside the cell, `False` otherwise.

##### `volume`

```python
@property
def volume(self) -> float:
```

Property to get the pre-calculated volume of the cell.

- **Returns**:
  - The volume of the cell as a `float`.
## Dependencies

- `abc`: Provides the `ABC` class for defining abstract base classes.
- `dataclasses`: Provides the `@dataclass` decorator for automatic generation of `__init__` and other special methods.
- `typing`: Provides the `Tuple` type for type annotations.
- `numpy`: Used for array operations and handling coordinates.

## Notes

- Subclasses of `BaseCell` must implement all abstract methods and properties to ensure proper functionality.
- The `origin` attribute is expected to be a 2D point (x, y). Any other shape will raise a `ValueError`.