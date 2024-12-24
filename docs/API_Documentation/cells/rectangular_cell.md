# `rectangular_cell.py` Documentation

## Overview

The `rectangular_cell.py` module defines a `RectangularCell` class, which represents a rectangular cell in 3D space. This class inherits from the `BaseCell` class and provides functionality to validate the cell's dimensions, calculate its volume, determine its center, and check if a given point lies within the cell.

## Classes

### `RectangularCell`

```python
@dataclass
class RectangularCell(BaseCell):
```

Represents a rectangular cell in 3D space.

#### Attributes

- **`origin`** (`np.ndarray`):  
  The (x, y) coordinates of the cell's origin in the XY plane. This attribute is inherited from the `BaseCell` class.

- **`dimensions`** (`np.ndarray`):  
  The (length, width, height) of the cell.

#### Methods

##### `_validate_specific(self) -> None`

Validates rectangle-specific parameters.

- Ensures that the `dimensions` attribute is a 3D vector with positive values.
- Raises a `ValueError` if the dimensions are not a 3D vector or if any dimension is non-positive.

##### `_calculate_volume(self) -> float`

Calculates the volume of the rectangular cell.

- Returns the product of the cell's dimensions as a `float`.

##### `center(self) -> np.ndarray`

A property that returns the center point of the rectangular cell.

- The center is calculated as the midpoint of the cell's dimensions in the XY plane, with the Z-coordinate set to `0.0`.

##### `get_bounds(self) -> Tuple[np.ndarray, np.ndarray]`

Gets the minimum and maximum points that define the cell's bounds.

- Returns a tuple containing two `np.ndarray` objects:
  - `min_point`: The minimum point in the cell's bounds.
  - `max_point`: The maximum point in the cell's bounds.

##### `contains_point(self, point: np.ndarray) -> bool`

Checks if a given point lies within the rectangular cell.

- **Args:**
  - `point` (`np.ndarray`): A 3D point to check.

- **Raises:**
  - `ValueError`: If the point is not a 3D vector.

- **Returns:**
  - `bool`: `True` if the point is inside the cell, `False` otherwise.

## Usage Example

```python
from rectangular_cell import RectangularCell
import numpy as np

# Create a rectangular cell with origin at (0, 0) and dimensions (2, 3, 4)
cell = RectangularCell(origin=np.array([0, 0]), dimensions=np.array([2, 3, 4]))

# Validate the cell's dimensions
cell._validate_specific()

# Calculate the volume
volume = cell._calculate_volume()
print(f"Volume: {volume}")

# Get the center of the cell
center = cell.center
print(f"Center: {center}")

# Get the bounds of the cell
min_point, max_point = cell.get_bounds()
print(f"Min Point: {min_point}, Max Point: {max_point}")

# Check if a point is inside the cell
point = np.array([1, 1, 1])
is_inside = cell.contains_point(point)
print(f"Point {point} is inside: {is_inside}")
```

## Dependencies

- `dataclasses`: Used to define the `RectangularCell` class with default dataclass behavior.
- `numpy`: Used for array operations and calculations.
- `typing`: Used for type annotations, specifically for the `Tuple` type.
- `base_cell`: The base class from which `RectangularCell` inherits.

## Notes

- The `origin` attribute is inherited from the `BaseCell` class and is not explicitly defined in `RectangularCell`.
- The Z-coordinate of the cell's center is always `0.0`, as the cell is defined in the XY plane.
- The `contains_point` method assumes the cell extends symmetrically in the Z-direction from `-height/2` to `height/2`.