
# Module Documentation: `probability_functions.py`

## Overview

The `probability_functions.py` module provides tools for handling probability distributions, particularly for "top-hat" shaped subspaces within a larger spatial environment. The module includes functions for generating random points based on a probability distribution and a class for managing multiple top-hat probability functions.

## Functions

### `generate_points`

```python
def generate_points(
    pdf: callable,
    total_points: int,
    min_x: float,
    max_x: float,
    center: np.ndarray,
    radius: float,
    bias_subspace_x: float,
    space_prob: float,
    density_dif: float,
) -> np.ndarray:
```

#### Description

Generates random (x, y) points using the accept/reject method based on a given probability density function (pdf).

#### Parameters

- **`pdf`** (`callable`): The probability density function to sample from.
- **`total_points`** (`int`): The number of points to generate.
- **`min_x`** (`float`): The minimum x value for sampling.
- **`max_x`** (`float`): The maximum x value for sampling.
- **`center`** (`np.ndarray`): The coordinates of the center of the top-hat distribution.
- **`radius`** (`float`): The radius of the top-hat region.
- **`bias_subspace_x`** (`float`): The probability at the top of the top-hat.
- **`space_prob`** (`float`): The probability outside the top-hat region.
- **`density_dif`** (`float`): The scaling factor for density differences.

#### Returns

- **`np.ndarray`**: An array of generated (x, y) points.

---

### `generate_points_from_cls`

```python
def generate_points_from_cls(
    pdf: callable,
    total_points: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    density_dif: float,
) -> np.ndarray:
```

#### Description

Generates random (x, y, z) points using the accept/reject method based on a given probability density function (pdf).

#### Parameters

- **`pdf`** (`callable`): The probability density function to sample from.
- **`total_points`** (`int`): The number of points to generate.
- **`min_x`** (`float`): The minimum x value for sampling.
- **`max_x`** (`float`): The maximum x value for sampling.
- **`min_y`** (`float`): The minimum y value for sampling.
- **`max_y`** (`float`): The maximum y value for sampling.
- **`min_z`** (`float`): The minimum z value for sampling.
- **`max_z`** (`float`): The maximum z value for sampling.
- **`density_dif`** (`float`): The scaling factor for density differences.

#### Returns

- **`np.ndarray`**: An array of generated (x, y, z) points.

---

## Class: `multiple_top_hat_probability`

### Description

A class for handling the probability function of multiple top-hat-shaped subspaces within a larger spatial environment. The class calculates and retrieves probability values based on input positions.

### Methods

#### `__init__`

```python
def __init__(
    self,
    num_subspace: int,
    subspace_centers: np.ndarray,
    subspace_radius: np.ndarray,
    density_dif: float,
    cell: CellType,
) -> None:
```

##### Parameters

- **`num_subspace`** (`int`): The number of subspaces.
- **`subspace_centers`** (`np.ndarray`): The centers of each subspace.
- **`subspace_radius`** (`np.ndarray`): The radius of each subspace.
- **`density_dif`** (`float`): The difference in density between subspaces and non-subspaces.
- **`cell`** (`CellType`): The cell object defining the boundary.

#### `__call__`

```python
def __call__(self, position: np.ndarray, **kwargs) -> float:
```

##### Description

Returns the probability at a given position.

##### Parameters

- **`position`** (`np.ndarray`): The coordinates of the position.

##### Returns

- **`float`**: The probability at the given position.

#### `update_parameters`

```python
def update_parameters(
    self,
    num_subspace: int | None = None,
    subspace_centers: np.ndarray | None = None,
    subspace_radius: np.ndarray | None = None,
    density_dif: float | None = None,
    cell: CellType | None = None,
) -> None:
```

##### Description

Updates the parameters of the probability function.

##### Parameters

- **`num_subspace`** (`int | None`): The number of subspaces.
- **`subspace_centers`** (`np.ndarray | None`): The centers of each subspace.
- **`subspace_radius`** (`np.ndarray | None`): The radius of each subspace.
- **`density_dif`** (`float | None`): The difference in density between subspaces and non-subspaces.
- **`cell`** (`CellType | None`): The cell object defining the boundary.