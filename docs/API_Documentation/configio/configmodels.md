# Configuration Models Documentation

## Overview
The configuration models module defines Pydantic BaseModel classes for validating and managing microscopy simulation parameters. Each model class represents a specific set of configuration parameters with built-in validation and type conversion.

## Models

### CellParameters
Defines the physical parameters of the cell being simulated.

```python
class CellParameters(BaseModel)
```

#### Fields
- `cell_space: List[List[float]]`
  - Description: Cell space dimensions in micrometers
  - Format: 2D array defining spatial boundaries
  - Automatically converted to numpy array

- `cell_axial_radius: float`
  - Description: Axial radius in micrometers
  - Defines the cell's axial dimension

### MoleculeParameters
Defines parameters for molecular motion and behavior simulation.

```python
class MoleculeParameters(BaseModel)
```

#### Fields
- `num_molecules: List[int]`
  - Description: Number of molecules for each type

- `track_type: List[Literal["fbm", "constant"]]`
  - Description: Type of molecular motion
  - Valid values: "fbm" (fractional Brownian motion) or "constant"

- `diffusion_coefficient: List[List[float]]`
  - Description: Diffusion coefficients in μm²/s
  - 2D array automatically converted to numpy array

- `diffusion_track_amount: List[List[float]]`
  - Description: Amount of diffusive tracking for each state
  - 2D array automatically converted to numpy array

- `hurst_exponent: List[List[float]]`
  - Description: Hurst exponents for fractional Brownian motion
  - 2D array automatically converted to numpy array

- `hurst_track_amount: List[List[float]]`
  - Description: Amount of Hurst tracking for each state
  - 2D array automatically converted to numpy array

- `allow_transition_probability: List[bool]`
  - Description: Whether to allow state transitions

- `transition_matrix_time_step: List[int]`
  - Description: Time step in milliseconds for transition matrices

- `diffusion_transition_matrix: List[List[List[float]]]`
  - Description: State transition probabilities for diffusion
  - 3D array automatically converted to numpy array

- `hurst_transition_matrix: List[List[List[float]]]`
  - Description: State transition probabilities for Hurst exponent
  - 3D array automatically converted to numpy array

- `state_probability_diffusion: List[List[float]]`
  - Description: Initial state probabilities for diffusion
  - 2D array automatically converted to numpy array

- `state_probability_hurst: List[List[float]]`
  - Description: Initial state probabilities for Hurst exponent
  - 2D array automatically converted to numpy array

### GlobalParameters
Defines global simulation parameters.

```python
class GlobalParameters(BaseModel)
```

#### Fields
- `sample_plane_dim: List[float]`
  - Description: Sample plane dimensions in micrometers
  - Automatically converted to numpy array

- `cycle_count: int`
  - Description: Number of simulation cycles

- `exposure_time: int`
  - Description: Exposure time in milliseconds

- `interval_time: int`
  - Description: Interval time in milliseconds

- `oversample_motion_time: int`
  - Description: Oversample motion time in milliseconds

### CondensateParameters
Defines parameters for molecular condensate simulation.

```python
class CondensateParameters(BaseModel)
```

#### Fields
- `initial_centers: List[List[float]]`
  - Description: Initial centers in micrometers
  - 2D array automatically converted to numpy array

- `initial_scale: List[float]`
  - Description: Initial scale in micrometers
  - Automatically converted to numpy array

- `diffusion_coefficient: List[float]`
  - Description: Diffusion coefficients in μm²/s
  - Automatically converted to numpy array

- `hurst_exponent: List[float]`
  - Description: Hurst exponents for motion
  - Automatically converted to numpy array

- `density_dif: int`
  - Description: Density difference parameter

### OutputParameters
Defines parameters for simulation output.

```python
class OutputParameters(BaseModel)
```

#### Fields
- `output_path: str`
  - Description: Path for output files

- `output_name: str`
  - Description: Base name for output files

- `subsegment_type: str`
  - Description: Type of subsegmentation

- `subsegment_number: int`
  - Description: Number of subsegments

### ConfigList
Container model that combines all parameter models.

```python
class ConfigList(BaseModel)
```

#### Fields
- `CellParameters: CellParameters`
- `MoleculeParameters: MoleculeParameters`
- `GlobalParameters: GlobalParameters`
- `CondensateParameters: CondensateParameters`
- `OutputParameters: OutputParameters`

## Validation Features

All models include automatic validation and conversion:
- Numeric arrays are automatically converted to numpy arrays
- Field types are strictly enforced
- Required fields must be present
- Literal fields must match specified values

## Usage Example

```python
# Create configuration instance
config = ConfigList(
    CellParameters=CellParameters(
        cell_space=[[0, 10], [0, 10]],
        cell_axial_radius=5.0
    ),
    MoleculeParameters=MoleculeParameters(
        num_molecules=[100],
        track_type=["fbm"],
        # ... other required fields ...
    ),
    GlobalParameters=GlobalParameters(
        sample_plane_dim=[20.0, 20.0],
        cycle_count=100,
        exposure_time=100,
        interval_time=100,
        oversample_motion_time=10
    ),
    CondensateParameters=CondensateParameters(
        initial_centers=[[5.0, 5.0]],
        initial_scale=[2.0],
        diffusion_coefficient=[0.1],
        hurst_exponent=[0.7],
        density_dif=2
    ),
    OutputParameters=OutputParameters(
        output_path="./output",
        output_name="simulation",
        subsegment_type="uniform",
        subsegment_number=10
    )
)
```

## Dependencies
- `pydantic`: For data validation and settings management
- `numpy`: For array operations
- `typing`: For type hints