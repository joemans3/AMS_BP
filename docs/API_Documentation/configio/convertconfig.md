# AMS_BP Configuration and Setup Documentation

## ConfigLoader Class

### Overview
The `ConfigLoader` class handles loading and parsing configuration files for microscopy simulation setup. It manages the creation and configuration of various components including fluorophores, PSF engines, lasers, filters, and detectors.

### Constructor
```python
def __init__(self, config_path: Union[str, Path, dict])
```
- **Parameters:**
  - `config_path`: Path to TOML configuration file or configuration dictionary
- **Raises:**
  - `FileNotFoundError`: If configuration file not found

### Methods

#### `create_dataclass_schema`
```python
def create_dataclass_schema(self, dataclass_schema: type[BaseModel], config: Dict[str, Any]) -> BaseModel
```
Populates a dataclass schema with configuration data.
- **Parameters:**
  - `dataclass_schema`: Type of BaseModel to create
  - `config`: Configuration dictionary
- **Returns:** Populated BaseModel instance

#### `create_experiment_from_config`
```python
def create_experiment_from_config(self, config: Dict[str, Any]) -> Tuple[BaseExpConfig, Callable]
```
Creates experiment configuration and associated callable function.
- **Parameters:**
  - `config`: Configuration dictionary
- **Returns:** 
  - Tuple containing experiment configuration and experiment function
- **Raises:**
  - `TypeError`: If experiment type not supported

#### `create_fluorophores_from_config`
```python
def create_fluorophores_from_config(self, config: Dict[str, Any]) -> List[Fluorophore]
```
Creates fluorophore instances from configuration.
- **Parameters:**
  - `config`: Configuration dictionary
- **Returns:** List of Fluorophore instances
- **Raises:**
  - `ValueError`: If no fluorophore configuration found

#### `create_psf_from_config`
```python
def create_psf_from_config(self, config: Dict[str, Any]) -> Tuple[Callable, Dict[str, Any]]
```
Creates PSF engine instance from configuration.
- **Parameters:**
  - `config`: Configuration dictionary
- **Returns:**
  - Tuple containing PSF engine function and additional parameters
- **Raises:**
  - `ValueError`: If no PSF configuration found

#### `create_laser_from_config`
```python
def create_laser_from_config(self, laser_config: Dict[str, Any], preset: str) -> LaserProfile
```
Creates laser profile instance from configuration.
- **Parameters:**
  - `laser_config`: Laser configuration dictionary
  - `preset`: Laser preset name
- **Returns:** LaserProfile instance
- **Raises:**
  - `ValueError`: If unsupported laser type specified

## Utility Functions

### `load_config`
```python
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]
```
Loads and parses TOML configuration file.
- **Parameters:**
  - `config_path`: Path to configuration file
- **Returns:** Parsed configuration dictionary
- **Raises:**
  - `FileNotFoundError`: If config file not found
  - `tomli.TOMLDecodeError`: If TOML file invalid

### `make_cell`
```python
def make_cell(cell_params) -> BaseCell
```
Creates cell instance from parameters.
- **Parameters:**
  - `cell_params`: Cell parameters
- **Returns:** BaseCell instance

### `make_sample`
```python
def make_sample(global_params, cell_params) -> SamplePlane
```
Creates sample plane from parameters.
- **Parameters:**
  - `global_params`: Global parameters
  - `cell_params`: Cell parameters
- **Returns:** SamplePlane instance

### `make_condensatedict`
```python
def make_condensatedict(condensate_params, cell) -> List[dict]
```
Creates condensate dictionaries from parameters.
- **Parameters:**
  - `condensate_params`: Condensate parameters
  - `cell`: Cell instance
- **Returns:** Condensate dictionaries

### `get_tracks`
```python
def get_tracks(molecule_params, global_params, initial_positions, track_generator)
```
Generates molecular tracks based on parameters.
- **Parameters:**
  - `molecule_params`: Molecule parameters
  - `global_params`: Global parameters
  - `initial_positions`: Initial molecular positions
  - `track_generator`: Track generator instance
- **Returns:** Tuple of tracks collection and points per time collection

## Dependencies

- `tomli`: For TOML file parsing
- `pydantic`: For data validation using BaseModel
- `pathlib`: For path handling
- Various internal modules:
  - `AMS_BP.optics.filters`
  - `AMS_BP.cells`
  - `AMS_BP.motion`
  - `AMS_BP.optics.camera`
  - `AMS_BP.sample`

## Usage Example

```python
# Create config loader
config_loader = ConfigLoader("config.toml")

# Setup microscope
microscope_config = config_loader.setup_microscope()

# Access components
vm = microscope_config["microscope"]
base_config = microscope_config["base_config"]
psf = microscope_config["psf"]
```

## Notes

- The configuration system expects TOML format for input files
- All numeric parameters should be provided in SI units unless otherwise specified
- The system supports multiple types of:
  - Lasers (Gaussian, Widefield, HiLo)
  - Detectors (CMOS, EMCCD)
  - Filters (Bandpass, Tophat, Allow-all)
  - Experiments (Time-series, Z-stack)
