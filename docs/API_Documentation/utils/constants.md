## `constants.py`

### Constants

#### `N_A`

```python
N_A = 6.02214076e23  # Avogadro's number
```

- **Description**: Avogadro's number, representing the number of particles (atoms, molecules, ions, etc.) in one mole of a substance.

#### `C_LIGHT_M_S`

```python
C_LIGHT_M_S = 299792458  # Speed of light in m/s
```

- **Description**: The speed of light in a vacuum, measured in meters per second.

#### `K_BOLTZMANN`

```python
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant in J/K
```

- **Description**: The Boltzmann constant, relating the average kinetic energy of particles in a gas to the temperature of the gas.

#### `H_PLANCK`

```python
H_PLANCK = 6.62607015e-34  # Planck constant in J*s
```

- **Description**: The Planck constant, a fundamental constant in quantum mechanics.

#### `H_C_COM`

```python
H_C_COM = 0.1 / (H_PLANCK * C_LIGHT_M_S)
```

- **Description**: A derived constant calculated as `0.1 / (H_PLANCK * C_LIGHT_M_S)`.

---