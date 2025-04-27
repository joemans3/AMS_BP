# Module Documentation: `markov_chain.py`

## Overview

The `markov_chain.py` module provides functions for simulating Markov Chain Monte Carlo (MCMC) state transitions and utility functions for converting between rates and probabilities. The module is designed to facilitate simulations of state transitions based on a given transition matrix and to handle time-dependent probability calculations.

## Functions

### `MCMC_state_selection`

```python
def MCMC_state_selection(
    initial_state_index: int,
    transition_matrix: np.ndarray,
    possible_states: np.ndarray,
    n: int,
) -> np.ndarray:
```

#### Description

Simulates state transitions using a Markov Chain Monte Carlo (MCMC) method. The function selects the next state based on the current state and a transition matrix over `n` iterations. The transition matrix defines the probability of switching to a new state in each time step.

#### Parameters

- **`initial_state_index`** (`int`): The index of the initial state in the `possible_states` array.
- **`transition_matrix`** (`np.ndarray`): A square matrix representing the transition probabilities between states.
- **`possible_states`** (`np.ndarray`): An array of possible states for the system.
- **`n`** (`int`): The number of iterations to perform.

#### Returns

- **`np.ndarray`**: An array of selected states at each iteration.

---

### `MCMC_state_selection_rate`

```python
def MCMC_state_selection_rate(
    initial_state_index: int,
    transition_matrix: np.ndarray,  # in rate, (1/s) s= seconds
    possible_states: np.ndarray,
    n: int,
    time_unit: int,  # amount of time (ms) in one n; ms = milliseconds
):
```

#### Description

Simulates state transitions using a Markov Chain Monte Carlo (MCMC) method, where the transition matrix is given in terms of rates (1/s). The function converts the rates to probabilities before performing the state selection.

#### Parameters

- **`initial_state_index`** (`int`): The index of the initial state in the `possible_states` array.
- **`transition_matrix`** (`np.ndarray`): A square matrix representing the transition rates between states (in 1/s).
- **`possible_states`** (`np.ndarray`): An array of possible states for the system.
- **`n`** (`int`): The number of iterations to perform.
- **`time_unit`** (`int`): The amount of time (in milliseconds) for one iteration.

#### Returns

- **`np.ndarray`**: An array of selected states at each iteration.

---

### `rate_to_probability`

```python
@cache
def rate_to_probability(rate: float, dt: float) -> float:
```

#### Description

Converts a rate (1/s) to a probability (0-1) based on a given time step.

#### Parameters

- **`rate`** (`float`): The rate (1/s).
- **`dt`** (`float`): The time step (s) for the probability calculation.

#### Returns

- **`float`**: The probability (0-1).

---

### `probability_to_rate`

```python
@cache
def probability_to_rate(probability: float, dt: float) -> float:
```

#### Description

Converts a probability (0-1) to a rate (1/s) based on a given time step.

#### Parameters

- **`probability`** (`float`): The probability (0-1).
- **`dt`** (`float`): The time step (s) for the probability calculation.

#### Returns

- **`float`**: The rate (1/s).

---

### `change_prob_time`

```python
def change_prob_time(
    probability: np.ndarray | float, dt: float, dt_prime: float
) -> np.ndarray:
```

#### Description

Changes the probability defined for a time step `dt` to a new time step `dt_prime`.

#### Parameters

- **`probability`** (`np.ndarray | float`): The probability (0-1).
- **`dt`** (`float`): The original time step (s).
- **`dt_prime`** (`float`): The new time step (s).

#### Returns

- **`np.ndarray | float`**: The probability adjusted for the new time step.

---