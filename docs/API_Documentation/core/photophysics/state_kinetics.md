# Module Documentation: `state_kinetics.py`

## Overview

The `state_kinetics.py` module provides functionality for simulating state transitions of fluorescent objects using a Markov Chain Monte Carlo (MCMC) approach. It includes classes and methods to calculate state transitions, track fluorescence history, and handle errors during simulation.

## Classes

### `ErnoMsg`

A data model representing the result of a state transition simulation, including success status, time, and end state.

#### Attributes:
- `success` (`bool`): Indicates whether the simulation was successful.
- `erno_time` (`Optional[float]`): The time at which the error occurred (if any).
- `erno_end_state` (`Optional[State]`): The state at which the error occurred (if any).

---

### `StateTransitionCalculator`

A class responsible for calculating state transitions of a fluorescent object over a given time duration.

#### Attributes:
- `flurophoreobj` (`FluorescentObject`): The fluorescent object for which state transitions are calculated.
- `time_duration` (`int | float`): The duration of the simulation in seconds.
- `current_global_time` (`int`): The current global time in milliseconds.
- `laser_intensity_generator` (`Callable`): A function to generate laser intensities.
- `fluorescent_state_history` (`dict`): A dictionary tracking fluorescence state history.

#### Methods:
- `__init__(self, flurophoreobj: FluorescentObject, time_duration: int | float, current_global_time: int, laser_intensity_generator: Callable) -> None`:
  Initializes the `StateTransitionCalculator` with the given parameters.

- `__call__(self) -> Tuple[dict, State, ErnoMsg]`:
  Executes the state transition calculation and returns the fluorescence state history, final state, and error message.

- `_initialize_state_hist(self, time_pos: int, time_laser: float) -> dict`:
  Initializes the state history dictionary with laser intensities.

- `MCMC(self) -> Tuple[State, ErnoMsg]`:
  Performs the MCMC simulation to calculate state transitions and returns the final state and error message.

- `_find_transitions(self, statename: str) -> list`:
  Cached method to find transitions from a given state.

---

## Functions

### `ssa_step(reaction_rates: Sequence[float | int]) -> tuple[float, int]`

Performs one step of the Stochastic Simulation Algorithm (SSA) to determine the next reaction and time step.

#### Parameters:
- `reaction_rates` (`Sequence[float | int]`): List of reaction rates.

#### Returns:
- `dt` (`float`): The time step to advance.
- `next_event` (`int`): The index of the next reaction.

---
