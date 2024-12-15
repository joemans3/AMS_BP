from functools import cache

import numpy as np
from scipy.linalg import fractional_matrix_power


def MCMC_state_selection(
    initial_state_index: int,
    transition_matrix: np.ndarray,
    possible_states: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Markov Chain Monte Carlo (MCMC) state selection.

    This function simulates state transitions using a Markov Chain Monte Carlo method.
    It selects the next state based on the current state and a transition matrix over `n` iterations.
    The probability in the transition matrix is the probability of switching to a new state in the "time" step from n-1 -> n.

    Parameters:
    -----------
    initial_state_index : int
        The index of the initial state in the possible states.
    transition_matrix : np.ndarray
        A square matrix representing the transition probabilities between states.
    possible_states : np.ndarray
        An array of possible states for the system.
    n : int
        The number of iterations to perform.

    Returns:
    --------
    np.ndarray
        An array of selected states at each iteration.
    """
    # initialize the state selection
    state_selection = np.zeros(n)
    # initialize the current state
    current_state = possible_states[initial_state_index]
    current_state_index = initial_state_index
    # iterate through the number of iterations
    for i in range(n):
        # find the probability of switching to each state
        state_probability = transition_matrix[current_state_index]
        # find the next state
        next_state_index = np.random.choice(
            np.arange(len(possible_states)), p=state_probability
        )
        next_state = possible_states[next_state_index]
        # update the current state
        current_state = next_state
        current_state_index = next_state_index
        state_selection[i] = current_state
    return state_selection


def MCMC_state_selection_rate(
    initial_state_index: int,
    transition_matrix: np.ndarray,  # in rate, (1/s) s= seconds
    possible_states: np.ndarray,
    n: int,
    time_unit: int,  # amount of time (ms) in one n; ms = milliseconds
):
    # convert transition_matrix to probability
    # divide elementwise to convert 1/s -> 1/ms */1000
    transition_matrix = transition_matrix * (1.0 / 1000.0)
    # convert to prob
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix[i])):
            transition_matrix[i][j] = rate_to_probability(
                transition_matrix[i][j], time_unit
            )

    assert np.sum(transition_matrix, axis=0) == 1.0

    # apply "MCMC_state_selection
    return MCMC_state_selection(
        initial_state_index=initial_state_index,
        transition_matrix=transition_matrix,
        possible_states=possible_states,
        n=n,
    )


# convert from rate (1/s) to probability (0-1)
@cache
def rate_to_probability(rate: float, dt: float) -> float:
    """Convert from rate (1/s) to probability (0-1)

    Parameters:
    -----------
    rate : float
        The rate (1/s)
    dt : float
        The time step (s) for the probability calculation

    Returns:
    --------
    float
        The probability (0-1)
    """
    return 1 - np.exp(-rate * dt)


# convert from probability (0-1) to rate (1/s)
@cache
def probability_to_rate(probability: float, dt: float) -> float:
    """Convert from probability (0-1) to rate (1/s)

    Parameters:
    -----------
    probability : float
        The probability (0-1)
    dt : float
        The time step (s) for the probability calculation
    """
    return -np.log(1 - probability) / dt


# fractional probability util
def change_prob_time(
    probability: np.ndarray | float, dt: float, dt_prime: float
) -> np.ndarray:
    """Change the probability defined for dt to dt'

    Parameters:
    -----------
    probability : np.ndarray | float
        The probability (0-1)
    dt : float
        The time step (s) for the probability calculation
    dt_prime : float
        The new time step (s) for the probability calculation

    Returns:
    --------
    np.ndarray | float
        The probability (0-1)
    """
    if isinstance(probability, np.ndarray):
        return fractional_matrix_power(probability, dt_prime / dt)
    else:
        return probability ** (dt_prime / dt)
