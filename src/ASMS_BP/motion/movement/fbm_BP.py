import numpy as np
from .boundary_conditions import _refecting_boundary, _absorbing_boundary
from ...probabilityfuncs.markov_chain import MCMC_state_selection

BOUNDARY_CONDITIONS = {
    "reflecting": _refecting_boundary,
    "absorbing": _absorbing_boundary,
}


class FBM_BP:
    """
    Fractional Brownian Motion (FBM) simulation with a Markov process for
    diffusion coefficients and Hurst exponents.

    This class simulates the motion of particles using a fractional Brownian motion model
    with adjustable parameters for diffusion and the Hurst exponent.

    Parameters:
    -----------
    n : int
        Number of time steps in the simulation.
    dt : float
        Time step duration in milliseconds.
    diffusion_parameters : np.ndarray
        Array of diffusion coefficients for the FBM simulation.
    hurst_parameters : np.ndarray
        Array of Hurst exponents for the FBM simulation.
    diffusion_parameter_transition_matrix : np.ndarray
        Transition matrix for diffusion coefficients.
    hurst_parameter_transition_matrix : np.ndarray
        Transition matrix for Hurst exponents.
    state_probability_diffusion : np.ndarray
        Initial probabilities of different diffusion states.
    state_probability_hurst : np.ndarray
        Initial probabilities of different Hurst states.
    space_lim : np.ndarray
        Space limits (min, max) for the FBM.

    Methods:
    --------
    _autocovariance(k: int, hurst: float) -> float:
        Computes the autocovariance function for fractional Gaussian noise (fGn).

    _setup() -> None:
        Sets up the simulation by precomputing the autocovariance matrix and initial states.

    fbm() -> np.ndarray:
        Runs the FBM simulation and returns the positions at each time step.
    """

    def __init__(
        self,
        n: int,
        dt: float,
        diffusion_parameters: np.ndarray,
        hurst_parameters: np.ndarray,
        diffusion_parameter_transition_matrix: np.ndarray,
        hurst_parameter_transition_matrix: np.ndarray,
        state_probability_diffusion: np.ndarray,
        state_probability_hurst: np.ndarray,
        space_lim: np.ndarray,
    ):
        self.n = int(n)
        self.dt = dt  # ms
        self.diffusion_parameter = diffusion_parameters
        self.hurst_parameter = hurst_parameters
        # state probability of the diffusion parameter
        self.diffusion_parameter_transition_matrix = (
            diffusion_parameter_transition_matrix
        )
        # state probability of the hurst parameter
        self.hurst_parameter_transition_matrix = hurst_parameter_transition_matrix
        # probability of the initial state, this approximates the population distribution
        self.state_probability_diffusion = state_probability_diffusion
        # probability of the initial state, this approximates the population distribution
        self.state_probability_hurst = state_probability_hurst
        # space lim (min, max) for the FBM
        self.space_lim = np.array(space_lim, dtype=float)
        # initialize the autocovariance matrix and the diffusion parameter
        self._setup()

    def _autocovariance(self, k: int, hurst: float) -> float:
        """
        Autocovariance function for fractional Gaussian noise (fGn).

        Parameters:
        -----------
        k : int
            Lag in time steps.
        hurst : float
            Hurst parameter, which controls the roughness of the trajectory.

        Returns:
        --------
        float
            The autocovariance value for the given lag.
        """
        return 0.5 * (
            abs(k - 1) ** (2 * hurst)
            - 2 * abs(k) ** (2 * hurst)
            + abs(k + 1) ** (2 * hurst)
        )

    def _setup(self) -> None:
        """
        Precomputes the autocovariance matrix and sets up initial diffusion and Hurst parameters.

        This method initializes the state selection using Markov Chain Monte Carlo (MCMC)
        and avoids recomputation of the autocovariance matrix during the simulation.
        """
        self._cov = np.zeros(self.n)
        self._diff_a_n = np.zeros(self.n)
        self._hurst_n = np.zeros(self.n)
        # catch if the diffusion or hurst parameter sets are singular
        if len(self.diffusion_parameter) == 1:
            self._diff_a_n = np.full(self.n, self.diffusion_parameter[0])
        else:
            diff_a_start = np.random.choice(
                self.diffusion_parameter, p=self.state_probability_diffusion
            )
            self._diff_a_n[0] = diff_a_start
            self._diff_a_n[1:] = MCMC_state_selection(
                np.where(self.diffusion_parameter == diff_a_start)[0][0],
                self.diffusion_parameter_transition_matrix,
                self.diffusion_parameter,
                self.n - 1,
            )

        if len(self.hurst_parameter) == 1:
            self._hurst_n = np.full(self.n, self.hurst_parameter[0])
        else:
            hurst_start = np.random.choice(
                self.hurst_parameter, p=self.state_probability_hurst
            )
            self._hurst_n[0] = hurst_start
            self._hurst_n[1:] = MCMC_state_selection(
                np.where(self.hurst_parameter == hurst_start)[0][0],
                self.hurst_parameter_transition_matrix,
                self.hurst_parameter,
                self.n - 1,
            )
        for i in range(self.n):
            self._cov[i] = self._autocovariance(i, self._hurst_n[i])

    def fbm(self) -> np.ndarray:
        """
        Simulates fractional Brownian motion (FBM) over `n` time steps.

        If the Hurst exponent is 0.5 for all time steps, it performs a simple Gaussian
        random walk. Otherwise, it uses the precomputed autocovariance matrix to generate
        fractional Gaussian noise (fGn) and simulates FBM.

        Returns:
        --------
        np.ndarray
            An array representing the simulated FBM positions over time.
        """
        fgn = np.zeros(self.n)
        fbm_store = np.zeros(self.n)
        phi = np.zeros(self.n)
        psi = np.zeros(self.n)
        # construct a gaussian noise vector
        gn = (
            np.random.normal(0, 1, self.n)
            * np.sqrt(self.dt * 2 * self._diff_a_n)
            * (self.dt ** (2 * self._hurst_n))
        )
        # catch is all hurst are 0.5 then use the gaussian noise vector corresponding to the scale defined by the diffusion parameter
        if np.all(self._hurst_n == 0.5):
            # each gn is then pulled from a normal distribution with mean 0 and standard deviation diff_a_n
            # ignore the fbm calculations but keep the reflection
            for i in range(1, self.n):
                fbm_candidate = fbm_store[i - 1] + gn[i]
                # check if this is outside the space limit by using the reflecting boundary condition
                fbm_store[i] = _boundary_conditions(
                    fbm_store[i - 1], fbm_candidate, self.space_lim, "reflecting"
                )
            return fbm_store

        fbm_store[0] = 0
        fgn[0] = gn[0]
        v = 1
        phi[0] = 0

        for i in range(1, self.n):
            phi[i - 1] = self._cov[i]
            for j in range(i - 1):
                psi[j] = phi[j]
                phi[i - 1] -= psi[j] * self._cov[i - j - 1]
            phi[i - 1] /= v
            for j in range(i - 1):
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]
            v *= 1 - phi[i - 1] * phi[i - 1]
            for j in range(i):
                fgn[i] += phi[j] * fgn[i - j - 1]
            fgn[i] += np.sqrt(np.abs(v)) * gn[i]
            # add to the fbm
            fbm_candidate = fbm_store[i - 1] + fgn[i]

            # check if this is outside the space limit by using the reflecting boundary condition
            fbm_store[i] = _boundary_conditions(
                fbm_store[i - 1], fbm_candidate, self.space_lim, "reflecting"
            )
            if fbm_store[i] != fbm_candidate:
                # update the fgn based on the new difference
                fgn[i] = fbm_store[i] - fbm_store[i - 1]
        return fbm_store


def _boundary_conditions(
    fbm_store_last: float,
    fbm_candidate: float,
    space_lim: np.ndarray,
    condition_type: str,
) -> float:
    """
    Apply boundary conditions to the FBM simulation.

    Parameters:
    -----------
    fbm_store_last : float
        The last value of the fractional Brownian motion (FBM) trajectory.
    fbm_candidate : float
        The candidate value for the next step in the FBM trajectory.
    space_lim : np.ndarray
        A 2-element array representing the minimum and maximum space limits.
    condition_type : str
        The type of boundary condition to apply, either "reflecting" or "absorbing".

    Returns:
    --------
    float
        The new value for the FBM trajectory, adjusted by the specified boundary condition.
    """
    # check if the condition type is valid
    if condition_type not in BOUNDARY_CONDITIONS:
        raise ValueError(
            "Invalid condition type: "
            + condition_type
            + "! Must be one of: "
            + str(BOUNDARY_CONDITIONS.keys())
        )
    return BOUNDARY_CONDITIONS[condition_type](fbm_store_last, fbm_candidate, space_lim)
