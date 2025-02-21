import numpy as np

from ...cells.base_cell import BaseCell
from ...probabilityfuncs.markov_chain import MCMC_state_selection
from .boundary_conditions import _absorbing_boundary, _refecting_boundary

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
        Time step duration in seconds.
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
    cell: BaseCell
        BaseCell Object or derivative

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
        cell: BaseCell,
    ):
        self.n = int(n)
        self.dt = dt
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

    def fbm(self, dims=3) -> np.ndarray:
        """
        Simulates 3D fractional Brownian motion (FBM) over `n` time steps using coroutines.

        This implementation uses a coroutine approach to:
        1. Generate candidate positions for each dimension independently
        2. Combine them into a 3D position vector
        3. Check boundary conditions in 3D space
        4. Send accepted positions back to the generators

        Parameters:
        -----------
        dims : int, default=3
            Number of dimensions for the FBM (default: 3 for x,y,z)

        Returns:
        --------
        np.ndarray
            An array of shape (n, dims) representing the simulated FBM positions over time.
        """
        # Initialize storage for the trajectory
        fbm_store = np.zeros((self.n, dims))

        # Create coroutines for each dimension
        dimension_generators = [self._fbm_dimension_generator() for _ in range(dims)]

        # Initialize all generators
        next_candidates = []
        for gen in dimension_generators:
            next(gen)  # Prime the generator
            next_candidates.append(gen.send(None))  # Get first position

        # Set initial position
        fbm_store[0] = np.zeros(dims)

        # Run the simulation
        for i in range(1, self.n):
            # Get candidate positions for each dimension
            candidate_pos = np.array(next_candidates)
            prev_pos = fbm_store[i - 1]

            # Check if the candidate position is within boundaries
            if self._is_within_boundaries(candidate_pos):
                accepted_pos = candidate_pos
            else:
                # Apply boundary condition in 3D
                accepted_pos = self._apply_3d_boundary_condition(
                    prev_pos, candidate_pos, self.space_lim, "reflecting"
                )

            # Store the accepted position
            fbm_store[i] = accepted_pos

            # Send the accepted position back to each generator and get next candidates
            for d, gen in enumerate(dimension_generators):
                next_candidates[d] = gen.send(accepted_pos[d])

        return fbm_store

    def _fbm_dimension_generator(self):
        """
        Generator for a single dimension of FBM.

        Yields candidate positions and receives accepted positions.


        Yields:
        -------
        float
            Next candidate position for this dimension
        """
        # Initialize for this dimension
        fgn = np.zeros(self.n)
        fbm_values = np.zeros(self.n)
        phi = np.zeros(self.n)
        psi = np.zeros(self.n)

        # Construct Gaussian noise vector for this dimension
        gn = np.random.normal(0, 1, self.n) * np.sqrt(
            2 * self._diff_a_n * (self.dt ** (2 * self._hurst_n))
        )

        # First value is zero
        fbm_values[0] = 0
        accepted_pos = yield fbm_values[0]  # Initial yield and receive

        # Fast path for Brownian motion (H=0.5)
        if np.all(self._hurst_n == 0.5):
            for i in range(1, self.n):
                # Generate candidate
                fbm_candidate = fbm_values[i - 1] + gn[i]
                # Yield candidate and receive accepted position
                accepted_pos = yield fbm_candidate
                # Update based on accepted position
                fbm_values[i] = accepted_pos
                # Adjust noise if needed
                if accepted_pos != fbm_candidate:
                    gn[i] = accepted_pos - fbm_values[i - 1]
            return

        # Full FBM simulation
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

            # Generate candidate position
            fbm_candidate = fbm_values[i - 1] + fgn[i]

            # Yield candidate and receive accepted position
            accepted_pos = yield fbm_candidate

            # Update stored values based on accepted position
            fbm_values[i] = accepted_pos

            # If position was modified, adjust the noise
            if accepted_pos != fbm_candidate:
                fgn[i] = accepted_pos - fbm_values[i - 1]

    def _is_within_boundaries(self, position):
        """
        Check if a position is within the specified boundaries.

        Parameters:
        -----------
        position : np.ndarray
            Position vector to check

        Returns:
        --------
        bool
            True if position is within boundaries, False otherwise
        """
        # Assuming self.space_lim is now shape (dims, 2)
        dims = len(position)
        for d in range(dims):
            if position[d] < self.space_lim[d, 0] or position[d] > self.space_lim[d, 1]:
                return False
        return True

    def _apply_3d_boundary_condition(
        self, prev_pos, candidate_pos, space_lim, condition_type="reflecting"
    ):
        """
        Apply boundary conditions to 3D position.

        Parameters:
        -----------
        prev_pos : np.ndarray
            Previous position vector
        candidate_pos : np.ndarray
            Candidate position vector
        space_lim : np.ndarray
            Spatial boundaries for each dimension
        condition_type : str
            Type of boundary condition to apply

        Returns:
        --------
        np.ndarray
            Corrected position vector
        """
        if condition_type == "reflecting":
            corrected_pos = candidate_pos.copy()
            for d in range(len(candidate_pos)):
                if candidate_pos[d] < space_lim[d, 0]:
                    corrected_pos[d] = 2 * space_lim[d, 0] - candidate_pos[d]
                elif candidate_pos[d] > space_lim[d, 1]:
                    corrected_pos[d] = 2 * space_lim[d, 1] - candidate_pos[d]
            return corrected_pos
        elif condition_type == "absorbing":
            # For absorbing boundaries, return previous position if any dimension crosses boundary
            for d in range(len(candidate_pos)):
                if (
                    candidate_pos[d] < space_lim[d, 0]
                    or candidate_pos[d] > space_lim[d, 1]
                ):
                    return prev_pos
            return candidate_pos
        else:
            raise ValueError(f"Unknown boundary condition: {condition_type}")
