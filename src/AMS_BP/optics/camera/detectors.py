from abc import ABC, abstractmethod
from typing import Tuple, overload

import numpy as np


@overload
def photon_noise(photons: float) -> float: ...
@overload
def photon_noise(photons: np.ndarray) -> np.ndarray: ...
def photon_noise(photons: np.ndarray | float) -> np.ndarray | float:
    """
    Calculate photons with Poisson noise.
    """
    # non-negative floats
    return np.random.poisson(lam=photons)


class Detector(ABC):
    """Base class for microscopy camera detectors."""

    def __init__(
        self,
        pixel_size: float,
        dark_current: float,
        readout_noise: float,
        pixel_count: Tuple[int, int],
        pixel_detector_size: float | int,
        magnification: float | int,
        bit_depth: int = 16,
        sensitivity: float = 1.0,
        base_adu: int = 100,
        binning_size: int = 1,
    ):
        """
        Initialize detector parameters.

        Args:
            pixel_size: Size of each pixel in microns
            dark_current: Dark current in electrons/pixel/second
            readout_noise: RMS readout noise in electrons
            pixel_count: Tuple of (width, height) in pixels
            pixel_detector_size: Size of the detector in microns
            magnification: Magnification of the microscope
            bit_depth: Number of bits for analog-to-digital conversion
            sensitivity: Conversion gain in electrons/ADU
            base_adu: base ADU to avoid negative values due to photon arrival rate
        """
        self.pixel_size = pixel_size
        self.dark_current = dark_current
        self.readout_noise = readout_noise
        self.pixel_count = pixel_count
        self.bit_depth = bit_depth
        self.sensitivity = sensitivity
        self._max_counts = 2**bit_depth - 1
        self.pixel_detector_size = pixel_detector_size
        self.magnification = magnification
        self.base_adu = base_adu
        self.binning_size = binning_size
        self.binning_function = create_binning_function(
            self.pixel_count, self.binning_size, mode="sum"
        )

    def base_frame(self, base_adu: int) -> np.ndarray:
        return np.zeros(self.pixel_count) + base_adu

    def electrons_to_counts(self, electrons: np.ndarray) -> np.ndarray:
        """
        Convert electrons to digital counts (ADU). This is rounded to the integer, but not bounded by the bit depth.

        Args:
            electrons: Array of electron values

        Returns:
            Array of digital counts
        """
        # Convert electrons to counts using sensitivity
        counts = electrons / self.sensitivity
        # Clip to valid range and round to integers
        return np.round(counts)

    def clipADU(self, counts: np.ndarray) -> np.ndarray:
        if self.bit_depth == 8:
            return np.clip(
                self.base_adu + self.binning_function(counts),
                0,
                self._max_counts,
            ).astype(np.uint8)
        elif self.bit_depth == 16:
            return np.clip(
                self.base_adu + self.binning_function(counts),
                0,
                self._max_counts,
            ).astype(np.uint16)
        elif self.bit_depth == 32:
            return np.clip(
                self.base_adu + self.binning_function(counts),
                0,
                self._max_counts,
            ).astype(np.uint32)
        elif self.bit_depth == 64:
            return np.clip(
                self.base_adu + self.binning_function(counts),
                0,
                self._max_counts,
            ).astype(np.uint64)
        else:
            raise ValueError(f"bit depth: {self.bit_depth} is not supported.")

    @abstractmethod
    def capture_frame(self, photons: np.ndarray, exposure_time: float) -> np.ndarray:
        """
        Capture a frame with the detector.

        Args:
            photons: 2D array of incident photons
            exposure_time: Exposure time in seconds

        Returns:
            2D array of measured counts
        """
        pass


class EMCCDDetector(Detector):
    """Electron Multiplying CCD detector implementation."""

    def __init__(
        self,
        pixel_size: float,
        dark_current: float,
        readout_noise: float,
        pixel_count: Tuple[int, int],
        em_gain: float,
        clock_induced_charge: float,
        pixel_detector_size: float | int,
        magnification: float | int,
        sensitivity: float = 1.0,
        bit_depth: int = 16,
        base_adu: int = 0,
        binning_size: int = 1,
    ):
        """
        Initialize EMCCD detector.

        Args:
            em_gain: Electron multiplication gain
            clock_induced_charge: Clock-induced charge in electrons/pixel/frame
            sensitivity: Conversion gain in electrons/ADU
            bit_depth: Number of bits for analog-to-digital conversion
        """
        super().__init__(
            pixel_size,
            dark_current,
            readout_noise,
            pixel_count,
            pixel_detector_size,
            magnification,
            bit_depth,
            sensitivity,
            base_adu,
            binning_size,
        )
        self.em_gain = em_gain
        self.clock_induced_charge = clock_induced_charge

    def capture_frame(self, photons: np.ndarray, exposure_time: float) -> np.ndarray:
        # Validate input dimensions
        if photons.shape != (self.pixel_count[1], self.pixel_count[0]):  # height, width
            raise ValueError(
                f"Photons array shape {photons.shape} does not match detector dimensions "
                f"({self.pixel_count[1]}, {self.pixel_count[0]})"
            )

        # Convert photons to electrons using quantum efficiency
        electrons = photons

        # Add dark current
        electrons += self.dark_current * exposure_time

        # Add clock-induced charge
        electrons += self.clock_induced_charge

        # Apply EM gain (with stochastic multiplication)
        electrons = np.random.gamma(electrons, self.em_gain)

        # Add readout noise
        electrons += np.random.normal(0, self.readout_noise, electrons.shape)

        # Convert electrons to digital counts (ADU)
        counts = self.electrons_to_counts(electrons)

        return self.clipADU(counts)


class CMOSDetector(Detector):
    """CMOS detector implementation."""

    def __init__(
        self,
        pixel_size: float,
        dark_current: float,
        readout_noise: float,
        pixel_count: Tuple[int, int],
        pixel_detector_size: float | int,
        magnification: float | int,
        sensitivity: float = 1.0,
        bit_depth: int = 16,
        base_adu: int = 0,
        binning_size: int = 1,
    ):
        """
        Initialize CMOS detector.

        Args:
            dark_current: Dark current in electrons/pixel/second
            readout_noise: RMS readout noise in electrons
            pixel_count: Tuple of (width, height) in pixels
            sensitivity: Conversion gain in electrons/ADU
            bit_depth: Number of bits for analog-to-digital conversion
        """
        super().__init__(
            pixel_size,
            dark_current,
            readout_noise,
            pixel_count,
            pixel_detector_size,
            magnification,
            bit_depth,
            sensitivity,
            base_adu,
            binning_size,
        )

    def capture_frame(self, photons: np.ndarray, exposure_time: float) -> np.ndarray:
        # Validate input dimensions
        if photons.shape != (self.pixel_count[1], self.pixel_count[0]):  # height, width
            raise ValueError(
                f"Photons array shape {photons.shape} does not match detector dimensions "
                f"({self.pixel_count[1]}, {self.pixel_count[0]})"
            )

        # Convert photons to electrons using quantum efficiency
        electrons = photons

        # Add dark current
        electrons += self.dark_current * exposure_time

        # Add readout noise
        electrons += np.random.normal(0, self.readout_noise, electrons.shape)

        # Convert electrons to digital counts (ADU)
        counts = self.electrons_to_counts(electrons)

        return self.clipADU(counts)


def create_binning_function(input_shape, binning_size, mode="sum"):
    """
    Creates an optimized partial function for binning arrays of a specific shape.

    Parameters:
    -----------
    input_shape : tuple
        Shape of the input arrays that will be binned
    binning_size : int
        Size of the binning window (e.g., 2 for 2x2 binning)
    mode : str, optional
        Method for binning. Currently only supports 'sum'

    Returns:
    --------
    function
        A specialized function that only takes an array as input and performs binning
    """
    # Pre-calculate all the constants we'll need
    original_shape = np.array(input_shape)
    output_shape = np.ceil(original_shape / binning_size).astype(int)
    effective_shape = (output_shape * binning_size) - original_shape

    # Pre-calculate padding configuration if needed
    needs_padding = np.any(effective_shape > 0)
    pad_width = [(0, int(s)) for s in effective_shape] if needs_padding else None

    # Pre-calculate reshape dimensions
    new_shape = []
    working_shape = output_shape * binning_size if needs_padding else original_shape
    for dim in working_shape:
        new_shape.extend([dim // binning_size, binning_size])

    # Pre-calculate sum axes
    sum_axes = tuple(range(1, len(new_shape), 2))

    def optimized_binning(array):
        """
        Optimized binning function for arrays of a specific shape.

        Parameters:
        -----------
        array : numpy.ndarray
            Input array to be binned (must match input_shape)

        Returns:
        --------
        numpy.ndarray
            Binned array
        """
        if array.shape != tuple(input_shape):
            raise ValueError(
                f"Input array shape {array.shape} does not match expected shape {input_shape}"
            )

        # Apply padding if needed
        if needs_padding:
            array = np.pad(array, pad_width, mode="constant")

        # Perform binning
        return array.reshape(new_shape).sum(axis=sum_axes)

    return optimized_binning
