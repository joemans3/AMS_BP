import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from scipy.ndimage import center_of_mass, shift
from scipy.interpolate import RegularGridInterpolator
from .psf_engine import PSFParameters, PSFEngine
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, Literal
from typing_extensions import Annotated

class PSFMetadata(BaseModel):
    """Container for PSF metadata with validation"""
    wavelength: Annotated[
        float, 
        Field(gt=0, description="Light wavelength in nanometers")
    ]
    numerical_aperture: Annotated[
        float, 
        Field(gt=0, le=1.5, description="Microscope numerical aperture")
    ]
    pixel_size: Annotated[
        float, 
        Field(gt=0, description="Camera pixel size in nanometers")
    ]
    z_step: Annotated[
        float, 
        Field(gt=0, description="Axial step size in nanometers")
    ]
    refractive_index: Annotated[
        float, 
        Field(default=1.0, gt=0, description="Medium refractive index")
    ]
    microscope_type: Optional[str] = Field(None, description="Type of microscope")
    immersion_medium: Optional[str] = Field(None, description="Immersion medium type")
    additional_params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional parameters"
    )

    @field_validator('microscope_type')
    @classmethod
    def validate_microscope_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.lower()
            valid_types = {'widefield', 'confocal', 'two-photon', 'lightsheet'}
            if v not in valid_types:
                raise ValueError(f"Invalid microscope type. Must be one of: {valid_types}")
        return v

    @field_validator('immersion_medium')
    @classmethod
    def validate_immersion_medium(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.lower()
            valid_media = {'air', 'water', 'oil', 'glycerol'}
            if v not in valid_media:
                raise ValueError(f"Invalid immersion medium. Must be one of: {valid_media}")
        return v


class PSFValidator:
    """Validates PSF data"""
    
    @staticmethod
    def validate_psf(psf_data: np.ndarray) -> bool:
        """
        Validate PSF data
        
        Args:
            psf_data: PSF array to validate
            
        Returns:
            bool: True if valid, raises exception if invalid
        """
        # Check data type
        if not isinstance(psf_data, np.ndarray):
            raise TypeError("PSF must be a numpy array")
            
        # Check dimensions
        if psf_data.ndim not in [2, 3]:
            raise ValueError("PSF must be 2D or 3D")
            
        # Check for NaN or Inf values
        if np.any(~np.isfinite(psf_data)):
            raise ValueError("PSF contains NaN or Inf values")
            
        # Check for negative values
        if np.any(psf_data < 0):
            raise ValueError("PSF contains negative values")
            
        # Check for zero sum
        if np.sum(psf_data) <= 0:
            raise ValueError("PSF sum must be positive")
            
        return True

class PSFProcessor:
    """Processes PSF data for alignment, interpolation, and normalization"""
    
    @staticmethod
    def center_psf(psf_data: np.ndarray) -> np.ndarray:
        """Center PSF using center of mass"""
        # Calculate center of mass
        com = center_of_mass(psf_data)
        center = np.array(psf_data.shape) / 2
        
        # Calculate shift
        shift_values = center - com
        
        # Apply shift
        return shift(psf_data, shift_values, mode='constant')
    
    @staticmethod
    def interpolate_psf(psf_data: np.ndarray, 
                       original_spacing: Tuple[float, ...],
                       target_spacing: Tuple[float, ...]) -> np.ndarray:
        """
        Interpolate PSF to new spacing
        
        Args:
            psf_data: Input PSF
            original_spacing: Original (z,y,x) or (y,x) spacing
            target_spacing: Target spacing
        """
        # Create coordinate grids
        grids = [np.arange(s) * sp for s, sp in zip(psf_data.shape, original_spacing)]
        
        # Create interpolator
        interpolator = RegularGridInterpolator(grids, psf_data, bounds_error=False, fill_value=0)
        
        # Create target coordinate grid
        target_shape = [int(s * o / t) for s, o, t in 
                       zip(psf_data.shape, original_spacing, target_spacing)]
        target_grids = [np.arange(s) * sp for s, sp in zip(target_shape, target_spacing)]
        target_coords = np.meshgrid(*target_grids, indexing='ij')
        
        # Interpolate
        points = np.stack([x.ravel() for x in target_coords], axis=-1)
        interpolated = interpolator(points).reshape(target_shape)
        
        return interpolated

class PSFLoader:
    """Enhanced PSF loader with validation and processing capabilities"""
    
    def __init__(self, metadata: PSFMetadata):
        """Initialize with metadata"""
        self.metadata = metadata
        self.validator = PSFValidator()
        self.processor = PSFProcessor()
    
    def load_psf_from_array(self, psf_data: np.ndarray, 
                           center: bool = True) -> PSFEngine:
        """Enhanced PSF loading with validation and processing"""
        # Validate
        self.validator.validate_psf(psf_data)
        self.validator.validate_metadata(self.metadata)
        
        # Process
        if center:
            psf_data = self.processor.center_psf(psf_data)
        
        # Create parameters
        params = PSFParameters(
            wavelength=self.metadata.wavelength,
            numerical_aperture=self.metadata.numerical_aperture,
            pixel_size=self.metadata.pixel_size,
            z_step=self.metadata.z_step,
            refractive_index=self.metadata.refractive_index
        )
        
        # Create engine
        return CustomPSFEngine(params, psf_data, self.metadata)

    def load_psf_from_file(self, file_path: str, 
                          center: bool = True) -> PSFEngine:
        """Enhanced file loading with metadata handling"""
        file_path = Path(file_path)
        
        # Load data and metadata
        if file_path.suffix == '.npy':
            psf_data = np.load(file_path)
            # Look for metadata file
            meta_path = file_path.with_suffix('.json')
            if meta_path.exists():
                import json
                with open(meta_path) as f:
                    self.metadata.additional_params.update(json.load(f))
        elif file_path.suffix in ['.tif', '.tiff']:
            try:
                from tifffile import TiffFile
                with TiffFile(file_path) as tif:
                    psf_data = tif.asarray()
                    # Extract metadata from TIFF tags
                    if tif.is_imagej:
                        self.metadata.additional_params.update(tif.imagej_metadata)
            except ImportError:
                raise ImportError("tifffile package required for reading TIF files")
        else:
            raise ValueError("Unsupported file format")
            
        return self.load_psf_from_array(psf_data, center)

class CustomPSFEngine(PSFEngine):
    """Enhanced custom PSF engine with interpolation support"""
    
    def __init__(self, params: PSFParameters, psf_data: np.ndarray, 
                 metadata: PSFMetadata):
        super().__init__(params)
        self.psf_data = psf_data
        self.shape = psf_data.shape
        self.metadata = metadata
    
    def get_psf(self, size: Optional[Tuple[int, ...]] = None,
                spacing: Optional[Tuple[float, ...]] = None) -> np.ndarray:
        """
        Get PSF with optional resizing and resampling
        
        Args:
            size: Optional target size
            spacing: Optional target spacing
        """
        if size is None and spacing is None:
            return self.psf_data
            
        if spacing is not None:
            # Use interpolation for spacing change
            original_spacing = (self.params.z_step,) * 2 + (self.params.pixel_size,)
            return PSFProcessor.interpolate_psf(self.psf_data, original_spacing, spacing)
        else:
            # Use zoom for size change
            from scipy.ndimage import zoom
            scale_factors = [new_s / old_s for new_s, old_s in zip(size, self.shape)]
            return zoom(self.psf_data, scale_factors)