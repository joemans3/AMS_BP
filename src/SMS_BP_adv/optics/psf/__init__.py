from .psf_engine import PSFEngine, PSFParameters
from .load_psf import (
    PSFMetadata,
    PSFLoader,
    PSFProcessor,
    PSFValidator,
    CustomPSFEngine
)
from ._psf_database import PSFDatabase


# Create a default instance of the database
default_psf_db = PSFDatabase()
# store sample PSFs if empty
if not default_psf_db.list_psfs():
    default_psf_db.store_sample_psfs()

# Expose key database operations as module-level functions
def get_psf(name: str):
    """Load a PSF by name from the default database."""
    return default_psf_db.load_psf(name)

def save_psf(name: str, psf_data, metadata: PSFMetadata, 
             is_sample: bool = False, description: str = "") -> None:
    """Save a PSF to the default database."""
    default_psf_db.store_psf(name, psf_data, metadata, is_sample, description)

def list_psfs(sample_only: bool = False) -> list:
    """List all available PSFs in the default database."""
    return default_psf_db.list_psfs(sample_only)

def delete_psf(name: str) -> None:
    """Delete a PSF from the default database."""
    default_psf_db.delete_psf(name)

__all__ = [
    # Core classes
    'PSFEngine',
    'PSFParameters',
    'PSFMetadata',
    'PSFLoader',
    'PSFProcessor',
    'PSFValidator',
    'CustomPSFEngine',
    'PSFDatabase',
    
    # Default database instance
    'default_psf_db',
    
    # Convenience functions
    'get_psf',
    'save_psf',
    'list_psfs',
    'delete_psf',
]