from ._filter_database import FilterDatabase
from .filters import (
    FilterSet,
    FilterSpectrum,
    create_allow_all_filter,
    create_bandpass_filter,
    create_tophat_filter,
)

# Create a default instance of the database
default_filter_db = FilterDatabase()


# Expose key database operations as module-level functions
def get_filter_set(name: str) -> FilterSet:
    """Load a filter set by name from the default database."""
    return default_filter_db.load_filter_set(name)


def save_filter_set(
    filter_set: FilterSet, description: str = "", is_sample: bool = False
) -> int:
    """Save a filter set to the default database."""
    return default_filter_db.store_filter_set(
        filter_set, description=description, is_sample=is_sample
    )


def list_filter_sets(sample_only: bool = False) -> list:
    """List all available filter sets in the default database."""
    return default_filter_db.list_filter_sets(sample_only=sample_only)


def delete_filter_set(name: str) -> None:
    """Delete a filter set from the default database."""
    default_filter_db.delete_filter_set(name)


# Initialize sample data if needed
def init_sample_data():
    """Initialize the database with sample filter sets."""
    default_filter_db.store_sample_filter_sets()


__all__ = [
    # Core classes
    "FilterSpectrum",
    "FilterSet",
    # Filter creation functions
    "create_bandpass_filter",
    "create_tophat_filter",
    "create_allow_all_filter",
    # Database class
    "FilterDatabase",
    # Default database instance
    "default_filter_db",
    # Convenience functions
    "get_filter_set",
    "save_filter_set",
    "list_filter_sets",
    "delete_filter_set",
    "init_sample_data",
]
