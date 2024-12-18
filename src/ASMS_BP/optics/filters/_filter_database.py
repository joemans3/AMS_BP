import sqlite3
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import json
from datetime import datetime
from pathlib import Path
from .filters import FilterSpectrum, FilterSet

class FilterDatabase:
    _instance = None

    def __new__(cls, db_path: Optional[str] = None) -> 'FilterDatabase':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[str] = None) -> None:
        # Only initialize once
        if self._initialized:
            return
            
        if db_path is None:
            # Create a directory for data storage in the module's directory
            module_dir = Path(__file__).parent
            data_dir = module_dir / "data"
            data_dir.mkdir(exist_ok=True)
            # Use the data directory for the database file
            self._db_path: str = str(data_dir / "_filters.db")
        else:
            self._db_path: str = db_path
            
        self._init_db()
        self._initialized = True

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS filter_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    creation_date TEXT,
                    is_sample BOOLEAN
                );

                CREATE TABLE IF NOT EXISTS filter_spectra (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filter_set_id INTEGER,
                    filter_type TEXT NOT NULL,  -- 'excitation', 'emission', or 'dichroic'
                    name TEXT NOT NULL,
                    wavelengths BLOB NOT NULL,
                    transmission BLOB NOT NULL,
                    FOREIGN KEY (filter_set_id) REFERENCES filter_sets(id)
                );
            """)

    def store_filter_set(self, filter_set: FilterSet, description: str = "", 
                        is_sample: bool = False) -> int:
        """Store a filter set in the database."""
        with sqlite3.connect(self._db_path) as conn:
            # Store filter set metadata
            cursor = conn.execute("""
                INSERT INTO filter_sets (name, description, creation_date, is_sample)
                VALUES (?, ?, ?, ?)
            """, (
                filter_set.name,
                description,
                datetime.now().isoformat(),
                is_sample
            ))
            
            filter_set_id = cursor.lastrowid
            
            # Store individual filters
            filters = {
                'excitation': filter_set.excitation,
                'emission': filter_set.emission,
                'dichroic': filter_set.dichroic
            }
            
            for filter_type, spectrum in filters.items():
                wavelengths_bytes = spectrum.wavelengths.tobytes()
                transmission_bytes = spectrum.transmission.tobytes()
                
                conn.execute("""
                    INSERT INTO filter_spectra 
                    (filter_set_id, filter_type, name, wavelengths, transmission)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    filter_set_id,
                    filter_type,
                    spectrum.name,
                    wavelengths_bytes,
                    transmission_bytes
                ))
            
            return filter_set_id

    def load_filter_set(self, name: str) -> FilterSet:
        """Load a filter set from the database."""
        with sqlite3.connect(self._db_path) as conn:
            # Get filter set ID
            cursor = conn.execute(
                "SELECT id FROM filter_sets WHERE name = ?", 
                (name,)
            )
            result = cursor.fetchone()
            if result is None:
                raise ValueError(f"Filter set '{name}' not found")
                
            filter_set_id = result[0]
            
            # Load all filter spectra for this set
            cursor = conn.execute("""
                SELECT filter_type, name, wavelengths, transmission
                FROM filter_spectra
                WHERE filter_set_id = ?
            """, (filter_set_id,))
            
            filters = {}
            for row in cursor.fetchall():
                wavelengths = np.frombuffer(row[2], dtype=np.float64)
                transmission = np.frombuffer(row[3], dtype=np.float64)
                
                spectrum = FilterSpectrum(
                    wavelengths=wavelengths,
                    transmission=transmission,
                    name=row[1]
                )
                filters[row[0]] = spectrum
            
            return FilterSet(
                name=name,
                excitation=filters['excitation'],
                emission=filters['emission'],
                dichroic=filters['dichroic']
            )

    def list_filter_sets(self, sample_only: bool = False) -> List[Dict[str, Any]]:
        """List all stored filter sets."""
        query = """
            SELECT name, description, creation_date, is_sample
            FROM filter_sets
        """
        if sample_only:
            query += " WHERE is_sample = 1"
            
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def delete_filter_set(self, name: str) -> None:
        """Delete a filter set from the database."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM filter_sets WHERE name = ?", 
                (name,)
            )
            result = cursor.fetchone()
            if result is None:
                raise ValueError(f"Filter set '{name}' not found")
                
            filter_set_id = result[0]
            
            conn.execute(
                "DELETE FROM filter_spectra WHERE filter_set_id = ?", 
                (filter_set_id,)
            )
            conn.execute(
                "DELETE FROM filter_sets WHERE id = ?", 
                (filter_set_id,)
            )

    def store_sample_filter_sets(self) -> None:
        """Store sample filter sets in the database."""
        from .filters import create_bandpass_filter, create_tophat_filter
        
        # Create a sample FITC filter set
        excitation = create_bandpass_filter(
            center_wavelength=495,
            bandwidth=25,
            name="FITC-Ex"
        )
        
        emission = create_bandpass_filter(
            center_wavelength=519,
            bandwidth=35,
            name="FITC-Em"
        )
        
        dichroic = create_tophat_filter(
            center_wavelength=509,
            bandwidth=40,
            edge_steepness=10,
            name="FITC-Di"
        )
        
        fitc_set = FilterSet(
            name="FITC Filter Set",
            excitation=excitation,
            emission=emission,
            dichroic=dichroic
        )
        
        self.store_filter_set(
            fitc_set,
            description="Sample FITC filter set",
            is_sample=True
        ) 