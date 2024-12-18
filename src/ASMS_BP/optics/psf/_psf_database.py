import sqlite3
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import json
from datetime import datetime
import io
from pathlib import Path
from .load_psf import PSFMetadata
from .psf_engine import PSFParameters, PSFEngine

class PSFDatabase:
    _instance = None

    def __new__(cls, db_path: Optional[str] = None) -> 'PSFDatabase':
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
            self._db_path: str = str(data_dir / "_psfs.db")
        else:
            self._db_path: str = db_path
            
        self._init_db()
        self._initialized = True

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS psf_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    wavelength REAL,
                    numerical_aperture REAL,
                    pixel_size REAL,
                    z_step REAL,
                    refractive_index REAL,
                    microscope_type TEXT,
                    immersion_medium TEXT,
                    additional_params TEXT,
                    creation_date TEXT,
                    is_sample BOOLEAN,
                    description TEXT
                );

                CREATE TABLE IF NOT EXISTS psf_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metadata_id INTEGER,
                    data BLOB NOT NULL,
                    shape TEXT NOT NULL,
                    FOREIGN KEY (metadata_id) REFERENCES psf_metadata(id)
                );
            """)

    def store_psf(self, name: str, psf_data: np.ndarray, metadata: PSFMetadata, 
                  is_sample: bool = False, description: str = "") -> int:
        """Store PSF and its metadata in the database."""
        metadata_dict = metadata.model_dump()
        additional_params = json.dumps(metadata_dict.get('additional_params', {}))
        
        with sqlite3.connect(self._db_path) as conn:
            # Store metadata
            cursor = conn.execute("""
                INSERT INTO psf_metadata (
                    name, wavelength, numerical_aperture, pixel_size, z_step,
                    refractive_index, microscope_type, immersion_medium,
                    additional_params, creation_date, is_sample, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, metadata.wavelength, metadata.numerical_aperture,
                metadata.pixel_size, metadata.z_step, metadata.refractive_index,
                metadata.microscope_type, metadata.immersion_medium,
                additional_params, datetime.now().isoformat(),
                is_sample, description
            ))
            
            metadata_id = cursor.lastrowid
            
            # Store PSF data
            psf_bytes = io.BytesIO()
            np.save(psf_bytes, psf_data)
            shape = json.dumps(psf_data.shape)
            
            conn.execute("""
                INSERT INTO psf_data (metadata_id, data, shape)
                VALUES (?, ?, ?)
            """, (metadata_id, psf_bytes.getvalue(), shape))
            
            return metadata_id

    def load_psf(self, name: str) -> Tuple[np.ndarray, PSFMetadata]:
        """Load PSF and its metadata from the database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT m.*, d.data, d.shape
                FROM psf_metadata m
                JOIN psf_data d ON m.id = d.metadata_id
                WHERE m.name = ?
            """, (name,))
            
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"PSF '{name}' not found in database")
            
            # Convert row to metadata
            metadata_dict = {
                'wavelength': row['wavelength'],
                'numerical_aperture': row['numerical_aperture'],
                'pixel_size': row['pixel_size'],
                'z_step': row['z_step'],
                'refractive_index': row['refractive_index'],
                'microscope_type': row['microscope_type'],
                'immersion_medium': row['immersion_medium'],
                'additional_params': json.loads(row['additional_params']) if row['additional_params'] else {}
            }
            metadata = PSFMetadata(**metadata_dict)
            
            # Load PSF data
            psf_bytes = io.BytesIO(row['data'])
            psf_data = np.load(psf_bytes)
            
            return psf_data, metadata

    def list_psfs(self, sample_only: bool = False) -> List[Dict[str, Any]]:
        """List all stored PSFs."""
        query = """
            SELECT name, wavelength, numerical_aperture, creation_date, 
                   is_sample, description
            FROM psf_metadata
        """
        if sample_only:
            query += " WHERE is_sample = 1"
            
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def delete_psf(self, name: str) -> None:
        """Delete a PSF and its metadata from the database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                DELETE FROM psf_data 
                WHERE metadata_id IN (
                    SELECT id FROM psf_metadata WHERE name = ?
                )
            """, (name,))
            conn.execute("DELETE FROM psf_metadata WHERE name = ?", (name,))

    def store_sample_psfs(self) -> None:
        """Store sample PSFs in the database."""
        params = PSFParameters(
            wavelength=488,
            numerical_aperture=1.4,
            pixel_size=65,
            z_step=100,
            refractive_index=1.518
        )
        engine = PSFEngine(params)
        
        # Store sample 2D PSFs
        psf_2d = engine.airy_disk((64, 64))
        metadata = PSFMetadata(
            wavelength=488,
            numerical_aperture=1.4,
            pixel_size=65,
            z_step=100,
            microscope_type='widefield',
            description="Sample 2D Airy disk PSF"
        )
        self.store_psf("sample_airy_2d", psf_2d, metadata, is_sample=True)
        
        # Store sample 3D PSFs
        psf_3d = engine.airy_disk_3d((32, 64, 64))
        metadata = PSFMetadata(
            wavelength=488,
            numerical_aperture=1.4,
            pixel_size=65,
            z_step=100,
            description="Sample 3D Airy disk PSF"
        )
        self.store_psf("sample_airy_3d", psf_3d, metadata, is_sample=True)
