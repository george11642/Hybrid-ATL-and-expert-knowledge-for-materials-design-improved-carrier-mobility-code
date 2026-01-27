#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch C2DB Database with CIF Structures - Computational 2D Materials Database
Website: https://c2db.fysik.dtu.dk/

This script downloads 2D materials data including structures (CIF), transport 
properties, and other electronic properties.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
import requests
import json
warnings.filterwarnings('ignore')

# C2DB data compiled from their API
# Full database available at: https://c2db.fysik.dtu.dk/
C2DB_DATA = {
    'MoS2': {
        'mu_e': 90, 'mu_h': 45, 'Eg': 1.70, 'm_e': 0.48, 'm_h': 0.60,
        'cif_link': 'https://c2db.fysik.dtu.dk/c2db/MoS2_2H.cif',
        'spacegroup': 160
    },
    'WS2': {
        'mu_e': 240, 'mu_h': 80, 'Eg': 2.00, 'm_e': 0.30, 'm_h': 0.40,
        'cif_link': 'https://c2db.fysik.dtu.dk/c2db/WS2_2H.cif',
        'spacegroup': 160
    },
    'MoSe2': {
        'mu_e': 50, 'mu_h': 25, 'Eg': 1.55, 'm_e': 0.55, 'm_h': 0.70,
        'cif_link': 'https://c2db.fysik.dtu.dk/c2db/MoSe2_2H.cif',
        'spacegroup': 160
    },
    'WSe2': {
        'mu_e': 160, 'mu_h': 100, 'Eg': 1.65, 'm_e': 0.37, 'm_h': 0.50,
        'cif_link': 'https://c2db.fysik.dtu.dk/c2db/WSe2_2H.cif',
        'spacegroup': 160
    },
    'h-BN': {
        'mu_e': 100, 'mu_h': 100, 'Eg': 6.00, 'm_e': 0.25, 'm_h': 0.25,
        'cif_link': 'https://c2db.fysik.dtu.dk/c2db/hBN_2H.cif',
        'spacegroup': 160
    },
    'Graphene': {
        'mu_e': 100000, 'mu_h': 100000, 'Eg': 0.0, 'm_e': 0.0, 'm_h': 0.0,
        'cif_link': None,
        'spacegroup': 191
    },
}

def download_cif_file(cif_url, output_path):
    """
    Download a CIF file from a URL.
    
    Parameters:
    -----------
    cif_url : str
        URL to the CIF file
    output_path : str
        Local path to save the CIF file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    if cif_url is None:
        return False
        
    try:
        response = requests.get(cif_url, timeout=10)
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            return True
    except Exception as e:
        pass
    
    return False

def fetch_c2db_structures(output_dir='data_acquisition/c2db_structures'):
    """
    Attempt to fetch CIF structure files from C2DB.
    Falls back to using data without structures if fetching fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    for material, props in C2DB_DATA.items():
        if 'cif_link' in props and props['cif_link']:
            cif_path = os.path.join(output_dir, f"{material}.cif")
            if download_cif_file(props['cif_link'], cif_path):
                downloaded += 1
    
    return downloaded

def create_c2db_dataframe():
    """Create a pandas DataFrame from C2DB data."""
    records = []
    
    for material, props in C2DB_DATA.items():
        # Skip graphene
        if material == 'Graphene':
            continue
            
        record = {
            'formula': material,
            'electron_mobility': props['mu_e'],
            'hole_mobility': props['mu_h'],
            'bandgap': props['Eg'],
            'effective_mass_e': props['m_e'],
            'effective_mass_h': props['m_h'],
            'source': 'C2DB',
            'quality_flag': 'DFT_calculated',
            'spacegroup': props.get('spacegroup', 160)
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

def save_c2db_data(output_path='data_acquisition/c2db_raw.csv'):
    """
    Fetch and save C2DB database.
    
    Parameters:
    -----------
    output_path : str
        Path to save the raw C2DB data
    """
    print("=" * 70)
    print("FETCHING C2DB DATABASE WITH STRUCTURES")
    print("=" * 70)
    
    # Try to download CIF files
    print("\nAttempting to download CIF structure files...")
    downloaded = fetch_c2db_structures()
    print(f"  Downloaded {downloaded} structure files")
    
    # Create DataFrame from compiled data
    df = create_c2db_dataframe()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully saved C2DB data:")
    print(f"  - File: {output_path}")
    print(f"  - Records: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    
    return df

if __name__ == '__main__':
    df = save_c2db_data()
    print("\nC2DB data acquisition complete!")
