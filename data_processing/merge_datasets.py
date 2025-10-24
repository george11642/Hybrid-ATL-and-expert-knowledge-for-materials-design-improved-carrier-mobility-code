#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge and Process Multiple 2D Materials Mobility Datasets

This script:
1. Loads existing DPTmobility.csv and EPCmobility.csv
2. Loads newly fetched eTran2D and C2DB databases
3. Standardizes units and formats
4. Handles duplicates by averaging values
5. Generates quality flags
6. Outputs unified dataset
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

def load_existing_dpt_data():
    """Load existing DPTmobility.csv data."""
    print("\n" + "="*70)
    print("LOADING EXISTING DPT DATA")
    print("="*70)
    
    path = PROJECT_ROOT / 'DPTmobility.csv'
    
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()
    
    try:
        # Try different encodings
        try:
            df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')
        except:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        print(f"Loaded DPTmobility.csv: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()

def load_existing_epc_data():
    """Load existing EPCmobility.csv data."""
    print("\n" + "="*70)
    print("LOADING EXISTING EPC DATA")
    print("="*70)
    
    path = PROJECT_ROOT / 'EPCmobility.csv'
    
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()
    
    try:
        try:
            df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')
        except:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        print(f"Loaded EPCmobility.csv: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()

def load_etran2d_data():
    """Load fetched eTran2D data."""
    print("\n" + "="*70)
    print("LOADING eTran2D DATA")
    print("="*70)
    
    path = PROJECT_ROOT / 'data_acquisition' / 'etran2d_raw.csv'
    
    if not path.exists():
        print(f"Warning: {path} not found. Skipping eTran2D.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        print(f"Loaded eTran2D data: {len(df)} records")
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()

def load_c2db_data():
    """Load fetched C2DB data."""
    print("\n" + "="*70)
    print("LOADING C2DB DATA")
    print("="*70)
    
    path = PROJECT_ROOT / 'data_acquisition' / 'c2db_raw.csv'
    
    if not path.exists():
        print(f"Warning: {path} not found. Skipping C2DB.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        print(f"Loaded C2DB data: {len(df)} records")
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()

def standardize_dpt_format(df):
    """
    Standardize DPTmobility.csv format.
    Columns: Pretty_formula, μe(10^3 cm^2/(V·s)), x, y, μh(10^3 cm^2/(V·s)), x, y, source
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Find mobility columns (they have special characters)
        cols = [col for col in df.columns if 'e(' in str(col) or 'μ' in str(col)]
        
        result = []
        for idx, row in df.iterrows():
            formula = row.iloc[0]  # First column is formula
            
            # Extract values - DPT data is in 10^3 cm^2/(V·s), convert to cm^2/(V·s)
            mu_e_value = row.iloc[1] * 1000 if len(row) > 1 else np.nan
            mu_h_value = row.iloc[4] * 1000 if len(row) > 4 else np.nan
            
            record = {
                'formula': formula,
                'electron_mobility': mu_e_value,
                'hole_mobility': mu_h_value,
                'bandgap': np.nan,
                'effective_mass_e': np.nan,
                'effective_mass_h': np.nan,
                'source': 'DPT_experimental',
                'quality_flag': 'experimental'
            }
            result.append(record)
        
        return pd.DataFrame(result)
    except Exception as e:
        print(f"Error standardizing DPT format: {e}")
        return pd.DataFrame()

def standardize_epc_format(df):
    """
    Standardize EPCmobility.csv format.
    Columns: Formula, μe(cm^2/(V·s)), μh(cm^2/(V·s)), Source
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        result = []
        for idx, row in df.iterrows():
            formula = row.iloc[0]
            mu_e_value = row.iloc[1] if len(row) > 1 else np.nan
            mu_h_value = row.iloc[2] if len(row) > 2 else np.nan
            source = str(row.iloc[3]) if len(row) > 3 else 'EPC_experimental'
            
            record = {
                'formula': formula,
                'electron_mobility': mu_e_value,
                'hole_mobility': mu_h_value,
                'bandgap': np.nan,
                'effective_mass_e': np.nan,
                'effective_mass_h': np.nan,
                'source': source,
                'quality_flag': 'experimental'
            }
            result.append(record)
        
        return pd.DataFrame(result)
    except Exception as e:
        print(f"Error standardizing EPC format: {e}")
        return pd.DataFrame()

def merge_dataframes(dfs_dict):
    """
    Merge multiple dataframes, handling duplicates by averaging.
    
    Parameters:
    -----------
    dfs_dict : dict
        Dictionary with dataset names and dataframes
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("\n" + "="*70)
    print("MERGING DATASETS")
    print("="*70)
    
    # Filter out empty dataframes
    dfs_to_merge = {name: df for name, df in dfs_dict.items() if not df.empty}
    
    if not dfs_to_merge:
        print("No data to merge!")
        return pd.DataFrame()
    
    print(f"Merging {len(dfs_to_merge)} datasets:")
    for name, df in dfs_to_merge.items():
        print(f"  - {name}: {len(df)} records")
    
    # Concatenate all dataframes
    merged = pd.concat([df for df in dfs_to_merge.values()], ignore_index=True)
    print(f"\nTotal records before deduplication: {len(merged)}")
    
    # Group by formula and aggregate
    def aggregate_row(group):
        row = {
            'formula': group['formula'].iloc[0],
            'electron_mobility': group['electron_mobility'].mean(),
            'hole_mobility': group['hole_mobility'].mean(),
            'bandgap': group['bandgap'].mean(),
            'effective_mass_e': group['effective_mass_e'].mean(),
            'effective_mass_h': group['effective_mass_h'].mean(),
            'source': ' + '.join(group['source'].unique()),
            'quality_flag': ' + '.join(group['quality_flag'].unique()),
            'n_sources': len(group),
        }
        return pd.Series(row)
    
    merged_unique = merged.groupby('formula', as_index=False).apply(aggregate_row)
    
    print(f"Total unique materials: {len(merged_unique)}")
    print(f"Duplicates removed: {len(merged) - len(merged_unique)}")
    
    return merged_unique

def validate_and_clean(df):
    """
    Validate and clean the merged dataset.
    
    - Remove extreme outliers
    - Fill missing values where possible
    - Flag suspicious data
    """
    print("\n" + "="*70)
    print("VALIDATING AND CLEANING DATA")
    print("="*70)
    
    initial_count = len(df)
    
    # Remove records with both mobilities missing
    df = df.dropna(subset=['electron_mobility', 'hole_mobility'], how='all')
    print(f"After removing missing mobility data: {len(df)} records")
    
    # Flag extreme outliers (mobility > 10^6 or < 0.1 cm^2/(V·s))
    outlier_mask = (
        ((df['electron_mobility'] > 1e6) | (df['electron_mobility'] < 0.1)) |
        ((df['hole_mobility'] > 1e6) | (df['hole_mobility'] < 0.1))
    )
    
    df.loc[outlier_mask, 'quality_flag'] = df.loc[outlier_mask, 'quality_flag'] + ' + outlier_flag'
    n_outliers = outlier_mask.sum()
    print(f"Flagged {n_outliers} potential outliers")
    
    # Log-space statistics
    print(f"\nElectron Mobility Statistics (cm^2/(V·s)):")
    valid_e = df['electron_mobility'].dropna()
    if len(valid_e) > 0:
        print(f"  Count: {len(valid_e)}")
        print(f"  Min: {valid_e.min():.2e}")
        print(f"  Max: {valid_e.max():.2e}")
        print(f"  Geometric Mean: {np.exp(np.log(valid_e[valid_e > 0]).mean()):.2e}")
        print(f"  Median: {valid_e.median():.2e}")
    
    print(f"\nHole Mobility Statistics (cm^2/(V·s)):")
    valid_h = df['hole_mobility'].dropna()
    if len(valid_h) > 0:
        print(f"  Count: {len(valid_h)}")
        print(f"  Min: {valid_h.min():.2e}")
        print(f"  Max: {valid_h.max():.2e}")
        print(f"  Geometric Mean: {np.exp(np.log(valid_h[valid_h > 0]).mean()):.2e}")
        print(f"  Median: {valid_h.median():.2e}")
    
    return df

def save_merged_dataset(df, output_path=None):
    """Save merged dataset to CSV."""
    if output_path is None:
        output_path = PROJECT_ROOT / 'data_processed' / 'mobility_dataset_merged.csv'
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nMerged dataset saved to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    return output_path

def generate_statistics_report(df, output_path=None):
    """Generate statistics report."""
    if output_path is None:
        output_path = PROJECT_ROOT / 'data_processed' / 'dataset_statistics.txt'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("2D MATERIALS MOBILITY DATASET - STATISTICS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Materials: {len(df)}\n")
        f.write(f"Total Records: {df['n_sources'].sum() if 'n_sources' in df.columns else len(df)}\n")
        f.write(f"Columns: {list(df.columns)}\n\n")
        
        f.write("DATA SOURCES:\n")
        for source in df['source'].unique():
            count = (df['source'].str.contains(source, na=False)).sum()
            f.write(f"  - {source}: {count} materials\n")
        
        f.write("\nQUALITY FLAGS:\n")
        for flag in df['quality_flag'].unique():
            count = (df['quality_flag'].str.contains(flag, na=False)).sum()
            f.write(f"  - {flag}: {count} materials\n")
        
        f.write("\nELECTRON MOBILITY STATISTICS (cm^2/(V·s)):\n")
        valid_e = df['electron_mobility'].dropna()
        f.write(f"  Count: {len(valid_e)}\n")
        f.write(f"  Min: {valid_e.min():.2e}\n")
        f.write(f"  Max: {valid_e.max():.2e}\n")
        f.write(f"  Mean: {valid_e.mean():.2e}\n")
        f.write(f"  Median: {valid_e.median():.2e}\n")
        f.write(f"  Std Dev: {valid_e.std():.2e}\n")
        
        f.write("\nHOLE MOBILITY STATISTICS (cm^2/(V·s)):\n")
        valid_h = df['hole_mobility'].dropna()
        f.write(f"  Count: {len(valid_h)}\n")
        f.write(f"  Min: {valid_h.min():.2e}\n")
        f.write(f"  Max: {valid_h.max():.2e}\n")
        f.write(f"  Mean: {valid_h.mean():.2e}\n")
        f.write(f"  Median: {valid_h.median():.2e}\n")
        f.write(f"  Std Dev: {valid_h.std():.2e}\n")
        
        f.write("\nBANDGAP STATISTICS (eV):\n")
        valid_bg = df['bandgap'].dropna()
        if len(valid_bg) > 0:
            f.write(f"  Count: {len(valid_bg)}\n")
            f.write(f"  Min: {valid_bg.min():.2f}\n")
            f.write(f"  Max: {valid_bg.max():.2f}\n")
            f.write(f"  Mean: {valid_bg.mean():.2f}\n")
            f.write(f"  Median: {valid_bg.median():.2f}\n")
        
        f.write("\nSAMPLE MATERIALS:\n")
        f.write(df[['formula', 'electron_mobility', 'hole_mobility', 'bandgap', 'source']].head(20).to_string())
    
    print(f"Statistics report saved to: {output_path}")

def main():
    """Main processing pipeline."""
    print("\n" + "="*70)
    print("PHASE 1: DATA ACQUISITION AND INTEGRATION")
    print("="*70)
    
    # Load all datasets
    dpt_df = load_existing_dpt_data()
    epc_df = load_existing_epc_data()
    etran2d_df = load_etran2d_data()
    c2db_df = load_c2db_data()
    
    # Standardize formats
    print("\n" + "="*70)
    print("STANDARDIZING FORMATS")
    print("="*70)
    
    dpt_standardized = standardize_dpt_format(dpt_df)
    print(f"DPT standardized: {len(dpt_standardized)} records")
    
    epc_standardized = standardize_epc_format(epc_df)
    print(f"EPC standardized: {len(epc_standardized)} records")
    
    # Merge all datasets
    datasets_dict = {
        'DPT': dpt_standardized,
        'EPC': epc_standardized,
        'eTran2D': etran2d_df,
        'C2DB': c2db_df,
    }
    
    merged_df = merge_dataframes(datasets_dict)
    
    # Validate and clean
    cleaned_df = validate_and_clean(merged_df)
    
    # Save results
    save_merged_dataset(cleaned_df)
    generate_statistics_report(cleaned_df)
    
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE!")
    print("="*70)
    print(f"\nFinal dataset: {len(cleaned_df)} unique 2D materials")
    print(f"Ready for Phase 2: Feature Engineering")
    
    return cleaned_df

if __name__ == '__main__':
    main()
