#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 2-4 COMPLETE TRAINING PIPELINE
=====================================

This comprehensive script:
1. Loads merged dataset from Phase 1
2. Extracts and engineers features (Phase 2)
3. Trains separate electron/hole models (Phase 3)
4. Implements ensemble methods (Phase 4)
5. Performs cross-validation and evaluation

Expected runtime: 2-3 hours on CPU
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from joblib import dump, load
import json

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent

print("\n" + "="*80)
print("PHASE 2-4: COMPLETE MODEL TRAINING PIPELINE")
print("="*80)

# =============================================================================
# PHASE 2: FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*80)
print("PHASE 2: ENHANCED FEATURE ENGINEERING")
print("="*80)

def load_merged_dataset():
    """Load the merged dataset from Phase 1."""
    path = PROJECT_ROOT / 'data_processed' / 'mobility_dataset_merged.csv'
    df = pd.read_csv(path)
    print(f"Loaded merged dataset: {len(df)} materials")
    return df

def engineer_features(df):
    """
    Engineer rich features from material properties.
    
    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (40+ features per sample)
    feature_names : list
        Names of features for interpretability
    """
    print("\nEngineering features...")
    
    features = []
    feature_names = []
    
    # 1. DIRECT PROPERTIES (8 features)
    for col in ['bandgap', 'effective_mass_e', 'effective_mass_h']:
        if col in df.columns:
            val = df[col].fillna(df[col].mean()).values
            features.append(val)
            feature_names.append(col)
    
    # 2. DERIVED ELECTRONIC FEATURES (5 features)
    # Electron/hole mass ratio
    mass_ratio = df['effective_mass_e'].fillna(1.0) / (df['effective_mass_h'].fillna(1.0) + 1e-6)
    features.append(np.log1p(mass_ratio))
    feature_names.append('log_mass_ratio')
    
    # Mass sum and difference
    mass_sum = df['effective_mass_e'].fillna(0) + df['effective_mass_h'].fillna(0)
    features.append(mass_sum)
    feature_names.append('mass_sum')
    
    mass_diff = np.abs(df['effective_mass_e'].fillna(0) - df['effective_mass_h'].fillna(0))
    features.append(mass_diff)
    feature_names.append('mass_diff')
    
    # Bandgap categories
    bg = df['bandgap'].fillna(0)
    features.append((bg > 2.0).astype(float))
    feature_names.append('large_bandgap_flag')
    
    features.append((bg > 5.0).astype(float))
    feature_names.append('insulator_flag')
    
    # 3. COMPOSITION-BASED FEATURES (via formula parsing - 8 features)
    for idx, row in df.iterrows():
        formula = str(row['formula'])
        
        # Count atoms
        n_atoms = len([c for c in formula if c.isupper()])
        n_letters = len([c for c in formula if c.isalpha()])
        
        if idx == 0:
            atom_counts = []
            letter_counts = []
        
        atom_counts.append(n_atoms)
        letter_counts.append(n_letters)
    
    features.append(np.array(atom_counts))
    feature_names.append('n_atoms')
    
    features.append(np.array(letter_counts))
    feature_names.append('n_elements')
    
    # Complexity metrics
    complexity = np.array(atom_counts) * np.array(letter_counts)
    features.append(complexity)
    feature_names.append('material_complexity')
    
    # 4. SOURCE AND QUALITY FEATURES (3 features)
    is_experimental = (df['quality_flag'].str.contains('experimental', case=False, na=False)).astype(float)
    features.append(is_experimental)
    feature_names.append('is_experimental')
    
    is_dft = (df['quality_flag'].str.contains('DFT', case=False, na=False)).astype(float)
    features.append(is_dft)
    feature_names.append('is_dft_calculated')
    
    n_sources = df['n_sources'].fillna(1).values
    features.append(np.log1p(n_sources))
    feature_names.append('log_n_sources')
    
    # 5. BANDGAP REGIONS (6 features - one-hot-ish encoding)
    bg = df['bandgap'].fillna(2.0)
    
    regions = {
        'semimetal': (bg < 0.5),
        'narrow_gap': ((bg >= 0.5) & (bg < 1.5)),
        'direct_gap': ((bg >= 1.5) & (bg < 3.0)),
        'wide_gap': ((bg >= 3.0) & (bg < 5.0)),
        'insulator': (bg >= 5.0)
    }
    
    for region_name, mask in regions.items():
        features.append(mask.astype(float))
        feature_names.append(f'region_{region_name}')
    
    # Stack all features
    X = np.column_stack(features)
    
    print(f"Generated {X.shape[1]} features")
    print(f"Feature names: {feature_names}")
    
    return X, feature_names

# Load and process data
df = load_merged_dataset()
X, feature_names = engineer_features(df)

# Prepare targets
y_electron = df['electron_mobility'].values
y_hole = df['hole_mobility'].values

# Remove rows with NaN targets
valid_idx = (~np.isnan(y_electron)) & (~np.isnan(y_hole))
X = X[valid_idx]
y_electron = y_electron[valid_idx]
y_hole = y_hole[valid_idx]

print(f"\nFinal dataset: {len(X)} samples, {X.shape[1]} features")

# =============================================================================
# PHASE 3 & 4: MODEL TRAINING
# =============================================================================
print("\n" + "="*80)
print("PHASE 3-4: SEPARATE ELECTRON/HOLE MODEL TRAINING WITH ENSEMBLES")
print("="*80)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
dump(scaler, PROJECT_ROOT / 'models' / 'feature_scaler.joblib')

# Set up cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

def train_model_ensemble(X, y, target_name, k_splits=10):
    """
    Train ensemble of models for a single target.
    
    Models:
    1. XGBoost
    2. Random Forest
    3. Gradient Boosting
    4. Ridge Regression (meta-learner)
    """
    print(f"\n{'='*70}")
    print(f"Training ensemble for {target_name.upper()} MOBILITY")
    print(f"{'='*70}")
    
    results = {
        'model': target_name,
        'n_samples': len(X),
        'cv_scores': {},
        'base_models': {},
        'cv_metrics': []
    }
    
    # Base model parameters
    xgb_params = {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'verbosity': 0
    }
    
    rf_params = {
        'n_estimators': 150,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    gb_params = {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'random_state': 42
    }
    
    # Lists to store predictions for stacking
    xgb_cv_preds = np.zeros(len(X))
    rf_cv_preds = np.zeros(len(X))
    gb_cv_preds = np.zeros(len(X))
    meta_features = np.zeros((len(X), 3))
    
    cv_idx = 0
    
    # K-Fold Cross-Validation
    for train_idx, test_idx in kfold.split(X):
        cv_idx += 1
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"\n--- Fold {cv_idx}/10 ---")
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, verbose=False)
        xgb_pred = xgb_model.predict(X_test)
        xgb_cv_preds[test_idx] = xgb_pred
        meta_features[test_idx, 0] = xgb_pred
        
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_r2 = r2_score(y_test, xgb_pred)
        print(f"  XGBoost - RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.4f}")
        
        # Random Forest
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_cv_preds[test_idx] = rf_pred
        meta_features[test_idx, 1] = rf_pred
        
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        print(f"  Random Forest - RMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}")
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(**gb_params)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_cv_preds[test_idx] = gb_pred
        meta_features[test_idx, 2] = gb_pred
        
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
        gb_r2 = r2_score(y_test, gb_pred)
        print(f"  Gradient Boosting - RMSE: {gb_rmse:.2f}, R²: {gb_r2:.4f}")
        
        # Ensemble average
        ensemble_pred = np.mean([xgb_pred, rf_pred, gb_pred], axis=0)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        print(f"  Ensemble Average - RMSE: {ensemble_rmse:.2f}, R²: {ensemble_r2:.4f}")
        
        cv_metrics = {
            'fold': cv_idx,
            'xgb_rmse': float(xgb_rmse),
            'xgb_r2': float(xgb_r2),
            'rf_rmse': float(rf_rmse),
            'rf_r2': float(rf_r2),
            'gb_rmse': float(gb_rmse),
            'gb_r2': float(gb_r2),
            'ensemble_rmse': float(ensemble_rmse),
            'ensemble_r2': float(ensemble_r2)
        }
        results['cv_metrics'].append(cv_metrics)
    
    # Train final models on full dataset
    print(f"\n--- Training final models on full dataset ---")
    
    xgb_final = xgb.XGBRegressor(**xgb_params)
    xgb_final.fit(X_scaled, y, verbose=False)
    
    rf_final = RandomForestRegressor(**rf_params)
    rf_final.fit(X_scaled, y)
    
    gb_final = GradientBoostingRegressor(**gb_params)
    gb_final.fit(X_scaled, y)
    
    # Save base models
    models_path = PROJECT_ROOT / 'models' / 'final'
    models_path.mkdir(parents=True, exist_ok=True)
    
    dump(xgb_final, models_path / f'xgboost_{target_name}.joblib')
    dump(rf_final, models_path / f'random_forest_{target_name}.joblib')
    dump(gb_final, models_path / f'gradient_boosting_{target_name}.joblib')
    
    print(f"Saved base models for {target_name}")
    
    # Summary statistics
    summary = {
        'xgb': {
            'rmse_mean': np.mean([m['xgb_rmse'] for m in results['cv_metrics']]),
            'rmse_std': np.std([m['xgb_rmse'] for m in results['cv_metrics']]),
            'r2_mean': np.mean([m['xgb_r2'] for m in results['cv_metrics']]),
            'r2_std': np.std([m['xgb_r2'] for m in results['cv_metrics']])
        },
        'rf': {
            'rmse_mean': np.mean([m['rf_rmse'] for m in results['cv_metrics']]),
            'rmse_std': np.std([m['rf_rmse'] for m in results['cv_metrics']]),
            'r2_mean': np.mean([m['rf_r2'] for m in results['cv_metrics']]),
            'r2_std': np.std([m['rf_r2'] for m in results['cv_metrics']])
        },
        'gb': {
            'rmse_mean': np.mean([m['gb_rmse'] for m in results['cv_metrics']]),
            'rmse_std': np.std([m['gb_rmse'] for m in results['cv_metrics']]),
            'r2_mean': np.mean([m['gb_r2'] for m in results['cv_metrics']]),
            'r2_std': np.std([m['gb_r2'] for m in results['cv_metrics']])
        },
        'ensemble': {
            'rmse_mean': np.mean([m['ensemble_rmse'] for m in results['cv_metrics']]),
            'rmse_std': np.std([m['ensemble_rmse'] for m in results['cv_metrics']]),
            'r2_mean': np.mean([m['ensemble_r2'] for m in results['cv_metrics']]),
            'r2_std': np.std([m['ensemble_r2'] for m in results['cv_metrics']])
        }
    }
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY FOR {target_name.upper()}")
    print(f"{'='*70}")
    
    for model_name, metrics in summary.items():
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE: {metrics['rmse_mean']:.2f} ± {metrics['rmse_std']:.2f}")
        print(f"  R²:   {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
    
    return summary, results

# Train models for both targets
electron_summary, electron_results = train_model_ensemble(X_scaled, y_electron, 'electron')
hole_summary, hole_results = train_model_ensemble(X_scaled, y_hole, 'hole')

# Save results
results_path = PROJECT_ROOT / 'evaluation' / 'training_results.json'
results_path.parent.mkdir(parents=True, exist_ok=True)

training_summary = {
    'feature_names': feature_names,
    'n_samples': len(X),
    'n_features': X.shape[1],
    'electron_mobility_results': electron_summary,
    'hole_mobility_results': hole_summary
}

with open(results_path, 'w') as f:
    json.dump(training_summary, f, indent=2)

print(f"\n{'='*80}")
print("PHASE 2-4 COMPLETE!")
print(f"{'='*80}")
print(f"\nAll models trained and saved to: {PROJECT_ROOT / 'models' / 'final'}")
print(f"Results saved to: {results_path}")
print(f"\nNext: Run evaluation and analysis scripts")

