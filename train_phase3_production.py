#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 3 PRODUCTION TRAINING - SAVE MODELS
===========================================

Trains Phase 3 models on full DPT dataset and saves for production use.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "data/processed/mobility_dataset_merged.csv"
OUTPUT_DIR = "models/phase3"
RESULTS_DIR = "evaluation/phase3"

MAX_MOBILITY_OUTLIER = 500000
USE_LOG_TRANSFORM = True
ENHANCE_FEATURES = True
N_FOLDS = 5  # Reduced from 20 due to smaller dataset (32 samples)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(row, use_interactions=True):
    """
    Engineer feature set from INPUT variables only.

    IMPORTANT: Only uses bandgap (eg), effective_mass_e (m_e), effective_mass_h (m_h).
    Does NOT use electron_mobility or hole_mobility (the targets) to avoid data leakage.
    """
    features = np.zeros(45)

    try:
        # Input features ONLY - no target variables!
        eg = row['bandgap'] if pd.notna(row['bandgap']) else 1.5
        m_e = row['effective_mass_e'] if pd.notna(row['effective_mass_e']) else 0.5
        m_h = row['effective_mass_h'] if pd.notna(row['effective_mass_h']) else 0.5

        # Core features (0-11): Basic properties and ratios
        features[0] = eg
        features[1] = m_e
        features[2] = m_h
        features[3] = m_e / (m_h + 1e-6)
        features[4] = 1.0 / (m_e + m_h + 1e-6)
        features[5] = eg ** 2
        features[6] = m_e * m_h
        features[7] = (m_e + m_h) / 2
        features[8] = max(m_e, m_h) - min(m_e, m_h)
        features[9] = eg * m_e
        features[10] = eg * m_h
        features[11] = eg / (m_e + m_h + 1e-6)

        if use_interactions:
            # Polynomial interactions (12-21)
            features[12] = eg * m_e * m_h
            features[13] = eg ** 2 * m_e
            features[14] = eg ** 2 * m_h
            features[15] = eg / (m_e + 1e-6)
            features[16] = eg / (m_h + 1e-6)
            features[17] = m_e ** 2
            features[18] = m_h ** 2
            features[19] = (m_e ** 2 + m_h ** 2) / 2
            features[20] = (m_e * m_h) ** 0.5
            features[21] = m_e / (m_h ** 2 + 1e-6)

            # Higher order terms (22-26)
            features[22] = eg ** 3
            features[23] = m_e ** 3 + m_h ** 3
            features[24] = (m_e + m_h) ** 2
            features[25] = eg * (m_e + m_h) ** 2
            features[26] = eg ** 2 / ((m_e + m_h) ** 2 + 1e-6)

            # Nonlinear transforms (27-31)
            features[27] = 1.0 / (1.0 + np.exp(-eg))
            features[28] = np.exp(-m_e)
            features[29] = np.exp(-m_h)
            features[30] = np.sin(eg)
            features[31] = np.cos(m_e)

            # Fractional powers (32-36)
            features[32] = m_e ** (1/3)
            features[33] = m_h ** (1/3)
            features[34] = eg ** (1/2) if eg > 0 else 0
            features[35] = (m_e * eg) / (m_h + 1e-6)
            features[36] = (m_h * eg) / (m_e + 1e-6)

            # Combined terms (37-41)
            features[37] = (m_e + m_h) * eg
            features[38] = (m_e + m_h) / (eg + 1e-6)
            features[39] = m_e * eg + m_h * eg
            features[40] = m_e / (eg + 1e-6) + m_h / (eg + 1e-6)
            features[41] = (m_e + m_h) * (eg ** 2)

            # Log and mixed terms (42-44)
            features[42] = np.log(1 + m_e * m_h)
            features[43] = np.log(1 + eg * (m_e + m_h))
            features[44] = (m_e * m_h) ** 0.75

    except Exception as e:
        pass

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_and_save_models():
    """Train models on materials with complete input features"""

    print("\n" + "="*80)
    print("PHASE 3 PRODUCTION TRAINING - NO DATA LEAKAGE")
    print("="*80 + "\n")

    # Load data
    print("[*] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Loaded {len(df)} total materials")

    # CRITICAL: Only use materials with COMPLETE input features
    # Model needs: bandgap, effective_mass_e, effective_mass_h
    print("\n[*] Filtering to materials with complete input features...")
    print("    Required: bandgap, effective_mass_e, effective_mass_h")
    df_complete = df.dropna(subset=[
        'bandgap', 'effective_mass_e', 'effective_mass_h',
        'electron_mobility', 'hole_mobility'
    ]).copy()
    print(f"[OK] Materials with complete data: {len(df_complete)}")

    print("\n[*] Removing outliers (mobility > 500,000 cm2/(V*s))...")
    initial_count = len(df_complete)
    df_complete = df_complete[
        (df_complete['electron_mobility'] < MAX_MOBILITY_OUTLIER) &
        (df_complete['hole_mobility'] < MAX_MOBILITY_OUTLIER)
    ].copy()
    removed = initial_count - len(df_complete)
    print(f"[OK] Removed {removed} outliers, {len(df_complete)} remaining")

    # CRITICAL: Remove SiC from training data - we want to PREDICT its mobility
    print("\n[*] Removing SiC from training data (target for prediction)...")
    sic_mask = df_complete['formula'].str.contains('SiC', case=False, na=False)
    sic_removed = df_complete[sic_mask]
    if len(sic_removed) > 0:
        print(f"    Excluded materials: {sic_removed['formula'].tolist()}")
    df_complete = df_complete[~sic_mask].copy()
    print(f"[OK] Removed {sic_mask.sum()} SiC entries, {len(df_complete)} remaining")

    print(f"\n[OK] Final dataset: {len(df_complete)} materials with complete features")
    print("\n    Data sources:")
    for src, cnt in df_complete['source'].value_counts().items():
        print(f"      - {src}: {cnt}")
    
    # Extract features
    print("\n[*] Engineering 45D feature set (no target leakage)...")
    features_list = []
    for idx, row in tqdm(df_complete.iterrows(), total=len(df_complete)):
        feat = engineer_features(row, use_interactions=ENHANCE_FEATURES)
        features_list.append(feat)
    X = np.array(features_list)
    print(f"[OK] Feature matrix shape: {X.shape}")

    # Log-transform targets
    print("\n[*] Preparing targets (log-transform for scale reduction)...")
    y_electron = np.log(df_complete['electron_mobility'].values)
    y_hole = np.log(df_complete['hole_mobility'].values)
    print(f"[OK] Targets ready (log-transformed)")
    print(f"    Electron mobility range: {np.exp(y_electron.min()):.1f} - {np.exp(y_electron.max()):.1f} cm2/(V*s)")
    print(f"    Hole mobility range: {np.exp(y_hole.min()):.1f} - {np.exp(y_hole.max()):.1f} cm2/(V*s)")
    
    # Standardize features
    print("\n[*] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    scaler_path = os.path.join(OUTPUT_DIR, 'feature_scaler_phase3.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Scaler saved to {scaler_path}")
    
    # ========================================================================
    # TRAIN ELECTRON MODELS ON FULL DPT DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("TRAINING ELECTRON MOBILITY MODELS (FULL DPT DATA)")
    print("="*80 + "\n")
    
    # Random Forest for Electron
    print("[*] Training Random Forest (electron)...")
    rf_electron = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_electron.fit(X_scaled, y_electron)
    rf_e_path = os.path.join(OUTPUT_DIR, 'random_forest_electron.joblib')
    joblib.dump(rf_electron, rf_e_path)
    print(f"[OK] Saved to {rf_e_path}")
    
    # Gradient Boosting for Electron
    print("\n[*] Training Gradient Boosting (electron)...")
    gb_electron = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_electron.fit(X_scaled, y_electron)
    gb_e_path = os.path.join(OUTPUT_DIR, 'gradient_boosting_electron.joblib')
    joblib.dump(gb_electron, gb_e_path)
    print(f"[OK] Saved to {gb_e_path}")
    
    # ========================================================================
    # TRAIN HOLE MODELS ON FULL DPT DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("TRAINING HOLE MOBILITY MODELS (FULL DPT DATA)")
    print("="*80 + "\n")
    
    # Random Forest for Hole
    print("[*] Training Random Forest (hole)...")
    rf_hole = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_hole.fit(X_scaled, y_hole)
    rf_h_path = os.path.join(OUTPUT_DIR, 'random_forest_hole.joblib')
    joblib.dump(rf_hole, rf_h_path)
    print(f"[OK] Saved to {rf_h_path}")
    
    # Gradient Boosting for Hole
    print("\n[*] Training Gradient Boosting (hole)...")
    gb_hole = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_hole.fit(X_scaled, y_hole)
    gb_h_path = os.path.join(OUTPUT_DIR, 'gradient_boosting_hole.joblib')
    joblib.dump(gb_hole, gb_h_path)
    print(f"[OK] Saved to {gb_h_path}")
    
    # ========================================================================
    # EVALUATE WITH 20-FOLD CV
    # ========================================================================
    
    print("\n" + "="*80)
    print("EVALUATING WITH 20-FOLD CROSS-VALIDATION")
    print("="*80 + "\n")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Evaluate Electron
    print("[*] Evaluating electron mobility models...")
    rf_e_r2 = []
    gb_e_r2 = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_electron[train_idx], y_electron[val_idx]
        
        rf_e_tmp = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=3, 
                                         min_samples_leaf=1, random_state=42, n_jobs=-1)
        rf_e_tmp.fit(X_train, y_train)
        y_pred = rf_e_tmp.predict(X_val)
        r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
        rf_e_r2.append(r2)
        
        gb_e_tmp = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=7,
                                             min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42)
        gb_e_tmp.fit(X_train, y_train)
        y_pred = gb_e_tmp.predict(X_val)
        r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
        gb_e_r2.append(r2)
        
        if (fold + 1) % 5 == 0:
            print(f"  Fold {fold + 1}/{N_FOLDS}...")
    
    print(f"\n[OK] Electron R2: RF={np.mean(rf_e_r2):.4f}+/-{np.std(rf_e_r2):.4f}, GB={np.mean(gb_e_r2):.4f}+/-{np.std(gb_e_r2):.4f}")
    
    # Evaluate Hole
    print("\n[*] Evaluating hole mobility models...")
    rf_h_r2 = []
    gb_h_r2 = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_hole[train_idx], y_hole[val_idx]
        
        rf_h_tmp = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=3, 
                                         min_samples_leaf=1, random_state=42, n_jobs=-1)
        rf_h_tmp.fit(X_train, y_train)
        y_pred = rf_h_tmp.predict(X_val)
        r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
        rf_h_r2.append(r2)
        
        gb_h_tmp = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=7,
                                             min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42)
        gb_h_tmp.fit(X_train, y_train)
        y_pred = gb_h_tmp.predict(X_val)
        r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
        gb_h_r2.append(r2)
        
        if (fold + 1) % 5 == 0:
            print(f"  Fold {fold + 1}/{N_FOLDS}...")
    
    print(f"\n[OK] Hole R2: RF={np.mean(rf_h_r2):.4f}+/-{np.std(rf_h_r2):.4f}, GB={np.mean(gb_h_r2):.4f}+/-{np.std(gb_h_r2):.4f}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 3 PRODUCTION TRAINING COMPLETE!")
    print("="*80 + "\n")
    
    print("TRAINED MODELS SAVED:")
    print(f"  [OK] Random Forest (Electron): {rf_e_path}")
    print(f"  [OK] Gradient Boosting (Electron): {gb_e_path}")
    print(f"  [OK] Random Forest (Hole): {rf_h_path}")
    print(f"  [OK] Gradient Boosting (Hole): {gb_h_path}")
    print(f"  [OK] Feature Scaler: {scaler_path}")
    
    print("\nREADY FOR PRODUCTION INFERENCE!")
    print("\nUsage: python predict_mobility_production.py")

if __name__ == '__main__':
    train_and_save_models()
