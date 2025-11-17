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

DATASET_PATH = "data_processed/mobility_dataset_merged.csv"
OUTPUT_DIR = "models/phase3"
RESULTS_DIR = "evaluation/phase3"

MAX_MOBILITY_OUTLIER = 500000
USE_LOG_TRANSFORM = True
ENHANCE_FEATURES = True
N_FOLDS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(row, use_interactions=True):
    """Engineer 60D feature set with interaction terms"""
    features = np.zeros(60)
    
    try:
        eg = row['bandgap'] if pd.notna(row['bandgap']) else 1.5
        m_e = row['effective_mass_e'] if pd.notna(row['effective_mass_e']) else 0.5
        m_h = row['effective_mass_h'] if pd.notna(row['effective_mass_h']) else 0.5
        mu_e = row['electron_mobility']
        mu_h = row['hole_mobility']
        
        features[0] = eg
        features[1] = m_e
        features[2] = m_h
        features[3] = mu_e / (mu_h + 1e-6)
        features[4] = m_e / (m_h + 1e-6)
        features[5] = 1.0 / (m_e + m_h + 1e-6)
        features[6] = eg ** 2
        features[7] = m_e * m_h
        features[8] = (m_e + m_h) / 2
        features[9] = max(m_e, m_h) - min(m_e, m_h)
        features[10] = eg * m_e
        features[11] = eg * m_h
        features[12] = eg / (m_e + m_h + 1e-6)
        features[13] = np.log(mu_e + 1)
        features[14] = np.log(mu_h + 1)
        
        if use_interactions:
            features[15] = eg * m_e * m_h
            features[16] = eg ** 2 * m_e
            features[17] = eg ** 2 * m_h
            features[18] = eg / m_e
            features[19] = eg / m_h
            
            features[20] = m_e ** 2
            features[21] = m_h ** 2
            features[22] = (m_e ** 2 + m_h ** 2) / 2
            features[23] = (m_e * m_h) ** 0.5
            features[24] = m_e / (m_h ** 2 + 1e-6)
            
            features[25] = mu_e / (m_e + 1e-6)
            features[26] = mu_h / (m_h + 1e-6)
            features[27] = mu_e * mu_h / ((m_e + m_h) ** 2 + 1e-6)
            features[28] = np.log(mu_e * mu_h + 1)
            features[29] = (mu_e + mu_h) / 2
            
            features[30] = eg ** 3
            features[31] = m_e ** 3 + m_h ** 3
            features[32] = (m_e + m_h) ** 2
            features[33] = eg * (m_e + m_h) ** 2
            features[34] = eg ** 2 / ((m_e + m_h) ** 2 + 1e-6)
            
            features[35] = 1.0 / (1.0 + np.exp(-eg))
            features[36] = np.exp(-m_e)
            features[37] = np.exp(-m_h)
            features[38] = np.sin(eg)
            features[39] = np.cos(m_e)
            
            features[40] = m_e ** (1/3)
            features[41] = m_h ** (1/3)
            features[42] = eg ** (1/2)
            features[43] = (m_e * eg) / (m_h + 1e-6)
            features[44] = (m_h * eg) / (m_e + 1e-6)
            
            features[45] = (m_e + m_h) * eg
            features[46] = (m_e + m_h) / eg
            features[47] = m_e * eg + m_h * eg
            features[48] = m_e / eg + m_h / eg
            features[49] = (m_e + m_h) * (eg ** 2)
            
            features[50] = m_e * np.exp(-eg)
            features[51] = m_h * np.exp(-eg)
            features[52] = eg * np.exp(-(m_e + m_h))
            features[53] = (m_e + m_h) / np.log(eg + 1.01)
            features[54] = eg / np.log(m_e + m_h + 1.01)
            
            features[55:60] = [
                np.log(1 + m_e * m_h),
                np.log(1 + eg * (m_e + m_h)),
                eg ** 2.5,
                (m_e + m_h) ** 1.5,
                (m_e * m_h) ** 0.75
            ]
    
    except Exception as e:
        pass
    
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_and_save_models():
    """Train Phase 3 models on full DPT dataset and save them"""
    
    print("\n" + "="*80)
    print("PHASE 3 PRODUCTION TRAINING - SAVE MODELS FOR INFERENCE")
    print("="*80 + "\n")
    
    # Load and filter data
    print("[*] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Loaded {len(df)} total materials")
    
    print("\n[*] Filtering to DPT experimental data...")
    df_dpt = df[df['source'].str.contains('DPT', case=False, na=False)].copy()
    print(f"[OK] DPT materials: {len(df_dpt)}")
    
    print("\n[*] Removing outliers (mobility > 500,000 cm2/(V*s))...")
    initial_count = len(df_dpt)
    df_dpt = df_dpt[
        (df_dpt['electron_mobility'] < MAX_MOBILITY_OUTLIER) &
        (df_dpt['hole_mobility'] < MAX_MOBILITY_OUTLIER)
    ].copy()
    removed = initial_count - len(df_dpt)
    print(f"[OK] Removed {removed} outliers, {len(df_dpt)} remaining")
    
    print("\n[*] Removing rows with missing mobility data...")
    df_dpt = df_dpt.dropna(subset=['electron_mobility', 'hole_mobility'])
    print(f"[OK] Final dataset: {len(df_dpt)} materials (clean DPT subset)")
    
    # Extract features
    print("\n[*] Engineering 60D feature set with interaction terms...")
    features_list = []
    for idx, row in tqdm(df_dpt.iterrows(), total=len(df_dpt)):
        feat = engineer_features(row, use_interactions=ENHANCE_FEATURES)
        features_list.append(feat)
    X = np.array(features_list)
    print(f"[OK] Feature matrix shape: {X.shape}")
    
    # Log-transform targets
    print("\n[*] Preparing targets (log-transform for scale reduction)...")
    y_electron = np.log(df_dpt['electron_mobility'].values)
    y_hole = np.log(df_dpt['hole_mobility'].values)
    print(f"[OK] Targets ready (log-transformed)")
    
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
