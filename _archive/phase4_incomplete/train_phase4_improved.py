#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 4 IMPROVED TRAINING - ENHANCED ML MODELS
===============================================

Improvements over Phase 3:
1. Physics-informed features based on Deformation Potential Theory
2. Gaussian Process regression for principled uncertainty quantification
3. Feature selection to reduce overfitting (45 -> top N features)
4. XGBoost for ensemble diversity
5. Leave-one-out validation for similar materials
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "data_processed/mobility_dataset_merged.csv"
OUTPUT_DIR = "models/phase4"
RESULTS_DIR = "evaluation/phase4"

MAX_MOBILITY_OUTLIER = 500000
USE_LOG_TRANSFORM = True
N_FOLDS = 5
N_SELECTED_FEATURES = 20  # Reduce from 45 to prevent overfitting

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# PHYSICS-INFORMED FEATURE ENGINEERING
# ============================================================================

def engineer_features_v2(row, use_physics=True):
    """
    Enhanced feature engineering with physics-informed features.

    Based on Deformation Potential Theory:
    μ = (e × ℏ³ × C2D) / (kB × T × m*² × E1²)

    Key physics insights:
    - μ ∝ 1/m*² (inverse square of effective mass)
    - μ ∝ C2D (elastic modulus - not available for all materials)
    - μ ∝ 1/E1² (deformation potential)
    """
    features = {}

    try:
        # Input features
        eg = row['bandgap'] if pd.notna(row['bandgap']) else 1.5
        m_e = row['effective_mass_e'] if pd.notna(row['effective_mass_e']) else 0.5
        m_h = row['effective_mass_h'] if pd.notna(row['effective_mass_h']) else 0.5

        # Prevent division by zero
        eps = 1e-6

        # =====================================================================
        # CORE FEATURES (0-2)
        # =====================================================================
        features['eg'] = eg
        features['m_e'] = m_e
        features['m_h'] = m_h

        # =====================================================================
        # PHYSICS-INFORMED FEATURES (DPT-based)
        # =====================================================================
        if use_physics:
            # Key DPT relationship: μ ∝ 1/m*²
            features['inv_me_sq'] = 1.0 / (m_e ** 2 + eps)
            features['inv_mh_sq'] = 1.0 / (m_h ** 2 + eps)

            # Combined mass terms
            features['inv_me_mh'] = 1.0 / (m_e * m_h + eps)
            features['reduced_mass'] = (m_e * m_h) / (m_e + m_h + eps)
            features['inv_reduced_mass_sq'] = 1.0 / (features['reduced_mass'] ** 2 + eps)

            # Mobility proxy (μ ∝ 1/(m*² × E_g) for some scattering mechanisms)
            features['mobility_proxy_e'] = 1.0 / (m_e ** 2 * (eg + 0.1) + eps)
            features['mobility_proxy_h'] = 1.0 / (m_h ** 2 * (eg + 0.1) + eps)

            # Bandgap-mass interaction (higher gap often means lower mobility)
            features['eg_me_ratio'] = eg / (m_e + eps)
            features['eg_mh_ratio'] = eg / (m_h + eps)

            # Conductivity mass (density of states effective mass)
            features['dos_mass'] = (m_e ** (2/3) + m_h ** (2/3)) ** 1.5

        # =====================================================================
        # RATIO FEATURES
        # =====================================================================
        features['me_mh_ratio'] = m_e / (m_h + eps)
        features['mh_me_ratio'] = m_h / (m_e + eps)
        features['mass_asymmetry'] = abs(m_e - m_h) / (m_e + m_h + eps)
        features['inv_total_mass'] = 1.0 / (m_e + m_h + eps)

        # =====================================================================
        # POLYNOMIAL FEATURES
        # =====================================================================
        features['eg_sq'] = eg ** 2
        features['eg_sqrt'] = np.sqrt(max(eg, 0))
        features['me_sq'] = m_e ** 2
        features['mh_sq'] = m_h ** 2
        features['me_mh_prod'] = m_e * m_h
        features['me_mh_sqrt'] = np.sqrt(m_e * m_h)

        # Cross terms
        features['eg_me'] = eg * m_e
        features['eg_mh'] = eg * m_h
        features['eg_me_mh'] = eg * m_e * m_h

        # =====================================================================
        # HIGHER ORDER PHYSICS TERMS
        # =====================================================================
        features['eg_over_mass_sum'] = eg / (m_e + m_h + eps)
        features['eg_sq_over_mass'] = eg ** 2 / (m_e + m_h + eps)
        features['mass_sum_sq'] = (m_e + m_h) ** 2

        # Power law terms (common in transport)
        features['me_pow_1p5'] = m_e ** 1.5
        features['mh_pow_1p5'] = m_h ** 1.5
        features['me_pow_neg1p5'] = 1.0 / (m_e ** 1.5 + eps)
        features['mh_pow_neg1p5'] = 1.0 / (m_h ** 1.5 + eps)

        # =====================================================================
        # NONLINEAR TRANSFORMS
        # =====================================================================
        features['exp_neg_eg'] = np.exp(-eg)
        features['exp_neg_me'] = np.exp(-m_e)
        features['exp_neg_mh'] = np.exp(-m_h)
        features['log_me'] = np.log(m_e + eps)
        features['log_mh'] = np.log(m_h + eps)
        features['log_eg'] = np.log(eg + eps)

        # Sigmoid-like (bounded)
        features['sigmoid_eg'] = 1.0 / (1.0 + np.exp(-eg))

        # =====================================================================
        # FRACTIONAL POWERS
        # =====================================================================
        features['me_cbrt'] = m_e ** (1/3)
        features['mh_cbrt'] = m_h ** (1/3)
        features['eg_cbrt'] = eg ** (1/3) if eg > 0 else 0

    except Exception as e:
        print(f"[WARN] Feature engineering error: {e}")

    # Convert to array
    feature_names = list(features.keys())
    feature_array = np.array([features.get(name, 0.0) for name in feature_names])

    return np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6), feature_names


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_improved_models():
    """Train improved models with physics-informed features and GP uncertainty"""

    print("\n" + "="*80)
    print("PHASE 4 IMPROVED TRAINING")
    print("="*80 + "\n")

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    print("[*] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Loaded {len(df)} total materials")

    # Filter for complete data
    print("\n[*] Filtering to materials with complete input features...")
    df_complete = df.dropna(subset=[
        'bandgap', 'effective_mass_e', 'effective_mass_h',
        'electron_mobility', 'hole_mobility'
    ]).copy()
    print(f"[OK] Materials with complete data: {len(df_complete)}")

    # Remove outliers
    print("\n[*] Removing outliers (mobility > 500,000 cm²/(V·s))...")
    initial_count = len(df_complete)
    df_complete = df_complete[
        (df_complete['electron_mobility'] < MAX_MOBILITY_OUTLIER) &
        (df_complete['hole_mobility'] < MAX_MOBILITY_OUTLIER)
    ].copy()
    removed = initial_count - len(df_complete)
    print(f"[OK] Removed {removed} outliers, {len(df_complete)} remaining")

    # CRITICAL: Remove SiC from training
    print("\n[*] Removing SiC from training data (target for prediction)...")
    sic_mask = df_complete['formula'].str.contains('SiC', case=False, na=False)
    sic_data = df_complete[sic_mask].copy()
    if len(sic_data) > 0:
        print(f"    Excluded: {sic_data['formula'].tolist()}")
    df_complete = df_complete[~sic_mask].copy()
    print(f"[OK] Removed {sic_mask.sum()} SiC entries, {len(df_complete)} remaining")

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    print("\n[*] Engineering physics-informed features...")
    features_list = []
    feature_names = None
    for idx, row in tqdm(df_complete.iterrows(), total=len(df_complete)):
        feat, names = engineer_features_v2(row, use_physics=True)
        features_list.append(feat)
        if feature_names is None:
            feature_names = names
    X = np.array(features_list)
    print(f"[OK] Feature matrix shape: {X.shape}")
    print(f"    Total features: {len(feature_names)}")

    # Targets (log-transformed)
    y_electron = np.log(df_complete['electron_mobility'].values)
    y_hole = np.log(df_complete['hole_mobility'].values)
    print(f"\n[OK] Targets ready (log-transformed)")
    print(f"    Electron mobility range: {np.exp(y_electron.min()):.1f} - {np.exp(y_electron.max()):.1f} cm²/(V·s)")
    print(f"    Hole mobility range: {np.exp(y_hole.min()):.1f} - {np.exp(y_hole.max()):.1f} cm²/(V·s)")

    # =========================================================================
    # FEATURE SELECTION
    # =========================================================================

    print(f"\n[*] Selecting top {N_SELECTED_FEATURES} features...")

    # Use mutual information for feature selection
    selector_e = SelectKBest(score_func=mutual_info_regression, k=N_SELECTED_FEATURES)
    selector_e.fit(X, y_electron)

    selector_h = SelectKBest(score_func=mutual_info_regression, k=N_SELECTED_FEATURES)
    selector_h.fit(X, y_hole)

    # Get selected feature indices
    selected_e = selector_e.get_support(indices=True)
    selected_h = selector_h.get_support(indices=True)

    # Union of selected features for both tasks
    selected_features = list(set(selected_e.tolist() + selected_h.tolist()))
    selected_features.sort()

    print(f"[OK] Selected {len(selected_features)} features (union)")
    print(f"    Top electron features: {[feature_names[i] for i in selected_e[:5]]}")
    print(f"    Top hole features: {[feature_names[i] for i in selected_h[:5]]}")

    # Apply selection
    X_selected = X[:, selected_features]
    selected_feature_names = [feature_names[i] for i in selected_features]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Save scaler and feature info
    scaler_path = os.path.join(OUTPUT_DIR, 'feature_scaler_phase4.joblib')
    joblib.dump(scaler, scaler_path)
    joblib.dump(selected_features, os.path.join(OUTPUT_DIR, 'selected_feature_indices.joblib'))
    joblib.dump(feature_names, os.path.join(OUTPUT_DIR, 'all_feature_names.joblib'))
    print(f"[OK] Scaler saved to {scaler_path}")

    # =========================================================================
    # TRAIN MODELS
    # =========================================================================

    print("\n" + "="*80)
    print("TRAINING ELECTRON MOBILITY MODELS")
    print("="*80 + "\n")

    # Random Forest
    print("[*] Training Random Forest (electron)...")
    rf_electron = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_electron.fit(X_scaled, y_electron)
    joblib.dump(rf_electron, os.path.join(OUTPUT_DIR, 'random_forest_electron.joblib'))
    print("[OK] Saved Random Forest (electron)")

    # Gradient Boosting
    print("\n[*] Training Gradient Boosting (electron)...")
    gb_electron = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_electron.fit(X_scaled, y_electron)
    joblib.dump(gb_electron, os.path.join(OUTPUT_DIR, 'gradient_boosting_electron.joblib'))
    print("[OK] Saved Gradient Boosting (electron)")

    # XGBoost
    print("\n[*] Training XGBoost (electron)...")
    xgb_electron = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_electron.fit(X_scaled, y_electron)
    joblib.dump(xgb_electron, os.path.join(OUTPUT_DIR, 'xgboost_electron.joblib'))
    print("[OK] Saved XGBoost (electron)")

    # Gaussian Process
    print("\n[*] Training Gaussian Process (electron)...")
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    gp_electron = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42
    )
    gp_electron.fit(X_scaled, y_electron)
    joblib.dump(gp_electron, os.path.join(OUTPUT_DIR, 'gaussian_process_electron.joblib'))
    print("[OK] Saved Gaussian Process (electron)")

    # -------------------------------------------------------------------------
    # HOLE MODELS
    # -------------------------------------------------------------------------

    print("\n" + "="*80)
    print("TRAINING HOLE MOBILITY MODELS")
    print("="*80 + "\n")

    # Random Forest
    print("[*] Training Random Forest (hole)...")
    rf_hole = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_hole.fit(X_scaled, y_hole)
    joblib.dump(rf_hole, os.path.join(OUTPUT_DIR, 'random_forest_hole.joblib'))
    print("[OK] Saved Random Forest (hole)")

    # Gradient Boosting
    print("\n[*] Training Gradient Boosting (hole)...")
    gb_hole = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_hole.fit(X_scaled, y_hole)
    joblib.dump(gb_hole, os.path.join(OUTPUT_DIR, 'gradient_boosting_hole.joblib'))
    print("[OK] Saved Gradient Boosting (hole)")

    # XGBoost
    print("\n[*] Training XGBoost (hole)...")
    xgb_hole = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_hole.fit(X_scaled, y_hole)
    joblib.dump(xgb_hole, os.path.join(OUTPUT_DIR, 'xgboost_hole.joblib'))
    print("[OK] Saved XGBoost (hole)")

    # Gaussian Process
    print("\n[*] Training Gaussian Process (hole)...")
    gp_hole = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42
    )
    gp_hole.fit(X_scaled, y_hole)
    joblib.dump(gp_hole, os.path.join(OUTPUT_DIR, 'gaussian_process_hole.joblib'))
    print("[OK] Saved Gaussian Process (hole)")

    # =========================================================================
    # CROSS-VALIDATION EVALUATION
    # =========================================================================

    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION EVALUATION")
    print("="*80 + "\n")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Storage for results
    results = {
        'rf_e': [], 'gb_e': [], 'xgb_e': [], 'gp_e': [], 'ensemble_e': [],
        'rf_h': [], 'gb_h': [], 'xgb_h': [], 'gp_h': [], 'ensemble_h': []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_e_train, y_e_val = y_electron[train_idx], y_electron[val_idx]
        y_h_train, y_h_val = y_hole[train_idx], y_hole[val_idx]

        # Train models for this fold
        rf_e = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=3,
                                     min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
        gb_e = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                         min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42)
        xgb_e = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        gp_e = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42)

        rf_e.fit(X_train, y_e_train)
        gb_e.fit(X_train, y_e_train)
        xgb_e.fit(X_train, y_e_train)
        gp_e.fit(X_train, y_e_train)

        # Predictions
        pred_rf_e = rf_e.predict(X_val)
        pred_gb_e = gb_e.predict(X_val)
        pred_xgb_e = xgb_e.predict(X_val)
        pred_gp_e = gp_e.predict(X_val)
        pred_ensemble_e = (pred_rf_e + pred_gb_e + pred_xgb_e + pred_gp_e) / 4

        results['rf_e'].append(r2_score(y_e_val, pred_rf_e))
        results['gb_e'].append(r2_score(y_e_val, pred_gb_e))
        results['xgb_e'].append(r2_score(y_e_val, pred_xgb_e))
        results['gp_e'].append(r2_score(y_e_val, pred_gp_e))
        results['ensemble_e'].append(r2_score(y_e_val, pred_ensemble_e))

        # Hole models
        rf_h = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=3,
                                     min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
        gb_h = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                         min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42)
        xgb_h = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        gp_h = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42)

        rf_h.fit(X_train, y_h_train)
        gb_h.fit(X_train, y_h_train)
        xgb_h.fit(X_train, y_h_train)
        gp_h.fit(X_train, y_h_train)

        pred_rf_h = rf_h.predict(X_val)
        pred_gb_h = gb_h.predict(X_val)
        pred_xgb_h = xgb_h.predict(X_val)
        pred_gp_h = gp_h.predict(X_val)
        pred_ensemble_h = (pred_rf_h + pred_gb_h + pred_xgb_h + pred_gp_h) / 4

        results['rf_h'].append(r2_score(y_h_val, pred_rf_h))
        results['gb_h'].append(r2_score(y_h_val, pred_gb_h))
        results['xgb_h'].append(r2_score(y_h_val, pred_xgb_h))
        results['gp_h'].append(r2_score(y_h_val, pred_gp_h))
        results['ensemble_h'].append(r2_score(y_h_val, pred_ensemble_h))

        print(f"  Fold {fold+1}/{N_FOLDS}: Electron R²={results['ensemble_e'][-1]:.3f}, Hole R²={results['ensemble_h'][-1]:.3f}")

    print("\n" + "-"*60)
    print("ELECTRON MOBILITY R² (5-fold CV):")
    print(f"  Random Forest:      {np.mean(results['rf_e']):.3f} ± {np.std(results['rf_e']):.3f}")
    print(f"  Gradient Boosting:  {np.mean(results['gb_e']):.3f} ± {np.std(results['gb_e']):.3f}")
    print(f"  XGBoost:            {np.mean(results['xgb_e']):.3f} ± {np.std(results['xgb_e']):.3f}")
    print(f"  Gaussian Process:   {np.mean(results['gp_e']):.3f} ± {np.std(results['gp_e']):.3f}")
    print(f"  ENSEMBLE (4-model): {np.mean(results['ensemble_e']):.3f} ± {np.std(results['ensemble_e']):.3f}")

    print("\nHOLE MOBILITY R² (5-fold CV):")
    print(f"  Random Forest:      {np.mean(results['rf_h']):.3f} ± {np.std(results['rf_h']):.3f}")
    print(f"  Gradient Boosting:  {np.mean(results['gb_h']):.3f} ± {np.std(results['gb_h']):.3f}")
    print(f"  XGBoost:            {np.mean(results['xgb_h']):.3f} ± {np.std(results['xgb_h']):.3f}")
    print(f"  Gaussian Process:   {np.mean(results['gp_h']):.3f} ± {np.std(results['gp_h']):.3f}")
    print(f"  ENSEMBLE (4-model): {np.mean(results['ensemble_h']):.3f} ± {np.std(results['ensemble_h']):.3f}")

    # =========================================================================
    # LEAVE-ONE-OUT FOR SIMILAR MATERIALS
    # =========================================================================

    print("\n" + "="*80)
    print("LEAVE-ONE-OUT VALIDATION FOR SiC-SIMILAR MATERIALS")
    print("="*80 + "\n")

    similar_materials = ['GeC', 'AlAs', 'AlP', 'MoS2', 'WS2', 'WSe2', 'SnC']

    for material in similar_materials:
        mask = df_complete['formula'].str.match(f'^{material}$', case=False, na=False)
        if mask.sum() == 0:
            continue

        # Leave this material out
        df_loo = df_complete[~mask]
        target = df_complete[mask].iloc[0]

        # Prepare data
        X_loo = []
        for idx, row in df_loo.iterrows():
            feat, _ = engineer_features_v2(row, use_physics=True)
            X_loo.append(feat)
        X_loo = np.array(X_loo)[:, selected_features]
        y_loo_e = np.log(df_loo['electron_mobility'].values)
        y_loo_h = np.log(df_loo['hole_mobility'].values)

        # Target features
        X_target, _ = engineer_features_v2(target, use_physics=True)
        X_target = X_target[selected_features].reshape(1, -1)

        # Scale
        scaler_loo = StandardScaler()
        X_loo_s = scaler_loo.fit_transform(X_loo)
        X_target_s = scaler_loo.transform(X_target)

        # Train and predict
        rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_loo_s, y_loo_e)
        pred_e = np.exp(rf.predict(X_target_s)[0])

        rf.fit(X_loo_s, y_loo_h)
        pred_h = np.exp(rf.predict(X_target_s)[0])

        actual_e = target['electron_mobility']
        actual_h = target['hole_mobility']
        error_e = abs(pred_e - actual_e) / actual_e * 100
        error_h = abs(pred_h - actual_h) / actual_h * 100

        print(f"  {material:6s}: mu_e pred={pred_e:7.1f} actual={actual_e:7.1f} err={error_e:5.1f}% | "
              f"mu_h pred={pred_h:7.1f} actual={actual_h:7.1f} err={error_h:5.1f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 4 TRAINING COMPLETE!")
    print("="*80 + "\n")

    print("MODELS SAVED:")
    print(f"  [OK] Random Forest (electron/hole)")
    print(f"  [OK] Gradient Boosting (electron/hole)")
    print(f"  [OK] XGBoost (electron/hole)")
    print(f"  [OK] Gaussian Process (electron/hole)")
    print(f"  [OK] Feature scaler and selection indices")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nRun predict_mobility_phase4.py for predictions with uncertainty!")


if __name__ == '__main__':
    train_improved_models()
