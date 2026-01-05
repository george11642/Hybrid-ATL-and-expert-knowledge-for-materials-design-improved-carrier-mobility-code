#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 4 IMPROVED PREDICTION INTERFACE
======================================

Uses improved models with:
1. Physics-informed features
2. 4-model ensemble (RF, GB, XGBoost, GP)
3. Principled uncertainty from Gaussian Process
4. Feature selection for better generalization
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = Path("models/phase4")


# ============================================================================
# FEATURE ENGINEERING (must match training)
# ============================================================================

def engineer_features_v2(bandgap, m_e, m_h):
    """
    Enhanced feature engineering with physics-informed features.
    Must match train_phase4_improved.py exactly.
    """
    features = {}
    eg = bandgap
    eps = 1e-6

    # Core features
    features['eg'] = eg
    features['m_e'] = m_e
    features['m_h'] = m_h

    # Physics-informed features (DPT-based)
    features['inv_me_sq'] = 1.0 / (m_e ** 2 + eps)
    features['inv_mh_sq'] = 1.0 / (m_h ** 2 + eps)
    features['inv_me_mh'] = 1.0 / (m_e * m_h + eps)
    features['reduced_mass'] = (m_e * m_h) / (m_e + m_h + eps)
    features['inv_reduced_mass_sq'] = 1.0 / (features['reduced_mass'] ** 2 + eps)
    features['mobility_proxy_e'] = 1.0 / (m_e ** 2 * (eg + 0.1) + eps)
    features['mobility_proxy_h'] = 1.0 / (m_h ** 2 * (eg + 0.1) + eps)
    features['eg_me_ratio'] = eg / (m_e + eps)
    features['eg_mh_ratio'] = eg / (m_h + eps)
    features['dos_mass'] = (m_e ** (2/3) + m_h ** (2/3)) ** 1.5

    # Ratio features
    features['me_mh_ratio'] = m_e / (m_h + eps)
    features['mh_me_ratio'] = m_h / (m_e + eps)
    features['mass_asymmetry'] = abs(m_e - m_h) / (m_e + m_h + eps)
    features['inv_total_mass'] = 1.0 / (m_e + m_h + eps)

    # Polynomial features
    features['eg_sq'] = eg ** 2
    features['eg_sqrt'] = np.sqrt(max(eg, 0))
    features['me_sq'] = m_e ** 2
    features['mh_sq'] = m_h ** 2
    features['me_mh_prod'] = m_e * m_h
    features['me_mh_sqrt'] = np.sqrt(m_e * m_h)
    features['eg_me'] = eg * m_e
    features['eg_mh'] = eg * m_h
    features['eg_me_mh'] = eg * m_e * m_h

    # Higher order physics terms
    features['eg_over_mass_sum'] = eg / (m_e + m_h + eps)
    features['eg_sq_over_mass'] = eg ** 2 / (m_e + m_h + eps)
    features['mass_sum_sq'] = (m_e + m_h) ** 2
    features['me_pow_1p5'] = m_e ** 1.5
    features['mh_pow_1p5'] = m_h ** 1.5
    features['me_pow_neg1p5'] = 1.0 / (m_e ** 1.5 + eps)
    features['mh_pow_neg1p5'] = 1.0 / (m_h ** 1.5 + eps)

    # Nonlinear transforms
    features['exp_neg_eg'] = np.exp(-eg)
    features['exp_neg_me'] = np.exp(-m_e)
    features['exp_neg_mh'] = np.exp(-m_h)
    features['log_me'] = np.log(m_e + eps)
    features['log_mh'] = np.log(m_h + eps)
    features['log_eg'] = np.log(eg + eps)
    features['sigmoid_eg'] = 1.0 / (1.0 + np.exp(-eg))

    # Fractional powers
    features['me_cbrt'] = m_e ** (1/3)
    features['mh_cbrt'] = m_h ** (1/3)
    features['eg_cbrt'] = eg ** (1/3) if eg > 0 else 0

    # Convert to array (order must match training)
    feature_names = list(features.keys())
    feature_array = np.array([features.get(name, 0.0) for name in feature_names])

    return np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6), feature_names


# ============================================================================
# PREDICTION WITH UNCERTAINTY
# ============================================================================

def load_models():
    """Load all phase 4 models"""
    models = {}

    try:
        models['scaler'] = joblib.load(MODEL_DIR / 'feature_scaler_phase4.joblib')
        models['selected_features'] = joblib.load(MODEL_DIR / 'selected_feature_indices.joblib')

        # Electron models
        models['rf_e'] = joblib.load(MODEL_DIR / 'random_forest_electron.joblib')
        models['gb_e'] = joblib.load(MODEL_DIR / 'gradient_boosting_electron.joblib')
        models['xgb_e'] = joblib.load(MODEL_DIR / 'xgboost_electron.joblib')
        models['gp_e'] = joblib.load(MODEL_DIR / 'gaussian_process_electron.joblib')

        # Hole models
        models['rf_h'] = joblib.load(MODEL_DIR / 'random_forest_hole.joblib')
        models['gb_h'] = joblib.load(MODEL_DIR / 'gradient_boosting_hole.joblib')
        models['xgb_h'] = joblib.load(MODEL_DIR / 'xgboost_hole.joblib')
        models['gp_h'] = joblib.load(MODEL_DIR / 'gaussian_process_hole.joblib')

        return models

    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print(f"        Make sure to run train_phase4_improved.py first!")
        return None


def predict_mobility(material_name, bandgap, m_e, m_h, verbose=True):
    """
    Predict electron and hole mobility with uncertainty quantification.

    Returns dict with predictions, confidence intervals, and model agreement.
    """

    if verbose:
        print("\n" + "="*80)
        print(f"PHASE 4 MOBILITY PREDICTION: {material_name.upper()}")
        print("="*80 + "\n")

    # Load models
    models = load_models()
    if models is None:
        return None

    if verbose:
        print("[OK] Models loaded (RF, GB, XGBoost, GP)")

    # Engineer features
    if verbose:
        print("\n[*] Engineering physics-informed features...")
    features, _ = engineer_features_v2(bandgap, m_e, m_h)

    # Apply feature selection
    selected = models['selected_features']
    features_selected = features[selected]

    # Scale features
    X = models['scaler'].transform(features_selected.reshape(1, -1))

    if verbose:
        print(f"[OK] Features: {len(features_selected)} selected from {len(features)}")
        print(f"    - Bandgap: {bandgap:.3f} eV")
        print(f"    - Electron mass: {m_e:.3f} m0")
        print(f"    - Hole mass: {m_h:.3f} m0")

    # =========================================================================
    # ELECTRON MOBILITY PREDICTIONS
    # =========================================================================

    if verbose:
        print("\n[*] Predicting electron mobility...")

    # Individual model predictions (log-space)
    pred_rf_e = models['rf_e'].predict(X)[0]
    pred_gb_e = models['gb_e'].predict(X)[0]
    pred_xgb_e = models['xgb_e'].predict(X)[0]
    pred_gp_e, std_gp_e = models['gp_e'].predict(X, return_std=True)
    pred_gp_e = pred_gp_e[0]
    std_gp_e = std_gp_e[0]

    # Ensemble prediction (weighted average - GP gets lower weight due to limited data)
    weights = [0.3, 0.25, 0.25, 0.2]  # RF, GB, XGB, GP
    pred_ensemble_e = (weights[0] * pred_rf_e + weights[1] * pred_gb_e +
                       weights[2] * pred_xgb_e + weights[3] * pred_gp_e)

    # Model spread (disagreement)
    all_preds_e = [pred_rf_e, pred_gb_e, pred_xgb_e, pred_gp_e]
    model_std_e = np.std(all_preds_e)

    # Combined uncertainty (GP std + model disagreement)
    combined_std_e = np.sqrt(std_gp_e**2 + model_std_e**2)

    # Convert to linear scale
    mu_e = np.exp(pred_ensemble_e)
    mu_e_lower = np.exp(pred_ensemble_e - 2 * combined_std_e)  # 95% CI
    mu_e_upper = np.exp(pred_ensemble_e + 2 * combined_std_e)

    # =========================================================================
    # HOLE MOBILITY PREDICTIONS
    # =========================================================================

    if verbose:
        print("[*] Predicting hole mobility...")

    pred_rf_h = models['rf_h'].predict(X)[0]
    pred_gb_h = models['gb_h'].predict(X)[0]
    pred_xgb_h = models['xgb_h'].predict(X)[0]
    pred_gp_h, std_gp_h = models['gp_h'].predict(X, return_std=True)
    pred_gp_h = pred_gp_h[0]
    std_gp_h = std_gp_h[0]

    pred_ensemble_h = (weights[0] * pred_rf_h + weights[1] * pred_gb_h +
                       weights[2] * pred_xgb_h + weights[3] * pred_gp_h)

    all_preds_h = [pred_rf_h, pred_gb_h, pred_xgb_h, pred_gp_h]
    model_std_h = np.std(all_preds_h)
    combined_std_h = np.sqrt(std_gp_h**2 + model_std_h**2)

    mu_h = np.exp(pred_ensemble_h)
    mu_h_lower = np.exp(pred_ensemble_h - 2 * combined_std_h)
    mu_h_upper = np.exp(pred_ensemble_h + 2 * combined_std_h)

    # =========================================================================
    # RESULTS
    # =========================================================================

    if verbose:
        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80 + "\n")

        print("ELECTRON MOBILITY:")
        print(f"  Ensemble prediction: {mu_e:.1f} cm^2/(V*s)")
        print(f"  95% confidence interval: [{mu_e_lower:.1f}, {mu_e_upper:.1f}] cm^2/(V*s)")
        print(f"  Individual models:")
        print(f"    RF:  {np.exp(pred_rf_e):.1f} cm^2/(V*s)")
        print(f"    GB:  {np.exp(pred_gb_e):.1f} cm^2/(V*s)")
        print(f"    XGB: {np.exp(pred_xgb_e):.1f} cm^2/(V*s)")
        print(f"    GP:  {np.exp(pred_gp_e):.1f} +/- {np.exp(pred_gp_e)*std_gp_e:.1f} cm^2/(V*s)")
        print(f"  Model agreement (std): {model_std_e:.3f} (log-scale)")

        print("\nHOLE MOBILITY:")
        print(f"  Ensemble prediction: {mu_h:.1f} cm^2/(V*s)")
        print(f"  95% confidence interval: [{mu_h_lower:.1f}, {mu_h_upper:.1f}] cm^2/(V*s)")
        print(f"  Individual models:")
        print(f"    RF:  {np.exp(pred_rf_h):.1f} cm^2/(V*s)")
        print(f"    GB:  {np.exp(pred_gb_h):.1f} cm^2/(V*s)")
        print(f"    XGB: {np.exp(pred_xgb_h):.1f} cm^2/(V*s)")
        print(f"    GP:  {np.exp(pred_gp_h):.1f} +/- {np.exp(pred_gp_h)*std_gp_h:.1f} cm^2/(V*s)")
        print(f"  Model agreement (std): {model_std_h:.3f} (log-scale)")

        print("\nMOBILITY RATIO (electron/hole):")
        print(f"  Ratio: {mu_e/mu_h:.2f}")

        # Confidence assessment
        avg_model_std = (model_std_e + model_std_h) / 2
        if avg_model_std < 0.3:
            confidence = "HIGH"
        elif avg_model_std < 0.6:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        print(f"\nPREDICTION CONFIDENCE: {confidence}")
        print(f"  (Based on model agreement: std={avg_model_std:.3f})")

        print("\n" + "="*80)

    return {
        'material': material_name,
        'bandgap': bandgap,
        'electron_mass': m_e,
        'hole_mass': m_h,
        # Electron
        'mu_e': mu_e,
        'mu_e_lower': mu_e_lower,
        'mu_e_upper': mu_e_upper,
        'mu_e_rf': np.exp(pred_rf_e),
        'mu_e_gb': np.exp(pred_gb_e),
        'mu_e_xgb': np.exp(pred_xgb_e),
        'mu_e_gp': np.exp(pred_gp_e),
        'mu_e_gp_std': std_gp_e,
        'mu_e_model_std': model_std_e,
        # Hole
        'mu_h': mu_h,
        'mu_h_lower': mu_h_lower,
        'mu_h_upper': mu_h_upper,
        'mu_h_rf': np.exp(pred_rf_h),
        'mu_h_gb': np.exp(pred_gb_h),
        'mu_h_xgb': np.exp(pred_xgb_h),
        'mu_h_gp': np.exp(pred_gp_h),
        'mu_h_gp_std': std_gp_h,
        'mu_h_model_std': model_std_h,
        # Ratio
        'mu_ratio': mu_e / mu_h
    }


# ============================================================================
# MAIN - PREDICT FOR 2D SiC
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("PHASE 4 IMPROVED MODEL - MOBILITY PREDICTION")
    print("="*80)

    # Predict for 2D SiC Monolayer
    print("\nPredicting for 2D SiC Monolayer...")

    # 2D SiC parameters from group_iv_iv_raw.csv (Peng 2020)
    # Bandgap: 2.55 eV (direct gap, hexagonal structure)
    # Electron effective mass: 0.42 m0
    # Hole effective mass: 0.45 m0

    result = predict_mobility(
        material_name="2D SiC Monolayer",
        bandgap=2.55,
        m_e=0.42,
        m_h=0.45
    )

    if result:
        print("\n[OK] Prediction successful!")

        # Compare with DPT theoretical values
        print("\n" + "-"*60)
        print("COMPARISON WITH DPT THEORETICAL VALUES:")
        print("-"*60)
        print("DPT estimates (from group_iv_iv_raw.csv):")
        print("  Electron: 135.7 cm^2/(V*s)")
        print("  Hole:     246.6 cm^2/(V*s)")
        print(f"\nML predictions (this model):")
        print(f"  Electron: {result['mu_e']:.1f} cm^2/(V*s) [{result['mu_e_lower']:.1f}-{result['mu_e_upper']:.1f}]")
        print(f"  Hole:     {result['mu_h']:.1f} cm^2/(V*s) [{result['mu_h_lower']:.1f}-{result['mu_h_upper']:.1f}]")

    # Also predict for some reference materials
    print("\n" + "="*80)
    print("REFERENCE PREDICTIONS (for validation)")
    print("="*80)

    references = [
        ("MoS2", 1.66, 0.50, 0.56, 100, 50),      # Well-characterized TMD
        ("WS2", 1.97, 0.28, 0.39, 246, 607),       # Well-characterized TMD
        ("GeC", 2.07, 0.26, 0.20, 548, 2720),      # Group IV-IV (similar to SiC)
    ]

    for name, eg, me, mh, ref_e, ref_h in references:
        result = predict_mobility(name, eg, me, mh, verbose=False)
        if result:
            err_e = abs(result['mu_e'] - ref_e) / ref_e * 100
            err_h = abs(result['mu_h'] - ref_h) / ref_h * 100
            print(f"\n{name}:")
            print(f"  Electron: pred={result['mu_e']:.1f}, ref={ref_e}, error={err_e:.1f}%")
            print(f"  Hole:     pred={result['mu_h']:.1f}, ref={ref_h}, error={err_h:.1f}%")
