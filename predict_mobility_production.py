#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRODUCTION PREDICTION INTERFACE - Phase 3 Models
================================================

Loads trained Phase 3 models and provides predictions for 2D materials.
Includes uncertainty quantification and detailed output.
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

MODEL_DIR = Path("models/phase3")
SCALER_PATH = MODEL_DIR / "feature_scaler_phase3.joblib"

# Load models
RF_ELECTRON_PATH = MODEL_DIR / "random_forest_electron.joblib"
GB_ELECTRON_PATH = MODEL_DIR / "gradient_boosting_electron.joblib"
RF_HOLE_PATH = MODEL_DIR / "random_forest_hole.joblib"
GB_HOLE_PATH = MODEL_DIR / "gradient_boosting_hole.joblib"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features_for_prediction(bandgap, m_e, m_h):
    """
    Engineer 45D feature set for prediction.

    IMPORTANT: Only uses input variables (bandgap, m_e, m_h).
    Does NOT use mobility values to avoid data leakage.
    """
    features = np.zeros(45)

    try:
        eg = bandgap

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
        print(f"[WARN] Feature engineering error: {e}")

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# PREDICTION
# ============================================================================

def predict_mobility(material_name, bandgap, m_e, m_h):
    """Predict electron and hole mobility for a 2D material"""
    
    print("\n" + "="*80)
    print(f"MOBILITY PREDICTION FOR {material_name.upper()}")
    print("="*80 + "\n")
    
    # Load models and scaler
    try:
        scaler = joblib.load(str(SCALER_PATH))
        rf_e = joblib.load(str(RF_ELECTRON_PATH))
        gb_e = joblib.load(str(GB_ELECTRON_PATH))
        rf_h = joblib.load(str(RF_HOLE_PATH))
        gb_h = joblib.load(str(GB_HOLE_PATH))
        print("[OK] Models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return None
    
    # Engineer features
    print("\n[*] Engineering features...")
    features = engineer_features_for_prediction(bandgap, m_e, m_h)
    X = scaler.transform(features.reshape(1, -1))
    print(f"[OK] Features: {features.shape[0]}D")
    print(f"    - Bandgap: {bandgap:.3f} eV")
    print(f"    - Electron mass: {m_e:.3f} m0")
    print(f"    - Hole mass: {m_h:.3f} m0")
    
    # Predict electron mobility (log-transformed)
    print("\n[*] Predicting electron mobility (log-transformed)...")
    pred_e_rf = rf_e.predict(X)[0]
    pred_e_gb = gb_e.predict(X)[0]
    pred_e_avg = (pred_e_rf + pred_e_gb) / 2
    pred_e_unc = abs(pred_e_rf - pred_e_gb) / 2
    
    # Predict hole mobility (log-transformed)
    print("[*] Predicting hole mobility (log-transformed)...")
    pred_h_rf = rf_h.predict(X)[0]
    pred_h_gb = gb_h.predict(X)[0]
    pred_h_avg = (pred_h_rf + pred_h_gb) / 2
    pred_h_unc = abs(pred_h_rf - pred_h_gb) / 2
    
    # Convert back from log-transform
    mu_e = np.exp(pred_e_avg)
    mu_e_lower = np.exp(pred_e_avg - pred_e_unc)
    mu_e_upper = np.exp(pred_e_avg + pred_e_unc)
    
    mu_h = np.exp(pred_h_avg)
    mu_h_lower = np.exp(pred_h_avg - pred_h_unc)
    mu_h_upper = np.exp(pred_h_avg + pred_h_unc)
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80 + "\n")
    
    print("ELECTRON MOBILITY:")
    print(f"  Predicted: {mu_e:.1f} cm2/(V*s)")
    print(f"  Range: {mu_e_lower:.1f} - {mu_e_upper:.1f} cm2/(V*s)")
    print(f"  Uncertainty: +/- {(mu_e_upper - mu_e_lower)/2:.1f} cm2/(V*s)")
    print(f"  Models: RF={np.exp(pred_e_rf):.1f}, GB={np.exp(pred_e_gb):.1f}")
    
    print("\nHOLE MOBILITY:")
    print(f"  Predicted: {mu_h:.1f} cm2/(V*s)")
    print(f"  Range: {mu_h_lower:.1f} - {mu_h_upper:.1f} cm2/(V*s)")
    print(f"  Uncertainty: +/- {(mu_h_upper - mu_h_lower)/2:.1f} cm2/(V*s)")
    print(f"  Models: RF={np.exp(pred_h_rf):.1f}, GB={np.exp(pred_h_gb):.1f}")
    
    print("\nMOBILITY RATIO (electron/hole):")
    mobility_ratio = mu_e / mu_h
    print(f"  Ratio: {mobility_ratio:.2f}")
    
    print("\n" + "="*80)
    
    return {
        'material': material_name,
        'bandgap': bandgap,
        'electron_mass': m_e,
        'hole_mass': m_h,
        'mu_e': mu_e,
        'mu_e_lower': mu_e_lower,
        'mu_e_upper': mu_e_upper,
        'mu_h': mu_h,
        'mu_h_lower': mu_h_lower,
        'mu_h_upper': mu_h_upper,
        'mu_ratio': mobility_ratio
    }

# ============================================================================
# MAIN - PREDICT FOR 2D SIC
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("PHASE 3 PRODUCTION MODEL - MOBILITY PREDICTION")
    print("="*80)
    
    # Predict for actual 2D SiC Monolayer
    print("\nPredicting for 2D SiC Monolayer...")

    # 2D SiC parameters from group_iv_iv_raw.csv (Peng 2020)
    # Bandgap: 2.55 eV (direct gap, hexagonal structure)
    # Electron effective mass: 0.42 m0
    # Hole effective mass: 0.45 m0
    # These are consistent with DPT calculations in the training data

    result = predict_mobility(
        material_name="2D SiC Monolayer",
        bandgap=2.55,
        m_e=0.42,
        m_h=0.45
    )
    
    if result:
        print("\n[OK] Prediction successful!")
        print("\nSaved prediction data:")
        print(f"  Material: {result['material']}")
        print(f"  Electron mobility: {result['mu_e']:.1f} cm2/(V*s)")
        print(f"  Hole mobility: {result['mu_h']:.1f} cm2/(V*s)")
        print(f"  Ratio: {result['mu_ratio']:.2f}")
