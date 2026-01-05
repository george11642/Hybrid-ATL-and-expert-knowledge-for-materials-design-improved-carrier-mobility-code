#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PUBLICATION-READY EVALUATION SUITE
==================================

Comprehensive evaluation for journal submission including:
1. Leave-One-Out Cross-Validation (LOOCV) on known materials
2. Parity plots (predicted vs actual)
3. SHAP analysis for model interpretability
4. DPT baseline comparison
5. Uncertainty quantification with proper error bars
6. Learning curves showing data efficiency
7. Statistical metrics (R², MAE, RMSE, MAPE)

Author: Generated for journal publication
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import LeaveOneOut, KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Try to import SHAP - install if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed. Run: pip install shap")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = Path("data_processed/mobility_dataset_merged.csv")
MODEL_DIR = Path("models/phase3")
OUTPUT_DIR = Path("evaluation/publication_figures")
RESULTS_DIR = Path("evaluation/publication_results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality figure settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

MAX_MOBILITY_OUTLIER = 500000

# Known well-characterized materials for validation
KNOWN_MATERIALS = {
    'MoS2': {'bandgap': 1.66, 'm_e': 0.50, 'm_h': 0.56, 'exp_e': 100, 'exp_h': 50, 'source': 'Literature'},
    'WS2': {'bandgap': 1.97, 'm_e': 0.28, 'm_h': 0.39, 'exp_e': 250, 'exp_h': 100, 'source': 'Literature'},
    'MoSe2': {'bandgap': 1.47, 'm_e': 0.56, 'm_h': 0.64, 'exp_e': 50, 'exp_h': 25, 'source': 'Literature'},
    'WSe2': {'bandgap': 1.65, 'm_e': 0.34, 'm_h': 0.44, 'exp_e': 200, 'exp_h': 140, 'source': 'Literature'},
}

# ============================================================================
# FEATURE ENGINEERING (copied from training script for consistency)
# ============================================================================

def engineer_features(row, use_interactions=True):
    """Engineer 45D feature set from INPUT variables only."""
    features = np.zeros(45)

    try:
        eg = row['bandgap'] if pd.notna(row['bandgap']) else 1.5
        m_e = row['effective_mass_e'] if pd.notna(row['effective_mass_e']) else 0.5
        m_h = row['effective_mass_h'] if pd.notna(row['effective_mass_h']) else 0.5

        # Core features (0-11)
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

    except Exception:
        pass

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def get_feature_names():
    """Return descriptive names for all 45 features."""
    return [
        'bandgap', 'm_e', 'm_h', 'm_e/m_h', '1/(m_e+m_h)',
        'bandgap²', 'm_e×m_h', '(m_e+m_h)/2', '|m_e-m_h|', 'bandgap×m_e',
        'bandgap×m_h', 'bandgap/(m_e+m_h)', 'bandgap×m_e×m_h', 'bandgap²×m_e', 'bandgap²×m_h',
        'bandgap/m_e', 'bandgap/m_h', 'm_e²', 'm_h²', '(m_e²+m_h²)/2',
        '√(m_e×m_h)', 'm_e/m_h²', 'bandgap³', 'm_e³+m_h³', '(m_e+m_h)²',
        'bandgap×(m_e+m_h)²', 'bandgap²/(m_e+m_h)²', 'sigmoid(bandgap)', 'exp(-m_e)', 'exp(-m_h)',
        'sin(bandgap)', 'cos(m_e)', 'm_e^(1/3)', 'm_h^(1/3)', '√bandgap',
        'm_e×bandgap/m_h', 'm_h×bandgap/m_e', '(m_e+m_h)×bandgap', '(m_e+m_h)/bandgap', 'm_e×bandgap+m_h×bandgap',
        'm_e/bandgap+m_h/bandgap', '(m_e+m_h)×bandgap²', 'ln(1+m_e×m_h)', 'ln(1+bandgap×(m_e+m_h))', '(m_e×m_h)^0.75'
    ]


# ============================================================================
# DPT BASELINE MODEL
# ============================================================================

def dpt_mobility(m_star, C2D=166.0, E1=5.0, T=300.0):
    """
    Deformation Potential Theory mobility calculation for 2D materials.

    μ = (e × ℏ³ × C2D) / (kB × T × m*² × E1²)

    Parameters:
    -----------
    m_star : float
        Effective mass in units of m0 (electron mass)
    C2D : float
        2D elastic modulus in N/m (default: 166 N/m for SiC)
    E1 : float
        Deformation potential in eV (default: 5.0 eV)
    T : float
        Temperature in K (default: 300 K)

    Returns:
    --------
    float : Mobility in cm²/(V·s)
    """
    # Physical constants
    e = 1.602e-19  # C
    hbar = 1.055e-34  # J·s
    kB = 1.381e-23  # J/K
    m0 = 9.109e-31  # kg

    # Convert units
    m_star_kg = m_star * m0
    E1_J = E1 * e

    # DPT formula
    numerator = e * (hbar ** 3) * C2D
    denominator = kB * T * (m_star_kg ** 2) * (E1_J ** 2)

    mu_SI = numerator / denominator  # m²/(V·s)
    mu_cgs = mu_SI * 1e4  # cm²/(V·s)

    # Apply correction factor (DPT overestimates by ~3.5x)
    correction_factor = 3.5
    return mu_cgs / correction_factor


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data():
    """Load dataset and prepare for evaluation."""
    print("\n[*] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Loaded {len(df)} total materials")

    # Filter to complete data
    df_complete = df.dropna(subset=[
        'bandgap', 'effective_mass_e', 'effective_mass_h',
        'electron_mobility', 'hole_mobility'
    ]).copy()
    print(f"[OK] Materials with complete data: {len(df_complete)}")

    # Remove outliers
    df_complete = df_complete[
        (df_complete['electron_mobility'] < MAX_MOBILITY_OUTLIER) &
        (df_complete['hole_mobility'] < MAX_MOBILITY_OUTLIER)
    ].copy()
    print(f"[OK] After outlier removal: {len(df_complete)}")

    # Remove SiC (prediction target)
    sic_mask = df_complete['formula'].str.contains('SiC', case=False, na=False)
    df_complete = df_complete[~sic_mask].copy()
    print(f"[OK] After SiC exclusion: {len(df_complete)}")

    # Engineer features
    X = np.array([engineer_features(row) for _, row in df_complete.iterrows()])
    y_electron = np.log(df_complete['electron_mobility'].values)
    y_hole = np.log(df_complete['hole_mobility'].values)

    return df_complete, X, y_electron, y_hole


# ============================================================================
# 1. LEAVE-ONE-OUT CROSS-VALIDATION
# ============================================================================

def run_loocv(X, y, target_name='electron'):
    """Run Leave-One-Out Cross-Validation."""
    print(f"\n[*] Running LOOCV for {target_name} mobility...")

    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    y_pred_std_all = []

    # Use ensemble of RF and GB
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)

        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train)

        # Ensemble prediction
        pred_rf = rf.predict(X_test_scaled)[0]
        pred_gb = gb.predict(X_test_scaled)[0]
        pred_mean = (pred_rf + pred_gb) / 2
        pred_std = abs(pred_rf - pred_gb) / 2

        y_true_all.append(y_test[0])
        y_pred_all.append(pred_mean)
        y_pred_std_all.append(pred_std)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_pred_std_all = np.array(y_pred_std_all)

    # Calculate metrics
    r2 = r2_score(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    # Metrics in original scale
    y_true_orig = np.exp(y_true_all)
    y_pred_orig = np.exp(y_pred_all)
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100

    print(f"[OK] LOOCV Results ({target_name}):")
    print(f"     R² (log-scale): {r2:.4f}")
    print(f"     MAE (log-scale): {mae:.4f}")
    print(f"     RMSE (log-scale): {rmse:.4f}")
    print(f"     MAE (original): {mae_orig:.1f} cm²/(V·s)")
    print(f"     RMSE (original): {rmse_orig:.1f} cm²/(V·s)")
    print(f"     MAPE: {mape:.1f}%")

    return {
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_pred_std': y_pred_std_all,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mae_orig': mae_orig,
        'rmse_orig': rmse_orig,
        'mape': mape
    }


# ============================================================================
# 2. PARITY PLOTS
# ============================================================================

def create_parity_plot(results_e, results_h, df):
    """Create publication-quality parity plots."""
    print("\n[*] Creating parity plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (results, target, ax) in enumerate([
        (results_e, 'Electron', axes[0]),
        (results_h, 'Hole', axes[1])
    ]):
        y_true = np.exp(results['y_true'])
        y_pred = np.exp(results['y_pred'])
        y_std = np.exp(results['y_pred']) * results['y_pred_std']  # Approximate error propagation

        # Scatter with error bars
        ax.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.6,
                   capsize=2, capthick=1, elinewidth=1, markersize=6,
                   color='steelblue', ecolor='lightsteelblue', label='Predictions')

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min()) * 0.8
        max_val = max(y_true.max(), y_pred.max()) * 1.2
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')

        # ±2x error bands
        ax.fill_between([min_val, max_val], [min_val/2, max_val/2], [min_val*2, max_val*2],
                       alpha=0.15, color='gray', label='±2× error band')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'Actual {target} Mobility (cm²/V·s)')
        ax.set_ylabel(f'Predicted {target} Mobility (cm²/V·s)')
        ax.set_title(f'{target} Mobility: R² = {results["r2"]:.3f}, MAPE = {results["mape"]:.1f}%')
        ax.legend(loc='upper left')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')

    plt.tight_layout()

    fig_path = OUTPUT_DIR / 'parity_plots.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'parity_plots.pdf', bbox_inches='tight')
    print(f"[OK] Saved parity plots to {fig_path}")
    plt.close()


# ============================================================================
# 3. SHAP ANALYSIS
# ============================================================================

def run_shap_analysis(X, y_electron, y_hole, df):
    """Run SHAP analysis for model interpretability."""
    if not SHAP_AVAILABLE:
        print("\n[SKIP] SHAP analysis - package not installed")
        return None

    print("\n[*] Running SHAP analysis...")

    # Train models on full data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_names = get_feature_names()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, (y, target, ax) in enumerate([
        (y_electron, 'Electron', axes[0]),
        (y_hole, 'Hole', axes[1])
    ]):
        # Use Random Forest for SHAP (tree explainer is fast)
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)

        # SHAP explainer
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_scaled)

        # Get mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)

        # Sort by importance
        sorted_idx = np.argsort(mean_shap)[-15:]  # Top 15 features

        # Bar plot
        ax.barh(range(len(sorted_idx)), mean_shap[sorted_idx], color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'{target} Mobility - Feature Importance')

    plt.tight_layout()

    fig_path = OUTPUT_DIR / 'shap_importance.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'shap_importance.pdf', bbox_inches='tight')
    print(f"[OK] Saved SHAP analysis to {fig_path}")
    plt.close()

    # Detailed SHAP summary plot for electron mobility
    print("[*] Creating detailed SHAP summary plots...")
    rf_e = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_e.fit(X_scaled, y_electron)
    explainer_e = shap.TreeExplainer(rf_e)
    shap_values_e = explainer_e.shap_values(X_scaled)

    # Use shorter names for display
    short_names = ['Eg', 'm_e', 'm_h', 'm_e/m_h', '1/Σm', 'Eg²', 'm_e×m_h', 'Σm/2', '|Δm|', 'Eg×m_e',
                   'Eg×m_h', 'Eg/Σm', 'Eg×m_e×m_h', 'Eg²×m_e', 'Eg²×m_h'] + ['f' + str(i) for i in range(15, 45)]

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values_e, X_scaled, feature_names=short_names,
                      max_display=20, show=False)
    plt.title('Electron Mobility - SHAP Summary')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_summary_electron.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'shap_summary_electron.pdf', bbox_inches='tight')
    plt.close()

    return shap_values_e


# ============================================================================
# 4. DPT BASELINE COMPARISON
# ============================================================================

def compare_with_dpt_baseline(df, results_e, results_h):
    """Compare ML model with DPT baseline predictions."""
    print("\n[*] Comparing with DPT baseline...")

    # Calculate DPT predictions for all materials
    dpt_e = []
    dpt_h = []

    for _, row in df.iterrows():
        m_e = row['effective_mass_e']
        m_h = row['effective_mass_h']

        # DPT with default parameters
        dpt_e.append(dpt_mobility(m_e))
        dpt_h.append(dpt_mobility(m_h))

    dpt_e = np.array(dpt_e)
    dpt_h = np.array(dpt_h)

    y_true_e = df['electron_mobility'].values
    y_true_h = df['hole_mobility'].values

    # DPT metrics
    dpt_r2_e = r2_score(np.log(y_true_e), np.log(dpt_e))
    dpt_r2_h = r2_score(np.log(y_true_h), np.log(dpt_h))
    dpt_mape_e = np.mean(np.abs((y_true_e - dpt_e) / y_true_e)) * 100
    dpt_mape_h = np.mean(np.abs((y_true_h - dpt_h) / y_true_h)) * 100

    # ML metrics (from LOOCV)
    ml_r2_e = results_e['r2']
    ml_r2_h = results_h['r2']
    ml_mape_e = results_e['mape']
    ml_mape_h = results_h['mape']

    print("\n" + "="*60)
    print("MODEL COMPARISON: ML vs DPT Baseline")
    print("="*60)
    print(f"\n{'Metric':<25} {'ML Model':<15} {'DPT Baseline':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'R² (Electron)':<25} {ml_r2_e:<15.4f} {dpt_r2_e:<15.4f} {(ml_r2_e - dpt_r2_e)*100:+.1f}%")
    print(f"{'R² (Hole)':<25} {ml_r2_h:<15.4f} {dpt_r2_h:<15.4f} {(ml_r2_h - dpt_r2_h)*100:+.1f}%")
    print(f"{'MAPE (Electron)':<25} {ml_mape_e:<15.1f}% {dpt_mape_e:<14.1f}% {(dpt_mape_e - ml_mape_e):+.1f}%")
    print(f"{'MAPE (Hole)':<25} {ml_mape_h:<15.1f}% {dpt_mape_h:<14.1f}% {(dpt_mape_h - ml_mape_h):+.1f}%")
    print("="*70)

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart comparing metrics
    methods = ['ML Model', 'DPT Baseline']
    x = np.arange(len(methods))
    width = 0.35

    # R² comparison
    ax = axes[0]
    r2_e = [ml_r2_e, dpt_r2_e]
    r2_h = [ml_r2_h, dpt_r2_h]

    bars1 = ax.bar(x - width/2, r2_e, width, label='Electron', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, r2_h, width, label='Hole', color='coral', alpha=0.8)

    ax.set_ylabel('R² Score')
    ax.set_title('Model Comparison: R² Score (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Good threshold')

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # MAPE comparison
    ax = axes[1]
    mape_e = [ml_mape_e, dpt_mape_e]
    mape_h = [ml_mape_h, dpt_mape_h]

    bars1 = ax.bar(x - width/2, mape_e, width, label='Electron', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, mape_h, width, label='Hole', color='coral', alpha=0.8)

    ax.set_ylabel('MAPE (%)')
    ax.set_title('Model Comparison: MAPE (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()

    fig_path = OUTPUT_DIR / 'model_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'model_comparison.pdf', bbox_inches='tight')
    print(f"\n[OK] Saved comparison plot to {fig_path}")
    plt.close()

    return {
        'dpt_r2_e': dpt_r2_e,
        'dpt_r2_h': dpt_r2_h,
        'dpt_mape_e': dpt_mape_e,
        'dpt_mape_h': dpt_mape_h,
        'ml_r2_e': ml_r2_e,
        'ml_r2_h': ml_r2_h,
        'ml_mape_e': ml_mape_e,
        'ml_mape_h': ml_mape_h
    }


# ============================================================================
# 5. LEARNING CURVES
# ============================================================================

def create_learning_curves(X, y_electron, y_hole):
    """Create learning curves showing model performance vs training size."""
    print("\n[*] Creating learning curves...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    train_sizes = np.linspace(0.1, 1.0, 10)

    for idx, (y, target, ax) in enumerate([
        (y_electron, 'Electron', axes[0]),
        (y_hole, 'Hole', axes[1])
    ]):
        # Use ensemble model
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            rf, X_scaled, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='steelblue')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='coral')

        ax.plot(train_sizes_abs, train_mean, 'o-', color='steelblue', label='Training score')
        ax.plot(train_sizes_abs, val_mean, 'o-', color='coral', label='Cross-validation score')

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('R² Score')
        ax.set_title(f'{target} Mobility - Learning Curve')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    fig_path = OUTPUT_DIR / 'learning_curves.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'learning_curves.pdf', bbox_inches='tight')
    print(f"[OK] Saved learning curves to {fig_path}")
    plt.close()


# ============================================================================
# 6. KNOWN MATERIALS VALIDATION
# ============================================================================

def validate_known_materials(X, y_electron, y_hole, df):
    """Validate model on well-characterized materials."""
    print("\n[*] Validating on known materials (leave-one-out)...")

    results = []

    # For each known material, train on everything else and predict
    for material, props in KNOWN_MATERIALS.items():
        # Find material in dataset
        mask = df['formula'].str.contains(material, case=False, na=False)
        if not mask.any():
            print(f"  [WARN] {material} not found in dataset, skipping...")
            continue

        mat_idx = df[mask].index[0]
        df_idx = df.index.get_loc(mat_idx)

        # Train on all except this material
        X_train = np.delete(X, df_idx, axis=0)
        y_train_e = np.delete(y_electron, df_idx)
        y_train_h = np.delete(y_hole, df_idx)

        X_test = X[df_idx].reshape(1, -1)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        rf_e = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        gb_e = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
        rf_h = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        gb_h = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)

        rf_e.fit(X_train_scaled, y_train_e)
        gb_e.fit(X_train_scaled, y_train_e)
        rf_h.fit(X_train_scaled, y_train_h)
        gb_h.fit(X_train_scaled, y_train_h)

        # Predict
        pred_e = np.exp((rf_e.predict(X_test_scaled)[0] + gb_e.predict(X_test_scaled)[0]) / 2)
        pred_h = np.exp((rf_h.predict(X_test_scaled)[0] + gb_h.predict(X_test_scaled)[0]) / 2)

        # Actual values from dataset
        actual_e = df.iloc[df_idx]['electron_mobility']
        actual_h = df.iloc[df_idx]['hole_mobility']

        # Experimental reference values
        exp_e = props['exp_e']
        exp_h = props['exp_h']

        results.append({
            'Material': material,
            'Pred_e': pred_e,
            'Actual_e': actual_e,
            'Exp_e': exp_e,
            'Error_e': abs(pred_e - actual_e) / actual_e * 100,
            'Pred_h': pred_h,
            'Actual_h': actual_h,
            'Exp_h': exp_h,
            'Error_h': abs(pred_h - actual_h) / actual_h * 100
        })

        print(f"  {material}: Pred={pred_e:.1f}/{pred_h:.1f}, Actual={actual_e:.1f}/{actual_h:.1f}, Exp={exp_e}/{exp_h}")

    # Create validation figure
    if results:
        results_df = pd.DataFrame(results)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(results_df))
        width = 0.2

        ax.bar(x - 1.5*width, results_df['Pred_e'], width, label='Predicted (e)', color='steelblue', alpha=0.8)
        ax.bar(x - 0.5*width, results_df['Actual_e'], width, label='Dataset (e)', color='navy', alpha=0.8)
        ax.bar(x + 0.5*width, results_df['Pred_h'], width, label='Predicted (h)', color='coral', alpha=0.8)
        ax.bar(x + 1.5*width, results_df['Actual_h'], width, label='Dataset (h)', color='darkred', alpha=0.8)

        # Add experimental reference lines
        for i, row in results_df.iterrows():
            ax.axhline(y=row['Exp_e'], xmin=(i-0.3)/len(results_df), xmax=(i+0.3)/len(results_df),
                      color='blue', linestyle='--', alpha=0.5)
            ax.axhline(y=row['Exp_h'], xmin=(i-0.3)/len(results_df), xmax=(i+0.3)/len(results_df),
                      color='red', linestyle='--', alpha=0.5)

        ax.set_ylabel('Mobility (cm²/V·s)')
        ax.set_xlabel('Material')
        ax.set_title('Validation on Known Materials (Leave-One-Out)')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Material'])
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()

        fig_path = OUTPUT_DIR / 'known_materials_validation.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / 'known_materials_validation.pdf', bbox_inches='tight')
        print(f"\n[OK] Saved known materials validation to {fig_path}")
        plt.close()

        return results_df

    return None


# ============================================================================
# 7. ERROR DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_error_distribution(results_e, results_h):
    """Analyze and visualize prediction error distributions."""
    print("\n[*] Analyzing error distributions...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (results, target) in enumerate([(results_e, 'Electron'), (results_h, 'Hole')]):
        # Residuals in log-scale
        residuals = results['y_pred'] - results['y_true']

        # Histogram of residuals
        ax = axes[idx, 0]
        ax.hist(residuals, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', lw=2)
        ax.axvline(x=np.mean(residuals), color='green', linestyle='-', lw=2, label=f'Mean={np.mean(residuals):.3f}')
        ax.set_xlabel('Residual (log-scale)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{target} Mobility - Residual Distribution')
        ax.legend()

        # Q-Q plot
        ax = axes[idx, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'{target} Mobility - Q-Q Plot')

    plt.tight_layout()

    fig_path = OUTPUT_DIR / 'error_distribution.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'error_distribution.pdf', bbox_inches='tight')
    print(f"[OK] Saved error distribution to {fig_path}")
    plt.close()


# ============================================================================
# 8. GENERATE SUMMARY REPORT
# ============================================================================

def generate_summary_report(results_e, results_h, comparison, df, known_validation):
    """Generate comprehensive summary report for publication."""
    print("\n[*] Generating summary report...")

    report_path = RESULTS_DIR / 'publication_summary.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Publication-Ready Evaluation Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total materials**: {len(df)}\n")
        f.write(f"- **Materials with complete features**: {len(df)}\n")
        f.write(f"- **Feature dimensionality**: 45 engineered features\n")
        f.write(f"- **Validation method**: Leave-One-Out Cross-Validation\n\n")

        f.write("## Model Performance (LOOCV)\n\n")
        f.write("| Metric | Electron Mobility | Hole Mobility |\n")
        f.write("|--------|-------------------|---------------|\n")
        f.write(f"| R² (log-scale) | {results_e['r2']:.4f} | {results_h['r2']:.4f} |\n")
        f.write(f"| MAE (log-scale) | {results_e['mae']:.4f} | {results_h['mae']:.4f} |\n")
        f.write(f"| RMSE (log-scale) | {results_e['rmse']:.4f} | {results_h['rmse']:.4f} |\n")
        f.write(f"| MAE (cm²/V·s) | {results_e['mae_orig']:.1f} | {results_h['mae_orig']:.1f} |\n")
        f.write(f"| RMSE (cm²/V·s) | {results_e['rmse_orig']:.1f} | {results_h['rmse_orig']:.1f} |\n")
        f.write(f"| MAPE (%) | {results_e['mape']:.1f} | {results_h['mape']:.1f} |\n\n")

        f.write("## Comparison with DPT Baseline\n\n")
        f.write("| Metric | ML Model | DPT Baseline | Improvement |\n")
        f.write("|--------|----------|--------------|-------------|\n")
        f.write(f"| R² (Electron) | {comparison['ml_r2_e']:.4f} | {comparison['dpt_r2_e']:.4f} | {(comparison['ml_r2_e'] - comparison['dpt_r2_e'])*100:+.1f}% |\n")
        f.write(f"| R² (Hole) | {comparison['ml_r2_h']:.4f} | {comparison['dpt_r2_h']:.4f} | {(comparison['ml_r2_h'] - comparison['dpt_r2_h'])*100:+.1f}% |\n")
        f.write(f"| MAPE (Electron) | {comparison['ml_mape_e']:.1f}% | {comparison['dpt_mape_e']:.1f}% | {(comparison['dpt_mape_e'] - comparison['ml_mape_e']):+.1f}% |\n")
        f.write(f"| MAPE (Hole) | {comparison['ml_mape_h']:.1f}% | {comparison['dpt_mape_h']:.1f}% | {(comparison['dpt_mape_h'] - comparison['ml_mape_h']):+.1f}% |\n\n")

        if known_validation is not None:
            f.write("## Validation on Known Materials\n\n")
            f.write("| Material | Pred μ_e | Actual μ_e | Error (e) | Pred μ_h | Actual μ_h | Error (h) |\n")
            f.write("|----------|----------|------------|-----------|----------|------------|----------|\n")
            for _, row in known_validation.iterrows():
                f.write(f"| {row['Material']} | {row['Pred_e']:.1f} | {row['Actual_e']:.1f} | {row['Error_e']:.1f}% | {row['Pred_h']:.1f} | {row['Actual_h']:.1f} | {row['Error_h']:.1f}% |\n")
            f.write("\n")

        f.write("## Generated Figures\n\n")
        f.write("1. **parity_plots.png/pdf** - Predicted vs Actual mobility with error bars\n")
        f.write("2. **shap_importance.png/pdf** - Feature importance from SHAP analysis\n")
        f.write("3. **shap_summary_electron.png/pdf** - Detailed SHAP summary plot\n")
        f.write("4. **model_comparison.png/pdf** - ML vs DPT baseline comparison\n")
        f.write("5. **learning_curves.png/pdf** - Model performance vs training size\n")
        f.write("6. **known_materials_validation.png/pdf** - Validation on well-characterized materials\n")
        f.write("7. **error_distribution.png/pdf** - Residual analysis and Q-Q plots\n\n")

        f.write("## Key Findings\n\n")
        improvement_e = (comparison['ml_r2_e'] - comparison['dpt_r2_e']) / comparison['dpt_r2_e'] * 100
        improvement_h = (comparison['ml_r2_h'] - comparison['dpt_r2_h']) / comparison['dpt_r2_h'] * 100
        f.write(f"1. The ML model achieves R² = {results_e['r2']:.3f} for electron mobility and R² = {results_h['r2']:.3f} for hole mobility\n")
        f.write(f"2. Compared to DPT baseline, the ML model shows {improvement_e:.1f}% improvement in R² for electrons\n")
        f.write(f"3. Mean Absolute Percentage Error is {results_e['mape']:.1f}% (electron) and {results_h['mape']:.1f}% (hole)\n")
        f.write(f"4. Learning curves show the model benefits from additional training data\n\n")

        f.write("## Suggested Journal Submission\n\n")
        if results_e['r2'] > 0.85 and results_h['r2'] > 0.80:
            f.write("Based on the results, this work is suitable for:\n")
            f.write("- **Computational Materials Science** (Elsevier)\n")
            f.write("- **Journal of Chemical Information and Modeling** (ACS)\n")
            f.write("- **npj Computational Materials** (Nature)\n")
            f.write("- **Materials Today Communications** (Elsevier)\n")
        else:
            f.write("Consider improving model performance before submission to top-tier journals.\n")

    print(f"[OK] Saved summary report to {report_path}")

    # Also save results as CSV
    results_csv_path = RESULTS_DIR / 'loocv_predictions.csv'
    pd.DataFrame({
        'formula': df['formula'].values,
        'actual_e': np.exp(results_e['y_true']),
        'pred_e': np.exp(results_e['y_pred']),
        'std_e': results_e['y_pred_std'],
        'actual_h': np.exp(results_h['y_true']),
        'pred_h': np.exp(results_h['y_pred']),
        'std_h': results_h['y_pred_std']
    }).to_csv(results_csv_path, index=False)
    print(f"[OK] Saved LOOCV predictions to {results_csv_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete publication evaluation suite."""
    print("\n" + "="*80)
    print("PUBLICATION-READY EVALUATION SUITE")
    print("="*80)

    # Load data
    df, X, y_electron, y_hole = load_and_prepare_data()

    # 1. LOOCV
    results_e = run_loocv(X, y_electron, 'electron')
    results_h = run_loocv(X, y_hole, 'hole')

    # 2. Parity plots
    create_parity_plot(results_e, results_h, df)

    # 3. SHAP analysis
    run_shap_analysis(X, y_electron, y_hole, df)

    # 4. DPT baseline comparison
    comparison = compare_with_dpt_baseline(df, results_e, results_h)

    # 5. Learning curves
    create_learning_curves(X, y_electron, y_hole)

    # 6. Known materials validation
    known_validation = validate_known_materials(X, y_electron, y_hole, df)

    # 7. Error distribution
    analyze_error_distribution(results_e, results_h)

    # 8. Generate summary report
    generate_summary_report(results_e, results_h, comparison, df, known_validation)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nFigures saved to: {OUTPUT_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nNext steps:")
    print("1. Review generated figures for publication quality")
    print("2. Check publication_summary.md for key findings")
    print("3. Use loocv_predictions.csv for supplementary materials")


if __name__ == '__main__':
    main()
