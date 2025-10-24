#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRODUCTION PREDICTION INTERFACE
================================

Predict 2D materials electron and hole mobility from:
1. Chemical formula
2. Bandgap
3. Effective masses

Usage:
    python predict_mobility.py --formula "MoS2" --bandgap 1.66 --mass_e 0.5 --mass_h 0.56
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import json

PROJECT_ROOT = Path(__file__).parent

class MobilityPredictor:
    """Predict 2D materials carrier mobility."""
    
    def __init__(self):
        """Load trained models and scaler."""
        models_dir = PROJECT_ROOT / 'models' / 'final'
        
        self.scaler = load(PROJECT_ROOT / 'models' / 'feature_scaler.joblib')
        
        # Load base models
        self.xgb_electron = load(models_dir / 'xgboost_electron.joblib')
        self.rf_electron = load(models_dir / 'random_forest_electron.joblib')
        self.gb_electron = load(models_dir / 'gradient_boosting_electron.joblib')
        
        self.xgb_hole = load(models_dir / 'xgboost_hole.joblib')
        self.rf_hole = load(models_dir / 'random_forest_hole.joblib')
        self.gb_hole = load(models_dir / 'gradient_boosting_hole.joblib')
        
        # Load training results for feature info
        with open(PROJECT_ROOT / 'evaluation' / 'training_results.json', 'r') as f:
            self.training_results = json.load(f)
        
        self.feature_names = self.training_results['feature_names']
        
        print("✓ Successfully loaded trained models")
        print(f"  Features: {len(self.feature_names)}")
    
    def engineer_features(self, formula, bandgap, mass_e, mass_h):
        """
        Engineer features for a single material.
        
        Parameters:
        -----------
        formula : str
            Chemical formula (e.g., "MoS2")
        bandgap : float
            Band gap in eV
        mass_e : float
            Electron effective mass (in units of free electron mass)
        mass_h : float
            Hole effective mass (in units of free electron mass)
        
        Returns:
        --------
        X : np.ndarray, shape (1, n_features)
            Feature vector for prediction
        """
        features = []
        
        # 1. Direct properties
        features.append(bandgap if bandgap > 0 else 2.0)  # bandgap
        features.append(mass_e)  # effective_mass_e
        features.append(mass_h)  # effective_mass_h
        
        # 2. Derived electronic features
        mass_ratio = mass_e / (mass_h + 1e-6)
        features.append(np.log1p(mass_ratio))
        
        mass_sum = mass_e + mass_h
        features.append(mass_sum)
        
        mass_diff = abs(mass_e - mass_h)
        features.append(mass_diff)
        
        features.append(float(bandgap > 2.0))  # large_bandgap_flag
        features.append(float(bandgap > 5.0))  # insulator_flag
        
        # 3. Composition features (estimate from formula)
        n_atoms = len([c for c in formula if c.isupper()])
        n_elements = len([c for c in formula if c.isalpha()])
        
        features.append(n_atoms)
        features.append(n_elements)
        features.append(n_atoms * n_elements)  # complexity
        
        # 4. Source and quality features (default values for prediction)
        features.append(0.5)  # is_experimental (assume unknown)
        features.append(0.5)  # is_dft_calculated (assume unknown)
        features.append(0.0)  # log_n_sources (single source)
        
        # 5. Bandgap regions
        bg = bandgap if bandgap > 0 else 2.0
        features.append(float(bg < 0.5))      # semimetal
        features.append(float((bg >= 0.5) & (bg < 1.5)))  # narrow_gap
        features.append(float((bg >= 1.5) & (bg < 3.0)))  # direct_gap
        features.append(float((bg >= 3.0) & (bg < 5.0)))  # wide_gap
        features.append(float(bg >= 5.0))     # insulator
        
        X = np.array(features).reshape(1, -1)
        return X
    
    def predict(self, formula, bandgap, mass_e, mass_h, use_ensemble=True):
        """
        Predict electron and hole mobility.
        
        Parameters:
        -----------
        formula : str
            Chemical formula
        bandgap : float
            Band gap (eV)
        mass_e : float
            Electron effective mass
        mass_h : float
            Hole effective mass
        use_ensemble : bool
            Use ensemble averaging (recommended)
        
        Returns:
        --------
        predictions : dict
            Dictionary with mobility predictions and model details
        """
        # Engineer features
        X = self.engineer_features(formula, bandgap, mass_e, mass_h)
        X_scaled = self.scaler.transform(X)
        
        # Electron mobility predictions
        pred_e_xgb = self.xgb_electron.predict(X_scaled)[0]
        pred_e_rf = self.rf_electron.predict(X_scaled)[0]
        pred_e_gb = self.gb_electron.predict(X_scaled)[0]
        
        if use_ensemble:
            pred_e = np.mean([pred_e_xgb, pred_e_rf, pred_e_gb])
        else:
            pred_e = pred_e_xgb
        
        # Hole mobility predictions
        pred_h_xgb = self.xgb_hole.predict(X_scaled)[0]
        pred_h_rf = self.rf_hole.predict(X_scaled)[0]
        pred_h_gb = self.gb_hole.predict(X_scaled)[0]
        
        if use_ensemble:
            pred_h = np.mean([pred_h_xgb, pred_h_rf, pred_h_gb])
        else:
            pred_h = pred_h_xgb
        
        # Calculate uncertainty (standard deviation of predictions)
        uncertainty_e = np.std([pred_e_xgb, pred_e_rf, pred_e_gb])
        uncertainty_h = np.std([pred_h_xgb, pred_h_rf, pred_h_gb])
        
        predictions = {
            'formula': formula,
            'bandgap_eV': bandgap,
            'effective_mass_electron': mass_e,
            'effective_mass_hole': mass_h,
            'electron_mobility_cm2_Vs': float(pred_e),
            'electron_mobility_uncertainty': float(uncertainty_e),
            'hole_mobility_cm2_Vs': float(pred_h),
            'hole_mobility_uncertainty': float(uncertainty_h),
            'model_details': {
                'xgboost_electron': float(pred_e_xgb),
                'random_forest_electron': float(pred_e_rf),
                'gradient_boosting_electron': float(pred_e_gb),
                'xgboost_hole': float(pred_h_xgb),
                'random_forest_hole': float(pred_h_rf),
                'gradient_boosting_hole': float(pred_h_gb),
            }
        }
        
        return predictions

def main():
    parser = argparse.ArgumentParser(
        description='Predict 2D materials electron and hole mobility'
    )
    parser.add_argument('--formula', type=str, required=True, help='Chemical formula (e.g., MoS2)')
    parser.add_argument('--bandgap', type=float, required=True, help='Band gap in eV')
    parser.add_argument('--mass_e', type=float, required=True, help='Electron effective mass')
    parser.add_argument('--mass_h', type=float, required=True, help='Hole effective mass')
    parser.add_argument('--no-ensemble', action='store_true', help='Use single XGBoost model instead of ensemble')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("2D MATERIALS MOBILITY PREDICTOR")
    print("="*80 + "\n")
    
    # Load predictor
    predictor = MobilityPredictor()
    
    # Make prediction
    print(f"\nPredicting for: {args.formula}")
    print(f"  Bandgap: {args.bandgap} eV")
    print(f"  Electron mass: {args.mass_e} m₀")
    print(f"  Hole mass: {args.mass_h} m₀")
    
    predictions = predictor.predict(
        formula=args.formula,
        bandgap=args.bandgap,
        mass_e=args.mass_e,
        mass_h=args.mass_h,
        use_ensemble=not args.no_ensemble
    )
    
    # Print results
    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)
    print(f"\nElectron Mobility: {predictions['electron_mobility_cm2_Vs']:.2f} ± {predictions['electron_mobility_uncertainty']:.2f} cm²/(V·s)")
    print(f"Hole Mobility:     {predictions['hole_mobility_cm2_Vs']:.2f} ± {predictions['hole_mobility_uncertainty']:.2f} cm²/(V·s)")
    
    if not args.no_ensemble:
        print("\nIndividual Model Predictions:")
        print(f"  Electron Mobility:")
        print(f"    XGBoost:          {predictions['model_details']['xgboost_electron']:.2f} cm²/(V·s)")
        print(f"    Random Forest:    {predictions['model_details']['random_forest_electron']:.2f} cm²/(V·s)")
        print(f"    Gradient Boosting: {predictions['model_details']['gradient_boosting_electron']:.2f} cm²/(V·s)")
        print(f"  Hole Mobility:")
        print(f"    XGBoost:          {predictions['model_details']['xgboost_hole']:.2f} cm²/(V·s)")
        print(f"    Random Forest:    {predictions['model_details']['random_forest_hole']:.2f} cm²/(V·s)")
        print(f"    Gradient Boosting: {predictions['model_details']['gradient_boosting_hole']:.2f} cm²/(V·s)")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()


