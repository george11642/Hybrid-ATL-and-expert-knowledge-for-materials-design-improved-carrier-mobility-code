#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 5: CROSS-VALIDATION COMPARISON AND ANALYSIS

This script:
1. Loads training results from Phase 2-4
2. Generates comprehensive comparison tables
3. Creates visualization plots
4. Identifies best-performing models
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent

def load_training_results():
    """Load training results from JSON."""
    path = PROJECT_ROOT / 'evaluation' / 'training_results.json'
    with open(path, 'r') as f:
        results = json.load(f)
    return results

def generate_comparison_report(results):
    """Generate comprehensive comparison report."""
    report_path = PROJECT_ROOT / 'evaluation' / 'model_comparison_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("2D MATERIALS MOBILITY PREDICTION - MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Samples: {results['n_samples']}\n")
        f.write(f"Total Features: {results['n_features']}\n")
        f.write(f"Feature Names:\n")
        for i, fname in enumerate(results['feature_names'], 1):
            f.write(f"  {i}. {fname}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ELECTRON MOBILITY PREDICTION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        electron_res = results['electron_mobility_results']
        
        f.write("MODEL PERFORMANCE (10-Fold Cross-Validation)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'RMSE':<20} {'R² Score':<20}\n")
        f.write("-"*80 + "\n")
        
        for model_name, metrics in electron_res.items():
            if model_name != 'xgb':  # Skip detailed model name
                rmse_str = f"{metrics['rmse_mean']:.2f} ± {metrics['rmse_std']:.2f}"
                r2_str = f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}"
                f.write(f"{model_name.upper():<20} {rmse_str:<20} {r2_str:<20}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("HOLE MOBILITY PREDICTION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        hole_res = results['hole_mobility_results']
        
        f.write("MODEL PERFORMANCE (10-Fold Cross-Validation)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'RMSE':<20} {'R² Score':<20}\n")
        f.write("-"*80 + "\n")
        
        for model_name, metrics in hole_res.items():
            if model_name != 'xgb':
                rmse_str = f"{metrics['rmse_mean']:.2f} ± {metrics['rmse_std']:.2f}"
                r2_str = f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}"
                f.write(f"{model_name.upper():<20} {rmse_str:<20} {r2_str:<20}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL SELECTION RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        # Find best models
        best_electron_model = min(electron_res.items(), key=lambda x: x[1]['rmse_mean'] if x[0] != 'xgb' else float('inf'))
        best_hole_model = min(hole_res.items(), key=lambda x: x[1]['rmse_mean'] if x[0] != 'xgb' else float('inf'))
        
        f.write(f"Best Electron Mobility Model: {best_electron_model[0].upper()}\n")
        f.write(f"  RMSE: {best_electron_model[1]['rmse_mean']:.2f} cm²/(V·s)\n")
        f.write(f"  R²: {best_electron_model[1]['r2_mean']:.4f}\n\n")
        
        f.write(f"Best Hole Mobility Model: {best_hole_model[0].upper()}\n")
        f.write(f"  RMSE: {best_hole_model[1]['rmse_mean']:.2f} cm²/(V·s)\n")
        f.write(f"  R²: {best_hole_model[1]['r2_mean']:.4f}\n\n")
        
        f.write("\nRECOMMENDATION:\n")
        f.write("Use ensemble average predictions for best generalization.\n")
        f.write("Individual models available for specific use cases.\n")
    
    print(f"Comparison report saved to: {report_path}")

def main():
    print("="*80)
    print("PHASE 5: MODEL COMPARISON AND EVALUATION")
    print("="*80)
    
    # Load results
    results = load_training_results()
    print(f"\nLoaded results for {results['n_samples']} samples with {results['n_features']} features")
    
    # Generate report
    generate_comparison_report(results)
    
    print("\n" + "="*80)
    print("PHASE 5 COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()


