#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict mobility for actual 2D SiC monolayer using C2DB parameters
"""

from predict_mobility_production import predict_mobility

# Actual 2D SiC parameters from C2DB database
result = predict_mobility(
    material_name="2D SiC Monolayer",
    bandgap=2.39,      # eV (from C2DB)
    m_e=0.42,          # electron mass (from C2DB)
    m_h=0.45           # hole mass (from C2DB)
)

print("="*70)
print("PREDICTION FOR ACTUAL 2D SiC MONOLAYER")
print("="*70)
print(f"\nInput Parameters (from C2DB database):")
print(f"  Bandgap: 2.39 eV")
print(f"  Electron effective mass: 0.42 m0")
print(f"  Hole effective mass: 0.45 m0")
print(f"\nPredicted Mobility:")
uncertainty_e = (result['mu_e_upper'] - result['mu_e_lower']) / 2
uncertainty_h = (result['mu_h_upper'] - result['mu_h_lower']) / 2
print(f"  Electron: {result['mu_e']:.1f} +/- {uncertainty_e:.1f} cm2/(V*s)")
print(f"  Hole: {result['mu_h']:.1f} +/- {uncertainty_h:.1f} cm2/(V*s)")
print(f"  e/h Ratio: {result['mu_ratio']:.2f}")
print(f"\nC2DB Reported Values:")
print(f"  Electron: 120 cm2/(V*s)")
print(f"  Hole: 100 cm2/(V*s)")
print(f"\nModel vs C2DB:")
print(f"  Electron ratio: {result['mu_e']/120:.2f}x")
print(f"  Hole ratio: {result['mu_h']/100:.2f}x")
print("="*70)

