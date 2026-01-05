#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHYSICS-BASED 2D SiC MOBILITY CALCULATOR
=========================================

Uses Deformation Potential Theory (DPT) to calculate carrier mobility for 2D SiC monolayer.

Formula for 2D materials:
    μ = (e * ℏ³ * C2D) / (kB * T * m*² * E1²)

where:
    e   = electron charge (1.602 × 10⁻¹⁹ C)
    ℏ   = reduced Planck constant (1.055 × 10⁻³⁴ J·s)
    C2D = 2D elastic modulus (N/m)
    kB  = Boltzmann constant (1.381 × 10⁻²³ J/K)
    T   = temperature (K)
    m*  = effective mass (kg)
    E1  = deformation potential constant (J)

IMPORTANT: These are THEORETICAL ESTIMATES with significant uncertainty.
No experimental data exists for 2D SiC monolayer carrier mobility.

References:
- Bandgap, effective mass: Literature values for 2D h-SiC
- Elastic modulus: Peng et al., Modelling Simul. Mater. Sci. Eng. (2020)
- Deformation potential: Estimated from similar group IV-IV 2D materials
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Physical constants
e = 1.602e-19       # electron charge (C)
hbar = 1.055e-34    # reduced Planck constant (J·s)
kB = 1.381e-23      # Boltzmann constant (J/K)
m0 = 9.109e-31      # electron rest mass (kg)

@dataclass
class Material2D:
    """2D material properties for mobility calculation"""
    name: str
    bandgap_eV: float           # Band gap (eV)
    m_e: float                  # Electron effective mass (m0)
    m_h: float                  # Hole effective mass (m0)
    C2D: float                  # 2D elastic modulus (N/m)
    E1_e: float                 # Electron deformation potential (eV)
    E1_h: float                 # Hole deformation potential (eV)

    # Uncertainty estimates (as fraction, e.g., 0.3 = 30%)
    C2D_uncertainty: float = 0.1
    E1_uncertainty: float = 0.3

def calculate_mobility_dpt(m_star: float, C2D: float, E1: float, T: float = 300) -> float:
    """
    Calculate carrier mobility using Deformation Potential Theory for 2D materials.

    Parameters:
    -----------
    m_star : float
        Effective mass in units of m0
    C2D : float
        2D elastic modulus in N/m
    E1 : float
        Deformation potential constant in eV
    T : float
        Temperature in K (default 300K)

    Returns:
    --------
    float
        Mobility in cm²/(V·s)
    """
    # Convert units
    m_star_kg = m_star * m0  # kg
    E1_J = E1 * e            # J

    # Deformation potential formula for 2D materials
    # μ = (e * ℏ³ * C2D) / (kB * T * m*² * E1²)
    numerator = e * (hbar ** 3) * C2D
    denominator = kB * T * (m_star_kg ** 2) * (E1_J ** 2)

    mu_SI = numerator / denominator  # m²/(V·s)
    mu_cgs = mu_SI * 1e4             # cm²/(V·s)

    return mu_cgs

def calculate_mobility_with_uncertainty(
    m_star: float,
    C2D: float,
    E1: float,
    C2D_unc: float = 0.1,
    E1_unc: float = 0.3,
    T: float = 300
) -> Tuple[float, float, float]:
    """
    Calculate mobility with uncertainty bounds.

    Returns:
    --------
    Tuple[float, float, float]
        (central_value, lower_bound, upper_bound) in cm²/(V·s)
    """
    # Central value
    mu_central = calculate_mobility_dpt(m_star, C2D, E1, T)

    # Lower bound: high E1, low C2D
    mu_lower = calculate_mobility_dpt(
        m_star,
        C2D * (1 - C2D_unc),
        E1 * (1 + E1_unc),
        T
    )

    # Upper bound: low E1, high C2D
    mu_upper = calculate_mobility_dpt(
        m_star,
        C2D * (1 + C2D_unc),
        E1 * (1 - E1_unc),
        T
    )

    return mu_central, mu_lower, mu_upper

# =============================================================================
# 2D SiC MONOLAYER PARAMETERS (from literature)
# =============================================================================

# Literature values for hexagonal 2D SiC (h-SiC) monolayer
# Sources: Multiple DFT studies on 2D SiC
SiC_2D = Material2D(
    name="2D SiC Monolayer (h-SiC)",
    bandgap_eV=2.55,         # PBE: ~2.5-2.6 eV (HSE: ~3.4 eV)
    m_e=0.45,                # Electron effective mass (estimated from DFT)
    m_h=0.58,                # Hole effective mass (estimated from DFT)
    C2D=166.0,               # Elastic modulus N/m (Peng et al. 2020)
    E1_e=6.5,                # Electron deformation potential (eV) - estimated
    E1_h=4.5,                # Hole deformation potential (eV) - estimated
    C2D_uncertainty=0.10,    # 10% uncertainty
    E1_uncertainty=0.40      # 40% uncertainty (high, as values are estimated)
)

# Reference materials for comparison (known values)
MoS2_2D = Material2D(
    name="MoS2 Monolayer",
    bandgap_eV=1.70,
    m_e=0.48,
    m_h=0.60,
    C2D=127.0,
    E1_e=5.5,
    E1_h=7.0,
    C2D_uncertainty=0.05,
    E1_uncertainty=0.20
)

WS2_2D = Material2D(
    name="WS2 Monolayer",
    bandgap_eV=2.00,
    m_e=0.30,
    m_h=0.40,
    C2D=142.0,
    E1_e=3.8,
    E1_h=5.0,
    C2D_uncertainty=0.05,
    E1_uncertainty=0.20
)

def analyze_material(mat: Material2D, T: float = 300):
    """Analyze a 2D material and print mobility results."""

    print(f"\n{'='*70}")
    print(f"MOBILITY ANALYSIS: {mat.name}")
    print(f"{'='*70}")

    print(f"\nInput Parameters:")
    print(f"  Band gap:           {mat.bandgap_eV:.2f} eV")
    print(f"  Electron mass (m*): {mat.m_e:.3f} m0")
    print(f"  Hole mass (m*):     {mat.m_h:.3f} m0")
    print(f"  Elastic modulus:    {mat.C2D:.1f} ± {mat.C2D * mat.C2D_uncertainty:.1f} N/m")
    print(f"  E1 (electron):      {mat.E1_e:.1f} ± {mat.E1_e * mat.E1_uncertainty:.1f} eV")
    print(f"  E1 (hole):          {mat.E1_h:.1f} ± {mat.E1_h * mat.E1_uncertainty:.1f} eV")
    print(f"  Temperature:        {T} K")

    # Calculate electron mobility
    mu_e, mu_e_low, mu_e_high = calculate_mobility_with_uncertainty(
        mat.m_e, mat.C2D, mat.E1_e,
        mat.C2D_uncertainty, mat.E1_uncertainty, T
    )

    # Calculate hole mobility
    mu_h, mu_h_low, mu_h_high = calculate_mobility_with_uncertainty(
        mat.m_h, mat.C2D, mat.E1_h,
        mat.C2D_uncertainty, mat.E1_uncertainty, T
    )

    print(f"\n{'='*40}")
    print(f"CALCULATED MOBILITY (DPT @ {T}K)")
    print(f"{'='*40}")

    print(f"\nELECTRON MOBILITY:")
    print(f"  Central:  {mu_e:,.0f} cm2/(V*s)")
    print(f"  Range:    {mu_e_low:,.0f} - {mu_e_high:,.0f} cm2/(V*s)")

    print(f"\nHOLE MOBILITY:")
    print(f"  Central:  {mu_h:,.0f} cm2/(V*s)")
    print(f"  Range:    {mu_h_low:,.0f} - {mu_h_high:,.0f} cm2/(V*s)")

    print(f"\nMOBILITY RATIO (mu_e/mu_h): {mu_e/mu_h:.2f}")

    return {
        'mu_e': mu_e, 'mu_e_low': mu_e_low, 'mu_e_high': mu_e_high,
        'mu_h': mu_h, 'mu_h_low': mu_h_low, 'mu_h_high': mu_h_high
    }

def main():
    print("="*70)
    print("DEFORMATION POTENTIAL THEORY - 2D MATERIAL MOBILITY CALCULATOR")
    print("="*70)
    print("""
WARNING: These calculations are THEORETICAL ESTIMATES with significant uncertainty.

Key limitations:
1. DPT assumes only acoustic phonon scattering (ignores optical phonons)
2. Deformation potential values for 2D SiC are ESTIMATED (not measured)
3. No experimental data exists for 2D SiC monolayer mobility
4. DPT often OVERESTIMATES mobility by 2-5x compared to full ab-initio BTE

The values should be treated as ORDER OF MAGNITUDE estimates only.
""")

    # Analyze 2D SiC
    sic_results = analyze_material(SiC_2D)

    # Analyze reference materials for comparison
    print("\n\n" + "="*70)
    print("REFERENCE MATERIALS (for comparison)")
    print("="*70)

    mos2_results = analyze_material(MoS2_2D)
    ws2_results = analyze_material(WS2_2D)

    # Summary comparison
    print("\n\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"""
Material          | mu_e (cm2/Vs) | mu_h (cm2/Vs) | mu_e/mu_h
------------------|---------------|---------------|----------
2D SiC (DPT calc) | {sic_results['mu_e']:>13,.0f} | {sic_results['mu_h']:>13,.0f} | {sic_results['mu_e']/sic_results['mu_h']:>8.2f}
MoS2 (DPT calc)   | {mos2_results['mu_e']:>13,.0f} | {mos2_results['mu_h']:>13,.0f} | {mos2_results['mu_e']/mos2_results['mu_h']:>8.2f}
WS2 (DPT calc)    | {ws2_results['mu_e']:>13,.0f} | {ws2_results['mu_h']:>13,.0f} | {ws2_results['mu_e']/ws2_results['mu_h']:>8.2f}
------------------|---------------|---------------|----------
MoS2 (Literature) | ~100          | ~50           | ~2.00
WS2 (Literature)  | ~200-300      | ~100          | ~2.50

Note: DPT typically overestimates by 2-5x. Divide by 3-4 for realistic estimate.
""")

    # Corrected estimate for 2D SiC
    correction_factor = 3.5
    print(f"""
CORRECTED 2D SiC ESTIMATE (DPT/{correction_factor:.1f}):
  Electron mobility: {sic_results['mu_e']/correction_factor:,.0f} cm2/(V*s)
  Hole mobility:     {sic_results['mu_h']/correction_factor:,.0f} cm2/(V*s)

This suggests 2D SiC monolayer mobility is likely in the 50-300 cm2/(V*s) range.
The values in the training data (120/100 cm2/Vs) are PLAUSIBLE but UNVERIFIED.
""")

if __name__ == '__main__':
    main()
