# DFT BTE Calculation Requirements

## Overview

This document outlines the requirements and recommendations for performing first-principles Boltzmann Transport Equation (BTE) calculations to validate and improve the carrier mobility predictions for 2D materials, particularly for materials like 2D SiC where experimental data is unavailable.

## Current Status

### Data Quality Assessment
- **Verified materials**: 5 (MoS2, WS2, MoSe2, WSe2, h-BN)
- **DPT-validated materials**: 18 (including Group IV-IV family)
- **Unverified theoretical estimates**: ~234 materials
- **Total dataset**: 257 materials

### Known Limitations of Current Approach
1. **Deformation Potential Theory (DPT)** overestimates mobility by 2-5x
2. Only acoustic phonon scattering considered
3. Optical phonon scattering ignored
4. Polar optical phonon (Fröhlich) interaction not included
5. No temperature dependence beyond T^-1 scaling

## Recommended DFT BTE Workflow

### 1. Electronic Structure Calculations

#### Software Options
- **VASP** + BoltzTraP2 / EPW
- **Quantum ESPRESSO** + EPW (recommended for electron-phonon)
- **ABINIT** + Abinit transport
- **SIESTA** for large systems

#### Required Calculations
```
1. Structure optimization (PBE/PBEsol)
2. Self-consistent field (SCF) calculation
3. Band structure on fine k-grid (24x24x1 minimum for 2D)
4. Effective mass extraction at CBM/VBM
5. Density of states
```

### 2. Phonon Calculations

#### Lattice Dynamics
```
1. Phonon dispersion (DFPT or finite differences)
2. Phonon density of states
3. Acoustic/optical mode identification
4. Deformation potential extraction
```

#### Key Parameters
- q-point grid: 12x12x1 minimum
- Supercell size: 4x4x1 or larger
- Force convergence: < 0.01 eV/Å

### 3. Electron-Phonon Coupling (EPC)

#### EPW Workflow (Recommended)
```bash
# Step 1: SCF calculation
pw.x < scf.in > scf.out

# Step 2: NSCF on coarse k-grid
pw.x < nscf.in > nscf.out

# Step 3: Phonon calculation on coarse q-grid
ph.x < ph.in > ph.out

# Step 4: EPW interpolation
epw.x < epw.in > epw.out
```

#### Critical EPW Parameters
```fortran
&inputepw
  ! Fine grids
  nkf1 = 100, nkf2 = 100, nkf3 = 1
  nqf1 = 100, nqf2 = 100, nqf3 = 1

  ! Transport
  scattering = .true.
  int_mob = .true.
  carrier = .true.

  ! Temperature
  nstemp = 1
  temps = 300

  ! 2D specific
  system_2d = 'gaussian'
/
```

### 4. Boltzmann Transport Equation

#### Relaxation Time Approximation (RTA)
```
σ = e² ∫ τ(ε) v²(ε) (-∂f/∂ε) D(ε) dε

where:
  τ(ε) = relaxation time
  v(ε) = group velocity
  f = Fermi-Dirac distribution
  D(ε) = density of states
```

#### Full Iterative BTE (Higher Accuracy)
- Accounts for momentum-conserving scattering
- Required for high-mobility materials
- Computationally more expensive

## Priority Materials for BTE Calculations

### High Priority (Group IV-IV)
| Material | Current μ_e (DPT) | Current μ_h (DPT) | Priority |
|----------|-------------------|-------------------|----------|
| SiC      | 136 cm²/Vs        | 247 cm²/Vs        | **HIGH** |
| GeC      | 548 cm²/Vs        | 2720 cm²/Vs       | **HIGH** |
| SnC      | 156 cm²/Vs        | 186 cm²/Vs        | HIGH     |
| SiGe     | 198 cm²/Vs        | 190 cm²/Vs        | MEDIUM   |

### Validation Materials (Known Experimental Values)
| Material | Exp. μ_e | Exp. μ_h | Use Case |
|----------|----------|----------|----------|
| MoS2     | 100      | 50       | Benchmark |
| WS2      | 200-300  | 80-100   | Benchmark |
| WSe2     | 150-200  | 100-150  | Benchmark |

## Computational Resources

### Minimum Requirements
- **CPU cores**: 64-128 for EPW calculations
- **Memory**: 256 GB RAM for fine grids
- **Storage**: 500 GB per material (temporary files)
- **Time**: 1-2 weeks per material on HPC

### Recommended HPC Configuration
```yaml
nodes: 4-8
cores_per_node: 32-64
memory_per_node: 256GB
gpu: Optional (for phonon calculations)
queue_time: 72-168 hours
```

## Expected Outcomes

### Validation Metrics
1. **Absolute error**: |μ_BTE - μ_DPT| / μ_BTE
2. **Correlation**: Pearson r between BTE and DPT values
3. **Systematic correction factor**: μ_BTE / μ_DPT

### Data to Generate
For each material:
- Electron mobility at 300K
- Hole mobility at 300K
- Temperature dependence (100-500K)
- Scattering rate breakdown (acoustic, optical, polar)
- Uncertainty estimates

## Alternative Approaches

### If HPC Resources Unavailable

1. **Machine Learning Correction**
   - Train model to correct DPT → BTE values
   - Use known BTE values as training data
   - Apply correction to all DPT estimates

2. **Semi-empirical Methods**
   - Use experimental data to calibrate DPT
   - Material-specific correction factors
   - Physics-informed scaling relationships

3. **Literature Mining**
   - Extract BTE values from published papers
   - Focus on materials with similar chemistry
   - Use values for validation

## References

1. Poncé et al., "EPW: Electron-phonon coupling from first principles," Comput. Phys. Commun. (2016)
2. Giustino, "Electron-phonon interactions from first principles," Rev. Mod. Phys. (2017)
3. Li et al., "Electrical transport limited by electron-phonon coupling from Boltzmann transport equation," Phys. Rev. B (2015)
4. Haastrup et al., "The Computational 2D Materials Database," 2D Mater. (2018)

## Next Steps

1. [ ] Identify HPC access options
2. [ ] Set up EPW workflow for MoS2 (benchmark)
3. [ ] Validate against experimental MoS2 mobility
4. [ ] Calculate SiC mobility with full BTE
5. [ ] Update ML training data with BTE values
