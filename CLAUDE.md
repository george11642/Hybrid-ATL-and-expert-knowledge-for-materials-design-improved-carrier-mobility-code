# CLAUDE.md - Project Guide for AI Assistants

## Project Overview

**Purpose**: Machine learning system for predicting electron and hole carrier mobility in 2D materials (semiconductors like MoS2, WS2, SiC, etc.)

**Key Goal**: Predict mobility for **2D SiC monolayer** - a material with NO experimental data. The model must generalize from similar materials.

**Current Status**: Production-ready with 257 materials, R² ~0.88 for electron mobility

## Critical Rules

### 1. SiC Must NEVER Be in Training Data
```python
# SiC is the TARGET for prediction - it cannot be in training
sic_mask = df['formula'].str.contains('SiC', case=False, na=False)
df_train = df[~sic_mask]
```
The whole point is to PREDICT SiC mobility from other materials. If SiC is in training, the model just memorizes it.

### 2. No Data Leakage
Features must ONLY use input variables:
- ✅ `bandgap`, `effective_mass_e`, `effective_mass_h`
- ❌ NEVER use `electron_mobility` or `hole_mobility` as features

Previous versions had this bug - it's been fixed but watch for it.

### 3. Log Transform Targets
Mobility values span 0.1 to 10^6 cm²/(V·s). Always use log transform:
```python
y = np.log(df['electron_mobility'])  # For training
mu = np.exp(prediction)               # For inference
```

## Project Structure

```
├── data_acquisition/           # Data fetching scripts
│   ├── c2db_raw.csv           # Original C2DB data (25 materials)
│   ├── c2db_expanded.csv      # Expanded dataset (63 materials)
│   ├── group_iv_iv_raw.csv    # Group IV-IV materials (10 materials)
│   ├── fetch_expanded_c2db.py # Generates expanded data
│   └── group_iv_iv_materials.py
│
├── data_processing/
│   └── merge_datasets.py      # Combines all data sources
│
├── data_processed/
│   └── mobility_dataset_merged.csv  # Final training data (257 materials)
│
├── models/phase3/             # Trained models
│   ├── random_forest_electron.joblib
│   ├── gradient_boosting_electron.joblib
│   ├── random_forest_hole.joblib
│   ├── gradient_boosting_hole.joblib
│   └── feature_scaler_phase3.joblib
│
├── train_phase3_production.py # Training script (excludes SiC)
├── predict_mobility_production.py  # Prediction interface
├── calculate_2d_sic_mobility.py    # DPT physics calculator
│
├── DPTmobility.csv            # 197 materials from literature
├── EPCmobility.csv            # 38 experimental materials
└── VALIDATION_SUMMARY.md      # Data quality documentation
```

## Common Tasks

### Retrain Models
```bash
python train_phase3_production.py
```
This automatically excludes SiC from training data.

### Predict SiC Mobility
```bash
python predict_mobility_production.py
```
Current prediction: μ_e = 144.5 cm²/(V·s), μ_h = 107.1 cm²/(V·s)

### Regenerate Dataset
```bash
python data_acquisition/fetch_expanded_c2db.py
python data_acquisition/group_iv_iv_materials.py
python data_processing/merge_datasets.py
```

### Add New Materials
1. Add to appropriate CSV in `data_acquisition/`
2. Run merge script
3. Retrain models

## Key Data Sources

| Source | Materials | Description |
|--------|-----------|-------------|
| DPTmobility.csv | 197 | Deformation potential theory estimates |
| EPCmobility.csv | 38 | Experimental measurements |
| eTran2D | 19 | High-throughput DFT database |
| c2db_expanded.csv | 63 | TMDs, III-V, MXenes, high-mobility |
| group_iv_iv_raw.csv | 10 | SiC family (GeC, SnC, SiGe, etc.) |

## Feature Engineering

45 features derived from 3 inputs (bandgap, m_e, m_h):
- Core: eg, m_e, m_h, ratios
- Polynomial: eg², m_e*m_h, cross terms
- Nonlinear: exp(-m), sin(eg), log transforms
- Physics-inspired: eg/(m_e+m_h), mobility-like terms

See `engineer_features()` in train_phase3_production.py

## Model Architecture

```
Input (bandgap, m_e, m_h)
    ↓
Feature Engineering (45D)
    ↓
StandardScaler
    ↓
┌─────────────┬─────────────┐
│ Random      │ Gradient    │
│ Forest      │ Boosting    │
│ (500 trees) │ (300 iter)  │
└─────────────┴─────────────┘
    ↓
Ensemble Average
    ↓
exp() → Mobility (cm²/V·s)
```

## Physics Context

### Deformation Potential Theory (DPT)
```
μ = (e × ℏ³ × C2D) / (kB × T × m*² × E1²)
```
- C2D = 2D elastic modulus (~166 N/m for SiC)
- E1 = deformation potential (~5-7 eV)
- DPT typically overestimates by 2-5x

### 2D SiC Properties
- Bandgap: 2.39-2.55 eV (direct)
- Electron mass: 0.42 m₀
- Hole mass: 0.45 m₀
- No experimental mobility data exists

## Gotchas & Warnings

1. **Outliers**: Some materials have μ > 10⁶ cm²/Vs (semimetals). Filter with `MAX_MOBILITY_OUTLIER = 500000`

2. **Missing Data**: Only 71/257 materials have complete bandgap + effective mass data. Others used for validation only.

3. **Unit Consistency**: All mobility in cm²/(V·s). DPTmobility.csv uses 10³ units - converted in merge script.

4. **Formula Matching**: Some formulas have subscripts (MoS2 vs Mo1S2). Use flexible matching.

5. **DPT Correction**: Raw DPT values need ~3.5x correction factor to match experiment.

## Testing Predictions

Validate against known materials:
```python
# MoS2 - well characterized
predict_mobility("MoS2", bandgap=1.66, m_e=0.50, m_h=0.56)
# Expected: ~100/50 cm²/Vs

# WS2 - well characterized
predict_mobility("WS2", bandgap=1.97, m_e=0.28, m_h=0.39)
# Expected: ~200-300/80-100 cm²/Vs
```

## References

- C2DB: https://c2db.fysik.dtu.dk/
- MatHub-2d: Yao et al., Sci. China Mater. 66, 2768-2776 (2023)
- DPT for 2D: Bardeen & Shockley (1950), adapted for 2D
- SiC elastic: Peng et al., Modelling Simul. Mater. Sci. Eng. (2020)

## Contact & Files

- Main documentation: README.md
- Validation details: VALIDATION_SUMMARY.md
- DFT requirements: DFT_BTE_REQUIREMENTS.md
