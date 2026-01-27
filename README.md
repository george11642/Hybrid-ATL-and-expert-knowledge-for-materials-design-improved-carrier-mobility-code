# Machine Learning Prediction of Carrier Mobility in 2D Materials

Machine learning framework for predicting electron and hole carrier mobility in two-dimensional materials using minimal electronic descriptors.

**Author:** George Teifel, University of New Mexico
**Contact:** gteifel@unm.edu

## Key Results

| Metric | Electron | Hole |
|--------|----------|------|
| R² (LOOCV) | 0.912 | 0.851 |
| MAPE | 22.6% | 27.3% |
| Improvement vs DPT | +110.7% | +18.6% |

**2D SiC Prediction:** μₑ = 141.7 ± 9.5 cm²/(V·s), μₕ = 121.2 ± 2.1 cm²/(V·s)

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn joblib

# Run prediction for 2D SiC
python predict_mobility_production.py
```

## Dataset

- **Total materials:** 257
- **Materials with complete features:** 70 (used for LOOCV)
- **Feature dimensions:** 45 physics-informed features

### Data Sources

| Source | Materials | Type |
|--------|-----------|------|
| DPT literature | 161 | Experimental |
| C2DB expanded | 47 | DFT-calculated |
| EPC measurements | 38 | Experimental |
| eTran2D | 19 | DFT-calculated |
| Group IV-IV | 10 | DPT theoretical |

## Model Architecture

```
Input: Bandgap (Eg), Electron mass (m*e), Hole mass (m*h)
           ↓
   45D Feature Engineering
           ↓
      StandardScaler
           ↓
    ┌──────┴──────┐
    ↓             ↓
Random Forest  Gradient Boosting
 (500 trees)    (300 iterations)
    ↓             ↓
    └──────┬──────┘
           ↓
    Ensemble Average
           ↓
   exp() → Mobility (cm²/V·s)
```

## Project Structure

```
├── predict_mobility_production.py   # Main prediction interface
├── train_phase3_production.py       # Model training script
├── calculate_2d_sic_mobility.py     # DPT physics calculator
│
├── models/phase3/                   # Trained models
│   ├── random_forest_electron.joblib
│   ├── gradient_boosting_electron.joblib
│   ├── random_forest_hole.joblib
│   ├── gradient_boosting_hole.joblib
│   └── feature_scaler_phase3.joblib
│
├── data/
│   ├── processed/                   # Training data
│   │   └── mobility_dataset_merged.csv
│   ├── raw/                         # Raw data sources
│   │   ├── c2db_expanded.csv
│   │   ├── etran2d_raw.csv
│   │   └── group_iv_iv_raw.csv
│   ├── external/                    # Literature data
│   │   ├── DPTmobility.csv
│   │   └── EPCmobility.csv
│   └── merge_datasets.py            # Data processing script
│
├── docs/                            # Documentation
│   ├── model_documentation.md
│   ├── validation_summary.md
│   ├── dft_bte_requirements.md
│   └── changelog.md
│
├── evaluation/
│   ├── publication_figures/         # Publication-ready figures
│   └── publication_results/         # LOOCV predictions
│
├── paper/                           # Manuscript
└── _archive/                        # Historical development files
```

## Usage

### Predict for a new material

```python
from predict_mobility_production import predict_mobility

result = predict_mobility(
    material_name="MoS2",
    bandgap=1.66,
    m_e=0.50,
    m_h=0.56
)

print(f"Electron: {result['mu_e']:.1f} cm²/(V·s)")
print(f"Hole: {result['mu_h']:.1f} cm²/(V·s)")
```

### Retrain models

```bash
python train_phase3_production.py
```

Note: 2D SiC is automatically excluded from training data.

## Validation

### Known Materials (TMDs)

| Material | Pred μₑ | Exp μₑ | Pred μₕ | Exp μₕ |
|----------|---------|--------|---------|--------|
| MoS₂ | 153 | ~100 | 116 | ~50 |
| WS₂ | 269 | ~246 | 166 | ~607 |
| MoSe₂ | 68 | ~52 | 35 | ~29 |
| WSe₂ | 221 | ~161 | 109 | ~108 |

### Physics Validation (DPT)

2D SiC DPT calculation: μₑ = 118 cm²/(V·s), μₕ = 149 cm²/(V·s) (after 3.5x correction)

ML prediction agrees within ~20%, confirming physical plausibility.

## References

1. Bardeen & Shockley, Phys. Rev. 80, 72-80 (1950) - Deformation Potential Theory
2. C2DB Database: https://c2db.fysik.dtu.dk/
3. eTran2D: https://sites.utexas.edu/yuanyue-liu/etran2d/
4. MatHub-2d: Yao et al., Sci. China Mater. 66, 2768-2776 (2023)

## Citation

```bibtex
@article{teifel2026mobility,
  title={Machine Learning Prediction of Carrier Mobility in Two-Dimensional
         Materials Using Minimal Electronic Descriptors},
  author={Teifel, George},
  year={2026}
}
```

## License

MIT License
