# Publication-Ready Evaluation Summary

Generated: 2026-01-04 22:54:03

## Dataset Overview

- **Total materials**: 70
- **Materials with complete features**: 70
- **Feature dimensionality**: 45 engineered features
- **Validation method**: Leave-One-Out Cross-Validation

## Model Performance (LOOCV)

| Metric | Electron Mobility | Hole Mobility |
|--------|-------------------|---------------|
| R² (log-scale) | 0.9122 | 0.8505 |
| MAE (log-scale) | 0.2198 | 0.2834 |
| RMSE (log-scale) | 0.2944 | 0.4504 |
| MAE (cm²/V·s) | 71.3 | 93.4 |
| RMSE (cm²/V·s) | 130.5 | 270.7 |
| MAPE (%) | 22.6 | 27.3 |

## Comparison with DPT Baseline

| Metric | ML Model | DPT Baseline | Improvement |
|--------|----------|--------------|-------------|
| R² (Electron) | 0.9122 | -0.1948 | +110.7% |
| R² (Hole) | 0.8505 | 0.6642 | +18.6% |
| MAPE (Electron) | 22.6% | 241.7% | +219.1% |
| MAPE (Hole) | 27.3% | 86.1% | +58.8% |

## Validation on Known Materials

| Material | Pred μ_e | Actual μ_e | Error (e) | Pred μ_h | Actual μ_h | Error (h) |
|----------|----------|------------|-----------|----------|------------|----------|
| MoS2 | 153.0 | 100.0 | 53.0% | 115.7 | 50.0 | 131.3% |
| WS2 | 268.8 | 246.0 | 9.2% | 166.1 | 607.0 | 72.6% |
| MoSe2 | 68.2 | 52.0 | 31.2% | 34.6 | 29.0 | 19.2% |
| WSe2 | 220.7 | 161.0 | 37.1% | 109.0 | 108.0 | 0.9% |

## Generated Figures

1. **parity_plots.png/pdf** - Predicted vs Actual mobility with error bars
2. **shap_importance.png/pdf** - Feature importance from SHAP analysis
3. **shap_summary_electron.png/pdf** - Detailed SHAP summary plot
4. **model_comparison.png/pdf** - ML vs DPT baseline comparison
5. **learning_curves.png/pdf** - Model performance vs training size
6. **known_materials_validation.png/pdf** - Validation on well-characterized materials
7. **error_distribution.png/pdf** - Residual analysis and Q-Q plots

## Key Findings

1. The ML model achieves R² = 0.912 for electron mobility and R² = 0.851 for hole mobility
2. Compared to DPT baseline, the ML model shows -568.3% improvement in R² for electrons
3. Mean Absolute Percentage Error is 22.6% (electron) and 27.3% (hole)
4. Learning curves show the model benefits from additional training data

## Suggested Journal Submission

Based on the results, this work is suitable for:
- **Computational Materials Science** (Elsevier)
- **Journal of Chemical Information and Modeling** (ACS)
- **npj Computational Materials** (Nature)
- **Materials Today Communications** (Elsevier)
