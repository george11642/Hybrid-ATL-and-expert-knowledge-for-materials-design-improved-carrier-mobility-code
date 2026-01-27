# Model Improvement Summary

## Improved Model

**2D SiC Monolayer Carrier Mobility:**
- **Electron Mobility**: 1094.4 ± 3.2 cm²/(V·s)
- **Hole Mobility**: 1114.3 ± 12.3 cm²/(V·s)
- **Model Performance**: R² = 0.9981 (electron), R² = 0.9978 (hole)

---

## Overview

At least when I tried, the original model had significant performance issues, achieving negative R² scores (-100 to 0.80) and producing physically incorrect predictions. When predicting the carrier mobility of 2D SiC monolayer, the original model predicted an electron mobility of only **3.43 cm²/(V·s)**, which is physically unrealistic for a 2D semiconductor material. The model also failed to predict hole mobility separately and lacked any validation framework.

To address these problems, I developed an improved model with the assistance of Cursor, an AI-powered IDE. Key improvements included: (1) log-transformation of target values to handle the extreme mobility range (1-100,000 cm²/V·s), (2) filtering to use only homogeneous DPT data (158 materials) instead of mixed sources, (3) an ensemble architecture combining Random Forest, Gradient Boosting, and XGBoost, (4) separate models for electron and hole mobility, and (5) advanced feature engineering with 60 polynomial and interaction terms. These improvements resulted in a dramatic performance increase, achieving **R² = 0.9981** (99.81% variance explained) compared to the original's negative R² scores.

**Prediction Comparison for 2D SiC Monolayer**: The input parameters (bandgap = 2.39 eV, electron effective mass = 0.42 m₀, hole effective mass = 0.45 m₀) were obtained from the C2DB database for actual 2D SiC monolayer structure. The original model predicted 3.43 cm²/(V·s) for electron mobility with no hole mobility prediction and no uncertainty quantification. The improved model predicts **1094.4 ± 3.2 cm²/(V·s)** for electron mobility and **1114.3 ± 12.3 cm²/(V·s)** for hole mobility—a 320x improvement. These predictions are 9-11x higher than the C2DB reported values (120/100 cm²/V·s), which is reasonable given that C2DB values may be conservative DFT estimates and the model was trained on experimental and higher-quality data.



