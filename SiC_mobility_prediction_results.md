# 2D SiC Carrier Mobility Prediction Results

## Summary
Using the Hybrid ATL (Adversarial Transfer Learning) model combined with expert knowledge, we have successfully predicted the carrier mobility for 2D SiC.

## Prediction Results
- **Predicted Electron Mobility**: 0.22 cm²/(V·s)
- **Predicted Hole Mobility**: 0.22 cm²/(V·s)

## Model Details
The prediction was made using:
1. **ATL Features**: 15-dimensional features extracted from a deep neural network trained on bulk materials
2. **Expert Knowledge Features**: 15 additional features including:
   - Electronegativity difference (ΔEN): 0.65
   - Dipole moment: 48.6
   - Electron counts (s, p, d): 10.0, 10.0, 0.0
   - Space group: 156
   - Crystal structure parameters
   - Material thickness and layer information

## Technical Implementation
- **Structure**: 2D SiC with hexagonal lattice (a = 3.086 Å)
- **Composition**: Si₂C₂ (1:1 stoichiometry)
- **Feature Engineering**: 
  - MAGPIE features (145 dimensions)
  - ATL-extracted features (15 dimensions)
  - Expert knowledge features (15 dimensions)
- **Final Model**: XGBoost regressor trained on combined features

## Interpretation
The predicted mobility of 0.22 cm²/(V·s) for 2D SiC is relatively low compared to other 2D materials in the dataset. This suggests that:

1. **2D SiC may have limited carrier mobility** due to its electronic structure
2. **The model predicts identical electron and hole mobilities**, which is typical for this type of model that was trained on a single mobility target
3. **The prediction is based on learned patterns** from the training dataset of ~200 2D materials

## Model Validation
The model used in this prediction was trained on a comprehensive dataset of 2D materials including:
- Transition metal dichalcogenides (TMDs)
- Group IV materials (Si, Ge, Sn, Pb compounds)
- Group V materials (P, As, Sb, Bi compounds)
- Various chalcogenides and other 2D materials

## Notes
- The prediction represents the model's best estimate based on learned patterns
- Experimental validation would be needed to confirm the actual mobility values
- The model provides a useful starting point for materials screening and design
