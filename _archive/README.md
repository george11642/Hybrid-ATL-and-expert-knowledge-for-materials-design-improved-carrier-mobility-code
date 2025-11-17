# Archive - Outdated Files

**Date Archived**: November 17, 2025

This folder contains outdated files from the development process that have been superseded by the Phase 3 production models.

## What's Archived

### old_models/
- **v2 models**: All `*_v2.joblib` files (intermediate versions)
- **XGBoost models**: `xgboost_*.joblib` (not used in final Phase 3)
- **Phase 2 models**: Entire `phase2/` folder (superseded by Phase 3)
- **Old ATL files**: `feature_extractor.pt`, `mean.npy`, `scale.npy`, `miuE.joblib` (from original Folder 1)
- **Duplicate scalers**: Various `feature_scaler_*.joblib` files
- **Production duplicates**: Models from `final/` folder (active ones in `models/phase3/`)

### old_scripts/
- **Old training**: `train_all_models.py`, `train_final_clean.py`, `train_final_production_model.py`, `train_improved_models.py`, `train_phase2_atl_integration.py`, `train_phase3_optimized.py`
- **Old prediction**: `Prediction.py`, `predict_2d_sic.py`, `predict_mobility.py`, `predict_sic_mobility.py`
- **Old analysis**: `ATL.py`, `View_TSNE.py`, `XGBoost + Hyperopt + SHAP.py`

### old_docs/
- **Debugging notes**: `Why_Folder3_Better_2D_SiC.md`, `Why_Folder4_Fails_2D_SiC.md`
- **Old presentations**: `Presentation_All_Folders.md`, `Presentation_Outline_NonTechnical.md`, `Presentation_Technical_5Slides.md`
- **Old comparisons**: `2D_SiC_Comparison.md`, `SiC_polytype_comparison.md`
- **Old validation**: `validation_report_6H_SiC.md`
- **Redundant summary**: `PROJECT_SUMMARY.md` (superseded by `PROFESSOR_BRIEFING.md`)

### old_evaluation/
- **Old results**: `training_results.json`, `training_results_improved.json`
- **Phase 2 results**: `phase2/` folder

## Current Production Setup

**Active Models**: `models/phase3/`
- `feature_scaler_phase3.joblib`
- `random_forest_electron.joblib`
- `gradient_boosting_electron.joblib`
- `random_forest_hole.joblib`
- `gradient_boosting_hole.joblib`

**Active Scripts**:
- `predict_mobility_production.py` - Main prediction API
- `train_phase3_production.py` - Training pipeline

**Active Documentation**:
- `PROFESSOR_BRIEFING.md` - Main briefing document
- `IMPROVEMENT_SUMMARY.md` - Summary of improvements
- `PRODUCTION_MODEL_SUMMARY.md` - Production details
- `README.md` - Project overview

## Why These Were Archived

1. **Superseded by Phase 3**: Better models with R² = 0.998
2. **Duplicate files**: Multiple versions of the same thing
3. **Old approach**: XGBoost-only models replaced by RF+GB ensemble
4. **Development artifacts**: Intermediate debugging and testing files
5. **Redundant documentation**: Consolidated into cleaner docs

## Can I Delete These?

**Keep for now** if you want historical reference. You can safely delete this entire `_archive/` folder if:
- Phase 3 models are working well
- No need to reference old development process
- Disk space is needed

**Recommendation**: Keep archive for 6-12 months, then delete if not needed.

