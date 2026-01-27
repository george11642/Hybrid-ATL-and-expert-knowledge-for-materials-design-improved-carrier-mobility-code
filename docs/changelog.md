# Changelog

## 2026-01-27 - Documentation Cleanup
- Consolidated scattered documentation into `docs/` folder
- Reorganized data files into unified `data/` structure
- Updated CLAUDE.md with correct prediction values
- Archived outdated development docs

## 2026-01-04 - Publication-Ready Evaluation
- Generated LOOCV evaluation (R² = 0.912 electron, 0.851 hole)
- Created publication figures (parity plots, SHAP analysis, model comparison)
- Validated against known TMD materials
- Compared with DPT baseline (+110.7% improvement)

## 2025-11-17 - Folder Cleanup
- Archived 40+ outdated files to `_archive/`
- Established clean production structure
- Kept only essential scripts and models

## 2025-10-26 - Phase 3 Production Model
- Fixed critical data leakage bug (mobility was being used as a feature)
- Implemented 45-feature physics-inspired engineering
- Trained ensemble model (Random Forest + Gradient Boosting)
- Achieved R² = 0.912 (LOOCV) for electron mobility
- SiC prediction: μ_e = 141.7 ± 9.5 cm²/(V·s)

## 2025-10-15 - Dataset Expansion
- Merged 5 data sources (DPT, EPC, eTran2D, C2DB, Group IV-IV)
- Total: 257 materials, 70 with complete features
- Excluded SiC from training data (prediction target)

## Earlier Development
- See `_archive/old_docs/` for historical development notes
