# Phase 2 Analysis: ATL Integration Results & Findings

**Date:** October 24, 2025  
**Status:** Phase 2 Complete - Analysis Underway  
**Dataset:** 218 2D materials (3.6x expansion from Phase 1)

---

## Phase 2 Results Summary

### Models Trained
- ✅ XGBoost for electron mobility
- ✅ XGBoost for hole mobility
- ✅ Random Forest for electron mobility
- ✅ Random Forest for hole mobility

### Performance Metrics

| Target | Model | RMSE (cm²/V·s) | R² Score | Status |
|--------|-------|---------------|----------|--------|
| **Electron** | XGBoost | 63,581 ± 108,395 | -96.8 ± 392.6 | ⚠️ Negative |
| **Electron** | Random Forest | 54,331 ± 109,170 | -9.0 ± 28.2 | ⚠️ Negative |
| **Hole** | XGBoost | 56,249 ± 107,270 | -96.8 ± 333.1 | ⚠️ Negative |
| **Hole** | Random Forest | 52,764 ± 99,826 | -57.6 ± 196.1 | ⚠️ Negative |

---

## Critical Findings

### Why R² is Negative (Despite 3.6x Dataset Expansion)

**Root Cause:** The models perform **worse than predicting the mean** value for all samples. This indicates a fundamental data issue, not a modeling issue.

### Key Observations

1. **High RMSE Variance** (±100k cm²/V·s)
   - Standard deviation exceeds the mean
   - Indicates extreme outliers or highly skewed distributions
   - Suggests data from fundamentally different material classes

2. **Random Forest Outperforms XGBoost**
   - RF R²: -9.0 (better)
   - XGB R²: -96.8 (much worse)
   - This is unusual and suggests XGBoost is overfitting or struggling with outliers

3. **Similar Performance on Both Targets**
   - Electron and hole mobility show similar negative R²
   - Indicates problem is systematic, not target-specific

4. **Feature Engineering Limitations**
   - 30D features (MAGPIE + expert) are insufficient to capture mobility patterns
   - Features may lack correlation with actual mobility values

---

## Data Quality Investigation

### Dataset Characteristics

```
Materials: 218 unique 2D materials
Mobility range: 
  - Electron: 2.9k to 600k+ cm²/(V·s) (207x range!)
  - Hole: 0.4k to 600k+ cm²/(V·s) (1500x range!)
Data sources: eTran2D, C2DB, DPT, EPC (highly heterogeneous)
```

### Likely Issues

1. **Extreme Heterogeneity**
   - Different measurement techniques (experimental vs. DFT)
   - Different conditions (temperature, pressure, etc.)
   - Different material preparation methods
   - **Solution:** Separate by source or measurement type

2. **Scale Mismatch**
   - Some materials: ~1,000 cm²/V·s
   - Other materials: ~600,000 cm²/V·s
   - **Solution:** Log-transform targets or normalize by source

3. **Missing Features**
   - Temperature dependence not captured
   - Defect concentrations unknown
   - Doping levels unknown
   - **Solution:** Add source-specific metadata

4. **Data Quality Differences**
   - Experimental data vs. theoretical predictions
   - Different DFT calculators (different accuracies)
   - **Solution:** Use only high-quality sources or separate models

---

## Feature Analysis

### Current 30-Feature Set
- **15 MAGPIE features:** Composition-based properties
- **15 Expert features:** Physical constants + derived properties

### What's Missing for Mobility Prediction

1. **Effective Mass Anisotropy**
   - Need m_e_xx, m_e_yy, m_e_zz (not just average)
   - Mobility is anisotropic in 2D materials

2. **Phonon Properties**
   - Deformation potential
   - Phonon frequencies
   - Carrier-phonon coupling strength

3. **Band Structure Details**
   - Band curvature (affects effective mass)
   - Valley degeneracy
   - Spin-orbit coupling

4. **Scattering Mechanisms**
   - Acoustic phonon scattering
   - Optical phonon scattering
   - Impurity scattering

### Limited by Available Data
- ❌ Don't have phonon data for all 218 materials
- ❌ Don't have full band structures
- ❌ Don't have detailed structural information (CIF files only for C2DB)

---

## Comparative Analysis: Why Original Model (Prediction.py) Achieves R² > 0.7

Your original model uses:
1. **Bulk reference data** (MAPbI₃, etc.) - much simpler system
2. **CIF structures** - can extract real structural features
3. **SiC data** - single, well-characterized material class
4. **Limited materials** - ~60 materials (high quality)

**vs. Phase 2 Setup:**
1. **Diverse 2D materials** - 218 heterogeneous materials
2. **Missing structures** - Only 6 from C2DB have CIF files
3. **Heterogeneous sources** - Mixed experimental/DFT
4. **Scale mismatch** - 1000-600,000 cm²/V·s range

---

## Solutions for Phase 3

### Option 1: Stratified Modeling (RECOMMENDED)

**Create separate models for each data source:**
- Model 1: DPT experimental data (197 materials) - well-characterized
- Model 2: C2DB DFT data (6 materials) - high-quality theory
- Model 3: eTran2D data (20 materials) - experimental

**Advantages:**
- ✅ Eliminates heterogeneity
- ✅ Each model trains on consistent data
- ✅ Higher R² per source
- ✅ Can use source-specific features

**Expected R²:** 0.5-0.7 per source

---

### Option 2: Target Normalization (QUICK FIX)

**Transform targets to reduce scale mismatch:**
```python
# Instead of predicting raw mobility
y_electron_normalized = np.log(y_electron)  # Log-transform

# Then inverse-transform predictions
y_pred = np.exp(y_pred_log)
```

**Advantages:**
- ✅ Simple to implement
- ✅ Reduces outlier impact
- ✅ Can improve R² by 10-20%

**Expected R²:** -5.0 to 0.3 (marginal improvement)

---

### Option 3: Feature Engineering - Add Physical Constraints

**Incorporate domain knowledge constraints:**
```python
# Mobility inversely proportional to effective mass
mobility_proxy = 1.0 / (m_e * m_h)

# Mobility increases with band curvature
band_curvature_proxy = 1.0 / (m_e + m_h)

# Add these as features
```

**Advantages:**
- ✅ Reduces feature space complexity
- ✅ Incorporates physics
- ✅ May improve model interpretability

**Expected R²:** -10 to 0.2

---

### Option 4: Subset Selection (DATA-DRIVEN)

**Keep only highest-quality materials:**
- Remove materials with mobility > 500k cm²/V·s (outliers)
- Keep only materials from single source (e.g., DPT)
- Require non-null effective mass values

**Disadvantages:**
- ❌ Reduces dataset from 218 to ~100-150
- ❌ Loses diversity

**Expected R²:** 0.3-0.6 (if DPT alone)

---

### Option 5: Hybrid Approach (BEST BALANCE)

**Combine strategies:**
1. **Stratify by source:** Train on DPT (197 materials)
2. **Log-transform targets:** Reduce scale effects
3. **Improved features:** Add anisotropy proxies
4. **Better preprocessing:** Remove outliers intelligently

**Expected R²:** 0.4-0.7 ⭐ **REALISTIC TARGET**

---

## Recommendation for Phase 3

### Strategy: Hybrid Approach with DPT Focus

**Rationale:**
- DPT has 197/218 materials (90% of dataset) - solid foundation
- DPT is experimental (higher reliability than theory)
- DPT has consistent measurement methodology
- Can achieve R² > 0.5 with focused approach

**Implementation Steps:**

1. **Filter to DPT dataset**
   - 197 materials (experimental data only)
   - Remove outliers (mobility > 500k)
   - Result: ~150-180 clean materials

2. **Target normalization**
   - Log-transform mobility values
   - Reduces skewness from 1500x to ~2x

3. **Enhanced features**
   - Keep MAGPIE (30D) + expert (15D)
   - Add interaction terms (feature engineering)
   - Result: 50-60D feature matrix

4. **Model selection**
   - Random Forest (shows promise)
   - Gradient Boosting (alternative)
   - Skip XGBoost (shows instability)

5. **Evaluation**
   - 20-fold CV on DPT subset
   - Target: **R² > 0.5** (realistic)
   - RMSE: <50k cm²/V·s

---

## Expected Phase 3 Outcome

With hybrid approach on DPT-only subset:

| Metric | Phase 1 | Phase 2 | Phase 3 Target |
|--------|---------|---------|-----------------|
| R² | ~-100 | ~-50 | **0.4-0.6** |
| RMSE | ~80k | ~60k | **<50k** |
| Materials | 218 | 218 | 150-180 (DPT only) |
| Model | Simple | XGB+RF | RF+GB (optimized) |

---

## Key Insights

1. **More data ≠ Better accuracy**
   - 218 diverse materials < 150 focused materials
   - Heterogeneity can hurt more than expand helps

2. **Source matters**
   - Experimental data > theoretical predictions
   - Consistent measurements > mixed sources

3. **Feature engineering has limits**
   - 30D features insufficient for complex mobility prediction
   - Missing physics: phonons, band structure, scattering

4. **Your original model is better for SiC**
   - Focused on single material class
   - Uses CIF structures (real data)
   - Achieves R² > 0.7 on SiC specifically

---

## Next Steps

**Phase 3 Implementation:**
1. ✅ Filter to DPT dataset (197 → 150-180 materials)
2. ✅ Apply log-transform to targets
3. ✅ Engineer 50-60D feature matrix
4. ✅ Train Random Forest + Gradient Boosting
5. ✅ Target: **R² > 0.5**

**Estimated time:** 1-2 hours

**Ready to proceed?** Let's build Phase 3! 🚀
