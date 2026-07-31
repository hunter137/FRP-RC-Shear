# Changelog

All notable changes to FRP-ShearPred are documented in this file.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] — 2026-04-11

### Added
- Eight ensemble ML algorithms: GBDT, XGBoost, LightGBM, CatBoost, Random Forest,
  Extra Trees, AdaBoost, KNN
- Three hyperparameter optimisation strategies: Bayesian (Optuna TPE), TLBO,
  NSGA-II multi-objective
- Five design code predictions: GB 50608-2020, ACI 440.1R-15, CSA S806-12,
  BISE (1999), JSCE (1997)
- 95 % split conformal prediction intervals (Vovk et al., 2005) attached to
  every ML prediction, with a distribution-free finite-sample marginal
  coverage guarantee; calibration uses held-out test-set residuals
- Applicability-domain check in the Prediction tab: per-input "?" buttons
  display the training-data range (extracted from the fitted MinMaxScaler)
  and flag out-of-range queries
- Non-negative output constraint on all design-code formulas and ML
  predictions (clip at 0 kN), reflecting the physical lower bound of shear
  capacity
- SHAP analysis dialog with beeswarm, bar, and waterfall plots
- Partial dependence plot (PDP) panel in the Interpretability tab
- Portable `.frpmdl` model bundle format (compressed joblib archive)
- Batch prediction workflow with CSV/Excel export
- Interactive beam cross-section schematic with live parameter annotation
- PyQt5/PySide6 compatibility shim (`qt_compat.py`)
- Command-line training interface (`train_frp_models.py`) with `--only`,
  `--trials`, and `--time-limit` flags
- Unit test suite for design code formulas and evaluation metrics (31 tests)
- `environment.yml` for one-command conda environment setup
- Example experimental database `data/testdata.xls` (728 specimens)
- Crash diagnostics tool `tools/diagnose_crash.py`

### Fixed
- OpenMP runtime conflict (`vcomp140.dll` vs `libiomp5md.dll`) causing silent
  process termination after training on Windows; resolved by explicit loky
  executor shutdown before Qt signal emission
