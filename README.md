# FRP-ShearPred

**An open-source platform for shear capacity prediction of stirrup-free FRP-reinforced concrete beams integrating design codes and ensemble machine learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Tests](https://github.com/hunter137/FRP-RC-Shear/actions/workflows/tests.yml/badge.svg)](https://github.com/hunter137/FRP-RC-Shear/actions/workflows/tests.yml)

**Author:** Deyu Liang  
**Affiliation:** Shenyang Jianzhu University; Southeast University  
**Version:** 1.0  
**License:** MIT

---

## Overview

FRP-ShearPred is a cross-platform desktop application that unifies five international design codes, eight ensemble machine learning algorithms, and a Bayesian-optimised empirical formula into a single graphical environment for shear strength assessment of fibre-reinforced polymer (FRP) reinforced concrete beams without shear reinforcement.

The software is intended for structural engineers, researchers, and students who need to:

- compare design-code predictions against experimental or field data;
- train and evaluate ensemble ML models on custom experimental databases;
- interpret model predictions via SHAP values, partial dependence plots, and feature importance rankings;
- export results for further statistical or reliability analysis.

---

## Screenshots

<!-- Add screenshots to docs/screenshots/ then update paths below -->
<!-- ![Prediction Tab](docs/screenshots/prediction_tab.png) -->
<!-- ![Training Tab](docs/screenshots/training_tab.png) -->
<!-- ![SHAP Dialog](docs/screenshots/shap_dialog.png) -->

---

## Features

- **Multi-code prediction**: GB 50608-2020, ACI 440.1R-15, CSA S806-12, BISE (1999), JSCE (1997)
- **Proposed empirical formula**: Bayesian-optimised formula incorporating size-effect, arch-action, and FRP reinforcement stiffness terms
- **8 ML algorithms**: GBDT, XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, AdaBoost, KNN
- **Hyperparameter optimisation**: Bayesian (Optuna TPE), TLBO, NSGA-II multi-objective
- **Model interpretability**: SHAP analysis, feature importance, partial dependence plots
- **Batch prediction**: Process entire databases with CSV/Excel export
- **Portable model bundles**: Save/load trained models as `.frpmdl` files
- **Interactive beam schematic**: Visual cross-section diagram with parameter annotations

---

## Installation

### Requirements

- Python 3.8 or later
- Operating system: Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)

### Option 1 — conda (recommended)

```bash
git clone https://github.com/hunter137/FRP-RC-Shear.git
cd FRP-RC-Shear
conda env create -f environment.yml
conda activate frpshear
python main.py
```

### Option 2 — pip

```bash
git clone https://github.com/hunter137/FRP-RC-Shear.git
cd FRP-RC-Shear
pip install -r requirements.txt
python main.py
```

### Optional: CatBoost

CatBoost is excluded from the default install due to its large package size (~400 MB). Install separately if needed:

```bash
pip install catboost
```

---

## Usage

### Single Prediction

1. Open the **Prediction** tab.
2. Enter beam parameters (a/d, d, b, f′c, ρf, Ef, FRP type).
3. Load a pre-trained model bundle (`.frpmdl`) or train your own in the Training tab.
4. Click **Predict** to obtain results from all design codes and the loaded ML model simultaneously.
5. Click **Export CSV** to save results.

### Model Training

1. Open the **Model Retraining** tab.
2. Load an experimental database (`.xls`, `.xlsx`, or `.csv`).
3. Map columns to required features via the interactive column-mapping dialog.
4. Select algorithms and an optimisation strategy (Bayesian / TLBO / NSGA-II).
5. Click **Train**; live metrics and a progress bar are displayed throughout.

### Command-Line Training

```bash
# Train all models with Bayesian optimisation (1000 trials, paper default)
python train_frp_models.py --data "database.xls"

# Train specific models only
python train_frp_models.py --data "database.xls" --only LightGBM KNN

# Reduce trials for a faster exploratory run
python train_frp_models.py --data "database.xls" --trials 200

# Set a wall-clock time limit (minutes)
python train_frp_models.py --data "database.xls" --time-limit 60
```

### Batch Prediction

1. Click **Batch Prediction** in the Prediction tab.
2. Select an Excel/CSV file containing beam parameters in the required column format.
3. Results are computed for all design codes and the loaded ML model.
4. Export to CSV for downstream reliability or statistical analysis.

---

## Input Parameters

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Shear span ratio | a/d | — | Ratio of shear span to effective depth |
| Effective depth | d | mm | Distance from compression face to centroid of tensile reinforcement |
| Beam width | b | mm | Width of the rectangular cross-section |
| Concrete compressive strength | f′c | MPa | Cylinder compressive strength |
| FRP reinforcement ratio | ρf | % | Longitudinal FRP reinforcement ratio |
| FRP elastic modulus | Ef | GPa | Elastic modulus of FRP bars |
| FRP material type | — | — | CFRP, GFRP, BFRP, or AFRP |

---

## Design Codes Implemented

| Code | Region | Full Reference |
|------|--------|----------------|
| GB 50608-2020 | China | Technical Standard for Application of Fiber Reinforced Polymer (FRP) in Construction |
| ACI 440.1R-15 | USA | Guide for the Design and Construction of Structural Concrete Reinforced with FRP Bars |
| CSA S806-12 | Canada | Design and Construction of Building Structures with Fibre-Reinforced Polymers |
| BISE (1999) | UK | Interim Guidance on the Design of Reinforced Concrete Structures Using Fibre Composite Reinforcement |
| JSCE (1997) | Japan | Recommendation for Design and Construction of Concrete Structures Using Continuous Fibre Reinforcing Materials |

---

## Model Bundle Format

Trained models are saved as `.frpmdl` files (compressed joblib archives). Each bundle contains:

- Fitted model object(s)
- `MinMaxScaler` fitted on training data
- Feature column names and `OneHotEncoder` for categorical inputs
- Training/test metrics and cross-validation scores
- SHAP calibration subsample (up to 400 rows)
- Metadata: algorithm name, hyperparameters, training timestamp, software version

---

## Project Structure

```
FRP-RC-Shear/
├── main.py                  # Application entry point
├── app.py                   # MainWindow: assembles all tabs
├── train_frp_models.py      # Command-line training script
├── formulas.py              # Design code formula implementations
├── config.py                # Global constants and colour palette
├── column_mapping.py        # Database column auto-detection and mapping
├── metrics.py               # Regression evaluation metrics (R², RMSE, MAE, …)
├── model_io.py              # Model bundle save/load (.frpmdl)
├── optimization.py          # Hyperparameter search: TLBO, Bayesian, NSGA-II
├── qt_compat.py             # PyQt5/PySide6 compatibility shim
├── widgets.py               # Shared UI helper widgets
├── requirements.txt         # pip dependencies
├── environment.yml          # conda environment specification
├── LICENSE                  # MIT License
├── README.md                # This file
├── CITATION.cff             # Machine-readable citation metadata
├── CHANGELOG.md             # Version history
├── models/                  # Pre-trained model bundles (.frpmdl)
├── docs/
│   └── screenshots/         # Application screenshots
├── tests/
│   ├── test_formulas.py     # Unit tests — design code formulas
│   └── test_metrics.py      # Unit tests — evaluation metrics
└── tabs/
    ├── data_tab.py
    ├── train_tab.py
    ├── eval_tab.py
    ├── code_tab.py
    ├── predict_tab.py
    └── interp_tab.py
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Citation

If you use FRP-ShearPred in your research, please cite the following SoftwareX article (forthcoming):

```bibtex
@article{liang2026frpshearpred,
  author  = {Liang, Deyu},
  title   = {{FRP-ShearPred}: An open-source platform for shear capacity
             prediction of stirrup-free {FRP}-reinforced concrete beams
             integrating design codes and ensemble machine learning},
  journal = {SoftwareX},
  year    = {2026},
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
