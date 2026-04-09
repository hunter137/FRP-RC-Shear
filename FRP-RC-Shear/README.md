# FRP-RC Shear Strength Prediction Platform

A desktop application for predicting the shear capacity of FRP (Fiber-Reinforced Polymer) reinforced concrete beams without stirrups. The platform integrates international design codes, machine learning models, and a proposed empirical formula.

**Author:** Deyu Liang  
**Version:** 2.0.0  
**License:** MIT

---

## Features

- **Multi-code prediction**: GB 50608-2020, ACI 440.1R-15, CSA S806-12, BISE (1999), JSCE (1997)
- **Proposed empirical formula**: Bayesian-optimised formula with size-effect, arch-action, and reinforcement terms
- **11 ML algorithms**: GBDT, XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, MLP, SVR, AdaBoost, KNN, 1D-CNN
- **Hyperparameter optimisation**: Bayesian (Optuna TPE), TLBO, NSGA-II multi-objective
- **Model interpretability**: SHAP analysis, feature importance, partial dependence plots
- **Batch prediction**: Process entire databases with CSV/Excel export
- **Portable model bundles**: Save/load trained models as `.frpmdl` files
- **Interactive beam schematic**: Visual cross-section diagram with parameter annotations

---

## Installation

### Requirements

- Python 3.8 or later
- Operating system: Windows / macOS / Linux

### Quick Start

```bash
# Clone the repository
git clone https://github.com/deyuliang/FRP-RC-Shear.git
cd FRP-RC-Shear

# Install dependencies
pip install -r requirements.txt

# Launch the application
python main.py
```

### Optional Dependencies

For full functionality, install the optional packages:

```bash
pip install xgboost lightgbm catboost optuna shap pymoo
```

For 1D-CNN model support:

```bash
pip install torch
```

---

## Usage

### Single Prediction

1. Open the **Prediction** tab
2. Enter beam parameters (a/d, d, b, f'c, ρf, Ef, FRP type)
3. Load a pre-trained model bundle (`.frpmdl`) or train your own
4. Click **Predict** to see results from all design codes and ML models
5. Click **Export CSV** to save results

### Model Training

1. Open the **Model Retraining** tab
2. Load an experimental database (`.xls`, `.xlsx`, or `.csv`)
3. Map columns via the interactive dialog
4. Select algorithms and optimisation strategy
5. Click **Train** to start (progress bar and live metrics shown)

### Command-Line Training

```bash
# Train all 11 models with Bayesian optimisation (1000 trials)
python train_frp_models.py --data "database.xls"

# Train specific models with a time limit
python train_frp_models.py --data "database.xls" --only LightGBM MLP --time-limit 60

# Reduce trials for faster training
python train_frp_models.py --data "database.xls" --trials 200
```

### Batch Prediction

1. Click **Batch Prediction** in the Prediction tab
2. Select an Excel/CSV file with beam parameters
3. Results are computed for all design codes and the loaded ML model
4. Export to CSV for further analysis

---

## Input Parameters

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Shear span ratio | a/d | — | Ratio of shear span to effective depth |
| Effective depth | d | mm | Distance from compression face to centroid of tensile reinforcement |
| Beam width | b | mm | Width of the rectangular cross-section |
| Concrete strength | f'c | MPa | Compressive strength of concrete (cylinder) |
| FRP reinforcement ratio | ρf | % | Longitudinal FRP reinforcement ratio |
| FRP elastic modulus | Ef | GPa | Elastic modulus of FRP reinforcement |
| FRP material type | — | — | CFRP, GFRP, BFRP, or AFRP |

---

## Design Codes Implemented

| Code | Region | Reference |
|------|--------|-----------|
| GB 50608-2020 | China | Technical Standard for FRP in Construction |
| ACI 440.1R-15 | USA | Guide for Design of Concrete with FRP Bars |
| CSA S806-12 | Canada | Design of FRP Structures |
| BISE (1999) | UK | ISIS Canada Design Manual No. 3 |
| JSCE (1997) | Japan | Standard Specification for Concrete |

---

## Model Bundle Format

Trained models are saved as `.frpmdl` files (compressed joblib archives) containing:
- Fitted model object(s)
- MinMaxScaler fitted on training data
- Feature column names and OneHotEncoder
- Training/test metrics and cross-validation scores
- SHAP calibration subsample (400 rows)
- Metadata (algorithm, hyperparameters, timestamp)

---

## Project Structure

```
FRP-RC-Shear/
├── main.py                 # Application entry point
├── app.py                  # MainWindow: assembles all tabs
├── train_frp_models.py     # Command-line training script
├── formulas.py             # Design code formulas (standalone module)
├── config.py               # Global constants and colour palette
├── column_mapping.py       # Database column auto-detection and mapping
├── metrics.py              # Regression evaluation metrics
├── model_io.py             # Model bundle save/load (.frpmdl)
├── optimization.py         # Hyperparameter search (TLBO, Bayesian, NSGA-II)
├── qt_compat.py            # PyQt5/PySide6 compatibility layer
├── widgets.py              # Shared UI helper widgets
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── README.md               # This file
├── models/                 # Pre-trained model bundles
├── tests/
│   ├── test_formulas.py    # Unit tests for design code formulas
│   └── test_metrics.py     # Unit tests for evaluation metrics
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

If you use this software in your research, please cite:

```bibtex
@article{liang2025frprc,
  author  = {Liang, Deyu},
  title   = {FRP-RC Shear Strength Prediction Platform: An Ensemble
             Learning Approach for Stirrup-Free Beams},
  journal = {SoftwareX},
  year    = {2025},
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
