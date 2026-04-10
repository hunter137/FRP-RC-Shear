# Pre-trained Model Bundles

This directory is the default location for `.frpmdl` model bundle files.

## Obtaining pre-trained models

Pre-trained models are distributed via the GitHub Releases page to avoid
storing large binary files in the repository.

1. Go to the [Releases page](https://github.com/hunter137/FRP-RC-Shear/releases).
2. Download the asset `pretrained_models_v2.0.0.zip`.
3. Extract the `.frpmdl` files into this directory.

Alternatively, you can train your own models using the **Model Retraining** tab
in the application or the command-line script:

```bash
python train_frp_models.py --data "your_database.xlsx"
```

## Bundle format

Each `.frpmdl` file is a compressed joblib archive containing:

| Key | Description |
|-----|-------------|
| `model` | Fitted scikit-learn / XGBoost / LightGBM estimator object |
| `scaler` | `MinMaxScaler` fitted on training features |
| `feature_cols` | Ordered list of feature column names |
| `encoder` | `OneHotEncoder` for the FRP material type category |
| `metrics` | Dict of R², RMSE, MAE for train, validation, and test splits |
| `cv_scores` | 5-fold cross-validation R² scores |
| `shap_sample` | 400-row subsample for fast SHAP background computation |
| `meta` | Algorithm name, hyperparameters, training timestamp, app version |
