# LLX2026 Concrete Drying Shrinkage Prediction

Version 1

This repository contains a Python implementation and desktop interface for the
LLX2026 concrete drying-shrinkage equation. The program supports individual
prediction, CSV batch prediction, development-curve plotting, and empirical
prediction intervals.

The model is associated with the manuscript *Metaheuristic-Based Parameter
Calibration of Empirical Concrete Drying Shrinkage Models: Systematic
Evaluation and Improved Formulation*. The manuscript has not been accepted or
published. This repository is a software record and should not be interpreted
as evidence of journal acceptance or peer-review endorsement.

## Interface

### Model formulation

![Model formulation](docs/screenshots/01_model_formulation.png)

### Individual prediction

![Individual prediction](docs/screenshots/02_individual_prediction.png)

### Batch prediction

![Batch prediction](docs/screenshots/03_batch_prediction.png)

### Development curve

![Development curve](docs/screenshots/04_development_curve.png)

## Requirements

- Python 3.8 or later
- Tkinter
- NumPy
- pandas
- Pillow
- Matplotlib

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run the program

From the repository directory:

```bash
python main.py
```

The interface display is retained from the original program. Numerical,
batch-processing, and plotting functions are also available from the
`llx2026` package for use in scripts or notebooks.

## Inputs

| Input | Meaning | Unit |
|---|---|---|
| `dt` | Drying duration | days |
| `t0` | Age at the start of drying | days |
| `RH` | Relative humidity | % |
| `VtoS` | Volume-to-surface ratio | mm |
| `wc` | Water-to-cement ratio | – |
| `agg_total` | Total aggregate content | kg/m³ |

An example batch-input file is provided at
[`examples/batch_input.csv`](examples/batch_input.csv).

## Python use

```python
from llx2026 import ShrinkageInputs, evaluate

inputs = ShrinkageInputs(
    drying_time=100,
    curing_age=7,
    relative_humidity=60,
    volume_to_surface=50,
    water_cement_ratio=0.45,
    aggregate_content=1860,
)

result = evaluate(inputs)
print(result.value)
print(result.pi90_lower, result.pi90_upper)
```

## Data note

The original calibration database is not included in this repository. The CSV
file in `examples/` only demonstrates the required input format.

## Model-use note

The program is provided for research and educational use. Prediction intervals
are empirical record-level intervals, not confidence intervals for the mean.
Predictions outside the calibration domain should be treated cautiously. The
software is not a design code and should not be used as the sole basis for
structural design or safety decisions.

## Zenodo record

A DOI has been reserved in a Zenodo draft, but the record has not yet been
published. Consequently, DOI resolver links are not active at present. The
planned Zenodo record address is:

[https://zenodo.org/records/21614015](https://zenodo.org/records/21614015)

After the Zenodo record is published, this section can be updated with the
active DOI citation.

## Authors

- Deyu Liang
- Jinlong Liu
- Lei Xu

## Repository structure

```text
LLX2026-drying-shrinkage-predictor/
├── main.py                  # application entry point
├── src/llx2026/
│   ├── gui.py               # original desktop interface
│   ├── model.py             # LLX2026 numerical model
│   ├── batch.py             # CSV and DataFrame helpers
│   └── plotting.py          # reusable plotting function
├── examples/                # example CSV input
├── docs/screenshots/        # four interface screenshots
├── tests/                   # automated checks
├── CITATION.cff
├── MODEL_CARD.md
└── LICENSE
```

After installing the project in editable mode, run the checks with:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## License

MIT License. See [`LICENSE`](LICENSE).
