# tools/

Utility scripts for development and diagnostics.  These scripts are
**not** required for normal use of the application.

---

## diagnose_crash.py — Crash Diagnostics Tool

Run this script instead of `main.py` whenever the application exits
silently, crashes without a visible error dialog, or hangs during
model training.

### Usage

```bash
# From the project root:
python tools/diagnose_crash.py
```

A log file called `crash_diag.log` is written to the project root.
Attach that file when opening a bug report.

### What it captures

| Category | Detail |
|---|---|
| C-level crashes | Segfaults and access violations via `faulthandler` |
| Uncaught exceptions | Full traceback via `sys.excepthook` |
| `sys.exit` call-sites | Stack trace showing who triggered the exit |
| Training thread | Entry/exit and any exception in `TrainingThread.run` |
| GUI callback | Entry/exit and any exception in `TrainTab._on_done_inner` |
| Thread snapshots | Active thread list every 5 seconds |
| Dependency versions | PyQt5, NumPy, scikit-learn, Optuna, XGBoost, LightGBM, … |
| BLAS back-ends | Thread-pool info from `threadpoolctl` |

### Common crash causes diagnosed by this tool

| Symptom | Likely cause shown in log |
|---|---|
| Silent exit during training | `KMP_DUPLICATE_LIB_OK` issue (duplicate OpenMP runtime) |
| `SystemExit(1)` in log | Exception swallowed by a Qt slot |
| Thread snapshot stops updating | Deadlock in worker thread |
| Segfault line in log | NumPy / BLAS version mismatch |

### Notes

- The script sets `KMP_DUPLICATE_LIB_OK=TRUE` and `KMP_WARNINGS=FALSE`
  **before** any imports, which is required for the Intel OpenMP runtime
  to honour them.
- It is safe to keep in the repository; it adds no overhead when the
  application is started normally via `main.py`.
