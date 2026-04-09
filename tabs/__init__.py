"""tabs — GUI tab widgets for the FRP-RC shear platform.

Module structure
----------------
data_tab.py          DataTab: database loading and column mapping
train_tab.py         TrainTab: model training orchestration
  train_constants.py   Shared algorithm catalogue and helper functions
  train_threads.py     Background training and TLBO worker threads
  train_hyperparams.py Algorithm hyperparameter configuration dialogs
  train_dialogs.py     Settings, pre-training, and summary dialogs
eval_tab.py          EvalTab: metrics table and diagnostic plots
  eval_dialogs.py      Scatter, response-surface, and percentage dialogs
code_tab.py          CodeTab: design-code vs ML model comparison
predict_tab.py       PredictTab: single-sample inference and batch prediction
  predict_helpers.py   _compute_pi utility and BeamSchematicWidget
  predict_dialogs.py   PredictionSetupDialog and BatchPredictionDialog
interp_tab.py        InterpTab: SHAP / PDP feature analysis
dist_dialog.py       DistributionDialog: histogram + KDE for each variable
shap_dialog.py       ShapBeeswarmDialog: SHAP beeswarm popup
"""
from .data_tab    import DataTab
from .train_tab   import TrainTab
from .eval_tab    import EvalTab
from .code_tab    import CodeTab
from .predict_tab import PredictTab
from .interp_tab  import InterpTab
