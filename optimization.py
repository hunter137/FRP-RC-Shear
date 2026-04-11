"""
optimization.py — Hyperparameter search backends.

Implements TLBO (Rao et al. 2011), Bayesian TPE via Optuna, and
NSGA-II via pymoo.  Also provides model factory functions and
parameter space definitions for all supported algorithms.
"""
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from config import HAS_XGB, HAS_LGB, HAS_CAT, HAS_OPTUNA, HAS_PYMOO
from metrics import calc_metrics

if HAS_XGB:    import xgboost as xgb
if HAS_LGB:    import lightgbm as lgb
if HAS_CAT:    import catboost as cb
if HAS_OPTUNA:
    import optuna
    import optuna.exceptions
if HAS_PYMOO:
    from pymoo.core.problem         import Problem as _Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2   as _NSGA2
    from pymoo.optimize             import minimize as _pymoo_min
    # pymoo 0.6.0+  : pymoo.termination.get_termination
    # pymoo 0.5.x   : pymoo.factory.get_termination  (or manual ctor)
    _get_termination = None
    try:
        from pymoo.termination import get_termination as _get_termination
    except ImportError:
        pass
    if _get_termination is None:
        try:
            from pymoo.factory import get_termination as _get_termination
        except ImportError:
            pass
    if _get_termination is None:
        # Last-resort: construct termination object manually
        try:
            from pymoo.termination.default import DefaultMultiObjectiveTermination as _DMOT
            def _get_termination(name, n):
                return _DMOT(n_max_gen=n)
        except Exception:
            def _get_termination(name, n):
                return n   # pymoo sometimes accepts plain int for n_gen

NSGA2_OBJECTIVES = {
    'R2':         ('R²  (coefficient of determination)',   'maximize'),
    'safety_pct': ('Safety %  (V_pred ≤ V_exp)',           'maximize'),
    'RMSE':       ('RMSE  (root-mean-square error, kN)',    'minimize'),
    'MAE':        ('MAE  (mean absolute error, kN)',        'minimize'),
    'MAPE':       ('MAPE  (mean abs. percentage error %)',  'minimize'),
    'cov':        ('CoV  (coefficient of variation)',       'minimize'),
    'mean_ratio': ('Mean ratio  (|k̄ − 1.0| minimised)',   'ratio'),
}

# sklearn < 1.2 uses sparse=False; >= 1.2 uses sparse_output=False.
def _ohe_sparse_kwarg():
    """Return the correct sparse keyword for OneHotEncoder."""
    import sklearn
    from packaging.version import Version
    try:
        if Version(sklearn.__version__) >= Version('1.2'):
            return {'sparse_output': False}
        return {'sparse': False}
    except Exception:
        from sklearn.preprocessing import OneHotEncoder
        try:
            OneHotEncoder(sparse_output=False)
            return {'sparse_output': False}
        except TypeError:
            return {'sparse': False}

# XGBoost >= 2.0 renamed tree_method='gpu_hist' to device='cuda'.
def _xgb_gpu_kwargs():
    """Return the correct GPU kwargs dict for the installed XGBoost."""
    if not HAS_XGB:
        return {}
    try:
        import xgboost as _xgb
        if Version(_xgb.__version__) >= Version('2.0.0'):
            return {'device': 'cuda'}
        else:
            return {'tree_method': 'gpu_hist'}
    except Exception:
        # Fall back to the new API; if it fails at runtime XGBoost will warn
        return {'device': 'cuda'}

def _ps_gbdt():
    """GBDT — Table 3-1.  Optimal (paper): n_est=2300, depth=3, lr=0.15, split=50, leaf=1."""
    return [
        ('n_estimators',      200,  3000, True),
        ('max_depth',           3,    20, True),
        ('learning_rate',    0.03,   0.2, False),
        ('min_samples_split',   2,   100, True),
        ('min_samples_leaf',    1,    10, True),
    ]

def _ps_xgb():
    """XGBoost — Table 3-2.  Optimal (paper): n_est=3000, depth=9, lr=0.12, mcw=2."""
    return [
        ('n_estimators',     200,  3000, True),
        ('max_depth',          3,    12, True),
        ('learning_rate',   0.03,   0.2, False),
        ('min_child_weight',   2,    30, True),
        ('subsample',        0.5,   1.0, False),
    ]

def _ps_lgb():
    """LightGBM — Table 3-3.  Optimal (paper): n_est=800, depth=3, lr=0.18, leaves=512, mcs=1."""
    return [
        ('n_estimators',     200,  3000, True),
        ('max_depth',          3,    20, True),
        ('learning_rate',   0.03,   0.2, False),
        ('num_leaves',        15,   512, True),
        ('min_child_samples',  1,    10, True),
    ]

def _ps_cat():
    """CatBoost — Table 3-4.  Optimal (paper): n_est=800, lr=0.13, leaf=5, l2=9.6, bag=0.82."""
    return [
        ('iterations',       200,  3000, True),
        ('learning_rate',   0.03,   0.2, False),
        ('min_data_in_leaf',   1,    10, True),
        ('l2_leaf_reg',        0,    10, False),
        ('bagging_temperature',0,     1, False),
    ]

def _ps_rf():
    """Random Forest — Table 3-5.  Optimal (paper): n_est=1200, depth=25, leaf=1, split=5."""
    return [
        ('n_estimators',     200,  3000, True),
        ('max_depth',          3,    30, True),
        ('min_samples_leaf',   1,    10, True),
        ('min_samples_split',  2,   100, True),
        ('max_features',     0.1,   1.0, False),
    ]

PARAM_SPACES = {
    'GBDT':          _ps_gbdt,
    'XGBoost':       _ps_xgb,
    'LightGBM':      _ps_lgb,
    'CatBoost':      _ps_cat,
    'Random Forest': _ps_rf,
    'Extra Trees':   lambda: [
        ('n_estimators',    200, 3000, True),
        ('max_depth',         3,   30, True),
        ('min_samples_leaf',  1,   10, True),
        ('min_samples_split', 2,  100, True),
        ('max_features',    0.1,  1.0, False),
    ],
    'AdaBoost':  lambda: [('n_estimators',50,500,True),
                           ('learning_rate',0.01,2.0,False)],
    'KNN':       lambda: [('n_neighbors',2,20,True),
                           ('leaf_size',10,60,True)],
}

# GPU-capable models use n_jobs=1 for CV to avoid multi-process VRAM contention.
_GPU_CAPABLE = {'XGBoost', 'LightGBM', 'CatBoost'}

def _lgb_gpu_kwargs():
    """LightGBM GPU device kwarg — version-aware (device_type >= 4.0, device < 4.0)."""
    if not HAS_LGB:
        return {}
    try:
        import lightgbm as _lgb
        if Version(_lgb.__version__) >= Version('4.0.0'):
            return {'device_type': 'gpu'}
        else:
            return {'device': 'gpu'}
    except Exception:
        return {'device': 'gpu'}

def _factory_for(name, seed, use_gpu=False):
    """Return a callable(**params) -> fitted model for the given algorithm."""
    if name == 'GBDT':
        # Paper uses min_samples_leaf (1-10) — pass through directly
        return lambda **p: GradientBoostingRegressor(random_state=seed, **p)
    if name == 'XGBoost':
        if use_gpu:
            gpu_kw = _xgb_gpu_kwargs()
            def _make_xgb_gpu(_gkw=gpu_kw, _seed=seed):
                def _factory(**p):
                    return xgb.XGBRegressor(
                        random_state=_seed, verbosity=0, n_jobs=1, **_gkw, **p)
                return _factory
            return _make_xgb_gpu()
        def _make_xgb_cpu(_seed=seed):
            def _factory(**p):
                return xgb.XGBRegressor(
                    random_state=_seed, verbosity=0, n_jobs=-1, **p)
            return _factory
        return _make_xgb_cpu()
    if name == 'LightGBM':
        if use_gpu:
            gpu_kw = _lgb_gpu_kwargs()
            # GPU model: n_jobs=1 avoids multi-process VRAM contention.
            # verbosity=-1 silences LightGBM C++-level "N warning generated." output.
            return lambda **p: lgb.LGBMRegressor(
                random_state=seed, verbose=-1, verbosity=-1,
                n_jobs=1, **gpu_kw, **p)
        return lambda **p: lgb.LGBMRegressor(
            random_state=seed, verbose=-1, verbosity=-1, n_jobs=-1, **p)
    if name == 'CatBoost':
        # bagging_temperature only works with bootstrap_type='Bayesian'.
        # Add it automatically so the parameter is never silently ignored.
        def _prep_cat(p):
            if 'bagging_temperature' in p:
                p = dict(p)
                p.setdefault('bootstrap_type', 'Bayesian')
            return p
        if use_gpu:
            gpu_kw = {'task_type': 'GPU', 'devices': '0', 'thread_count': -1}
            def _make_cat_gpu(_gkw=gpu_kw, _seed=seed):
                def _factory(**p):
                    return cb.CatBoostRegressor(
                        random_state=_seed, verbose=0, **_gkw, **_prep_cat(p))
                return _factory
            return _make_cat_gpu()
        def _make_cat_cpu(_seed=seed):
            def _factory(**p):
                return cb.CatBoostRegressor(
                    random_state=_seed, verbose=0,
                    thread_count=-1, **_prep_cat(p))
            return _factory
        return _make_cat_cpu()
    if name == 'Random Forest':
        return lambda **p: RandomForestRegressor(
            random_state=seed, n_jobs=-1, **p)
    if name == 'Extra Trees':
        return lambda **p: ExtraTreesRegressor(
            random_state=seed, n_jobs=-1, **p)
    if name == 'AdaBoost':
        return lambda **p: AdaBoostRegressor(random_state=seed, **p)
    if name == 'KNN':
        return lambda **p: KNeighborsRegressor(n_jobs=-1, **p)
    raise ValueError(f'Unknown algorithm: {name}')

def _score_vec(vec, space, factory, X, y, cv, stop_flag=None, cv_n_jobs=-1):
    if stop_flag is not None and stop_flag():
        return -1.0
    params = {n: int(round(v)) if is_int else float(v)
              for (n, lo, hi, is_int), v in zip(space, vec)}
    try:
        val = float(cross_val_score(
            factory(**params), X, y,
            cv=cv, scoring='r2', n_jobs=cv_n_jobs).mean())
        return val if np.isfinite(val) else -1.0
    except Exception:
        return -1.0

def tlbo_optimize(factory, space, X, y, cv=5,
                  n_pop=10, n_iter=20, seed=42,
                  log_fn=None, stop_flag=None, score_fn=None,
                  cv_n_jobs=-1):
    """
    Teaching-Learning-Based Optimisation (TLBO).
    Reference: Rao, R.V. et al. (2011). Information Sciences 183, 1–15.
    """
    rng         = np.random.RandomState(seed)
    n_p         = len(space)
    total_evals = n_pop * (1 + 2 * n_iter)
    eval_ct     = [0]
    best_running = [-np.inf]

    def _score(vec, tag=''):
        if stop_flag and stop_flag():
            return -1.0
        s = _score_vec(vec, space, factory, X, y, cv, stop_flag,
                       cv_n_jobs=cv_n_jobs)
        eval_ct[0] += 1
        if s > best_running[0]:
            best_running[0] = s
        # Log every 10 evals (or first) to avoid flooding the GUI log widget.
        # Bayesian already logs every 10 trials; TLBO now matches that cadence.
        if log_fn and (eval_ct[0] == 1 or eval_ct[0] % 10 == 0
                       or eval_ct[0] == total_evals):
            log_fn(f'    [{eval_ct[0]:>3d}/{total_evals}] {tag}  '
                   f'CV R² = {s:.4f}  (best={best_running[0]:.4f})')
        if score_fn:
            score_fn(eval_ct[0], best_running[0])
        return s

    def clip(pop):
        for j, (n, lo, hi, is_int) in enumerate(space):
            pop[:, j] = np.clip(pop[:, j], lo, hi)
            if is_int:
                pop[:, j] = np.round(pop[:, j])
        return pop

    if log_fn:
        log_fn(f'    Initialising population  ({n_pop} members × {cv}-fold CV)')
    pop    = clip(np.column_stack(
        [rng.uniform(lo, hi, n_pop) for _, lo, hi, _ in space]))
    scores = np.array([_score(pop[i], f'init [{i+1}/{n_pop}]')
                       for i in range(n_pop)])
    if stop_flag and stop_flag():
        return {n: lo for n, lo, hi, is_int in space}, -1.0, []
    best_v, best_s = pop[np.argmax(scores)].copy(), scores.max()
    history = [(0, best_s)]
    if log_fn:
        log_fn(f'    [INFO] Initialisation complete — best CV R² = {best_s:.4f}')

    for it in range(1, n_iter + 1):
        if stop_flag and stop_flag():
            break
        teacher = pop[np.argmax(scores)].copy()
        mean_p  = pop.mean(axis=0)
        Tf      = rng.randint(1, 3)
        r       = rng.uniform(0, 1, (n_pop, n_p))
        new_pop = clip(pop + r * (teacher - Tf * mean_p))
        ns      = np.array([_score(new_pop[i], f'iter {it:02d} / teach [{i+1}]')
                            for i in range(n_pop)])
        imp = ns > scores
        pop[imp], scores[imp] = new_pop[imp], ns[imp]
        for i in range(n_pop):
            if stop_flag and stop_flag():
                break
            j    = rng.choice([k for k in range(n_pop) if k != i])
            r_l  = rng.uniform(0, 1, n_p)
            cand = clip((pop[i] + r_l * (
                pop[i] - pop[j] if scores[i] > scores[j]
                else pop[j] - pop[i])).reshape(1, n_p))[0]
            sc = _score(cand, f'iter {it:02d} / learn [{i+1}]')
            if sc > scores[i]:
                pop[i], scores[i] = cand, sc
        bi = int(np.argmax(scores))
        if scores[bi] > best_s:
            best_v, best_s = pop[bi].copy(), scores[bi]
        history.append((it, best_s))
        if log_fn:
            log_fn(f'    [INFO] TLBO iteration {it:02d}/{n_iter} — '
                   f'best CV R² = {best_s:.4f}')

    best_params = {n: int(round(v)) if is_int else float(v)
                   for (n, lo, hi, is_int), v in zip(space, best_v)}
    return best_params, best_s, history

def _optuna_optimize(factory, space, X, y, cv=5,
                     n_trials=50, seed=42,
                     log_fn=None, stop_flag=None, score_fn=None,
                     n_startup_trials=10, multivariate=False,
                     cv_n_jobs=-1, early_stop=True, patience=80,
                     n_parallel_trials=1):
    """Bayesian hyperparameter search via Optuna TPE sampler."""
    if not HAS_OPTUNA:
        if log_fn:
            log_fn('    [WARN] optuna not installed — pip install optuna')
        return ({n: (int(round((lo+hi)/2)) if is_int else (lo+hi)/2)
                 for n, lo, hi, is_int in space}, 0.0, [])

    import threading
    _lock = threading.Lock()

    history  = []
    best_so  = [-np.inf]
    no_improve_count = [0]

    if log_fn:
        log_fn(f'    Startup trials: {n_startup_trials}, '
               f'multivariate: {"yes" if multivariate else "no"}')
        if early_stop:
            log_fn(f'    Early stopping enabled (patience={patience})')
        if n_parallel_trials > 1:
            log_fn(f'    Parallel trials: {n_parallel_trials}')

    def objective(trial):
        if stop_flag and stop_flag():
            raise optuna.exceptions.TrialPruned()
        params = {}
        for n, lo, hi, is_int in space:
            params[n] = (trial.suggest_int(n, int(lo), int(hi)) if is_int
                         else trial.suggest_float(
                             n, lo, hi, log=(lo > 0 and hi / lo > 10)))
        try:
            import joblib as _jlib
            with _jlib.parallel_backend('loky', n_jobs=cv_n_jobs):
                val = float(cross_val_score(
                    factory(**params), X, y,
                    cv=cv, scoring='r2', n_jobs=cv_n_jobs).mean())
        except Exception:
            val = -1.0

        with _lock:
            if val > best_so[0] + 1e-6:
                best_so[0] = val
                no_improve_count[0] = 0
            else:
                no_improve_count[0] += 1
            history.append((trial.number, best_so[0]))
            cur_best = best_so[0]
            cur_no_improve = no_improve_count[0]

        if score_fn:
            score_fn(trial.number + 1, cur_best)
        if log_fn and trial.number % 10 == 0:
            log_fn(f'    [INFO] trial {trial.number:03d}  best CV R² = {cur_best:.4f}')

        if (early_stop
                and trial.number >= n_startup_trials
                and cur_no_improve >= patience):
            if log_fn:
                log_fn(f'    [INFO] Early stop at trial {trial.number} '
                       f'(best CV R² = {cur_best:.4f})')
            trial.study.stop()

        return val

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            multivariate=multivariate,
        ))
    study.optimize(objective, n_trials=n_trials,
                   n_jobs=n_parallel_trials,
                   show_progress_bar=False)
    return study.best_params, study.best_value, history

def nsga2_optimize(factory, space, X_tr, X_te, y_tr, y_te,
                   cv=5, n_gen=20, pop_size=15, seed=42, log_fn=None,
                   objectives=None, stop_flag=None):
    """
    NSGA-II bi-objective search.
    Reference: Deb, K. et al. (2002). IEEE Trans. Evol. Comput. 6, 182–197.
    """
    if objectives is None or len(objectives) != 2:
        objectives = ['R2', 'safety_pct']

    if not HAS_PYMOO:
        if log_fn:
            log_fn('    [WARN] pymoo not installed — '
                   'falling back to Bayesian search.')
        return _optuna_optimize(factory, space, X_tr, y_tr, cv,
                                n_trials=pop_size * n_gen,
                                seed=seed, log_fn=log_fn,
                                stop_flag=None,
                                n_startup_trials=min(10, pop_size),
                                multivariate=False)

    def _metric_to_cost(met, key):
        direction = NSGA2_OBJECTIVES.get(key, ('', 'minimize'))[1]
        val = met.get(key, 0.0)
        if direction == 'maximize':
            return -val
        if direction == 'ratio':
            return abs(val - 1.0)
        return val

    def _eval(vec):
        if stop_flag is not None and stop_flag():
            return (0.0, 0.0)
        params = {n: int(round(np.clip(v, lo, hi))) if is_int
                  else float(np.clip(v, lo, hi))
                  for (n, lo, hi, is_int), v in zip(space, vec)}
        try:
            m = factory(**params)
            m.fit(X_tr, y_tr)
            te_pred = m.predict(X_te)
            met     = calc_metrics(y_te, te_pred)
            # When R2 is an objective, replace the simple test-set R² with
            # the cross-validated score so the CV setting is honoured.
            if 'R2' in objectives and cv > 1:
                try:
                    cv_r2 = float(cross_val_score(
                        factory(**params), X_tr, y_tr,
                        cv=min(cv, len(y_tr)), scoring='r2', n_jobs=1).mean())
                    met = dict(met)
                    met['R2'] = cv_r2
                except Exception:
                    pass  # fall back to test-set R²
            return (_metric_to_cost(met, objectives[0]),
                    _metric_to_cost(met, objectives[1]))
        except Exception:
            return (0.0, 0.0)

    class _HP(_Problem):
        def __init__(self):
            super().__init__(
                n_var=len(space), n_obj=2,
                xl=np.array([lo for _, lo, _, _ in space], float),
                xu=np.array([hi for _, _, hi, _ in space], float))
        def _evaluate(self, X, out, *args, **kwargs):
            out['F'] = np.array([_eval(row) for row in X])

    lbl1 = NSGA2_OBJECTIVES.get(objectives[0], (objectives[0],))[0]
    lbl2 = NSGA2_OBJECTIVES.get(objectives[1], (objectives[1],))[0]
    if log_fn:
        log_fn(f'    Objective 1: {lbl1}')
        log_fn(f'    Objective 2: {lbl2}')

    termination = _get_termination('n_gen', n_gen)
    res = _pymoo_min(
        _HP(), _NSGA2(pop_size=pop_size, eliminate_duplicates=True),
        termination, seed=seed, verbose=False)
    bi          = np.argmin(res.F[:, 0])
    best_params = {n: int(round(np.clip(v, lo, hi))) if is_int
                   else float(np.clip(v, lo, hi))
                   for (n, lo, hi, is_int), v in zip(space, res.X[bi])}
    if log_fn:
        log_fn(f'    [INFO] NSGA-II complete — '
               f'Pareto size = {len(res.X)}')
    return best_params, float(-res.F[bi, 0]), []
