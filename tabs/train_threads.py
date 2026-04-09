import warnings
import traceback
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from config import HAS_CUDA
from optimization import (
    tlbo_optimize, _optuna_optimize, nsga2_optimize,
    PARAM_SPACES, _GPU_CAPABLE,
)
from metrics import calc_metrics
from model_io import FittedModel

class _TLBOPreviewThread(QThread):
    progress = pyqtSignal(int, int, float)   # (eval_idx, total, best_cv)
    finished = pyqtSignal(dict, float)        # (best_params, best_score)
    log_line = pyqtSignal(str)

    def __init__(self, factory, space, X, y, n_pop, n_iter, cv, seed):
        super().__init__()
        self._fac    = factory
        self._space  = space
        self._X, self._y = X, y
        self._n_pop  = n_pop
        self._n_iter = n_iter
        self._cv     = cv
        self._seed   = seed
        self._abort  = False

    def stop(self): self._abort = True

    def run(self):
        try:
            bp, bs, _ = tlbo_optimize(
                self._fac, self._space, self._X, self._y,
                cv=self._cv, n_pop=self._n_pop, n_iter=self._n_iter,
                seed=self._seed,
                log_fn=self.log_line.emit,
                stop_flag=lambda: self._abort,
                score_fn=lambda i, s: self.progress.emit(
                    i, self._n_pop + self._n_iter * 2 * self._n_pop, s),
            )
            if not self._abort:
                self.finished.emit(bp, bs)
        except Exception as e:
            import traceback
            self.log_line.emit(f'[FATAL] TLBO preview error:\n{traceback.format_exc()}')

class _TLBOMultiThread(QThread):
    algo_started = pyqtSignal(str)               # algo name
    algo_done    = pyqtSignal(str, dict, float)  # name, best_params, best_score
    trial_update = pyqtSignal(str, int, int, float)  # name, i, total, best
    log_line     = pyqtSignal(str)
    all_done     = pyqtSignal()

    def __init__(self, tasks):
        super().__init__()
        self._tasks = tasks
        self._abort = False

    def stop(self):
        self._abort = True

    def run(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        try:
            self._run_tasks()
        except Exception:
            import traceback
            self.log_line.emit(f'[FATAL] Multi-TLBO error:\n{traceback.format_exc()}')
        finally:
            self.all_done.emit()

    def _run_tasks(self):
        for task in self._tasks:
            if self._abort:
                self.log_line.emit('[WARN] Optimisation aborted.')
                break
            name  = task['name']
            total = task['n_pop'] + task['n_iter'] * 2 * task['n_pop']
            self.algo_started.emit(name)
            self.log_line.emit(f'[INFO] ── {name} '
                               + '─' * max(0, 42 - len(name)))
            self.log_line.emit(
                '    Note: the first 20 trials are random initialisation.'
                '  Some R² values may be low or negative — this is expected.'
                '  The optimiser will converge rapidly after initialisation.')

            is_mlp    = (task['name'] == 'MLP')
            cv_n_jobs = 1 if (task.get('use_gpu', False) or is_mlp) else -1
            if cv_n_jobs == 1:
                self.log_line.emit(
                    f'    [sequential CV] folds run with n_jobs=1 '
                    f'(GPU model or MLP).')

            bp, bs, _ = tlbo_optimize(
                task['factory'], task['space'],
                task['X'], task['y'],
                cv=task['cv'],
                n_pop=task['n_pop'],
                n_iter=task['n_iter'],
                seed=task['seed'],
                log_fn=self.log_line.emit,
                stop_flag=lambda: self._abort,
                score_fn=lambda i, s, _n=name, _t=total:
                    self.trial_update.emit(_n, i, _t, s),
                cv_n_jobs=cv_n_jobs,
            )
            if not self._abort:
                self.log_line.emit(
                    f'[DONE] {name}  —  Best CV R² = {bs:.4f}  '
                    '(\u2713 values applied, parameters locked)')
                self.algo_done.emit(name, bp, bs)

class TrainingThread(QThread):
    progress    = pyqtSignal(int)
    log         = pyqtSignal(str)
    done        = pyqtSignal(dict)
    trial_score = pyqtSignal(str, int, float)   # (model_name, eval_idx, best_cv)

    def __init__(self, X_tr, X_te, y_tr, y_te, models_cfg,
                 cv_folds, X_all, y_all,
                 opt_strategy='none', opt_trials=50, seed=42,
                 custom_ranges=None, nsga2_objectives=None,
                 tlbo_settings=None, bayes_settings=None,
                 use_gpu=False, early_stop=True, patience=80):
        super().__init__()
        self.X_tr             = X_tr
        self.X_te             = X_te
        self.y_tr             = y_tr
        self.y_te             = y_te
        self.models_cfg       = models_cfg
        self.cv               = cv_folds
        self.X_all            = X_all
        self.y_all            = y_all
        self.opt_strategy     = opt_strategy
        self.opt_trials       = opt_trials
        self.seed             = seed
        self.custom_ranges    = custom_ranges or {}
        self.nsga2_objectives = nsga2_objectives or ['R2', 'safety_pct']
        self.tlbo_settings    = tlbo_settings  or {'mode': 'auto', 'n_pop': 10, 'n_iter': 10}
        self.bayes_settings   = bayes_settings or {'n_startup_trials': 10, 'multivariate': False}
        self.use_gpu          = use_gpu
        self.early_stop       = early_stop
        self.patience         = patience
        self._stop            = False

    def stop(self):
        self._stop = True
        self.log.emit('    [WARN] Stop requested …')

    def _stopped(self): return self._stop

    def _build_space(self, name, locked_params=None):
        """Return param space with custom ranges applied and locked params removed."""
        if name not in PARAM_SPACES:
            return None
        space   = list(PARAM_SPACES[name]())
        custom  = self.custom_ranges.get(name, {})
        if custom:
            space = [(p, custom.get(p, (lo, hi))[0],
                         custom.get(p, (lo, hi))[1], is_int)
                     for p, lo, hi, is_int in space]
        # Remove locked (fixed) parameters from search space
        if locked_params:
            space = [(p, lo, hi, is_int) for p, lo, hi, is_int in space
                     if p not in locked_params]
        return space if space else None

    def run(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        results = {}
        try:
            self._run_inner(results)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log.emit(f'\n[FATAL] Unexpected error in training thread:\n{tb}')
        finally:
            self.done.emit(results)

    def _run_inner(self, results):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        n_mod   = len(self.models_cfg)
        cv_tr   = max(2, min(self.cv, len(self.X_tr)))
        cv_all  = max(2, min(self.cv, len(self.X_all)))

        for idx, (name, cfg) in enumerate(self.models_cfg.items()):
            if self._stopped(): break
            factory      = cfg['factory']
            fixed_params = cfg['fixed_params']
            model_uses_gpu = cfg.get('use_gpu', False)
            is_mlp         = (name == 'MLP')
            cv_n_jobs      = 1 if (model_uses_gpu or is_mlp) else -1
            self.log.emit(f'\n{"="*55}')
            self.log.emit(f'[{idx+1} of {n_mod}]  {name}'
                          + ('  [GPU]' if model_uses_gpu else ''))

            best_params = dict(fixed_params)
            best_cv     = float('nan')
            locked_params = cfg.get('locked_params', set())
            space       = self._build_space(name, locked_params)

            # Log locked params so user knows what's fixed
            if locked_params and self.opt_strategy != 'none':
                self.log.emit(
                    f'    Locked (fixed) params: '
                    f'{", ".join(sorted(locked_params))}')

            # score_fn: emit trial signal so the live curve updates
            def _sfn(i, s, _n=name):
                self.trial_score.emit(_n, i, s)

            if self.opt_strategy in ('bayesian', 'tlbo', 'nsga2') \
                    and space is not None:
                if self.opt_strategy == 'tlbo':
                    ts = self.tlbo_settings
                    if ts.get('mode') == 'manual':
                        n_pop  = ts['n_pop']
                        n_iter = ts['n_iter']
                    else:
                        n_pop  = max(5, self.opt_trials // 5)
                        n_iter = max(3, self.opt_trials // 5)
                    total = n_pop + n_iter * 2 * n_pop
                    self.log.emit(
                        f'    TLBO (Teaching-Learning)  N_pop={n_pop}, N_iter={n_iter}, '
                        f'total≈{total} evals …')
                    opt_params, best_cv, _ = tlbo_optimize(
                        factory, space,
                        self.X_tr, self.y_tr, cv=cv_tr,
                        n_pop=n_pop,
                        n_iter=n_iter,
                        seed=self.seed,
                        log_fn=self.log.emit,
                        stop_flag=self._stopped,
                        score_fn=_sfn,
                        cv_n_jobs=cv_n_jobs)
                elif self.opt_strategy == 'bayesian':
                    bs = self.bayes_settings
                    self.log.emit(
                        f'    Bayesian Optimisation (TPE)  {self.opt_trials} trials, '
                        f'startup={bs["n_startup_trials"]}, '
                        f'multivariate={"yes" if bs["multivariate"] else "no"} …')
                    opt_params, best_cv, _ = _optuna_optimize(
                        factory, space,
                        self.X_tr, self.y_tr, cv=cv_tr,
                        n_trials=self.opt_trials,
                        seed=self.seed,
                        log_fn=self.log.emit,
                        stop_flag=self._stopped,
                        score_fn=_sfn,
                        n_startup_trials=bs['n_startup_trials'],
                        multivariate=bs['multivariate'],
                        cv_n_jobs=cv_n_jobs,
                        early_stop=self.early_stop,
                        patience=self.patience)
                elif self.opt_strategy == 'nsga2':
                    self.log.emit(
                        f'    NSGA-II (Multi-Objective)  {self.opt_trials} evaluations')
                    opt_params, best_cv, _ = nsga2_optimize(
                        factory, space,
                        self.X_tr, self.X_te,
                        self.y_tr, self.y_te,
                        cv=cv_tr,
                        n_gen=max(5, self.opt_trials // 10),
                        pop_size=10,
                        seed=self.seed,
                        log_fn=self.log.emit,
                        objectives=self.nsga2_objectives,
                        stop_flag=self._stopped)
                else:
                    opt_params = {}
                best_params.update(opt_params)
            else:
                try:
                    cv_scores = cross_val_score(
                        factory(**best_params),
                        self.X_tr, self.y_tr,
                        cv=cv_tr, scoring='r2', n_jobs=cv_n_jobs)
                    best_cv = float(cv_scores.mean())
                    self.log.emit(
                        f'    Fixed parameters  CV R2 = {best_cv:.4f} '
                        f'± {cv_scores.std():.4f}')
                except Exception as e:
                    self.log.emit(f'    [WARN] CV failed: {e}')

            if self._stopped(): break

            # Final fit
            try:
                model   = factory(**best_params)
                model.fit(self.X_tr, self.y_tr)
                tr_pred = model.predict(self.X_tr)
                te_pred = model.predict(self.X_te)
                tr_m    = calc_metrics(self.y_tr, tr_pred)
                te_m    = calc_metrics(self.y_te, te_pred)
                try:
                    cv_full = cross_val_score(
                        factory(**best_params), self.X_all, self.y_all,
                        cv=cv_all, scoring='r2', n_jobs=cv_n_jobs)
                    cv_mean, cv_std = float(cv_full.mean()), float(cv_full.std())
                except Exception:
                    cv_mean, cv_std = best_cv, float('nan')

                results[name] = {
                    'model':       FittedModel(model),
                    'best_params': best_params,
                    'tr_pred':    tr_pred,
                    'te_pred':    te_pred,
                    'tr_metrics': tr_m,
                    'te_metrics': te_m,
                    'cv_mean':    cv_mean,
                    'cv_std':     cv_std,
                    '_y_tr':      self.y_tr,
                    '_y_te':      self.y_te,
                }
                self.log.emit(
                    f'    Train  R² = {tr_m["R2"]:.4f}  '
                    f'RMSE = {tr_m["RMSE"]:.2f} kN')
                self.log.emit(
                    f'    Test   R² = {te_m["R2"]:.4f}  '
                    f'RMSE = {te_m["RMSE"]:.2f} kN  '
                    f'r = {te_m["r"]:.4f}')
                self.log.emit(
                    f'    Full CV  = {cv_mean:.4f} ± {cv_std:.4f}')
            except Exception as e:
                self.log.emit(f'    [ERROR] {name} failed: {e}')

            self.progress.emit(int((idx + 1) / n_mod * 100))

