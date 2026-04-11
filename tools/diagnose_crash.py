"""
FRP-ShearPred  –  Crash Diagnostics Tool
=========================================
Run this script instead of ``main.py`` whenever the application crashes
silently or exits without a visible error message.  It wraps the normal
startup with comprehensive logging so that the root cause can be identified
from the generated log file.

Usage
-----
From the project root directory:

    python tools/diagnose_crash.py

A file called ``crash_diag.log`` is written to the project root.  Share
that file when reporting a bug.

What it captures
----------------
* C-level crashes (segfaults, access violations) via ``faulthandler``
* Uncaught Python exceptions (``sys.excepthook``)
* ``sys.exit`` call-sites with full stack traces
* ``TrainingThread.run`` entry / exit and any exception it raises
* ``TrainTab._on_done_inner`` entry / exit and any exception it raises
* Periodic thread snapshots every 5 seconds
* Versions of all key dependencies at startup
* BLAS thread-pool back-end information (via ``threadpoolctl``)

Notes
-----
* ``KMP_DUPLICATE_LIB_OK`` and ``KMP_WARNINGS`` are set *before* any
  import so that the Intel OpenMP runtime picks them up at first load.
* The script is safe to leave in place during normal use; it adds no
  overhead unless you explicitly run it.
"""

# ── 0. Environment flags — must come before ALL other imports ────────────────
# KMP_DUPLICATE_LIB_OK must be set before Intel OpenMP (libiomp5md.dll /
# libiomp5.dylib) is first loaded.  Importing sklearn or joblib before
# setting this flag makes the variable arrive too late.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_WARNINGS']         = 'FALSE'

# ── 1. Standard-library imports ──────────────────────────────────────────────
import sys
import atexit
import datetime
import faulthandler
import io
import threading
import time
import traceback

# ── 2. Log file setup ────────────────────────────────────────────────────────
# Always write to the project root, one directory above this script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE      = os.path.join(_PROJECT_ROOT, 'crash_diag.log')

_log_fh = open(LOG_FILE, 'w', encoding='utf-8', buffering=1)

# Enable faulthandler output to the log file so C-level crashes are captured.
faulthandler.enable(file=_log_fh)


def _ts() -> str:
    return datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]


def _log(*args, **kwargs) -> None:
    line = ' '.join(str(a) for a in args)
    print(f'[{_ts()}] {line}', file=_log_fh, **kwargs)
    print(f'[{_ts()}] {line}', **kwargs)  # mirror to console
    _log_fh.flush()


_log('=' * 70)
_log('FRP-ShearPred  Crash Diagnostics Tool')
_log('=' * 70)
_log(f'Python  : {sys.version}')
_log(f'Platform: {sys.platform}')
_log(f'PID     : {os.getpid()}')
_log(f'Log     : {LOG_FILE}')
_log(f'Root    : {_PROJECT_ROOT}')

# ── 3. Ensure the project root is on sys.path ────────────────────────────────
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 4. Intercept sys.exit ────────────────────────────────────────────────────
_real_exit = sys.exit


def _patched_exit(code=0):
    _log(f'\n[!!!] sys.exit({code!r}) called — call-stack:')
    for line in traceback.format_stack():
        _log(line.rstrip())
    _log_fh.flush()
    _real_exit(code)


sys.exit = _patched_exit

# ── 5. Intercept sys.excepthook ─────────────────────────────────────────────
_orig_excepthook = sys.excepthook


def _diag_excepthook(exc_type, exc_val, exc_tb):
    _log('\n[!!!] Unhandled exception (sys.excepthook):')
    for line in traceback.format_exception(exc_type, exc_val, exc_tb):
        _log(line.rstrip())
    _log_fh.flush()
    _orig_excepthook(exc_type, exc_val, exc_tb)


sys.excepthook = _diag_excepthook

# ── 6. atexit: log on every kind of exit ────────────────────────────────────
@atexit.register
def _on_exit():
    _log('\n[===] Process exiting (atexit hook) — call-stack:')
    for line in traceback.format_stack():
        _log(line.rstrip())
    _log('=== Log end ===')
    _log_fh.flush()
    _log_fh.close()

# ── 7. Background thread monitor (snapshot every 5 s) ───────────────────────
def _thread_monitor():
    while True:
        time.sleep(5)
        names = [t.name for t in threading.enumerate()]
        _log(f'[threads] active={len(names)}: {names}')


_monitor = threading.Thread(
    target=_thread_monitor, daemon=True, name='DiagMonitor')
_monitor.start()

# ── 8. Patch TrainingThread ───────────────────────────────────────────────────
try:
    from tabs.train_threads import TrainingThread

    _orig_run = TrainingThread.run

    def _patched_run(self):
        _log(f'\n[TRAIN] TrainingThread.run() start  tid={threading.get_ident()}')
        _log(f'        models      : {list(self.models_cfg.keys())}')
        _log(f'        opt_strategy: {self.opt_strategy}  cv: {self.cv}')
        try:
            _orig_run(self)
        except BaseException as exc:
            _log(f'\n[TRAIN][!!!] run() raised {type(exc).__name__}: {exc}')
            _log(traceback.format_exc())
            raise
        finally:
            _log('[TRAIN] TrainingThread.run() finished')
            _log_fh.flush()

    TrainingThread.run = _patched_run
    _log('[patch] TrainingThread.run  — injected')

except Exception as exc:
    _log(f'[patch] TrainingThread  — FAILED: {exc}')
    _log(traceback.format_exc())

# ── 9. Patch TrainTab._on_done_inner ────────────────────────────────────────
try:
    from tabs.train_tab import TrainTab

    _orig_done_inner = TrainTab._on_done_inner

    def _patched_done_inner(self, results):
        _log(f'\n[GUI] _on_done_inner start  keys={list(results.keys())}')
        try:
            _orig_done_inner(self, results)
            _log('[GUI] _on_done_inner finished normally')
        except Exception as exc:
            _log(f'\n[GUI][!!!] _on_done_inner raised {type(exc).__name__}: {exc}')
            _log(traceback.format_exc())
            _log_fh.flush()
            raise

    TrainTab._on_done_inner = _patched_done_inner
    _log('[patch] TrainTab._on_done_inner — injected')

except Exception as exc:
    _log(f'[patch] TrainTab._on_done_inner — FAILED: {exc}')

# ── 10. Dependency version report ────────────────────────────────────────────
_log('\n[deps]')
for pkg in ['PyQt5', 'numpy', 'sklearn', 'scipy',
            'threadpoolctl', 'optuna', 'xgboost', 'lightgbm']:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '?')
        _log(f'  {pkg:<22} {ver}')
    except ImportError:
        _log(f'  {pkg:<22} NOT INSTALLED')

try:
    from threadpoolctl import threadpool_info
    _log('\n[threadpoolctl] BLAS back-ends:')
    for info in threadpool_info():
        _log(f'  {info}')
except Exception as exc:
    _log(f'[threadpoolctl] unavailable: {exc}')

# ── 11. Launch main application ──────────────────────────────────────────────
_log(f'\n{"=" * 70}')
_log('Launching main()  …')
_log(f'{"=" * 70}\n')
_log_fh.flush()

try:
    import main as _main_mod
    _main_mod.main()
except SystemExit as exc:
    _log(f'\n[!!!] Application exited via SystemExit({exc.code})')
    _log_fh.flush()
except Exception as exc:
    _log(f'\n[!!!] main() raised unhandled {type(exc).__name__}: {exc}')
    _log(traceback.format_exc())
    _log_fh.flush()
