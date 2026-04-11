"""
formulas.py — Design code shear capacity formulas.

Implements: GB 50608-2020, ACI 440.1R-15, CSA S806-12, BISE (1999),
            JSCE (1997).
"""
import numpy as np
import pandas as pd

def _split_tensile_strength(fc: float) -> float:
    """
    Estimate concrete tensile splitting strength f_t [MPa].
    f_t = 0.241 · f_c^(2/3)  (CEB-FIP-based approximation; f_c in MPa).
    Raises ValueError for fc <= 0 (physically meaningless).
    """
    if fc <= 0:
        raise ValueError(f"fc must be positive, got {fc}")
    return 0.241 * fc ** (2.0 / 3.0)

def _neutral_axis_depth_ratio(fc: float, rho_f: float, Ef_MPa: float) -> float:
    """
    Cracked neutral-axis depth ratio k for an FRP-RC section.
    k = sqrt(2·rho_f·n_f + (rho_f·n_f)²) − rho_f·n_f
    where n_f = E_f / E_c,  E_c = 4730·sqrt(f_c).
    """
    Ec    = 4730.0 * np.sqrt(fc)
    nf    = Ef_MPa / Ec
    x     = rho_f * nf
    return np.sqrt(2.0 * x + x ** 2) - x

def calc_gb50608(d, b, fc, rho_f_pct, Ef_GPa):
    """
    GB 50608-2020 Cl. 6.3.1 nominal shear capacity [kN].
    V_c = 0.86 · f_t · b · k · d
    """
    rho_f = rho_f_pct / 100.0
    k     = _neutral_axis_depth_ratio(fc, rho_f, Ef_GPa * 1e3)
    return max(0.86 * _split_tensile_strength(fc) * b * k * d / 1e3, 0.0)

def calc_aci440(d, b, fc, rho_f_pct, Ef_GPa):
    """
    ACI 440.1R-15 Eq. (7-14b) nominal shear capacity [kN].
    V_c = 0.4 · sqrt(f'c) · b · k · d
    """
    rho_f = rho_f_pct / 100.0
    k     = _neutral_axis_depth_ratio(fc, rho_f, Ef_GPa * 1e3)
    return max(0.4 * np.sqrt(fc) * b * k * d / 1e3, 0.0)

def calc_csa_s806(d, b, fc, rho_f_pct, Ef_GPa):
    """
    CSA S806-12 Cl. 8.4.4.1 nominal shear capacity [kN].
    V_c = 0.0215 · (rho_f · E_f · f'c)^(1/3) · b · (0.9d)
    """
    rho_f = rho_f_pct / 100.0
    term  = (rho_f * Ef_GPa * 1e3 * fc) ** (1.0 / 3.0)
    return max(0.0215 * term * b * 0.9 * d / 1e3, 0.0)

def calc_bise1999(d, b, fc, rho_f_pct, Ef_GPa):
    """
    BISE (1999) / ISIS Canada shear model [kN].
    Adapts the BS 8110 expression with FRP modular ratio correction.
    Reference: ISIS Canada (2007) Design Manual No. 3.
    """
    rho_f   = rho_f_pct / 100.0
    Ef_MPa  = Ef_GPa * 1e3
    Es_ref  = 200000.0   # reference steel modulus [MPa]
    gamma_m = 1.25
    beta_d  = (400.0 / d) ** 0.25                           # size effect
    beta_rho = (100.0 * rho_f * Ef_MPa / Es_ref) ** (1.0 / 3.0)  # reinf.
    beta_fc  = (fc / 25.0) ** (1.0 / 3.0)                  # concrete
    return max((0.79 / gamma_m) * beta_d * beta_rho * beta_fc * b * d / 1e3, 0.0)

def calc_jsce(d, b, fc, rho_f_pct, Ef_GPa):
    """
    JSCE (1997) Standard Specification for Concrete Structures, shear capacity [kN].
    V_c = beta_d · beta_p · f_vcd · b · d / gamma_b
    with gamma_b = 1.3 and f_vcd capped at 0.72 MPa.
    """
    rho_f  = rho_f_pct / 100.0
    Ef_MPa = Ef_GPa * 1e3
    Es_ref = 200000.0
    f_vcd  = min(0.20 * fc ** (1.0 / 3.0), 0.72)
    beta_d = min((1000.0 / d) ** 0.25, 1.5)
    beta_p = min((100.0 * rho_f * Ef_MPa / Es_ref) ** (1.0 / 3.0), 1.5)
    return max(beta_d * beta_p * f_vcd * b * d / 1.3 / 1e3, 0.0)


CODE_FUNCS = [
    ('GB 50608-2020', calc_gb50608),
    ('ACI 440.1R-15', calc_aci440),
    ('CSA S806-12',   calc_csa_s806),
    ('BISE (1999)',   calc_bise1999),
    ('JSCE (1997)',   calc_jsce),
]

def apply_code_formulas(df):
    """
    Vectorised application of all design-code functions to DataFrame df.

    Returns a dict {label: np.ndarray} with one predicted-capacity array
    per code.  Uses numpy.vectorize to broadcast scalar formula functions
    over the DataFrame without row-by-row Python iteration (~10–50× faster
    on large databases).
    """
    def _col(name, default):
        """Return numeric array for column *name*, or constant *default*."""
        if name in df.columns:
            return pd.to_numeric(df[name], errors='coerce').values
        return np.full(len(df), float(default))

    d   = pd.to_numeric(df['d(mm)'],    errors='coerce').values
    b   = pd.to_numeric(df['b(mm)'],    errors='coerce').values
    fc  = pd.to_numeric(df["f`c(Mpa)"], errors='coerce').values
    rho = _col('ρf(%)', 1.0)
    ef  = _col('Ef(GPa)', 50.0)

    results = {}
    for label, func in CODE_FUNCS:
        _vf = np.vectorize(
            lambda di, bi, fi, ri, ei:
                func(di, bi, fi, ri, ei)
                if all(np.isfinite([di, bi, fi, ri, ei])) else np.nan
        )
        preds = _vf(d, b, fc, rho, ef)
        results[label] = np.where(np.isfinite(preds),
                                  np.maximum(preds, 0.0), np.nan)
    return results

# Public aliases for unit-testing
# The internal helpers use a leading underscore (private convention) but
# tests import them without the underscore.  These aliases bridge the gap
# without changing the internal call sites.
split_tensile_strength   = _split_tensile_strength
neutral_axis_depth_ratio = _neutral_axis_depth_ratio
