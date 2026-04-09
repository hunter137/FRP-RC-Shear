"""
Unit tests for FRP-RC shear capacity design code formulas.

Reference values were independently verified by hand calculation
using the input set: d=300mm, b=200mm, f'c=40MPa, ρf=1.0%, Ef=60GPa.

Run:
    python -m pytest tests/test_formulas.py -v
"""

import sys, os
import pytest
import numpy as np

# Allow import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas import (
    split_tensile_strength,
    neutral_axis_depth_ratio,
    calc_gb50608,
    calc_aci440,
    calc_csa_s806,
    calc_bise1999,
    calc_jsce,
    calc_proposed,
)

# ── Standard test case ───────────────────────────────────────────────
# d=300mm, b=200mm, f'c=40MPa, ρf=1.0%, Ef=60GPa, a/d=3.0, GFRP
D, B, FC, RHO, EF, AD = 300, 200, 40.0, 1.0, 60.0, 3.0


class TestHelpers:
    """Tests for helper functions."""

    def test_split_tensile_strength_fc40(self):
        ft = split_tensile_strength(40.0)
        # 0.241 * 40^(2/3) = 0.241 * 11.696 = 2.8188
        assert abs(ft - 2.8188) < 0.01

    def test_split_tensile_strength_fc30(self):
        ft = split_tensile_strength(30.0)
        expected = 0.241 * 30.0 ** (2.0 / 3.0)
        assert abs(ft - expected) < 1e-6

    def test_split_tensile_strength_negative_raises(self):
        with pytest.raises(ValueError):
            split_tensile_strength(-5.0)

    def test_split_tensile_strength_zero_raises(self):
        with pytest.raises(ValueError):
            split_tensile_strength(0.0)

    def test_neutral_axis_depth_ratio(self):
        k = neutral_axis_depth_ratio(40.0, 0.01, 60000.0)
        # Ec=29915, nf=2.006, x=0.02006, k=0.1812
        assert abs(k - 0.1812) < 0.001


class TestGB50608:
    """GB 50608-2020 shear capacity formula."""

    def test_standard_case(self):
        V = calc_gb50608(D, B, FC, RHO, EF)
        assert abs(V - 26.36) < 0.1, f"Expected ~26.36, got {V:.2f}"

    def test_zero_reinforcement(self):
        # ρf=0 → k=0 → V=0
        V = calc_gb50608(D, B, FC, 0.0, EF)
        assert V == 0.0

    def test_higher_fc(self):
        V_low = calc_gb50608(D, B, 30.0, RHO, EF)
        V_high = calc_gb50608(D, B, 60.0, RHO, EF)
        assert V_high > V_low, "Higher f'c should give higher V"

    def test_returns_nonnegative(self):
        V = calc_gb50608(100, 100, 20.0, 0.1, 30.0)
        assert V >= 0.0


class TestACI440:
    """ACI 440.1R-15 shear capacity formula."""

    def test_standard_case(self):
        V = calc_aci440(D, B, FC, RHO, EF)
        assert abs(V - 27.51) < 0.1, f"Expected ~27.51, got {V:.2f}"

    def test_proportional_to_sqrt_fc(self):
        V1 = calc_aci440(D, B, 25.0, RHO, EF)
        V2 = calc_aci440(D, B, 100.0, RHO, EF)
        # V ∝ sqrt(fc) * k(fc), both increase with fc
        assert V2 > V1


class TestCSAS806:
    """CSA S806-12 shear capacity formula."""

    def test_standard_case(self):
        V = calc_csa_s806(D, B, FC, RHO, EF)
        assert abs(V - 33.49) < 0.1, f"Expected ~33.49, got {V:.2f}"

    def test_higher_ef_gives_higher_v(self):
        V_low = calc_csa_s806(D, B, FC, RHO, 40.0)
        V_high = calc_csa_s806(D, B, FC, RHO, 120.0)
        assert V_high > V_low


class TestBISE1999:
    """BISE (1999) shear capacity formula."""

    def test_standard_case(self):
        V = calc_bise1999(D, B, FC, RHO, EF)
        assert abs(V - 31.90) < 0.1, f"Expected ~31.90, got {V:.2f}"

    def test_size_effect(self):
        # Larger d → smaller beta_d → lower V/d ratio
        V_small = calc_bise1999(150, B, FC, RHO, EF)
        V_large = calc_bise1999(600, B, FC, RHO, EF)
        # V/d should decrease (size effect)
        assert V_small / 150 > V_large / 600


class TestJSCE:
    """JSCE (1997) shear capacity formula."""

    def test_standard_case(self):
        V = calc_jsce(D, B, FC, RHO, EF)
        assert abs(V - 28.56) < 0.1, f"Expected ~28.56, got {V:.2f}"

    def test_fvcd_cap(self):
        # f_vcd = min(0.20 * fc^(1/3), 0.72)
        # For fc=80: 0.20*80^(1/3) = 0.862 → capped at 0.72
        V_fc80 = calc_jsce(D, B, 80.0, RHO, EF)
        V_fc90 = calc_jsce(D, B, 90.0, RHO, EF)
        # Both capped → should be equal
        assert abs(V_fc80 - V_fc90) < 0.01


class TestProposed:
    """Proposed Bayesian-optimized empirical formula."""

    def test_standard_case(self):
        V = calc_proposed(D, B, FC, RHO, EF, AD)
        assert abs(V - 31.83) < 0.1, f"Expected ~31.83, got {V:.2f}"

    def test_arch_action_low_ad(self):
        # a/d < 2.72 → arch action boosts V
        V_low_ad = calc_proposed(D, B, FC, RHO, EF, 1.5)
        V_high_ad = calc_proposed(D, B, FC, RHO, EF, 4.0)
        assert V_low_ad > V_high_ad, "Low a/d should give higher V"

    def test_ad_above_transition(self):
        # a/d > 2.72 → lambda=1, no arch action boost
        V_3 = calc_proposed(D, B, FC, RHO, EF, 3.0)
        V_5 = calc_proposed(D, B, FC, RHO, EF, 5.0)
        assert abs(V_3 - V_5) < 0.01, "lambda=1 for both"

    def test_size_effect(self):
        V_small = calc_proposed(150, B, FC, RHO, EF, AD)
        V_large = calc_proposed(900, B, FC, RHO, EF, AD)
        assert V_small / 150 > V_large / 900, "Size effect should reduce V/d"

    def test_returns_nonnegative(self):
        V = calc_proposed(100, 100, 20, 0.1, 30, 5.0)
        assert V >= 0.0


class TestConsistencyAcrossCodes:
    """Cross-code sanity checks."""

    def test_all_codes_positive(self):
        for name, func in [
            ("GB50608", calc_gb50608),
            ("ACI440", calc_aci440),
            ("CSA", calc_csa_s806),
            ("BISE", calc_bise1999),
            ("JSCE", calc_jsce),
        ]:
            V = func(D, B, FC, RHO, EF)
            assert V > 0, f"{name} returned V={V}"

    def test_reasonable_range(self):
        """All codes should predict 10-100 kN for the standard case."""
        for name, func in [
            ("GB50608", calc_gb50608),
            ("ACI440", calc_aci440),
            ("CSA", calc_csa_s806),
            ("BISE", calc_bise1999),
            ("JSCE", calc_jsce),
        ]:
            V = func(D, B, FC, RHO, EF)
            assert 10 < V < 100, f"{name}: V={V:.1f} out of range"

    def test_proposed_in_code_range(self):
        """Proposed formula should be within 50% of code average."""
        code_vals = [f(D, B, FC, RHO, EF) for _, f in [
            ("GB50608", calc_gb50608), ("ACI440", calc_aci440),
            ("CSA", calc_csa_s806), ("BISE", calc_bise1999),
            ("JSCE", calc_jsce),
        ]]
        avg = np.mean(code_vals)
        V_prop = calc_proposed(D, B, FC, RHO, EF, AD)
        assert abs(V_prop - avg) / avg < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
