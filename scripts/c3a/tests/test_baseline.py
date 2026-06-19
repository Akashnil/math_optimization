"""Slow baseline reproduction test for PR #71 exact integers.

COST: count_diffset at (d=80, T=150) takes approximately 17 minutes.
This test is gated behind @pytest.mark.slow and is skipped by default.
Run with: pytest scripts/c3a/tests/test_baseline.py --runslow

The exact integers reproduced here are the authoritative certificate for
the G2026 record: theta_lo >= 1.1740744.
"""

import pytest
from decimal import Decimal

from scripts.c3a.construction import (
    no_carry_ok,
    max_U,
    count_sumset,
    count_diffset,
)
from scripts.c3a.certificate import certified_bound, build_certificate
from scripts.c3a.baseline import (
    BASELINE_B,
    BASELINE_A,
    BASELINE_D,
    BASELINE_T,
    BASELINE_S,
    BASELINE_DD,
    BASELINE_MAX_U,
)

# Register the slow mark (conftest.py in repo root may also do this)
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")


@pytest.mark.slow
class TestBaselineReproduction:
    """Reproduce the full PR #71 exact integers.

    EXPECTED RUNTIME: ~17 minutes (count_diffset dominates).
    Skip unless --runslow is passed.
    """

    def test_baseline_sumset(self):
        """count_sumset reproduces the PR #71 |U+U| integer exactly.

        This is fast (~20s).
        """
        assert no_carry_ok(BASELINE_B, BASELINE_A), "no-carry condition failed"
        s = count_sumset(BASELINE_A, BASELINE_D, BASELINE_T)
        assert s == BASELINE_S, (
            f"sumset mismatch:\n  got:      {s}\n  expected: {BASELINE_S}"
        )

    def test_baseline_max_U(self):
        """max_U reproduces the PR #71 max(U) integer exactly.

        This is instant (closed-form calculation).
        """
        mU = max_U(BASELINE_B, BASELINE_A, BASELINE_D, BASELINE_T)
        assert mU == BASELINE_MAX_U, (
            f"max_U mismatch:\n  got:      {mU}\n  expected: {BASELINE_MAX_U}"
        )

    def test_baseline_diffset(self):
        """count_diffset reproduces the PR #71 |U-U| integer exactly.

        WARNING: This takes approximately 17 minutes.
        """
        dd = count_diffset(BASELINE_A, BASELINE_D, BASELINE_T)
        assert dd == BASELINE_DD, (
            f"diffset mismatch:\n  got:      {dd}\n  expected: {BASELINE_DD}"
        )

    def test_baseline_theta_lo(self):
        """Full build_certificate for baseline gives theta_lo >= 1.1740744.

        WARNING: This takes approximately 17 minutes (diffset is the bottleneck).
        """
        mU = max_U(BASELINE_B, BASELINE_A, BASELINE_D, BASELINE_T)
        q = 2 * mU + 1
        theta_lo, theta_hi = certified_bound(BASELINE_S, BASELINE_DD, q)
        assert theta_lo >= Decimal("1.1740744"), (
            f"theta_lo={theta_lo} < 1.1740744"
        )
