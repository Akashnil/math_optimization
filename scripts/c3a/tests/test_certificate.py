"""Tests for the rigorous decimal-interval certificate.

Includes the REQUIRED cross-precision self-check: the prec=200 interval
must contain the prec=400 midpoint estimate.
"""

import pytest
from decimal import Decimal, localcontext

from scripts.c3a.certificate import (
    certified_bound,
    prec400_estimate,
    build_certificate,
    Certificate,
)
from scripts.c3a.construction import max_U

# ── Baseline integers (PR #71) ────────────────────────────────────────────────

BASELINE_S = 75448362167176243488362019935078206851619643198150854886920234689186981134888
BASELINE_DD = 195351744295266763842135520514417052287242446785296742323733058216909095059024572338564089814415
BASELINE_MAX_U = 2995805288150731620410662946668903948341032736664352641511848666717243994160370658179324073879212562136150
BASELINE_Q = 2 * BASELINE_MAX_U + 1

# ── Small case integers (Gerbicz worked example) ──────────────────────────────

SMALL_B = 7
SMALL_A = [0, 1, 2, 3]
SMALL_D = 4
SMALL_T = 8
SMALL_S = 2075
SMALL_DD = 2307


class TestCertificate:
    """Certificate correctness, cross-precision, and ordering tests."""

    def test_theta_lo_le_theta_hi_baseline(self):
        """theta_lo <= theta_hi for baseline integers."""
        lo, hi = certified_bound(BASELINE_S, BASELINE_DD, BASELINE_Q)
        assert lo <= hi, f"theta_lo={lo} > theta_hi={hi}"

    def test_baseline_theta_lo_ge_1174(self):
        """Baseline theta_lo >= 1.1740744 (the claimed PR #71 bound)."""
        lo, hi = certified_bound(BASELINE_S, BASELINE_DD, BASELINE_Q)
        assert lo >= Decimal("1.1740744"), (
            f"theta_lo={lo} < 1.1740744 for baseline"
        )

    def test_cross_precision_baseline(self):
        """Cross-precision: prec=200 interval must contain prec=400 estimate (baseline)."""
        lo200, hi200 = certified_bound(BASELINE_S, BASELINE_DD, BASELINE_Q, prec=200)
        mid400 = prec400_estimate(BASELINE_S, BASELINE_DD, BASELINE_Q)
        assert lo200 <= mid400 <= hi200, (
            f"Cross-precision test FAILED: prec=200 interval [{lo200}, {hi200}] "
            f"does not contain prec=400 estimate {mid400}"
        )

    def test_cross_precision_small(self):
        """Cross-precision: prec=200 interval must contain prec=400 estimate (small case)."""
        mU = max_U(SMALL_B, SMALL_A, SMALL_D, SMALL_T)
        q = 2 * mU + 1
        lo200, hi200 = certified_bound(SMALL_S, SMALL_DD, q, prec=200)
        mid400 = prec400_estimate(SMALL_S, SMALL_DD, q)
        assert lo200 <= mid400 <= hi200, (
            f"Cross-precision test FAILED for small case: "
            f"interval [{lo200}, {hi200}], estimate {mid400}"
        )

    def test_interval_width_is_small_baseline(self):
        """Interval width should be very small (~1e-198) for baseline at prec=200."""
        lo, hi = certified_bound(BASELINE_S, BASELINE_DD, BASELINE_Q, prec=200)
        width = hi - lo
        assert width < Decimal("1e-150"), f"Interval too wide: {width}"

    def test_small_case_exact_values(self):
        """Small case: certified_bound with known s, dd, q gives sensible theta."""
        mU = max_U(SMALL_B, SMALL_A, SMALL_D, SMALL_T)
        q = 2 * mU + 1
        lo, hi = certified_bound(SMALL_S, SMALL_DD, q)
        assert lo <= hi
        assert lo > Decimal("1"), "theta must be > 1 for a valid construction"
        # Gerbicz example: theta ~ 1.013631
        assert Decimal("1.01") < lo < Decimal("1.02"), f"Expected ~1.013, got lo={lo}"

    def test_build_certificate_small(self):
        """build_certificate end-to-end for the small Gerbicz case."""
        cert = build_certificate(SMALL_B, SMALL_A, SMALL_D, SMALL_T)
        assert cert.s == SMALL_S
        assert cert.dd == SMALL_DD
        assert cert.no_carry is True
        lo = Decimal(cert.theta_lo)
        hi = Decimal(cert.theta_hi)
        assert lo <= hi
        assert lo > Decimal("1")

    def test_certificate_json_roundtrip(self):
        """Certificate serializes and deserializes to JSON correctly."""
        cert = build_certificate(SMALL_B, SMALL_A, SMALL_D, SMALL_T)
        json_str = cert.to_json()
        cert2 = Certificate.from_json(json_str)
        assert cert.s == cert2.s
        assert cert.dd == cert2.dd
        assert cert.q == cert2.q
        assert cert.maxU == cert2.maxU
        assert cert.theta_lo == cert2.theta_lo
        assert cert.theta_hi == cert2.theta_hi
        assert cert.no_carry == cert2.no_carry
        assert cert.A == cert2.A

    def test_cross_precision_noncontiguous(self):
        """Cross-precision test for a non-contiguous digit set."""
        # A=[0,2,3,4,5], b=11, d=4, T=9
        from scripts.c3a.construction import count_sumset, count_diffset
        b, A, d, T = 11, [0, 2, 3, 4, 5], 4, 9
        s = count_sumset(A, d, T)
        dd = count_diffset(A, d, T)
        mU = max_U(b, A, d, T)
        q = 2 * mU + 1
        lo200, hi200 = certified_bound(s, dd, q, prec=200)
        mid400 = prec400_estimate(s, dd, q)
        assert lo200 <= mid400 <= hi200, (
            f"Cross-precision FAILED for noncontiguous case: "
            f"[{lo200}, {hi200}] does not contain {mid400}"
        )
