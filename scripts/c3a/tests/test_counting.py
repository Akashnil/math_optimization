"""Tests for exact counting correctness in C_3a construction.

Compares count_sumset / count_diffset against brute-force ground truth
for small (b, A, d, T) cases.
"""

import pytest

from scripts.c3a.construction import (
    no_carry_ok,
    max_U,
    count_sumset,
    count_diffset,
)
from scripts.c3a.certificate import build_certificate


class TestCounting:
    """Brute-force vs DP counting correctness for small cases."""

    def test_no_carry_ok_valid_contiguous(self):
        """Basic valid case: contiguous A, b=2*max+1."""
        assert no_carry_ok(7, [0, 1, 2, 3]) is True

    def test_no_carry_ok_valid_noncontiguous(self):
        """Non-contiguous A (missing digit 1) with minimal b."""
        assert no_carry_ok(11, [0, 2, 3, 4, 5]) is True

    def test_no_carry_ok_fails_b_too_small(self):
        """b < 2*max(A)+1 must fail."""
        assert no_carry_ok(9, [0, 2, 3, 4, 5]) is False   # need b >= 11
        assert no_carry_ok(10, [0, 2, 3, 4, 5]) is False  # need b >= 11
        assert no_carry_ok(11, [0, 2, 3, 4, 5]) is True   # exactly minimal

    def test_no_carry_ok_fails_zero_missing(self):
        """0 must be in A."""
        assert no_carry_ok(7, [1, 2, 3]) is False

    def test_no_carry_ok_fails_digit_out_of_range(self):
        """Digits must be in {0..b-1}."""
        assert no_carry_ok(7, [0, 1, 7]) is False

    def test_build_certificate_raises_on_no_carry_fail(self):
        """build_certificate must raise ValueError if no-carry fails."""
        with pytest.raises(ValueError, match="No-carry"):
            build_certificate(9, [0, 2, 3, 4, 5], 3, 6)

    def test_contiguous_small(self, brute_force):
        """Contiguous A=[0,1,2,3], b=7, d=3, T=5: DP matches brute force."""
        b, A, d, T = 7, [0, 1, 2, 3], 3, 5
        n, bf_s, bf_dd = brute_force(b, A, d, T)
        dp_s = count_sumset(A, d, T)
        dp_dd = count_diffset(A, d, T)
        assert dp_s == bf_s, f"sumset mismatch: DP={dp_s}, brute={bf_s}"
        assert dp_dd == bf_dd, f"diffset mismatch: DP={dp_dd}, brute={bf_dd}"

    def test_contiguous_gerbicz_checkpoint(self, brute_force):
        """Gerbicz worked example: A=[0,1,2,3], b=7, d=4, T=8.
        Expected: |U+U|=2075, |U-U|=2307.
        """
        b, A, d, T = 7, [0, 1, 2, 3], 4, 8
        n, bf_s, bf_dd = brute_force(b, A, d, T)
        dp_s = count_sumset(A, d, T)
        dp_dd = count_diffset(A, d, T)
        assert dp_s == 2075, f"sumset: got {dp_s}, expected 2075"
        assert dp_dd == 2307, f"diffset: got {dp_dd}, expected 2307"
        assert dp_s == bf_s
        assert dp_dd == bf_dd

    def test_noncontiguous_spec_example(self, brute_force):
        """Non-contiguous A=[0,2,3,4,5], b=11, d=4, T=9.
        Spec notes that min/max-interval approximation gives 3066 (wrong).
        The DP must give the correct value matching brute force.
        """
        b, A, d, T = 11, [0, 2, 3, 4, 5], 4, 9
        n, bf_s, bf_dd = brute_force(b, A, d, T)
        dp_s = count_sumset(A, d, T)
        dp_dd = count_diffset(A, d, T)
        assert dp_s == bf_s, f"sumset: DP={dp_s}, brute={bf_s}"
        assert dp_dd == bf_dd, f"diffset: DP={dp_dd}, brute={bf_dd}"
        # Specifically check the sumset is NOT the wrong 3066 (interval approx)
        # (The spec says the correct value is 3026, not 3066)
        assert dp_s == 3026, f"Expected 3026 (not 3066 from interval approx), got {dp_s}"

    def test_noncontiguous_various_T(self, brute_force):
        """Non-contiguous A=[0,2,3,4,5], b=11, various d and T."""
        b, A = 11, [0, 2, 3, 4, 5]
        for d, T in [(2, 4), (3, 6), (3, 8)]:
            n, bf_s, bf_dd = brute_force(b, A, d, T)
            dp_s = count_sumset(A, d, T)
            dp_dd = count_diffset(A, d, T)
            assert dp_s == bf_s, f"sumset d={d},T={T}: DP={dp_s}, brute={bf_s}"
            assert dp_dd == bf_dd, f"diffset d={d},T={T}: DP={dp_dd}, brute={bf_dd}"

    def test_max_U_formula(self):
        """max_U greedy formula: spot-check against known values."""
        # b=7, A=[0,1,2,3], d=4, T=8: max digits all 3 in top 2 positions + 2 in pos 1
        # rem=8: pos3=3(rem=5), pos2=3(rem=2), pos1=2(rem=0), pos0=0
        # max = 3*343 + 3*49 + 2*7 + 0 = 1029 + 147 + 14 = 1190
        mU = max_U(7, [0, 1, 2, 3], 4, 8)
        assert mU == 1190, f"max_U mismatch: {mU}"

    def test_T_cap_sumset(self, brute_force):
        """T < d*max(A) means not all digit combinations are allowed."""
        b, A, d, T = 7, [0, 1, 2, 3], 3, 4  # d*max(A)=9 but T=4
        n, bf_s, bf_dd = brute_force(b, A, d, T)
        dp_s = count_sumset(A, d, T)
        dp_dd = count_diffset(A, d, T)
        assert dp_s == bf_s
        assert dp_dd == bf_dd
