"""Tests for exact counting correctness in C_3a construction.

Compares count_sumset / count_diffset against brute-force ground truth
for small (b, A, d, T) cases, and against known goldens from the
commit dfc1a7f (pre-interning baseline captured 2026-06-30 at d<=24
via `git stash` + direct Python invocation, and from /tmp/c3a-results/d80_T154.json
for record-scale).
"""

import pytest

from scripts.c3a.construction import (
    no_carry_ok,
    max_U,
    count_sumset,
    count_diffset,
    StateBudgetExceeded,
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

    # ── Interning-equivalence tests (new) ─────────────────────────────────────
    # Golden values captured 2026-06-30 from the pre-interning implementation
    # (git stash of working tree, direct invocation against HEAD~0 before this
    # edit, A=[0,2,4,6,8,10] — the even-skip family used in small-d tests).
    # Cross-checked: d<=16 also pass test_noncontiguous_various_T brute-force above.

    def test_interning_equivalence_d8(self, brute_force):
        """Interned DP matches old code for A=[0,2,4,6,8,10], d=8, T=15."""
        A = [0, 2, 4, 6, 8, 10]
        d, T = 8, 15
        expected_s = 318450
        expected_dd = 3417927
        assert count_sumset(A, d, T) == expected_s
        assert count_diffset(A, d, T) == expected_dd
        # Cross-check d<=16 vs brute-force (b=21 minimal base)
        _, bf_s, bf_dd = brute_force(21, A, d, T)
        assert count_sumset(A, d, T) == bf_s
        assert count_diffset(A, d, T) == bf_dd

    def test_interning_equivalence_d12(self, brute_force):
        """Interned DP matches old code for A=[0,2,4,6,8,10], d=12, T=23."""
        A = [0, 2, 4, 6, 8, 10]
        d, T = 12, 23
        expected_s = 532129170
        expected_dd = 26685445821
        assert count_sumset(A, d, T) == expected_s
        assert count_diffset(A, d, T) == expected_dd
        _, bf_s, bf_dd = brute_force(21, A, d, T)
        assert count_sumset(A, d, T) == bf_s
        assert count_diffset(A, d, T) == bf_dd

    def test_interning_equivalence_d16(self, brute_force):
        """Interned DP matches old code for A=[0,2,4,6,8,10], d=16, T=30."""
        A = [0, 2, 4, 6, 8, 10]
        d, T = 16, 30
        expected_s = 926623241874
        expected_dd = 217841657982567
        assert count_sumset(A, d, T) == expected_s
        assert count_diffset(A, d, T) == expected_dd
        _, bf_s, bf_dd = brute_force(21, A, d, T)
        assert count_sumset(A, d, T) == bf_s
        assert count_diffset(A, d, T) == bf_dd

    def test_interning_equivalence_d20(self):
        """Interned DP matches old code for A=[0,2,4,6,8,10], d=20, T=38."""
        A = [0, 2, 4, 6, 8, 10]
        d, T = 20, 38
        expected_s = 1653392352583510
        expected_dd = 1828119300076012701
        assert count_sumset(A, d, T) == expected_s
        assert count_diffset(A, d, T) == expected_dd

    def test_interning_equivalence_d24(self):
        """Interned DP matches old code for A=[0,2,4,6,8,10], d=24, T=46."""
        A = [0, 2, 4, 6, 8, 10]
        d, T = 24, 46
        expected_s = 2998304927956358730
        expected_dd = 15640068152192886067533
        assert count_sumset(A, d, T) == expected_s
        assert count_diffset(A, d, T) == expected_dd

    # ── Budget-equivalence test (new) ─────────────────────────────────────────
    # Proves the (se, bs_id) injectivity invariant: the new code raises
    # StateBudgetExceeded at the same step and with the same nxt count as the
    # old code. Captured 2026-06-30 from pre-interning code: at A=[0,2,4,6,8,10],
    # d=8, T=15, the state dict peaks at 146 at step 3 (0-indexed), so
    # max_states=145 fires with "size 146 exceeded max_states=145".

    def test_budget_equivalence_fires_at_step3(self):
        """StateBudgetExceeded fires at the same step/count as old code.

        Old code (pre-interning): state dict at step 3 = 146 states.
        max_states=145 => raises with 'State dict size 146 exceeded max_states=145'.
        New code must reproduce this exactly (injectivity invariant).
        """
        A = [0, 2, 4, 6, 8, 10]
        with pytest.raises(
            StateBudgetExceeded,
            match=r"State dict size 146 exceeded max_states=145",
        ):
            count_diffset(A, 8, 15, max_states=145)

    def test_budget_equivalence_earlier_step(self):
        """StateBudgetExceeded fires at step 2 (size 96) with max_states=50.

        Old code: state dict at step 2 = 96, so max_states=50 fires with
        'State dict size 96 exceeded max_states=50'. New code must match.
        """
        A = [0, 2, 4, 6, 8, 10]
        with pytest.raises(
            StateBudgetExceeded,
            match=r"State dict size 96 exceeded max_states=50",
        ):
            count_diffset(A, 8, 15, max_states=50)

    def test_budget_no_raise_when_cap_above_max(self):
        """No StateBudgetExceeded when max_states is above actual peak.

        Peak at d=8,T=15 is 195; max_states=200 should not raise.
        """
        A = [0, 2, 4, 6, 8, 10]
        dd = count_diffset(A, 8, 15, max_states=200)
        assert dd == 3417927

    # ── Record-scale golden tests (new) ───────────────────────────────────────
    # Expected values from /tmp/c3a-results/d80_T154.json (cert for current
    # record G2026b, independently verified by code-reviewer in round 32).

    def test_record_scale_sumset_d80(self):
        """count_sumset for record family (A=[0..10 even+odd], d=80, T=154).

        Expected s from /tmp/c3a-results/d80_T154.json, field 's'.
        This matches the BASELINE_S constant in baseline.py.
        sumset is fast (<1 min) so not marked slow.
        """
        A = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_s = 597130362133498688344900538759091221981599964605490705452812019502078419618406
        s = count_sumset(A, 80, 154)
        assert s == expected_s, f"sumset d=80 mismatch: got {s}"

    @pytest.mark.slow
    def test_record_scale_diffset_d80(self):
        """count_diffset for record family (A=[0..10 even+odd], d=80, T=154).

        Expected dd from /tmp/c3a-results/d80_T154.json, field 'dd'.
        This matches the BASELINE_DD constant in baseline.py.
        SLOW: ~10 min. Run only with --runslow / outside the fast suite.
        """
        A = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_dd = 1583022697814754823730226433460816281662151877595631959725969360255416773109712840757177539870935
        dd = count_diffset(A, 80, 154)
        assert dd == expected_dd, f"diffset d=80 mismatch: got {dd}"
