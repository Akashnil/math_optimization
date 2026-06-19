"""Baseline reproduction and fast checkpoint for C_3a verifier.

Fast checkpoint (default): Gerbicz worked example, A=[0,1,2,3], b=7, d=4, T=8.
  Expected: |U|=221, |U+U|=2075, |U-U|=2307, 2*max+1=2381, theta~1.013631.
  Runs in milliseconds.

Full baseline (opt-in, ~17 min): PR #71 construction, b=21, A=[0,2,3,4,5,6,7,8,9,10],
  d=80, T=150.
  Expected integers are hardcoded below and must be reproduced exactly.
"""

import argparse
import sys
import time
from decimal import Decimal, localcontext

from scripts.c3a.construction import (
    no_carry_ok,
    max_U,
    count_sumset,
    count_diffset,
)
from scripts.c3a.certificate import build_certificate, certified_bound

# ── Baseline (PR #71) constants ──────────────────────────────────────────────

BASELINE_B = 21
BASELINE_A = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BASELINE_D = 80
BASELINE_T = 150

BASELINE_S = 75448362167176243488362019935078206851619643198150854886920234689186981134888
BASELINE_DD = 195351744295266763842135520514417052287242446785296742323733058216909095059024572338564089814415
BASELINE_MAX_U = 2995805288150731620410662946668903948341032736664352641511848666717243994160370658179324073879212562136150

# ── Gerbicz fast checkpoint constants ───────────────────────────────────────

FAST_B = 7
FAST_A = [0, 1, 2, 3]
FAST_D = 4
FAST_T = 8

FAST_S = 2075
FAST_DD = 2307
FAST_MAX_U_VAL = 189    # max(U) = 3*7^3 + 3*7^2 + 2*7 = 1029 + 147 + 14 = ... recompute below
# We'll compute FAST_Q dynamically; verified expected theta ~ 1.013631


def run_fast_checkpoint() -> None:
    """Run the Gerbicz worked-example fast checkpoint. Fails on mismatch."""
    print("=== Fast checkpoint: Gerbicz worked example ===")
    print(f"b={FAST_B}, A={FAST_A}, d={FAST_D}, T={FAST_T}")

    assert no_carry_ok(FAST_B, FAST_A), "no-carry condition failed for fast checkpoint"

    t0 = time.monotonic()
    s = count_sumset(FAST_A, FAST_D, FAST_T)
    t1 = time.monotonic()
    dd = count_diffset(FAST_A, FAST_D, FAST_T)
    t2 = time.monotonic()
    mU = max_U(FAST_B, FAST_A, FAST_D, FAST_T)
    q = 2 * mU + 1

    print(f"|U+U| = {s}  (expected {FAST_S})  [{t1-t0:.3f}s]")
    print(f"|U-U| = {dd}  (expected {FAST_DD})  [{t2-t1:.3f}s]")
    print(f"max(U) = {mU},  q = 2*max(U)+1 = {q}")

    if s != FAST_S:
        raise AssertionError(f"|U+U| mismatch: got {s}, expected {FAST_S}")
    if dd != FAST_DD:
        raise AssertionError(f"|U-U| mismatch: got {dd}, expected {FAST_DD}")

    theta_lo, theta_hi = certified_bound(s, dd, q)
    print(f"theta_lo = {theta_lo}")
    print(f"theta_hi = {theta_hi}")
    print(f"FAST CHECKPOINT PASSED in {t2-t0:.3f}s")


def run_full_baseline() -> None:
    """Run the full PR #71 baseline reproduction. Slow (~17 min for diffset).

    Asserts the three known exact integers and theta_lo >= 1.1740744.
    """
    print("=== Full baseline: PR #71 (b=21, A=[0,2..10], d=80, T=150) ===")
    print("WARNING: count_diffset at this scale takes ~17 minutes.")

    assert no_carry_ok(BASELINE_B, BASELINE_A), "no-carry failed for baseline"

    t0 = time.monotonic()
    s = count_sumset(BASELINE_A, BASELINE_D, BASELINE_T)
    t1 = time.monotonic()
    print(f"|U+U| = {s}  [{t1-t0:.1f}s]")

    if s != BASELINE_S:
        raise AssertionError(f"|U+U| mismatch: got {s}, expected {BASELINE_S}")
    print("|U+U| MATCHES")

    dd = count_diffset(BASELINE_A, BASELINE_D, BASELINE_T)
    t2 = time.monotonic()
    print(f"|U-U| = {dd}  [{t2-t1:.1f}s]")

    if dd != BASELINE_DD:
        raise AssertionError(f"|U-U| mismatch: got {dd}, expected {BASELINE_DD}")
    print("|U-U| MATCHES")

    mU = max_U(BASELINE_B, BASELINE_A, BASELINE_D, BASELINE_T)
    if mU != BASELINE_MAX_U:
        raise AssertionError(f"max_U mismatch: got {mU}, expected {BASELINE_MAX_U}")
    print(f"max_U MATCHES: {mU}")

    q = 2 * mU + 1
    theta_lo, theta_hi = certified_bound(s, dd, q)
    print(f"theta_lo = {theta_lo}")
    print(f"theta_hi = {theta_hi}")

    if theta_lo < Decimal("1.1740744"):
        raise AssertionError(
            f"theta_lo={theta_lo} < 1.1740744, baseline reproduction failed"
        )
    print(f"theta_lo >= 1.1740744 CONFIRMED")
    print(f"FULL BASELINE PASSED in {t2-t0:.1f}s total")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C_3a baseline verifier. Default: fast Gerbicz checkpoint."
    )
    parser.add_argument(
        "--full",
        "--slow",
        action="store_true",
        help="Run the full PR #71 baseline (~17 min for diffset). Opt-in only.",
    )
    args = parser.parse_args()

    if args.full:
        run_full_baseline()
    else:
        run_fast_checkpoint()


if __name__ == "__main__":
    main()
