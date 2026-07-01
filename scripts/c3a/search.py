"""Search for improved C_3a lower bounds within the benchmarked tractable regime.

Strategy:
- Fix b = 2*max(A)+1 (minimal no-carry base).
- Enumerate digit subsets A of {0..M} with 0 in A, M in A.
- For each A, scan (d, T) within the tractable regime.
- Rank candidates by Decimal-prec40 estimate (fast, no float).
- Exactly re-confirm the top-K via build_certificate.

Tractable regime (benchmarked): d <= 40, T <= 80, |A| <= 12.
Per-candidate state cap: 200_000 states in count_diffset (abort if exceeded).
"""

import argparse
import itertools
import time
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Optional

from scripts.c3a.construction import no_carry_ok, StateBudgetExceeded
from scripts.c3a.certificate import build_certificate, Certificate
from scripts.c3a.construction import max_U, count_sumset, count_diffset

# ── Constants ─────────────────────────────────────────────────────────────────

# Tractable regime limits (benchmarked)
MAX_D_FAST = 40
MAX_T_FAST = 80
MAX_M_FAST = 12  # max digit value (so |A| <= M+1 <= 13)

# Full baseline regime limits (opt-in --full)
MAX_D_FULL = 80
MAX_T_FULL = 150
MAX_M_FULL = 12

# Per-candidate state cap for count_diffset (abort if exceeded)
DEFAULT_MAX_STATES = 500_000

# Decimal precision for fast ranking (not for certified value)
RANK_PREC = 40

# Default results directory (under /tmp to avoid git tracking)
DEFAULT_RESULTS_DIR = Path("/tmp/c3a-results")


def _ln40(n: int) -> tuple[Decimal, Decimal]:
    """ln interval at prec=40 for ranking."""
    with localcontext() as ctx:
        ctx.prec = 40
        v = Decimal(n).ln()
        ulp = Decimal(10) ** (v.adjusted() - ctx.prec + 1)
        return v - 16 * ulp, v + 16 * ulp


def _enumerate_digit_sets(M: int) -> list[list[int]]:
    """Enumerate all subsets of {0..M} containing 0 and M.

    Returns sorted lists. Total: 2^(M-1) subsets.
    """
    middle = list(range(1, M))  # digits 1..M-1
    result: list[list[int]] = []
    for r in range(len(middle) + 1):
        for combo in itertools.combinations(middle, r):
            A = sorted([0] + list(combo) + [M])
            result.append(A)
    return result


def search(
    param_grid: Optional[dict] = None,
    top_k: int = 10,
    max_d: int = MAX_D_FAST,
    max_T: int = MAX_T_FAST,
    max_m: int = MAX_M_FAST,
    max_states: int = DEFAULT_MAX_STATES,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> list[Certificate]:
    """Search for best C_3a lower bounds in the tractable regime.

    Returns list of Certificate objects sorted by theta_lo descending.
    Writes certificate JSONs to results_dir.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # Collect (A, d, T, rank_estimate) candidates
    candidates: list[tuple[list[int], int, int, int, Decimal]] = []  # (A, b, d, T, est)

    print(f"Scanning digit sets for M in 3..{max_m}, d <= {max_d}, T <= {max_T} ...")

    for M in range(3, max_m + 1):
        b = 2 * M + 1
        digit_sets = _enumerate_digit_sets(M)
        print(f"  M={M}, b={b}: {len(digit_sets)} digit sets", flush=True)

        for A in digit_sets:
            if not no_carry_ok(b, A):
                continue

            # Quick scan: try a few representative (d, T) values for ranking
            best_for_A: list[tuple[int, int, Decimal]] = []
            for d in range(4, min(max_d, 20) + 1, 4):
                for T in range(d, min(max_T, 40) + 1, d // 2 + 1):
                    try:
                        s = count_sumset(A, d, T)
                        if s == 0:
                            continue
                        dd = count_diffset(A, d, T, max_states=max_states)
                        if dd == 0:
                            continue
                        mU = max_U(b, A, d, T)
                        q = 2 * mU + 1
                        with localcontext() as ctx:
                            ctx.prec = RANK_PREC
                            lo_d, hi_d = _ln40(dd)
                            lo_s, hi_s = _ln40(s)
                            lo_q, hi_q = _ln40(q)
                            theta_est = Decimal(1) + (lo_d - hi_s) / hi_q
                        best_for_A.append((d, T, theta_est))
                    except StateBudgetExceeded:
                        pass

            if best_for_A:
                best_for_A.sort(key=lambda x: x[2], reverse=True)
                # Keep only the top candidate per A for now
                d, T, est = best_for_A[0]
                candidates.append((A, b, d, T, est))

    if not candidates:
        print("No valid candidates found in the search regime.")
        return []

    # Sort all candidates by ranking estimate
    candidates.sort(key=lambda x: x[4], reverse=True)

    print(f"\nTop-{min(top_k, len(candidates))} candidates by Decimal-prec40 estimate:")
    for i, (A, b, d, T, est) in enumerate(candidates[:top_k]):
        print(f"  {i+1}. A={A}, b={b}, d={d}, T={T}, est={est:.10f}")

    # Exactly re-confirm the top-K via build_certificate
    certified: list[Certificate] = []
    print(f"\nExact re-confirmation of top {top_k} candidates ...")

    for i, (A, b, d, T, est) in enumerate(candidates[:top_k]):
        print(f"  Confirming {i+1}/{min(top_k, len(candidates))}: A={A}, d={d}, T={T} ...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            cert = build_certificate(b, A, d, T, max_states=max_states)
            elapsed = time.monotonic() - t0
            print(f"theta_lo={cert.theta_lo[:12]}... [{elapsed:.1f}s]")
            certified.append(cert)
            # Write certificate JSON
            fname = f"cert_M{max(A)}_d{d}_T{T}_A{'_'.join(map(str,A))}.json"
            (results_dir / fname).write_text(cert.to_json())
        except StateBudgetExceeded as e:
            elapsed = time.monotonic() - t0
            print(f"SKIPPED (state budget): {e}  [{elapsed:.1f}s]")
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"ERROR: {e}  [{elapsed:.1f}s]")

    # Sort by theta_lo descending (string comparison works for same-sign decimals
    # only if they have the same magnitude; use Decimal for correctness)
    certified.sort(key=lambda c: Decimal(c.theta_lo), reverse=True)
    return certified


def _run_full_search(
    max_d: int,
    max_T: int,
    max_m: int,
    top_k: int,
    max_states: int,
    results_dir: Path,
) -> list[Certificate]:
    """Full search with wider (d, T) grid including baseline-scale parameters."""
    print(f"FULL search: max_d={max_d}, max_T={max_T}, max_m={max_m}")
    return search(
        top_k=top_k,
        max_d=max_d,
        max_T=max_T,
        max_m=max_m,
        max_states=max_states,
        results_dir=results_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for improved C_3a lower bounds."
    )
    parser.add_argument(
        "--full",
        "--slow",
        action="store_true",
        help="Enable baseline-scale (d=80, T=150) search. Very slow.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to exactly re-confirm (default: 10).",
    )
    parser.add_argument(
        "--max-m",
        type=int,
        default=MAX_M_FAST,
        help=f"Maximum digit value M (default: {MAX_M_FAST}).",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=DEFAULT_MAX_STATES,
        help=f"Per-candidate state-dict cap for count_diffset (default: {DEFAULT_MAX_STATES}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for certificate JSON output (default: {DEFAULT_RESULTS_DIR}).",
    )
    args = parser.parse_args()

    if args.full:
        max_d, max_T = MAX_D_FULL, MAX_T_FULL
    else:
        max_d, max_T = MAX_D_FAST, MAX_T_FAST

    t0 = time.monotonic()
    certs = search(
        top_k=args.top_k,
        max_d=max_d,
        max_T=max_T,
        max_m=args.max_m,
        max_states=args.max_states,
        results_dir=args.results_dir,
    )
    elapsed = time.monotonic() - t0

    print(f"\n=== Search complete in {elapsed:.1f}s ===")

    if not certs:
        print("No certified results found.")
        return

    best = certs[0]
    print("\nBest certified result:")
    print(f"  A={best.A}, b={best.b}, d={best.d}, T={best.T}")
    print(f"  theta_lo = {best.theta_lo}")
    print(f"  theta_hi = {best.theta_hi}")
    print(f"  |U+U| = {best.s}")
    print(f"  |U-U| = {best.dd}")
    print(f"  q = {best.q}")

    baseline = Decimal("1.1740744")
    best_lo = Decimal(best.theta_lo)
    if best_lo > baseline:
        print(f"\n*** IMPROVEMENT: theta_lo={best_lo} > baseline={baseline} ***")
    else:
        print(f"\nBest theta_lo={best_lo:.10f} (baseline={baseline})")
        print("(Note: tractable regime typically cannot reach baseline scale d=80, T=150)")


if __name__ == "__main__":
    main()
