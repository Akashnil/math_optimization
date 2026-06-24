"""Generalized single-(b, A, d, T) certificate runner for C_3a lower bounds.

One certificate per invocation. Designed for background-job parallel execution
(one T per process). Reuses count_sumset, count_diffset, max_U, certified_bound,
and assert_claim verbatim — no new counting or certificate math.

Usage:
    python -m scripts.c3a.tune --d 90 --T 172
    python -m scripts.c3a.tune --d 90 --T 172 --results-dir /tmp/c3a-results
    python -m scripts.c3a.tune --smoke
    python -m scripts.c3a.tune --b 21 --A 0,2,3,4,5,6,7,8,9,10 --d 90 --T 172
"""

import argparse
import time
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

from scripts.c3a.certificate import Certificate, assert_claim, certified_bound
from scripts.c3a.construction import (
    count_diffset,
    count_sumset,
    max_U,
    no_carry_ok,
)

# ── Module constants ───────────────────────────────────────────────────────────

# Record family defaults (b=21, A=[0,2..10])
DEFAULT_B = 21
DEFAULT_A = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Current verified record — compare target for "BEATS RECORD" marker
RECORD = Decimal("1.1741713")

# Smoke-test parameters — tiny case that finishes in milliseconds
SMOKE_B = DEFAULT_B
SMOKE_A = DEFAULT_A
SMOKE_D = 8
SMOKE_T = 15


def _output_filename(b: int, A: list[int], d: int, T: int) -> str:
    """Return the JSON output filename for a given (b, A, d, T) tuple.

    Uses d{d}_T{T}.json for the record family (b=21, A=[0,2..10]) to keep
    filenames compatible with existing /tmp/c3a-results/d80_T*.json files.
    Uses b{b}_d{d}_T{T}.json for non-record families to avoid collisions.
    """
    if b == DEFAULT_B and A == DEFAULT_A:
        return f"d{d}_T{T}.json"
    return f"b{b}_d{d}_T{T}.json"


def run_one(
    b: int,
    A: list[int],
    d: int,
    T: int,
    results_dir: Path,
) -> Certificate:
    """Run one (b, A, d, T) certificate and write JSON to results_dir.

    Prints flushed heartbeat lines so the job is never silent for the full
    diffset duration. Calls count_sumset, count_diffset, and max_U directly
    (not via build_certificate) to interleave progress output between the fast
    sumset and the slow diffset.

    Raises ValueError if the no-carry condition fails (fail fast — before any
    counting runs).
    """
    if not no_carry_ok(b, A):
        raise ValueError(
            f"No-carry condition failed: b={b}, A={A}, max(A)={max(A)}, "
            f"requires b >= {2 * max(A) + 1}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[tune] start b={b} A={A} d={d} T={T} (expect ~17min+ diffset)...",
        flush=True,
    )
    t_start = time.monotonic()

    s = count_sumset(A, d, T)
    t_sumset = time.monotonic()
    elapsed_sumset = t_sumset - t_start
    print(
        f"[tune] T={T} sumset done s={s} ({elapsed_sumset:.1f}s); starting diffset...",
        flush=True,
    )

    dd = count_diffset(A, d, T)
    t_diffset = time.monotonic()
    elapsed_diffset = t_diffset - t_sumset

    mU = max_U(b, A, d, T)
    q = 2 * mU + 1

    theta_lo_dec, theta_hi_dec = certified_bound(s, dd, q)

    cert = Certificate(
        b=b,
        A=list(A),
        d=d,
        T=T,
        s=s,
        dd=dd,
        q=q,
        maxU=mU,
        theta_lo=str(theta_lo_dec),
        theta_hi=str(theta_hi_dec),
        no_carry=True,
    )

    beats_record = theta_lo_dec > RECORD
    record_marker = " *** BEATS RECORD ***" if beats_record else ""
    print(
        f"[tune] T={T} diffset done dd={dd} ({elapsed_diffset:.1f}s); "
        f"theta_lo={str(theta_lo_dec)[:16]}{record_marker}",
        flush=True,
    )

    filename = _output_filename(b, A, d, T)
    out_path = results_dir / filename
    out_path.write_text(cert.to_json())
    print(f"[tune] T={T} certificate written to {out_path}", flush=True)

    return cert


def _run_smoke(results_dir: Path) -> Certificate:
    """Smoke-test mode: run a tiny (d=SMOKE_D, T=SMOKE_T) case.

    Uses the default A and B but tiny d and T so it finishes in milliseconds.
    Verifies heartbeat lines are printed and JSON is written. Does NOT use
    full-scale parameters — this is a fast correctness check only.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if not no_carry_ok(SMOKE_B, SMOKE_A):
        raise ValueError(
            f"No-carry condition failed for smoke: b={SMOKE_B}, A={SMOKE_A}"
        )

    print(
        f"[tune] SMOKE start b={SMOKE_B} A={SMOKE_A} d={SMOKE_D} T={SMOKE_T} "
        f"(tiny case, fast)...",
        flush=True,
    )
    t_start = time.monotonic()

    s = count_sumset(SMOKE_A, SMOKE_D, SMOKE_T)
    t_sumset = time.monotonic()
    elapsed_sumset = t_sumset - t_start
    print(
        f"[tune] SMOKE T={SMOKE_T} sumset done s={s} ({elapsed_sumset:.3f}s); "
        f"starting diffset...",
        flush=True,
    )

    dd = count_diffset(SMOKE_A, SMOKE_D, SMOKE_T)
    t_diffset = time.monotonic()
    elapsed_diffset = t_diffset - t_sumset

    mU = max_U(SMOKE_B, SMOKE_A, SMOKE_D, SMOKE_T)
    q = 2 * mU + 1

    theta_lo_dec, theta_hi_dec = certified_bound(s, dd, q)

    cert = Certificate(
        b=SMOKE_B,
        A=list(SMOKE_A),
        d=SMOKE_D,
        T=SMOKE_T,
        s=s,
        dd=dd,
        q=q,
        maxU=mU,
        theta_lo=str(theta_lo_dec),
        theta_hi=str(theta_hi_dec),
        no_carry=True,
    )

    beats_record = theta_lo_dec > RECORD
    record_marker = " *** BEATS RECORD ***" if beats_record else ""
    print(
        f"[tune] SMOKE T={SMOKE_T} diffset done dd={dd} ({elapsed_diffset:.3f}s); "
        f"theta_lo={str(theta_lo_dec)[:16]}{record_marker}",
        flush=True,
    )

    out_path = results_dir / f"smoke_d{SMOKE_D}_T{SMOKE_T}.json"
    out_path.write_text(cert.to_json())
    print(f"[tune] SMOKE certificate written to {out_path}", flush=True)
    print("[tune] SMOKE PASSED — heartbeats and JSON write confirmed.", flush=True)

    return cert


def _parse_A(s: str) -> list[int]:
    """Parse a comma-separated string of integers into a sorted list.

    Raises ValueError if any token is not a valid integer.
    """
    try:
        return [int(tok.strip()) for tok in s.split(",")]
    except ValueError as exc:
        raise ValueError(f"--A must be comma-separated integers, got: {s!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ONE (b, A, d, T) certificate. "
            "Launch each T as a separate background job."
        )
    )
    parser.add_argument(
        "--b",
        type=int,
        default=DEFAULT_B,
        help=f"Base (default: {DEFAULT_B}).",
    )
    parser.add_argument(
        "--A",
        type=str,
        default=",".join(str(a) for a in DEFAULT_A),
        help=(
            f"Digit set as comma-separated ints "
            f"(default: {','.join(str(a) for a in DEFAULT_A)})."
        ),
    )
    parser.add_argument(
        "--d",
        type=int,
        default=None,
        help="Dimension (required unless --smoke).",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=None,
        help="T value to certify (required unless --smoke).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/tmp/c3a-results"),
        help="Directory for certificate JSON output (default: /tmp/c3a-results).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Smoke-test mode: run tiny d={SMOKE_D}, T={SMOKE_T} case to verify "
            "heartbeats and JSON write. Fast (<1s)."
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        cert = _run_smoke(args.results_dir)
        theta_lo_dec = Decimal(cert.theta_lo)
        claimed_str = theta_lo_dec.quantize(Decimal("1.0000000"), rounding=ROUND_FLOOR)
        print(
            f"[tune] SMOKE summary: d={SMOKE_D} T={SMOKE_T} "
            f"theta_lo={cert.theta_lo[:20]} "
            f"claimed_floor={claimed_str}"
        )
        return

    if args.d is None:
        parser.error("--d is required unless --smoke is specified.")
    if args.T is None:
        parser.error("--T is required unless --smoke is specified.")

    A = _parse_A(args.A)
    cert = run_one(args.b, A, args.d, args.T, args.results_dir)
    theta_lo_dec = Decimal(cert.theta_lo)
    beats_record = theta_lo_dec > RECORD

    # Produce claimed string via directed ROUND_FLOOR truncation — never round()
    claimed_str = theta_lo_dec.quantize(Decimal("1.0000000"), rounding=ROUND_FLOOR)
    assert_claim(theta_lo_dec, str(claimed_str))

    status = "BEATS RECORD" if beats_record else "below record"
    print(
        f"[tune] DONE b={args.b} d={args.d} T={args.T} "
        f"theta_lo={cert.theta_lo[:20]} "
        f"claimed_floor={claimed_str} [{status}]"
    )


if __name__ == "__main__":
    main()
