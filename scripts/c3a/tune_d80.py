"""Runner for d=80 T-sweep certificate jobs (one T per invocation).

Each invocation runs ONE (b=21, A=record, d=80, T) certificate and writes
the result to the results directory. Designed to be launched as a separate
background job per T value — never loop multiple T values inline.

Usage:
    python -m scripts.c3a.tune_d80 --T 152
    python -m scripts.c3a.tune_d80 --T 152 --results-dir /tmp/my-results
    python -m scripts.c3a.tune_d80 --smoke
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

B = 21
A = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
D = 80
RECORD = Decimal("1.1741713")

# Smoke-test parameters — tiny case that finishes in milliseconds
SMOKE_D = 8
SMOKE_T = 15


def run_one(T: int, results_dir: Path) -> Certificate:
    """Run one (b=B, A=A, d=D, T) certificate and write JSON to results_dir.

    Prints flushed heartbeat lines so the job is never silent for the full
    ~17-minute diffset duration. Calls count_sumset, count_diffset, and max_U
    directly (not via build_certificate) to interleave progress output between
    the fast sumset and the slow diffset.

    Raises ValueError if the no-carry condition fails (fail fast).
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if not no_carry_ok(B, A):
        raise ValueError(
            f"No-carry condition failed: b={B}, A={A}, max(A)={max(A)}, "
            f"requires b >= {2 * max(A) + 1}"
        )

    print(f"[tune] start T={T} d={D} (expect ~17min diffset)...", flush=True)
    t_start = time.monotonic()

    s = count_sumset(A, D, T)
    t_sumset = time.monotonic()
    elapsed_sumset = t_sumset - t_start
    print(
        f"[tune] T={T} sumset done s={s} ({elapsed_sumset:.1f}s); starting diffset...",
        flush=True,
    )

    dd = count_diffset(A, D, T)
    t_diffset = time.monotonic()
    elapsed_diffset = t_diffset - t_sumset

    mU = max_U(B, A, D, T)
    q = 2 * mU + 1

    theta_lo_dec, theta_hi_dec = certified_bound(s, dd, q)

    cert = Certificate(
        b=B,
        A=list(A),
        d=D,
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

    out_path = results_dir / f"d80_T{T}.json"
    out_path.write_text(cert.to_json())
    print(f"[tune] T={T} certificate written to {out_path}", flush=True)

    return cert


def _run_smoke(results_dir: Path) -> Certificate:
    """Smoke-test mode: run a tiny (d=SMOKE_D, T=SMOKE_T) case.

    Uses the same A and B but tiny d and T so it finishes in milliseconds.
    Verifies heartbeat lines are printed and JSON is written. Does NOT use
    the full d=80 parameters — this is a fast correctness check only.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if not no_carry_ok(B, A):
        raise ValueError(
            f"No-carry condition failed: b={B}, A={A}, max(A)={max(A)}, "
            f"requires b >= {2 * max(A) + 1}"
        )

    print(
        f"[tune] SMOKE start T={SMOKE_T} d={SMOKE_D} (tiny case, fast)...",
        flush=True,
    )
    t_start = time.monotonic()

    s = count_sumset(A, SMOKE_D, SMOKE_T)
    t_sumset = time.monotonic()
    elapsed_sumset = t_sumset - t_start
    print(
        f"[tune] SMOKE T={SMOKE_T} sumset done s={s} ({elapsed_sumset:.3f}s); "
        f"starting diffset...",
        flush=True,
    )

    dd = count_diffset(A, SMOKE_D, SMOKE_T)
    t_diffset = time.monotonic()
    elapsed_diffset = t_diffset - t_sumset

    mU = max_U(B, A, SMOKE_D, SMOKE_T)
    q = 2 * mU + 1

    theta_lo_dec, theta_hi_dec = certified_bound(s, dd, q)

    cert = Certificate(
        b=B,
        A=list(A),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ONE d=80 certificate for a given T. "
            "Launch each T as a separate background job."
        )
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

    if args.T is None:
        parser.error("--T is required unless --smoke is specified.")

    cert = run_one(args.T, args.results_dir)
    theta_lo_dec = Decimal(cert.theta_lo)
    beats_record = theta_lo_dec > RECORD

    # Produce claimed string via directed ROUND_FLOOR truncation — never round()
    claimed_str = theta_lo_dec.quantize(Decimal("1.0000000"), rounding=ROUND_FLOOR)
    assert_claim(theta_lo_dec, str(claimed_str))

    status = "BEATS RECORD" if beats_record else "below record"
    print(
        f"[tune] DONE T={args.T} theta_lo={cert.theta_lo[:20]} "
        f"claimed_floor={claimed_str} [{status}]"
    )


if __name__ == "__main__":
    main()
