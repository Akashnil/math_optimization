"""Rigorous decimal-interval certificate for C_3a lower bounds.

Given exact integers s=|U+U|, dd=|U-U|, q=2*max(U)+1, certifies theta >= V
using directed-rounding decimal arithmetic (no float in the certified value).
"""

import json
from dataclasses import dataclass
from decimal import Decimal, getcontext, localcontext, ROUND_FLOOR, ROUND_CEILING
from typing import Optional

from scripts.c3a.construction import (
    no_carry_ok,
    max_U,
    count_sumset,
    count_diffset,
)

# Default precision for the certificate (generous guard digits)
_DEFAULT_PREC = 200
_DEFAULT_PAD = 16


def ln_interval(n: int, pad: int = _DEFAULT_PAD) -> tuple[Decimal, Decimal]:
    """Return (lo, hi) such that true ln(n) is in [lo, hi].

    Uses CPython's decimal.ln() which is correctly rounded to context precision
    (differs from true ln by at most 1 ulp). We pad by `pad` ulps as a
    generous guard.
    """
    v = Decimal(n).ln()
    # ulp at the adjusted exponent of v
    ulp = Decimal(10) ** (v.adjusted() - getcontext().prec + 1)
    return v - pad * ulp, v + pad * ulp


def assert_claim(theta_lo: Decimal, claimed_str: str) -> None:
    """Assert that Decimal(claimed_str) <= theta_lo.

    Guards all doc edits: the claimed string must be a truncation of theta_lo
    toward zero (produced via ROUND_FLOOR quantize), never a rounding up.
    Raises AssertionError if the claimed value exceeds what the certificate proves.
    """
    if Decimal(claimed_str) > theta_lo:
        raise AssertionError(
            f"Claimed value {claimed_str} exceeds certified theta_lo={theta_lo}. "
            "The claimed string must be produced by theta_lo.quantize(..., rounding=ROUND_FLOOR)."
        )


def certified_bound(
    s: int,
    dd: int,
    q: int,
    prec: int = _DEFAULT_PREC,
    pad: int = _DEFAULT_PAD,
) -> tuple[Decimal, Decimal]:
    """Return (theta_lo, theta_hi) as a rigorous decimal interval.

    theta_lo is a certified lower bound on theta = 1 + (ln(dd)-ln(s)) / ln(q).
    theta_hi is a certified upper bound (for ranking/reporting).
    No math.log or float is used anywhere in this function.

    Uses ROUND_FLOOR for theta_lo (directed rounding toward -inf, valid because
    numerator (ld_lo - ls_hi) and denominator lq_hi are both positive — asserted
    below) and ROUND_CEILING for theta_hi (directed rounding toward +inf).
    """
    with localcontext() as ctx:
        ctx.prec = prec
        ld_lo, ld_hi = ln_interval(dd, pad)
        ls_lo, ls_hi = ln_interval(s, pad)
        lq_lo, lq_hi = ln_interval(q, pad)

        # Precondition: numerator (ld_lo - ls_hi) must be positive for the
        # ROUND_FLOOR directed-rounding argument to be valid. If this were
        # non-positive, ROUND_FLOOR of the quotient would not underestimate theta.
        assert ld_lo > ls_hi, (
            f"Positive-numerator precondition violated: ld_lo={ld_lo} <= ls_hi={ls_hi}. "
            "theta = 1 + (ln(dd)-ln(s))/ln(q) <= 1, so this construction cannot certify "
            "a meaningful lower bound above 1."
        )

        # Rigorous lower bound: numerator minimized (ld_lo - ls_hi), denominator maximized
        # (lq_hi), then rounded DOWN (ROUND_FLOOR) so result is a genuine underestimate.
        ctx.rounding = ROUND_FLOOR
        theta_lo = Decimal(1) + (ld_lo - ls_hi) / lq_hi

        # Rigorous upper bound: numerator maximized (ld_hi - ls_lo), denominator minimized
        # (lq_lo), then rounded UP (ROUND_CEILING) so result is a genuine overestimate.
        ctx.rounding = ROUND_CEILING
        theta_hi = Decimal(1) + (ld_hi - ls_lo) / lq_lo

    return theta_lo, theta_hi


def prec400_estimate(s: int, dd: int, q: int) -> Decimal:
    """Return a midpoint estimate of theta at prec=400 (for cross-precision test).

    Uses midpoints of the ln intervals at prec=400, so it is a point estimate
    (not a certified bound), but at 400 digits of precision.
    """
    with localcontext() as ctx:
        ctx.prec = 400
        ld_lo, ld_hi = ln_interval(dd, pad=_DEFAULT_PAD)
        ls_lo, ls_hi = ln_interval(s, pad=_DEFAULT_PAD)
        lq_lo, lq_hi = ln_interval(q, pad=_DEFAULT_PAD)
        ld_mid = (ld_lo + ld_hi) / 2
        ls_mid = (ls_lo + ls_hi) / 2
        lq_mid = (lq_lo + lq_hi) / 2
        return Decimal(1) + (ld_mid - ls_mid) / lq_mid


@dataclass
class Certificate:
    """Reproducible certificate for a C_3a lower bound claim.

    All integer fields are exact. theta_lo and theta_hi are decimal strings
    of the rigorous lower and upper bounds on theta.
    """

    b: int
    A: list[int]
    d: int
    T: int
    s: int        # |U+U|
    dd: int       # |U-U|
    q: int        # 2*max(U)+1
    maxU: int
    theta_lo: str
    theta_hi: str
    no_carry: bool

    def to_json(self) -> str:
        """Serialize to JSON string with all exact integers as strings."""
        data = {
            "b": self.b,
            "A": self.A,
            "d": self.d,
            "T": self.T,
            "s": str(self.s),
            "dd": str(self.dd),
            "q": str(self.q),
            "maxU": str(self.maxU),
            "theta_lo": self.theta_lo,
            "theta_hi": self.theta_hi,
            "no_carry": self.no_carry,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Certificate":
        """Deserialize from JSON string."""
        data = json.loads(text)
        return cls(
            b=int(data["b"]),
            A=list(data["A"]),
            d=int(data["d"]),
            T=int(data["T"]),
            s=int(data["s"]),
            dd=int(data["dd"]),
            q=int(data["q"]),
            maxU=int(data["maxU"]),
            theta_lo=data["theta_lo"],
            theta_hi=data["theta_hi"],
            no_carry=bool(data["no_carry"]),
        )


def build_certificate(
    b: int,
    A: list[int],
    d: int,
    T: int,
    *,
    max_states: Optional[int] = None,
) -> Certificate:
    """Build a rigorous certificate for the C_3a lower bound from (b, A, d, T).

    Raises ValueError if no_carry_ok fails (fail fast — never falls back).
    Raises StateBudgetExceeded (from count_diffset) if max_states cap is hit.
    """
    if not no_carry_ok(b, A):
        raise ValueError(
            f"No-carry condition failed: b={b}, A={A}, max(A)={max(A)}, "
            f"requires b >= {2 * max(A) + 1}"
        )

    s = count_sumset(A, d, T)
    dd = count_diffset(A, d, T, max_states=max_states)
    mU = max_U(b, A, d, T)
    q = 2 * mU + 1

    theta_lo_dec, theta_hi_dec = certified_bound(s, dd, q)

    return Certificate(
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
