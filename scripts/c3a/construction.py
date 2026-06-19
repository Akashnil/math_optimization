"""Exact-arithmetic construction and counting for C_3a digit constructions.

U(b, A, d, T) = { sum_{i=0}^{d-1} a_i * b^i : each a_i in A, sum_i a_i <= T }

No-carry condition: b >= 2*max(A) + 1 ensures digit-vectors uniquely represent
sums and differences (no carries or borrows), so counting digit-vectors is exact.
"""

from collections import defaultdict
from typing import Optional


class StateBudgetExceeded(Exception):
    """Raised when count_diffset state dict exceeds the max_states cap."""


def no_carry_ok(b: int, A: list[int]) -> bool:
    """Return True iff the no-carry condition holds and A is a valid digit set.

    Requires: b >= 2*max(A)+1, 0 in A, all elements of A in {0..b-1}.
    """
    if not A:
        return False
    if 0 not in A:
        return False
    mA = max(A)
    if b < 2 * mA + 1:
        return False
    for a in A:
        if a < 0 or a >= b:
            return False
    return True


def max_U(b: int, A: list[int], d: int, T: int) -> int:
    """Compute max(U) exactly using the greedy closed form.

    Places the largest digit mA=max(A) in the highest positions while
    the running digit-sum stays <= T.
    """
    mA = max(A)
    rem = T
    coords: list[int] = [0] * d
    for i in range(d - 1, -1, -1):
        v = min(mA, rem)
        coords[i] = v
        rem -= v
    return sum(coords[i] * (b ** i) for i in range(d))


def _minkowski(bs: int, abits: int) -> int:
    """Minkowski sum of two reachable sets encoded as bitsets (shift-OR).

    If bs has bit j set and abits has bit k set, the result has bit j+k set.
    """
    result = 0
    temp = abits
    bit = 0
    while temp:
        if temp & 1:
            result |= (bs << bit)
        temp >>= 1
        bit += 1
    return result


def _build_sumset_cvals(A: list[int]) -> dict[int, int]:
    """Build per-digit-value bitset table for sumset counting.

    For each ci in A+A: abits[ci] has bit a set for each a in A with (ci-a) in A.
    """
    A_set = set(A)
    A_sum = sorted({a1 + a2 for a1 in A for a2 in A})
    cvals: dict[int, int] = {}
    for ci in A_sum:
        abits = 0
        for a in A:
            if (ci - a) in A_set:
                abits |= (1 << a)
        if abits:
            cvals[ci] = abits
    return cvals


def _build_diffset_evals(A: list[int]) -> dict[int, int]:
    """Build per-digit-value bitset table for diffset counting.

    For each ei in A-A: abits[ei] has bit a set for each a in A with (a-ei) in A.
    """
    A_set = set(A)
    A_diff = sorted({a1 - a2 for a1 in A for a2 in A})
    evals: dict[int, int] = {}
    for ei in A_diff:
        abits = 0
        for a in A:
            if (a - ei) in A_set:
                abits |= (1 << a)
        if abits:
            evals[ei] = abits
    return evals


def count_sumset(A: list[int], d: int, T: int) -> int:
    """|U+U| via window-clipped bitset DP.

    b is not needed for counting once the no-carry condition is assumed;
    call no_carry_ok(b, A) before calling this function.

    State: dict keyed by (sum_c, clipped-bitset) -> count of c-prefixes.
    """
    cvals = _build_sumset_cvals(A)
    clip_mask = (1 << (T + 1)) - 1
    # Initial state: sum_c=0, reachable={0} (only digit-sum 0 reached), count 1
    cur: dict[tuple[int, int], int] = {(0, 1): 1}

    for _ in range(d):
        nxt: dict[tuple[int, int], int] = defaultdict(int)
        for (sc, bs), cnt in cur.items():
            for ci, abits in cvals.items():
                nsc = sc + ci
                if nsc > 2 * T:
                    continue  # prune: sum_c cannot exceed 2T
                nb = _minkowski(bs, abits)
                nb &= clip_mask  # WINDOW-CLIP before insert so equal states merge
                nxt[(nsc, nb)] += cnt
        cur = dict(nxt)

    s = 0
    for (sc, bs), cnt in cur.items():
        lo = max(0, sc - T)
        hi = T
        if lo > hi:
            continue
        window = ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)
        if bs & window:
            s += cnt
    return s


def count_diffset(
    A: list[int],
    d: int,
    T: int,
    *,
    max_states: Optional[int] = None,
) -> int:
    """|U-U| via window-clipped bitset DP with sum_e pruning.

    b is not needed for counting once the no-carry condition is assumed.
    max_states: if the state dict grows beyond this, raise StateBudgetExceeded.

    Optimizations applied:
    (1) Window-clip bitsets to [0,T] before inserting (clip-before-insert).
    (2) Prune dead sum_e values: if no valid final sum_e is reachable, drop prefix.
    (3) State merging is automatic: clip-before-insert ensures equal clipped bitsets
        hash equal and their counts are summed.
    """
    evals = _build_diffset_evals(A)
    mA = max(A)
    clip_mask = (1 << (T + 1)) - 1

    cur: dict[tuple[int, int], int] = {(0, 1): 1}

    for step in range(d):
        remaining = d - step - 1  # digits left after this step
        nxt: dict[tuple[int, int], int] = defaultdict(int)

        for (se, bs), cnt in cur.items():
            for ei, abits in evals.items():
                nse = se + ei
                # Prune: check if any final sum_e in [nse - mA*remaining, nse + mA*remaining]
                # can satisfy the window condition (sum_e in [-T, T])
                se_min = nse - mA * remaining
                se_max = nse + mA * remaining
                if se_min > T or se_max < -T:
                    continue  # dead prefix

                nb = _minkowski(bs, abits)
                nb &= clip_mask  # WINDOW-CLIP before insert
                nxt[(nse, nb)] += cnt

        if max_states is not None and len(nxt) > max_states:
            raise StateBudgetExceeded(
                f"State dict size {len(nxt)} exceeded max_states={max_states}"
            )

        cur = dict(nxt)

    dd = 0
    for (se, bs), cnt in cur.items():
        lo = max(0, se)
        hi = min(T, T + se)
        if lo > hi:
            continue
        window = ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)
        if bs & window:
            dd += cnt
    return dd
