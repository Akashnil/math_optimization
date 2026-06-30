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
    """|U+U| via window-clipped bitset DP with bitset interning + transition cache.

    b is not needed for counting once the no-carry condition is assumed;
    call no_carry_ok(b, A) before calling this function.

    State: dict keyed by (sum_c, bs_id) -> count of c-prefixes.
    Interning: each distinct clipped bitset is assigned a unique integer id.
    The (sum_c, bs_id) state count is identical to the old (sum_c, clipped_bitset)
    count because intern() is a bijection on clipped bitsets.
    """
    cvals = _build_sumset_cvals(A)
    clip_mask = (1 << (T + 1)) - 1

    # Bitset interning: assign int id to each distinct clipped bitset.
    # Initial bitset is 1 (only digit-sum 0 reached), clipped = 1.
    id2bs: list[int] = [1]
    bs2id: dict[int, int] = {1: 0}

    def intern_bs(bs: int) -> int:
        existing = bs2id.get(bs)
        if existing is not None:
            return existing
        new_id = len(id2bs)
        id2bs.append(bs)
        bs2id[bs] = new_id
        return new_id

    # Per-call transition cache: (bs_id, ci_index) -> nb_id
    cvals_list = list(cvals.items())
    trans: dict[tuple[int, int], int] = {}

    cur: dict[tuple[int, int], int] = {(0, 0): 1}

    for _ in range(d):
        nxt: dict[tuple[int, int], int] = defaultdict(int)
        for (sc, bs_id), cnt in cur.items():
            for ci_idx, (ci, abits) in enumerate(cvals_list):
                nsc = sc + ci
                if nsc > 2 * T:
                    continue  # prune: sum_c cannot exceed 2T
                cache_key = (bs_id, ci_idx)
                nb_id = trans.get(cache_key)
                if nb_id is None:
                    nb = _minkowski(id2bs[bs_id], abits) & clip_mask
                    nb_id = intern_bs(nb)
                    trans[cache_key] = nb_id
                nxt[(nsc, nb_id)] += cnt
        cur = dict(nxt)

    s = 0
    for (sc, bs_id), cnt in cur.items():
        bs = id2bs[bs_id]
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
    """|U-U| via window-clipped bitset DP with sum_e pruning + bitset interning.

    b is not needed for counting once the no-carry condition is assumed.
    max_states: if the state dict grows beyond this, raise StateBudgetExceeded.

    Optimizations applied:
    (1) Window-clip bitsets to [0,T] before inserting (clip-before-insert).
    (2) Prune dead sum_e values: if no valid final sum_e is reachable, drop prefix.
    (3) State merging is automatic: clip-before-insert ensures equal clipped bitsets
        hash equal and their counts are summed.
    (4) Bitset interning + transition cache: each distinct clipped bitset is
        assigned a unique int id. The (se, bs_id) state count is identical to
        the old (se, clipped_bitset) count because intern() is a bijection on
        clipped bitsets (injectivity invariant). The max_states cap applies to
        len(nxt) — the per-step (se, bs_id) count — NOT to len(id2bs).
    """
    evals = _build_diffset_evals(A)
    mA = max(A)
    clip_mask = (1 << (T + 1)) - 1

    # Bitset interning: assign int id to each distinct clipped bitset.
    # Initial bitset is 1 (only digit-sum 0 reached), clipped = 1.
    id2bs: list[int] = [1]
    bs2id: dict[int, int] = {1: 0}

    def intern_bs(bs: int) -> int:
        existing = bs2id.get(bs)
        if existing is not None:
            return existing
        new_id = len(id2bs)
        id2bs.append(bs)
        bs2id[bs] = new_id
        return new_id

    # Per-call transition cache: (bs_id, ei_index) -> nb_id
    evals_list = list(evals.items())
    trans: dict[tuple[int, int], int] = {}

    cur: dict[tuple[int, int], int] = {(0, 0): 1}

    for step in range(d):
        remaining = d - step - 1  # digits left after this step
        nxt: dict[tuple[int, int], int] = defaultdict(int)

        for (se, bs_id), cnt in cur.items():
            for ei_idx, (ei, abits) in enumerate(evals_list):
                nse = se + ei
                # Prune: check if any final sum_e in [nse - mA*remaining, nse + mA*remaining]
                # can satisfy the window condition (sum_e in [-T, T])
                se_min = nse - mA * remaining
                se_max = nse + mA * remaining
                if se_min > T or se_max < -T:
                    continue  # dead prefix

                cache_key = (bs_id, ei_idx)
                nb_id = trans.get(cache_key)
                if nb_id is None:
                    nb = _minkowski(id2bs[bs_id], abits) & clip_mask
                    nb_id = intern_bs(nb)
                    trans[cache_key] = nb_id
                nxt[(nse, nb_id)] += cnt

        if max_states is not None and len(nxt) > max_states:
            raise StateBudgetExceeded(
                f"State dict size {len(nxt)} exceeded max_states={max_states}"
            )

        cur = dict(nxt)

    dd = 0
    for (se, bs_id), cnt in cur.items():
        bs = id2bs[bs_id]
        lo = max(0, se)
        hi = min(T, T + se)
        if lo > hi:
            continue
        window = ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)
        if bs & window:
            dd += cnt
    return dd
