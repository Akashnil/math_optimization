"""Shared test fixtures and helpers for C_3a tests."""

import itertools
import pytest
from typing import Callable

from scripts.c3a.construction import no_carry_ok


def brute_force_sets(
    b: int,
    A: list[int],
    d: int,
    T: int,
) -> tuple[int, int, int]:
    """Compute |U|, |U+U|, |U-U| by brute force via itertools.product.

    REQUIRES no_carry_ok(b, A) — asserted before computation.
    A b < 2*max(A)+1 would produce carries, making this a counting test
    of the carries case rather than the digit-vector construction.
    """
    assert no_carry_ok(b, A), (
        f"brute_force_sets requires no_carry_ok: b={b}, A={A}, "
        f"max(A)={max(A)}, need b >= {2*max(A)+1}"
    )

    U: set[int] = set()
    for digits in itertools.product(A, repeat=d):
        if sum(digits) <= T:
            val = sum(digits[i] * (b ** i) for i in range(d))
            U.add(val)

    sumset: set[int] = {x + y for x in U for y in U}
    diffset: set[int] = {x - y for x in U for y in U}

    return len(U), len(sumset), len(diffset)


@pytest.fixture
def brute_force():
    """Fixture providing the brute_force_sets function."""
    return brute_force_sets
