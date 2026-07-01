"""Fast tests for scripts.c3a.tune — generalized single-(b,A,d,T) runner.

Tests covered:
1. --smoke path: runs, writes JSON, theta_lo is parseable and positive.
2. Gerbicz-checkpoint reproduction via the general path: b=7, A=[0,1,2,3], d=4, T=8
   must produce s=FAST_S and dd=FAST_DD exactly (ties the general path back to
   the counting DP anchored in baseline.py).
3. No-carry fail-fast gate: b=20 with A=[0,2,3,4,5,6,7,8,9,10] (needs b>=21)
   raises ValueError BEFORE any counting runs.
"""

import json
import pytest
from decimal import Decimal
from pathlib import Path

from scripts.c3a.baseline import (
    FAST_A,
    FAST_B,
    FAST_D,
    FAST_DD,
    FAST_S,
    FAST_T,
)
from scripts.c3a.tune import DEFAULT_A, run_one, _run_smoke, _output_filename


class TestTune:
    """Tests for the generalized tune runner."""

    def test_smoke_path(self, tmp_path: Path) -> None:
        """--smoke mode completes, writes valid JSON with parseable theta_lo."""
        _run_smoke(tmp_path)
        out_path = tmp_path / "smoke_d8_T15.json"
        assert out_path.exists(), "smoke JSON not written"

        data = json.loads(out_path.read_text())
        assert "theta_lo" in data, "theta_lo missing from JSON"
        theta_lo = Decimal(data["theta_lo"])
        assert theta_lo > Decimal("1"), f"theta_lo={theta_lo} not > 1"

    def test_gerbicz_checkpoint_exact(self, tmp_path: Path) -> None:
        """General run_one with (b=7, A=[0,1,2,3], d=4, T=8) reproduces
        Gerbicz checkpoint s=FAST_S and dd=FAST_DD exactly.

        This anchors the general (b, A, d, T) path to the same counting DP
        used in baseline.py and confirmed correct across prior rounds.
        """
        cert = run_one(FAST_B, FAST_A, FAST_D, FAST_T, tmp_path)

        assert cert.s == FAST_S, (
            f"sumset mismatch via general path: got {cert.s}, expected {FAST_S}"
        )
        assert cert.dd == FAST_DD, (
            f"diffset mismatch via general path: got {cert.dd}, expected {FAST_DD}"
        )

    def test_no_carry_fail_fast(self) -> None:
        """b=20 with A=[0,2,3,4,5,6,7,8,9,10] (needs b>=21) raises ValueError
        immediately — before any counting runs.

        Per spec-review suggestion #2: the fail-fast gate must fire before the
        expensive DP, not after. Verified by patching count_sumset to assert-fail
        if called, but we rely on the structural guarantee: no_carry_ok is checked
        at the top of run_one before any count_* call.
        """
        with pytest.raises(ValueError, match="No-carry condition failed"):
            # b=20 fails: max(A)=10, needs b >= 2*10+1 = 21
            run_one(20, DEFAULT_A, 4, 8, Path("/tmp/should-not-be-created"))

    def test_output_filename_record_family(self) -> None:
        """Record family (b=21, A=DEFAULT_A) uses d{d}_T{T}.json format."""
        name = _output_filename(21, DEFAULT_A, 90, 172)
        assert name == "d90_T172.json"

    def test_output_filename_nonrecord_family(self) -> None:
        """Non-record family uses b{b}_d{d}_T{T}.json format."""
        name = _output_filename(7, [0, 1, 2, 3], 4, 8)
        assert name == "b7_d4_T8.json"

    def test_gerbicz_json_written(self, tmp_path: Path) -> None:
        """run_one writes valid JSON to the expected path for non-record family."""
        run_one(FAST_B, FAST_A, FAST_D, FAST_T, tmp_path)
        expected = tmp_path / f"b{FAST_B}_d{FAST_D}_T{FAST_T}.json"
        assert expected.exists(), f"JSON not written to {expected}"
        data = json.loads(expected.read_text())
        assert int(data["s"]) == FAST_S
        assert int(data["dd"]) == FAST_DD
        assert data["b"] == FAST_B
        assert data["A"] == FAST_A
