"""Root conftest.py — shared pytest configuration."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (can take ~17 min).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skip unless --runslow)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="Skipped: pass --runslow to run slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
