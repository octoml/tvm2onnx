"""Provides common fixtures used through the tests."""
import os

import pytest
import tvm
import logging
import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_runtest_setup(item):
    for marker in item.iter_markers():
        """Checks if the target is cuda and if cuda exists. Skips the test if
        there is no cuda. cuda tests can be forced by setting the environment
        variable FORCE_CUDA=1"""
        if marker.name == "cuda" and int(os.getenv("FORCE_CUDA", 0)) == 0:
            try:
                # This is just to check if CUDA is available.
                tvm.cuda().compute_version
            except (ImportError, Exception):
                pytest.skip("PyCuda not installed or no CUDA device detected")
