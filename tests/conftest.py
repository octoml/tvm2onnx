"""Provides common fixtures used through the tests."""
import os

import pytest
import tvm

# include of xgboost avoids spurious failures on ARM
import xgboost  # noqa # pylint: disable=unused-import


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
