# Copyright 2023 OctoML
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides common fixtures used through the tests."""
import logging
import os

import pytest

logging.captureWarnings(True)


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
                # TODO(agladyshev): find another way to check, without import tvm
                import tvm

                tvm.cuda().compute_version
            except (ImportError, Exception):
                pytest.skip("PyCuda not installed or no CUDA device detected")
