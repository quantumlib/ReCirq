# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest import mock

from recirq.cpu_utils import get_available_cpu_count, set_threading_limits


def test_get_available_cpu_count_default():
    cpu_count = get_available_cpu_count()
    assert isinstance(cpu_count, int)
    assert cpu_count >= 1


def test_get_available_cpu_count_xdist():
    # Mocking environment and available CPUs
    with mock.patch.dict(os.environ, {"PYTEST_XDIST_WORKER_COUNT": "4"}):
        with mock.patch("os.cpu_count", return_value=8):
            # 1. Test when process_cpu_count exists
            with mock.patch("os.process_cpu_count", create=True, return_value=8):
                assert get_available_cpu_count() == 2

            # 2. Test when process_cpu_count doesn't exist
            # Save and temporarily remove process_cpu_count from os module if it exists
            orig_process_cpu_count = getattr(os, "process_cpu_count", None)
            if orig_process_cpu_count is not None:
                delattr(os, "process_cpu_count")
            try:
                # Test when sched_getaffinity exists
                with mock.patch("os.sched_getaffinity", create=True, return_value=set(range(8))):
                    assert get_available_cpu_count() == 2

                # Test when neither exists (fallback to cpu_count)
                orig_sched_getaffinity = getattr(os, "sched_getaffinity", None)
                if orig_sched_getaffinity is not None:
                    delattr(os, "sched_getaffinity")
                try:
                    assert get_available_cpu_count() == 2
                finally:
                    if orig_sched_getaffinity is not None:
                        os.sched_getaffinity = orig_sched_getaffinity
            finally:
                if orig_process_cpu_count is not None:
                    os.process_cpu_count = orig_process_cpu_count


def test_get_available_cpu_count_xdist_invalid():
    with mock.patch.dict(os.environ, {"PYTEST_XDIST_WORKER_COUNT": "invalid"}):
        cpu_count = get_available_cpu_count()
        assert isinstance(cpu_count, int)
        assert cpu_count >= 1


def test_set_threading_limits_xdist():
    with mock.patch.dict(os.environ, {"PYTEST_XDIST_WORKER_COUNT": "4"}), \
         mock.patch("recirq.cpu_utils.get_available_cpu_count", return_value=2):
        set_threading_limits()
        for var in ["MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
            assert os.environ[var] == "1"


def test_set_threading_limits_no_xdist():
    # Remove PYTEST_XDIST_WORKER_COUNT if it's there
    env_patch = os.environ.copy()
    env_patch.pop("PYTEST_XDIST_WORKER_COUNT", None)
    with mock.patch.dict(os.environ, env_patch, clear=True), \
         mock.patch("recirq.cpu_utils.get_available_cpu_count", return_value=4):
        set_threading_limits()
        for var in ["MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
            assert os.environ[var] == "3"  # get_available_cpu_count() - 1 = 3
