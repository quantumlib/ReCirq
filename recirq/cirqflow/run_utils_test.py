# Copyright 2022 Google
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

import datetime
import os

import pytest

from recirq.cirqflow.run_utils import get_unique_run_id


def test_get_unique_run_id(tmpdir):
    run_id = get_unique_run_id(base_data_dir=f'{tmpdir}')
    assert run_id == 'run-1'

    run_id = get_unique_run_id('myrun-{i}', base_data_dir=f'{tmpdir}')
    assert run_id == 'myrun-1'
    os.makedirs(f'{tmpdir}/{run_id}')

    run_id = get_unique_run_id('myrun-{i}', base_data_dir=f'{tmpdir}')
    assert run_id == 'myrun-2'

    with pytest.raises(ValueError):
        get_unique_run_id('myrun', base_data_dir=f'{tmpdir}')

    run_id = get_unique_run_id('myrun-{date}-{i}', base_data_dir=f'{tmpdir}')
    assert datetime.date.today().isoformat() in run_id
