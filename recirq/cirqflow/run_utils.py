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


def get_unique_run_id(fmt: str = 'run-{i}', base_data_dir: str = '.') -> str:
    """Find an unused run_id by checking for existing paths and incrementing `i` in `fmt` until
    an unused path name is found.

    Args:
        fmt: A format string containing {i} and optionally {date} to template trial run_ids.
        base_data_dir: The directory to check for the existence of files or directories.
    """
    if '{i}' not in fmt:
        raise ValueError("run_id format string must contain an '{i}' placeholder.")

    i = 1
    while True:
        run_id = fmt.format(i=i, date=datetime.date.today().isoformat())
        if not os.path.exists(f'{base_data_dir}/{run_id}'):
            break  # found an unused run_id
        i += 1

    return run_id
