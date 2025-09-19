# Copyright 2023 Google
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

from recirq.seniority_zero.scheduling import get_tqbg_groups, rotate_list


def test_rotate_list():
    a_list = [0, 1, 2, 3, 4]
    b_list = rotate_list(a_list, 2, clockwise=True)
    assert tuple(b_list) == (3, 4, 0, 1, 2)
    b_list = rotate_list(a_list, 2, clockwise=False)
    assert tuple(b_list) == (2, 3, 4, 0, 1)


def test_get_tqbg_groups():
    for nq in [6, 8, 10]:
        groups = get_tqbg_groups(nq)
        assert len(groups) == nq + 1
