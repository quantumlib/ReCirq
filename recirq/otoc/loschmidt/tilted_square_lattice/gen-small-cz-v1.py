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

"""Generate the QuantumExecutableGroup for the small-cz-v1 configuration of the
loschmidt.tilted_square_lattice benchmark.

The `small-cz-v1` configuration is A 'small' configuration for quick verification
of Loschmidt echos using the CZ gate.

This configuration uses small grid topologies (making it suitable for
running on simulators) and a small number of random instances making it
suitable for getting a quick reading on processor performance in ~minutes.

To run:

    python gen-small-cz-v1.py

"""

import numpy as np

import cirq
from recirq.otoc.loschmidt.tilted_square_lattice import \
    get_all_tilted_square_lattice_executables

EXES_FILENAME = 'loschmidt.tilted_square_lattice.small-cz-v1.json.gz'


def main():
    exes = get_all_tilted_square_lattice_executables(
        min_side_length=2, max_side_length=3, side_length_step=1,
        n_instances=3,
        macrocycle_depths=np.arange(0, 4 + 1, 1),
        twoq_gate_name='cz',
    )
    print(len(exes), 'executables')

    cirq.to_json_gzip(exes, EXES_FILENAME)
    print(f'Wrote {EXES_FILENAME}')


if __name__ == '__main__':
    main()
