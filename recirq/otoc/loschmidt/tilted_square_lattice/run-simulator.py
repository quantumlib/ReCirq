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

"""A script for executing the loschmidt.tilted_square_lattice benchmark on a simulator.

Practitioners should copy this script to provide their own parameters for
QuantumRuntimeConfiguration.
"""

import cirq
from cirq_google.workflow import (
    QuantumRuntimeConfiguration, SimulatedProcessorWithLocalDeviceRecord, RandomDevicePlacer,
    execute
)
from recirq.cirqflow.run_utils import get_unique_run_id
from recirq.otoc.loschmidt.tilted_square_lattice import TiltedSquareLatticeLoschmidtSpec

assert TiltedSquareLatticeLoschmidtSpec, 'register deserializer'

EXES_FILENAME = 'loschmidt.tilted_square_lattice.small-v1.json.gz'


def main():
    exegroup = cirq.read_json_gzip(EXES_FILENAME)
    rt_config = QuantumRuntimeConfiguration(
        processor_record=SimulatedProcessorWithLocalDeviceRecord('rainbow', noise_strength=0.005),
        qubit_placer=RandomDevicePlacer(),
        run_id=get_unique_run_id('simulated-{i}'),
        random_seed=52,
    )
    raw_results = execute(rt_config, exegroup)
    print("Finished run_id", raw_results.shared_runtime_info.run_id)


if __name__ == '__main__':
    main()
