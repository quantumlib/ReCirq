# Copyright 2020 Google
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

from recirq.quantum_volume.experiments.quantum_volume import QuantumVolumeTask, run_quantum_volume
import datetime


def main():
    """Main driver script entry point.

    This function contains configuration options and you will likely need
    to edit it to suit your needs. Of particular note, please make sure
    `dataset_id` and `device_name`
    are set how you want them. You may also want to change the values in
    the list comprehension to set the qubits.
    """
    # Uncomment below for an auto-generated unique dataset_id
    # dataset_id = datetime.datetime.now().isoformat(timespec='minutes')
    dataset_id = '2020-04-26'
    data_collection_tasks = [
        QuantumVolumeTask(
            dataset_id=dataset_id,
            device_name='Syc23-simulator',
            n_shots=10_000,
            n_circuits=1,
            n_qubits=4,
            depth=4,
            readout_error_correction=True,
        )
    ]

    for dc_task in data_collection_tasks:
        run_quantum_volume(dc_task)


if __name__ == '__main__':
    main()
