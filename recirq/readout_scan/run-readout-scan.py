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

# ---                 This file has been autogenerated                    --- #
# ---              from docs/Readout-Data-Collection.ipynb                --- #
# ---                   Do not edit this file directly                    --- #



from recirq.readout_scan.tasks import ReadoutScanTask, run_readout_scan
import datetime
import cirq_google as cg

MAX_N_QUBITS = 5

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
    dataset_id = '2020-02-tutorial'
    data_collection_tasks = [
        ReadoutScanTask(
            dataset_id=dataset_id,
            device_name='Syc23-simulator',
            n_shots=40_000,
            qubit=qubit,
            resolution_factor=6,
        )
        for qubit in sorted(cg.Sycamore23.metadata.qubit_set)[:MAX_N_QUBITS]
    ]

    for dc_task in data_collection_tasks:
        run_readout_scan(dc_task)


if __name__ == '__main__':
    main()

