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

from recirq.qaoa.experiments.problem_generation_tasks import SKProblemGenerationTask, \
    HardwareGridProblemGenerationTask, ThreeRegularProblemGenerationTask, \
    generate_3_regular_problem, generate_sk_problem, generate_hardware_grid_problem


def main():
    dataset_id = '2020-03-19'
    hardware_grid_problem_tasks = [
        HardwareGridProblemGenerationTask(
            dataset_id=dataset_id,
            device_name='Sycamore23',
            instance_i=i,
            n_qubits=n
        )
        for i in range(100)
        for n in range(2, 23 + 1)
    ]
    sk_problem_tasks = [
        SKProblemGenerationTask(
            dataset_id=dataset_id,
            instance_i=i,
            n_qubits=n
        )
        for i in range(100)
        for n in range(3, 17 + 1)
    ]
    three_regular_problem_tasks = [
        ThreeRegularProblemGenerationTask(
            dataset_id=dataset_id,
            instance_i=i,
            n_qubits=n
        )
        for i in range(100)
        for n in range(3, 23 + 1) if 3 * n % 2 == 0
    ]

    for task in hardware_grid_problem_tasks:
        generate_hardware_grid_problem(task)
    for task in sk_problem_tasks:
        generate_sk_problem(task)
    for task in three_regular_problem_tasks:
        generate_3_regular_problem(task)


if __name__ == '__main__':
    main()
