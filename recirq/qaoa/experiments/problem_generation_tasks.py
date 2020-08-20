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

import os
from functools import lru_cache
from typing import Union

import networkx as nx
import numpy as np

import cirq
import cirq.contrib.routing as ccr
import recirq
from recirq.qaoa.problems import get_all_hardware_grid_problems, get_all_sk_problems, \
    get_all_3_regular_problems

EXPERIMENT_NAME = 'qaoa-problems'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class HardwareGridProblemGenerationTask:
    """Generate 'Hardware Grid' problems for a named device.

    This is a subgraph of the device's hardware topology with random
    +-1 weights on edges.

    See Also:
        :py:func:`generate_hardware_grid_problem`

    Attributes:
        dataset_id: A unique identifier for this dataset.
        device_name: The device to generate problems for.
        instance_i: Generate random instances indexed by this number.
        n_qubits: Generate an n-qubit instance.
    """
    dataset_id: str
    device_name: str
    instance_i: int
    n_qubits: int

    @property
    def fn(self):
        return (f'{self.dataset_id}/hardware-grid-problems/'
                f'{self.device_name}/'
                f'instance-{self.instance_i}/'
                f'nq-{self.n_qubits}')


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class SKProblemGenerationTask:
    """Generate a Sherrington-Kirkpatrick problem.

    This is a complete (fully-connected) graph with random +-1
    weights on edges.

    See Also:
        :py:func:`generate_sk_problem`

    Attributes:
        dataset_id: A unique identifier for this dataset.
        instance_i: Generate random instances indexed by this number.
        n_qubits: Generate an n-qubit instance.
    """
    dataset_id: str
    instance_i: int
    n_qubits: int

    @property
    def fn(self):
        return (f'{self.dataset_id}/sk-problems/'
                f'nq-{self.n_qubits}/'
                f'instance-{self.instance_i}')


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class ThreeRegularProblemGenerationTask:
    """Generate a 3-regular MaxCut problem.

    This is a random 3-regular graph (edge weight 1).

    See Also:
        :py:func:`generate_3_regular_problem`

    Attributes:
        dataset_id: A unique identifier for this dataset.
        instance_i: Generate random instances indexed by this number.
        n_qubits: Generate an n-qubit instance.
    """
    dataset_id: str
    instance_i: int
    n_qubits: int

    @property
    def fn(self):
        return (f'{self.dataset_id}/3-regular-problems/'
                f'nq-{self.n_qubits}/'
                f'instance-{self.instance_i}')


def _get_device_graph(device_name: str) -> nx.Graph:
    """Helper function to get the qubit connectivity for a given named device"""
    device = recirq.get_device_obj_by_name(device_name)
    device_graph = ccr.gridqubits_to_graph_device(device.qubits)
    return device_graph


def _get_central_qubit(device_name: str) -> cirq.GridQubit:
    """Helper function to get the 'central' qubit from which we
    grow the hardware grid problems."""
    if device_name == 'Sycamore23':
        return cirq.GridQubit(6, 3)

    raise ValueError(f"Don't know what the central qubit is for {device_name}")


@lru_cache()
def _get_all_hardware_grid_problems(
        device_name: str,
        seed: int = 52,
        max_n_instances: int = 100
):
    """Helper function to get all hardware grid problems for a given named device.

    This is annotated with lru_cache so you can freely call this function
    multiple times without re-generating anything or messing up the random seed.

    We take great care to make sure our random number generation is both
    deterministic across runs but random among problems within a run. If you
    change `seed` or `max_n_instances`, the np.random.RandomState will be
    advanced in a different way and different problems will be generated,
    so avoid changing if possible and make sure you understand the consequences
    otherwise.

    Returns:
        A dictionary indexed by n_qubit, instance_i
    """
    rs = np.random.RandomState(seed)
    device_graph = _get_device_graph(device_name=device_name)
    central_qubit = _get_central_qubit(device_name)
    return get_all_hardware_grid_problems(
        device_graph=device_graph, central_qubit=central_qubit,
        n_instances=max_n_instances, rs=rs)


def generate_hardware_grid_problem(
        task: HardwareGridProblemGenerationTask,
        base_dir=None):
    """Execute a :py:class:`HardwareGridProblemGenerationTask` task."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    problem = _get_all_hardware_grid_problems(
        device_name=task.device_name)[task.n_qubits, task.instance_i]

    recirq.save(task=task, data={
        'problem': problem,
    }, base_dir=base_dir)
    print(f"{task.fn} complete.")


@lru_cache()
def _get_all_sk_problems(
        seed: int = 53,
        max_n_qubits: int = 50,
        max_n_instances: int = 100):
    """Helper function to get all random Sherrington-Kirkpatrick problem
    instances for all numbers of qubits.

    This is annotated with lru_cache so you can freely call this function
    multiple times without re-generating anything or messing up the random seed.

    We take great care to make sure our random number generation is both
    deterministic across runs but random among problems within a run. If you
    change `seed`, `max_n_qubits`, or `max_n_instances`, the
    np.random.RandomState will be  advanced in a different way and
    different problems will be generated, so avoid changing if possible
    and make sure you understand the consequences otherwise.

    Returns:
        A dictionary indexed by n_qubit, instance_i
    """
    rs = np.random.RandomState(seed)
    return get_all_sk_problems(max_n_qubits=max_n_qubits,
                               n_instances=max_n_instances,
                               rs=rs)


def generate_sk_problem(
        task: SKProblemGenerationTask,
        base_dir=None):
    """Execute a :py:class:`SKProblemGenerationTask` task."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    problem = _get_all_sk_problems()[task.n_qubits, task.instance_i]
    recirq.save(task=task, data={
        'problem': problem,
    }, base_dir=base_dir)
    print(f"{task.fn} complete.")


@lru_cache()
def _get_all_3_regular_problems(
        seed: int = 54,
        max_n_qubits: int = 100,
        max_n_instances: int = 100):
    """Helper function to get all random 3-regular graph problem
    instances for all numbers of qubits.

    This is annotated with lru_cache so you can freely call this function
    multiple times without re-generating anything or messing up the random seed.

    We take great care to make sure our random number generation is both
    deterministic across runs but random among problems within a run. If you
    change `seed`, `max_n_qubits`, or `max_n_instances`, the
    np.random.RandomState will be  advanced in a different way and
    different problems will be generated, so avoid changing if possible
    and make sure you understand the consequences otherwise.

    Returns:
        A dictionary indexed by n_qubit, instance_i
    """
    rs = np.random.RandomState(seed)
    return get_all_3_regular_problems(max_n_qubits=max_n_qubits,
                                      n_instances=max_n_instances,
                                      rs=rs)


def generate_3_regular_problem(
        task: ThreeRegularProblemGenerationTask,
        base_dir=None):
    """Execute a :py:class:`ThreeRegularProblemGenerationTask` task."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    problem = _get_all_3_regular_problems()[task.n_qubits, task.instance_i]
    recirq.save(task=task, data={
        'problem': problem,
    }, base_dir=base_dir)
    print(f"{task.fn} complete.")


ProblemGenerationTaskT = Union[HardwareGridProblemGenerationTask,
                               SKProblemGenerationTask,
                               ThreeRegularProblemGenerationTask]
