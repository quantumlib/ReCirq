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
from typing import Dict

import numpy as np

import recirq
from recirq.qaoa.classical_angle_optimization import OptimizationResult, \
    optimize_instance_interp_heuristic
from recirq.qaoa.experiments.problem_generation_tasks import ProblemGenerationTaskT, \
    DEFAULT_BASE_DIR as DEFAULT_PROBLEM_GENERATION_BASE_DIR
from recirq.qaoa.problems import HardwareGridProblem, ThreeRegularProblem, SKProblem

EXPERIMENT_NAME = 'qaoa-precomputation'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class AnglePrecomputationTask:
    """Pre-compute optimized angles classically for a given problem.

    See Also:
        :py:func:`precompute_angles`

    Attributes:
        dataset_id: A unique identifier for this dataset.
        generation_task: The input task which specifies the problem.
        p: QAOA depth hyperparameter p. The number of parameters is 2*p.
    """
    dataset_id: str
    generation_task: ProblemGenerationTaskT
    p: int

    @property
    def fn(self):
        parent_fn = self.generation_task.fn
        return (f'{self.dataset_id}/'
                f'from-{parent_fn}/'
                f'p-{self.p}')


@lru_cache()
def _get_optima(in_task: ProblemGenerationTaskT,
                problem_generation_base_dir,
                p_max: int = 5) \
        -> Dict[int, OptimizationResult]:
    """Helper function to get optimal parameters for a given instance
    of a given device's problem, subselected on a given number of qubits.

    This function is annotated with lru_cache so you can call it once for
    each p-value without re-doing the (expensive) optimization.
    optimize_instance_interp_heuristic uses low-p-values to bootstrap
    guesses for high-p-values, so it just does the optimization
    for everything up to p_max.

    This is the meat of `generate_problems` to get the optimal parameters
    for a given (device, instance, n_qubit, p) specification which are all
    the relevant parameters.
    """
    data = recirq.load(in_task, base_dir=problem_generation_base_dir)
    problem = data['problem']
    if isinstance(problem, HardwareGridProblem):
        param_guess = [np.pi / 8, -np.pi / 8]
    elif isinstance(problem, ThreeRegularProblem):
        param_guess = [np.pi / 8, -np.pi / 8]
    elif isinstance(problem, SKProblem):
        n = problem.graph.number_of_nodes()
        param_guess = [
            np.arccos(np.sqrt((1 + np.sqrt((n - 2) / (n - 1))) / 2)),
            -np.pi / 8
        ]
    else:
        raise ValueError("Unknown problem type: {}".format(problem))

    optima = optimize_instance_interp_heuristic(
        graph=problem.graph,
        p_max=p_max,
        param_guess_at_p1=param_guess,
        verbose=True,
    )
    return {op.p: op for op in optima}


def precompute_angles(task: AnglePrecomputationTask,
                      base_dir=None, problem_generation_base_dir=None):
    """Execute a :py:func:`AnglePrecomputationTask` task."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if problem_generation_base_dir is None:
        problem_generation_base_dir = DEFAULT_PROBLEM_GENERATION_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    optimum = _get_optima(
        in_task=task.generation_task,
        problem_generation_base_dir=problem_generation_base_dir,
    )[task.p]
    recirq.save(task=task, data={
        'optimum': optimum,
    }, base_dir=base_dir)
    print(f"{task.fn} complete.")
