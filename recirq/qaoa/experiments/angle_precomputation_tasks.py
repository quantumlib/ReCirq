import os
from functools import lru_cache
from typing import Dict, Union

import numpy as np

import recirq
from recirq.qaoa.classical_angle_optimization import OptimizationResult, \
    optimize_instance_interp_heuristic
from recirq.qaoa.experiments.problem_generation_tasks import HardwareGridProblemGenerationTask, \
    SKProblemGenerationTask, ThreeRegularProblemGenerationTask, \
    DEFAULT_BASE_DIR as DEFAULT_PROBLEM_GENERATION_BASE_DIR
from recirq.qaoa.problems import HardwareGridProblem, ThreeRegularProblem, SKProblem

ProblemGenerationTask = Union[HardwareGridProblemGenerationTask,
                              SKProblemGenerationTask,
                              ThreeRegularProblemGenerationTask]

EXPERIMENT_NAME = 'qaoa-precomputation'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class AnglePrecomputationTask:
    dataset_id: str
    generation_task: ProblemGenerationTask
    p: int

    @property
    def fn(self):
        parent_fn = self.generation_task.fn
        return (f'{self.dataset_id}/'
                f'from-{parent_fn}/'
                f'p-{self.p}')


@lru_cache()
def _get_optima(in_task: ProblemGenerationTask,
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
        dtype=np.float64,
    )
    return {op.p: op for op in optima}


def precompute_angles(task: AnglePrecomputationTask,
                      base_dir=None, problem_generation_base_dir=None):
    """Given a set of parameters that completely specifies a QAOA problem,
    generate and save a graph representation and optimal angles.
    """
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
