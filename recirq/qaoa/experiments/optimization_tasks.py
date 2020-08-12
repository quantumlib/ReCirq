from typing import Dict, Sequence, Optional, List, Tuple
import os
import shutil

import numpy as np
import cirq

import recirq
from recirq.qaoa.experiments.angle_precomputation_tasks import (
    AnglePrecomputationTask)
from recirq.qaoa.experiments.angle_precomputation_tasks import (
    DEFAULT_BASE_DIR as DEFAULT_PRECOMPUTATION_BASE_DIR)
from recirq.qaoa.experiments.problem_generation_tasks import (
    DEFAULT_BASE_DIR as DEFAULT_PROBLEM_GENERATION_BASE_DIR,
    ProblemGenerationTaskT, SKProblemGenerationTask)
from recirq.qaoa.placement import place_line_on_device
from recirq.qaoa.problem_circuits import (get_compiled_hardware_grid_circuit,
                                          get_compiled_sk_model_circuit,
                                          get_compiled_3_regular_maxcut_circuit)
from recirq.qaoa.problems import (ProblemT, HardwareGridProblem, SKProblem,
                                  ThreeRegularProblem)
from recirq.qaoa.simulation import hamiltonian_objectives

EXPERIMENT_NAME = 'qaoa-optimization'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


def _abbrev_n_shots(n_shots: int) -> str:
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class OptimizationAlgorithm:
    """An optimization algorithm.

    Attributes:
        method: The name of the optimization method.
        n_shots: The number of measurement shots per objective function
            evaluation the algorithm should request.
        options: Options for the optimization algorithm.
    """

    method: str
    n_shots: int
    options: Optional[Dict] = None

    @property
    def description(self):
        n_shots = _abbrev_n_shots(self.n_shots)
        description = f'{self.method}_n_shots-{n_shots}'
        if self.options is not None:
            for key, value in self.options.items():
                description += f'_{key}-{value}'
        return description


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class OptimizationTask:
    """An optimization task for the QAOA objective function.

    See Also:
        :py:func:`collect_optimization_data`

    Attributes:
        dataset_id: A unique identifier for this dataset.
        generation_task: The task specifying the problem to collect data for
        device_name: The device to run on.
        p: The number of iterations of problem + driver unitary.
        algorithm: The optimization algorithm to use.
        line_placement_strategy: Only used for SK model problems. Options
            include 'brute_force', 'random', 'greedy', 'anneal', 'mst',
            and 'mixed'. Defaults to 'mixed'.
    """
    dataset_id: str
    generation_task: ProblemGenerationTaskT
    device_name: str
    p: int
    algorithm: OptimizationAlgorithm
    x0: List[float]
    line_placement_strategy: Optional[str] = None

    @property
    def fn(self):
        fn = (f'{self.dataset_id}/'
              f'{self.device_name}/'
              f'from-{self.generation_task.fn}/'
              f'p-{self.p}/'
              f'{self.algorithm.description}/'
              f'x0-{self.x0}')
        if isinstance(self.generation_task, SKProblemGenerationTask):
            line_placement_strategy = self.line_placement_strategy or 'mixed'
            fn += f'/line_placement-{line_placement_strategy}'
        fn += '/result'
        return fn


def collect_optimization_data(
        task: OptimizationTask,
        base_dir: str = DEFAULT_BASE_DIR,
        problem_generation_base_dir: str = DEFAULT_PROBLEM_GENERATION_BASE_DIR,
        precomputation_base_dir: str = DEFAULT_PRECOMPUTATION_BASE_DIR) -> None:
    """Collect data for a QAOA optimization experiment.

    Args:
        task: The optimization task.
        base_dir: The base directory in which to save data.
        problem_generation_base_dir: The base directory of the problem.
    """
    if recirq.exists(task, base_dir=base_dir):
        print(f'{task.fn} already exists. Skipping.')
        return

    sampler = recirq.get_sampler_by_name(device_name=task.device_name)
    problem = recirq.load(
        task.generation_task,
        base_dir=problem_generation_base_dir)['problem']  # type: ProblemT

    lowest_energy_found = np.inf
    best_bitstring_found = None
    evaluated_points = []
    bitstring_lists = []
    energy_lists = []
    mean_energies = []

    def f(x):
        print('Evaluating objective ...')
        # Create circuit
        gammas = x[:task.p]
        betas = x[task.p:]
        initial_qubits, circuit, final_qubits = _get_circuit(
            problem, gammas, betas, task.device_name)
        # Sample bitstrings from circuit
        result = sampler.run(program=circuit,
                             repetitions=task.algorithm.n_shots)
        # Process bitstrings
        nonlocal lowest_energy_found
        nonlocal best_bitstring_found
        bitstrings = result.measurements['z']
        initial_indices = {q: i for i, q in enumerate(initial_qubits)}
        final_indices = [initial_indices[q] for q in final_qubits]
        nodelist = np.argsort(final_indices)
        energies = hamiltonian_objectives(bitstrings,
                                          problem.graph,
                                          nodelist=nodelist)
        lowest_energy_index = np.argmin(energies)
        lowest_energy = energies[lowest_energy_index]
        if lowest_energy < lowest_energy_found:
            lowest_energy_found = lowest_energy
            best_bitstring_found = bitstrings[lowest_energy_index]
        mean = np.mean(energies)
        # Save bitstrings and other data
        evaluated_points.append(recirq.NumpyArray(x))
        bitstring_lists.append(recirq.BitArray(bitstrings))
        energy_lists.append(recirq.NumpyArray(np.array(energies)))
        mean_energies.append(mean)
        print('Objective function: {}'.format(mean))
        print()
        return mean

    result = recirq.optimize.minimize(f,
                                      task.x0,
                                      method=task.algorithm.method,
                                      **(task.algorithm.options or {}))

    result_data = {
        'x': result.x,
        'fun': result.fun,
        'nit': result.nit,
        'nfev': result.nfev,
        'lowest_energy_found': lowest_energy_found,
        'best_bitstring_found': best_bitstring_found,
        'evaluated_points': evaluated_points,
        'bitstring_lists': bitstring_lists,
        'energy_lists': energy_lists,
        'mean_energies': mean_energies
    }
    if hasattr(result, 'x_iters'):
        result_data['x_iters'] = result['x_iters']
    if hasattr(result, 'func_vals'):
        result_data['func_vals'] = result['func_vals']
    if hasattr(result, 'model_vals'):
        result_data['model_vals'] = result['model_vals']

    # Evaluate at optimal angles if they have been precomputed
    angle_precomputation_task = AnglePrecomputationTask(
        dataset_id=task.dataset_id,
        generation_task=task.generation_task,
        p=task.p)
    if recirq.exists(angle_precomputation_task,
                     base_dir=precomputation_base_dir):
        optimum = recirq.load(angle_precomputation_task,
                              base_dir=precomputation_base_dir)['optimum']
        gammas = optimum.gammas
        betas = optimum.betas
        initial_qubits, circuit, final_qubits = _get_circuit(
            problem, gammas, betas, task.device_name)
        result = sampler.run(program=circuit, repetitions=50_000)
        bitstrings = result.measurements['z']
        initial_indices = {q: i for i, q in enumerate(initial_qubits)}
        final_indices = [initial_indices[q] for q in final_qubits]
        nodelist = np.argsort(final_indices)
        energies = hamiltonian_objectives(bitstrings,
                                          problem.graph,
                                          nodelist=nodelist)
        mean = np.mean(energies)
        result_data['optimal_angles'] = gammas + betas
        result_data['optimal_angles_fun'] = mean

    recirq.save(task=task, data=result_data, base_dir=base_dir)


def _get_circuit(problem: ProblemT, gammas: Sequence[float],
                 betas: Sequence[float], device_name: str
                ) -> Tuple[List[cirq.Qid], cirq.Circuit, List[cirq.Qid]]:
    if isinstance(problem, HardwareGridProblem):
        initial_qubits = [cirq.GridQubit(r, c) for r, c in problem.coordinates]
        circuit, final_qubits = get_compiled_hardware_grid_circuit(
            problem=problem,
            qubits=initial_qubits,
            gammas=gammas,
            betas=betas,
            non_negligible=False)
    elif isinstance(problem, SKProblem):
        initial_qubits = place_line_on_device(device_name=device_name,
                                              n=problem.graph.number_of_nodes(),
                                              line_placement_strategy='mixed')
        circuit, final_qubits = get_compiled_sk_model_circuit(
            problem=problem,
            qubits=initial_qubits,
            gammas=gammas,
            betas=betas,
            non_negligible=False)
    elif isinstance(problem, ThreeRegularProblem):
        device = recirq.get_device_obj_by_name(device_name=device_name)
        (initial_qubits, circuit,
         final_qubits) = get_compiled_3_regular_maxcut_circuit(problem=problem,
                                                               device=device,
                                                               gammas=gammas,
                                                               betas=betas)
    else:
        raise ValueError("Unknown problem: {}".format(problem))
    return initial_qubits, circuit, final_qubits
