import os
import time
import timeit
from typing import Optional, Iterable, Union, List

import cirq
import numpy as np
import recirq
from recirq.qaoa.circuit_structure import BadlyStructuredCircuitError, \
    find_circuit_structure_violations
from recirq.qaoa.classical_angle_optimization import OptimizationResult
from recirq.qaoa.experiments.angle_precomputation_tasks import AnglePrecomputationTask, \
    DEFAULT_BASE_DIR as DEFAULT_PRECOMPUTATION_BASE_DIR
from recirq.qaoa.experiments.problem_generation_tasks import \
    DEFAULT_BASE_DIR as DEFAULT_PROBLEM_GENERATION_BASE_DIR, ProblemGenerationTaskT, \
    SKProblemGenerationTask
from recirq.qaoa.gates_and_compilation import compile_to_non_negligible
from recirq.qaoa.placement import place_line_on_device
from recirq.qaoa.problem_circuits import get_compiled_hardware_grid_circuit, \
    get_compiled_sk_model_circuit, get_compiled_3_regular_maxcut_circuit
from recirq.qaoa.problems import ProblemT, HardwareGridProblem, SKProblem, ThreeRegularProblem

EXPERIMENT_NAME = 'qaoa-p1-landscape'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


def _abbrev_n_shots(n_shots: int) -> str:
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class P1LandscapeDataCollectionTask:
    """Collect data for a p=1 landscape

    This task does exactly one (beta, gamma) point. You will have
    to run many of these tasks to get a full landscape

    See Also:
        :py:func:`collect_p1_landscape_data`

    Attributes:
        dataset_id: A unique identifier for this dataset.
        generation_task: The task specifying the problem to collect data for
        device_name: The device to run on
        n_shots: The number of shots to take
        gamma: The problem unitary parameter gamma
        beta: The driver unitary parameter beta
        line_placement_strategy: Only used for SK model problems. Options
            include 'brute_force', 'random', 'greedy', 'anneal', 'mst',
            and 'mixed'.
    """
    dataset_id: str
    generation_task: ProblemGenerationTaskT

    device_name: str
    n_shots: int
    gamma: float
    beta: float
    line_placement_strategy: Optional[str] = None

    @property
    def fn(self):
        n_shots = _abbrev_n_shots(n_shots=self.n_shots)
        if isinstance(self.generation_task, SKProblemGenerationTask):
            line_placement_strategy = self.line_placement_strategy or 'mixed'
            line_placement = f'_lp-{line_placement_strategy}'
        else:
            line_placement = ''

        return (f'{self.dataset_id}/'
                f'{self.device_name}/'
                f'from-{self.generation_task.fn}/'
                f'gamma-{self.gamma:.6f}_beta-{self.beta:.6f}_shots-{n_shots}{line_placement}')


async def collect_p1_landscape_data(
        task: P1LandscapeDataCollectionTask,
        base_dir=None,
        problem_generation_base_dir=None) -> None:
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if problem_generation_base_dir is None:
        problem_generation_base_dir = DEFAULT_PROBLEM_GENERATION_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    generation_task = task.generation_task
    problem = recirq.load(generation_task, base_dir=problem_generation_base_dir)[
        'problem']  # type: ProblemT
    sampler = recirq.get_sampler_by_name(device_name=task.device_name)
    device = recirq.get_device_obj_by_name(device_name=task.device_name)
    if isinstance(problem, HardwareGridProblem):
        initial_qubits = [cirq.GridQubit(r, c) for r, c in problem.coordinates]
        circuit, final_qubits = get_compiled_hardware_grid_circuit(
            problem=problem,
            qubits=initial_qubits,
            gammas=[task.gamma],
            betas=[task.beta],
            non_negligible=False)
    elif isinstance(problem, SKProblem):
        initial_qubits = place_line_on_device(
            device_name=task.device_name,
            n=problem.graph.number_of_nodes(),
            line_placement_strategy='mixed')
        circuit, final_qubits = get_compiled_sk_model_circuit(
            problem=problem,
            qubits=initial_qubits,
            gammas=[task.gamma],
            betas=[task.beta],
            non_negligible=False)
    elif isinstance(problem, ThreeRegularProblem):
        initial_qubits, circuit, final_qubits = get_compiled_3_regular_maxcut_circuit(
            problem=problem,
            device=device,
            gammas=[task.gamma],
            betas=[task.beta])
    else:
        raise ValueError("Unknown problem: {}".format(problem))

    flipped_circuit = circuit.copy()
    measurement_op = flipped_circuit[-1].operations[0]
    flipped_circuit = flipped_circuit[:-1]
    flipped_circuit.append(cirq.Moment(cirq.X.on_each(*measurement_op.qubits)))
    flipped_circuit.append(
        measurement_op.gate.with_bits_flipped(*range(problem.graph.number_of_nodes())).on(
            *measurement_op.qubits))
    unmodified_n_shots = task.n_shots // 2
    flipped_n_shots = task.n_shots - unmodified_n_shots

    t0 = timeit.default_timer()
    circuit.program_id = task.fn
    unmodified_result = await sampler.run_async(program=circuit,
                                                repetitions=unmodified_n_shots)
    circuit.program_id = task.fn + '-flip'
    flipped_result = await sampler.run_async(program=flipped_circuit,
                                             repetitions=flipped_n_shots)
    t1 = timeit.default_timer()
    result = unmodified_result + flipped_result
    execution_time = t1 - t0
    print(f'Circuit execution time: {execution_time} s')

    t0 = timeit.default_timer()
    bitstrings = result.measurements['z']
    recirq.save(task=task, data={
        'bitstrings': recirq.BitArray(bitstrings),
        'qubits': initial_qubits,
        'final_qubits': final_qubits,
        'execution_time': execution_time
    }, base_dir=base_dir)
    t1 = timeit.default_timer()
    print(f'Time to save bitstrings: {t1 - t0} s')
    print()


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class ReadoutCalibrationTask:
    dataset_id: str
    device_name: str
    n_shots: int

    # Readout calibration is loosely coupled to data collection
    # set these to make sure unique calibrations get run when you want
    i: int
    epoch: str = None

    @property
    def fn(self):
        n_shots = _abbrev_n_shots(n_shots=self.n_shots)
        if self.epoch is None:
            epoch = ""
        else:
            epoch = f'{self.epoch}-'

        return (f'{self.dataset_id}/'
                f'{self.device_name}/'
                f'ro/'
                f'{epoch}i-{self.i}_{n_shots}')


async def _estimate_single_qubit_readout_errors_async(
        sampler: 'cirq.Sampler',
        *,
        qubits: Iterable['cirq.Qid'],
        repetitions: int = 1000) -> cirq.experiments.SingleQubitReadoutCalibrationResult:
    """Estimate single-qubit readout error.

    TODO: Commit an async version of this function to Cirq.

    For each qubit, prepare the |0⟩ state and measure. Calculate how often a 1
    is measured. Also, prepare the |1⟩ state and calculate how often a 0 is
    measured. The state preparations and measurements are done in parallel,
    i.e., for the first experiment, we actually prepare every qubit in the |0⟩
    state and measure them simultaneously.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: The qubits being tested.
        repetitions: The number of measurement repetitions to perform.

    Returns:
        A SingleQubitReadoutCalibrationResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.
    """
    qubits = list(qubits)

    if isinstance(sampler, cirq.DensityMatrixSimulator):
        # Do each qubit individually to make this simulable
        zero_state_errors = dict()
        one_state_errors = dict()
        for qubit in qubits:
            zero_circuit = cirq.Circuit(cirq.measure(qubit, key=repr(qubit)))
            one_circuit = cirq.Circuit(cirq.X(qubit),
                                       cirq.measure(qubit, key=repr(qubit)))
            zero_result = await sampler.run_async(zero_circuit, repetitions=repetitions)
            one_result = await sampler.run_async(one_circuit, repetitions=repetitions)
            zero_state_errors[qubit] = np.mean(zero_result.measurements[repr(qubit)])
            one_state_errors[qubit] = 1 - np.mean(one_result.measurements[repr(qubit)])
    else:
        zeros_circuit = cirq.Circuit(cirq.measure_each(*qubits, key_func=repr))
        ones_circuit = cirq.Circuit(cirq.X.on_each(*qubits),
                                    cirq.measure_each(*qubits, key_func=repr))

        zeros_result = await sampler.run_async(zeros_circuit, repetitions=repetitions)
        ones_result = await sampler.run_async(ones_circuit, repetitions=repetitions)

        zero_state_errors = {
            q: np.mean(zeros_result.measurements[repr(q)]) for q in qubits
        }
        one_state_errors = {
            q: 1 - np.mean(ones_result.measurements[repr(q)]) for q in qubits
        }

    timestamp = time.time()
    return cirq.experiments.SingleQubitReadoutCalibrationResult(
        zero_state_errors=zero_state_errors,
        one_state_errors=one_state_errors,
        repetitions=repetitions,
        timestamp=timestamp)


async def collect_readout_calibration(task: ReadoutCalibrationTask, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    sampler = recirq.get_sampler_by_name(device_name=task.device_name)
    device = recirq.get_device_obj_by_name(device_name=task.device_name)

    readout_calibration_result = await _estimate_single_qubit_readout_errors_async(
        sampler=sampler,
        qubits=sorted(device.qubits),
        repetitions=task.n_shots)

    recirq.save(task=task, data={
        'calibration': readout_calibration_result,
    }, base_dir=base_dir)


def interleave_ro_tasks(tasks: List[P1LandscapeDataCollectionTask],
                        n_shots: int, epoch: Optional[str], freq: int = 50) \
        -> List[Union[P1LandscapeDataCollectionTask, ReadoutCalibrationTask]]:
    """Interleave readout calibration tasks into the data collection tasks."""

    if freq is None:
        return tasks.copy()

    new_tasks = []
    for i, task in enumerate(tasks):
        if i % freq == 0:
            new_tasks += [ReadoutCalibrationTask(
                dataset_id=task.dataset_id,
                device_name=task.device_name,
                n_shots=n_shots,
                i=i,
                epoch=epoch,
            )]
        new_tasks += [task]
    return new_tasks


def get_data_collection_tasks_on_a_grid(*,
                                        dataset_id: str,
                                        pgen_task: ProblemGenerationTaskT,
                                        device_name: str,
                                        n_shots: int = 50_000,
                                        n_ro_shots: int = 1_000_000,
                                        gamma_res: int = 31,
                                        beta_res: int = 31,
                                        epoch: Optional[str] = None,
                                        ro_freq: int = 50) -> List[
    Union[P1LandscapeDataCollectionTask, ReadoutCalibrationTask]]:
    """Return a list of P1LandscapeDataCollectionTasks on a grid of (gamma,beta) points.

    This also interleaves ReadoutCalibration tasks.
    Use :py:func:`collect_either_landscape_or_cal` function to run the task.

    Args:
        dataset_id: A dataset id for the data collection tasks
        pgen_task: The problem generation task specifying the problem we wish to run.
        device_name: The device to collect data on
        n_shots: The number of shots to take during ordinary data collection
        n_ro_shots: The number of shots to take during readout calibration.
            Typically, this should be oversampled to provide an accurate
            correction.
        gamma_res: How many points to take along the gamma axis (resolution)
        beta_res: How many points to take along the beta axis (resolution).
            The total number of tasks is gamma_res * beta_res
        epoch: If you're using this function multiple times with a shared
            dataset_id (e.g. if you're running multiple problems) give each
            a unique `epoch` identifier so readout calibrations can be
            uniquely addressed
        ro_freq: The frequency (in number of data collection tasks) to perform
            readout correction
    """
    gamma_betas = np.array(np.meshgrid(
        np.linspace(0, np.pi / 2, gamma_res),
        np.linspace(-np.pi / 4, np.pi / 4, beta_res)
    )).T.reshape(-1, 2)
    rs = np.random.RandomState(52)
    rs.shuffle(gamma_betas)

    data_collection_tasks = [
        P1LandscapeDataCollectionTask(
            dataset_id=dataset_id,
            generation_task=pgen_task,
            device_name=device_name,
            n_shots=n_shots,
            gamma=gamma,
            beta=beta,
        )
        for gamma, beta in gamma_betas
    ]
    data_collection_tasks = interleave_ro_tasks(data_collection_tasks,
                                                n_shots=n_ro_shots,
                                                epoch=epoch,
                                                freq=ro_freq)
    return data_collection_tasks


async def collect_either_landscape_or_cal(
        task: Union[P1LandscapeDataCollectionTask, ReadoutCalibrationTask]):
    if isinstance(task, ReadoutCalibrationTask):
        return await collect_readout_calibration(task=task)
    elif isinstance(task, P1LandscapeDataCollectionTask):
        return await collect_p1_landscape_data(task=task)
    raise ValueError(task.fn)
