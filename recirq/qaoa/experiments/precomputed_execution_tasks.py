import os

import cirq
import recirq
from recirq.qaoa.circuit_structure import BadlyStructuredCircuitError, \
    find_circuit_structure_violations
from recirq.qaoa.classical_angle_optimization import OptimizationResult
from recirq.qaoa.experiments.angle_precomputation_tasks import AnglePrecomputationTask, \
    DEFAULT_BASE_DIR as DEFAULT_PRECOMPUTATION_BASE_DIR
from recirq.qaoa.experiments.problem_generation_tasks import \
    DEFAULT_BASE_DIR as DEFAULT_PROBLEM_GENERATION_BASE_DIR
from recirq.qaoa.gates_and_compilation import compile_to_non_negligible
from recirq.qaoa.placement import place_line_on_device
from recirq.qaoa.problem_circuits import get_compiled_hardware_grid_circuit, \
    get_compiled_sk_model_circuit, get_compiled_3_regular_maxcut_circuit
from recirq.qaoa.problems import ProblemT, HardwareGridProblem, SKProblem, ThreeRegularProblem

EXPERIMENT_NAME = 'qaoa-precomputed'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


def _abbrev_n_shots(n_shots: int) -> str:
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class PrecomputedDataCollectionTask:
    dataset_id: str
    precomputation_task: AnglePrecomputationTask

    device_name: str
    n_shots: int
    structured: bool = False
    echoed: bool = False

    @property
    def fn(self):
        n_shots = _abbrev_n_shots(n_shots=self.n_shots)
        generation_task = self.precomputation_task.generation_task
        if self.echoed:
            assert self.structured, 'must be structured'
            struc_echo = '_echoed'
        elif self.structured:
            struc_echo = '_structured'
        else:
            struc_echo = ''

        return (f'{self.dataset_id}/'
                f'{self.device_name}/'
                f'from-{generation_task.fn}/'
                f'p-{self.precomputation_task.p}_{n_shots}{struc_echo}')


async def collect_data(task: PrecomputedDataCollectionTask,
                       base_dir=None,
                       problem_generation_base_dir=None,
                       precomputation_base_dir=None,
                       ):
    """Collect and save data for the experiment specified by params.

    The associated problem generation data must already exist.
    """
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if problem_generation_base_dir is None:
        problem_generation_base_dir = DEFAULT_PROBLEM_GENERATION_BASE_DIR

    if precomputation_base_dir is None:
        precomputation_base_dir = DEFAULT_PRECOMPUTATION_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    precompute_task = task.precomputation_task
    generation_task = precompute_task.generation_task

    problem = recirq.load(generation_task, base_dir=problem_generation_base_dir)[
        'problem']  # type: ProblemT
    optimum = recirq.load(precompute_task, base_dir=precomputation_base_dir)[
        'optimum']  # type: OptimizationResult
    sampler = recirq.get_sampler_by_name(device_name=task.device_name)
    device = recirq.get_device_obj_by_name(device_name=task.device_name)

    try:
        if isinstance(problem, HardwareGridProblem):
            initial_qubits = [cirq.GridQubit(r, c) for r, c in problem.coordinates]
            circuit, final_qubits = get_compiled_hardware_grid_circuit(
                problem=problem,
                qubits=initial_qubits,
                gammas=optimum.gammas,
                betas=optimum.betas,
                non_negligible=False)
        elif isinstance(problem, SKProblem):
            initial_qubits = place_line_on_device(
                device_name=task.device_name,
                n=problem.graph.number_of_nodes(),
                line_placement_strategy='mixed')
            circuit, final_qubits = get_compiled_sk_model_circuit(
                problem=problem,
                qubits=initial_qubits,
                gammas=optimum.gammas,
                betas=optimum.betas,
                non_negligible=False)
        elif isinstance(problem, ThreeRegularProblem):
            initial_qubits, circuit, final_qubits = get_compiled_3_regular_maxcut_circuit(
                problem=problem,
                device=device,
                gammas=optimum.gammas,
                betas=optimum.betas)
        else:
            raise ValueError("Unknown problem: {}".format(problem))
    except BadlyStructuredCircuitError:
        print("!!!! Badly structured circuit: {}".format(task))
        # TODO https://github.com/quantumlib/Cirq/issues/2553
        return

    if not task.structured:
        # Left align
        circuit = compile_to_non_negligible(circuit)
        circuit = cirq.Circuit(circuit.all_operations())

    if task.echoed:
        assert task.structured
        raise NotImplementedError("To be implemented in follow-up PR")

    violation_indices = find_circuit_structure_violations(circuit)
    circuit.program_id = task.fn
    result = await sampler.run_async(program=circuit,
                                     repetitions=task.n_shots)
    bitstrings = result.measurements['z']

    recirq.save(task=task, data={
        'bitstrings': recirq.BitArray(bitstrings),
        'qubits': initial_qubits,
        'final_qubits': final_qubits,
        'circuit': circuit,
        'violation_indices': violation_indices,
    }, base_dir=base_dir)
    print(f"{task.fn} complete.")
