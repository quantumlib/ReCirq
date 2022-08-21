from typing import Sequence, Tuple, List

import networkx as nx

import cirq
import cirq_google as cg
from recirq.qaoa.gates_and_compilation import (
    ProblemUnitary,
    DriverUnitary,
    compile_problem_unitary_to_hardware_graph,
    compile_driver_unitary_to_rx,
    compile_to_syc,
    compile_out_virtual_z,
    compile_to_non_negligible,
    validate_well_structured,
    compile_problem_unitary_to_swap_network, compile_swap_network_to_zzswap,
    measure_with_final_permutation, compile_problem_unitary_to_arbitrary_zz)
from recirq.qaoa.placement import place_on_device
from recirq.qaoa.problems import HardwareGridProblem, SKProblem, ThreeRegularProblem


def get_generic_qaoa_circuit(
        problem_graph: nx.Graph,
        qubits: List[cirq.Qid],
        gammas: Sequence[float],
        betas: Sequence[float],
) -> cirq.Circuit:
    """Return a QAOA circuit with 'generic' N-body Problem and Driver unitaries

    Args:
        problem_graph: A graph with contiguous 0-indexed integer nodes and
            edges with a 'weight' attribute.
        qubits: The qubits to use in construction of the circuit.
        gammas: Gamma angles to use as parameters for problem unitaries
        betas: Beta angles to use as parameters for driver unitaries
    """
    layers = [cirq.H.on_each(*qubits)]
    assert len(gammas) == len(betas)
    for gamma, beta in zip(gammas, betas):
        layers.append(ProblemUnitary(problem_graph, gamma=gamma).on(*qubits))
        layers.append(DriverUnitary(num_qubits=len(qubits), beta=beta).on(*qubits))
    circuit = cirq.Circuit(layers)
    return circuit


def get_routed_hardware_grid_circuit(
        problem_graph: nx.Graph,
        qubits: List[cirq.Qid],
        coordinates: List[Tuple[int, int]],
        gammas: Sequence[float],
        betas: Sequence[float]) -> cirq.Circuit:
    """Return a QAOA circuit "routed" according to sequential activation
    of the four links in the degree-4 hardware graph.

    See Also:
        :py:func:`get_compiled_hardware_grid_circuit`

    Args:
        problem_graph: A graph with contiguous 0-indexed integer nodes and
            edges with a 'weight' attribute.
        qubits: The qubits to use in construction of the circuit.
        coordinates: A mapping from problem node to position on a 2D grid to
            help the "routing".
        gammas: Gamma angles to use as parameters for problem unitaries
        betas: Beta angles to use as parameters for driver unitaries
    """
    circuit = get_generic_qaoa_circuit(
        problem_graph=problem_graph,
        qubits=qubits,
        gammas=gammas,
        betas=betas)
    circuit = compile_problem_unitary_to_hardware_graph(circuit, coordinates)
    circuit = compile_driver_unitary_to_rx(circuit)
    return circuit


def get_compiled_hardware_grid_circuit(
        problem: HardwareGridProblem,
        qubits: List[cirq.Qid],
        gammas: Sequence[float],
        betas: Sequence[float],
        non_negligible=True) \
        -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    """Get a fully-compiled grid model circuit.

    Args:
        problem: A HardwareGridProblem
        qubits: The qubits to use in construction of the circuit.
        gammas: Gamma angles to use as parameters for problem unitaries
        betas: Beta angles to use as parameters for driver unitaries
        non_negligible: Whether to compile out negligible gates. This will
            preserve the quality that the returned circuit has homogeneous
            gate types in each moment but may make it so the predictable
            pattern of PhX, (SYC, PhX)*n, PhX may be disrupted. Set this
            argument to `False` if doing echo experiments.

    Returns:
        circuit: The final routed, gateset-targeted, and placed circuit with
            measurements.
        final_qubits: The qubits in their final logical order. This is for
            consistency with other problem types. This will be a copy of
            `qubits`.
    """
    circuit = get_routed_hardware_grid_circuit(
        problem_graph=problem.graph,
        qubits=qubits,
        coordinates=problem.coordinates,
        gammas=gammas,
        betas=betas)
    circuit = compile_to_syc(circuit)
    mcircuit = circuit + cirq.measure(*qubits, key='z')
    mcircuit = compile_out_virtual_z(mcircuit)
    if non_negligible:
        mcircuit = compile_to_non_negligible(mcircuit)
    validate_well_structured(mcircuit)
    final_qubits = qubits.copy()
    return mcircuit, final_qubits


def get_routed_sk_model_circuit(
        problem_graph: nx.Graph,
        qubits: List[cirq.Qid],
        gammas: Sequence[float],
        betas: Sequence[float],
) -> cirq.Circuit:
    """Get a QAOA circuit for a fully-connected problem using the linear swap
    network.

    See Also:
        :py:func:`get_compiled_sk_model_circuit`

    Args:
        problem_graph: A graph with contiguous 0-indexed integer nodes and
            edges with a 'weight' attribute.
        qubits: The qubits to use in construction of the circuit.
        gammas: Gamma angles to use as parameters for problem unitaries
        betas: Beta angles to use as parameters for driver unitaries
    """
    circuit = get_generic_qaoa_circuit(problem_graph, qubits, gammas, betas)
    circuit = compile_problem_unitary_to_swap_network(circuit)
    circuit = compile_swap_network_to_zzswap(circuit)
    circuit = compile_driver_unitary_to_rx(circuit)
    return circuit


def get_compiled_sk_model_circuit(
        problem: SKProblem,
        qubits: List[cirq.Qid],
        gammas: Sequence[float],
        betas: Sequence[float],
        *,
        non_negligible=True
) -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    """Get a fully-compiled SK model circuit.

    Args:
        problem: A SKProblem
        qubits: The qubits to use in construction of the circuit.
        gammas: Gamma angles to use as parameters for problem unitaries
        betas: Beta angles to use as parameters for driver unitaries
        non_negligible: Whether to compile out negligible gates. This will
            preserve the quality that the returned circuit has homogeneous
            gate types in each moment but may make it so the predictable
            pattern of PhX, (SYC, PhX)*n, PhX may be disrupted. Set this
            argument to `False` if doing echo experiments.

    Returns:
        circuit: The final routed, gateset-targeted, and placed circuit with
            measurements.
        final_qubits: The qubits in their final logical order.
    """
    circuit = get_routed_sk_model_circuit(problem.graph, qubits, gammas, betas)
    circuit = compile_to_syc(circuit)
    mcircuit, final_qubits = measure_with_final_permutation(circuit, qubits)
    mcircuit = compile_out_virtual_z(mcircuit)
    if non_negligible:
        mcircuit = compile_to_non_negligible(mcircuit)
    validate_well_structured(mcircuit)
    return mcircuit, final_qubits


def get_routed_3_regular_maxcut_circuit(
        problem_graph: nx.Graph,
        device: cirq.Device,
        gammas: Sequence[float],
        betas: Sequence[float],
) -> Tuple[List[cirq.Qid], cirq.Circuit, List[cirq.Qid]]:
    """Get a routed QAOA circuit for a 3-regular problem.

    See Also:
        :py:func:`get_compiled_3_regular_maxcut_circuit`

    Args:
        problem_graph: A graph with contiguous 0-indexed integer nodes and
            edges with a 'weight' attribute.
        qubits: The qubits to use in construction of the circuit.
        gammas: Gamma angles to use as parameters for problem unitaries
        betas: Beta angles to use as parameters for driver unitaries
    """
    dummy_qubits = cirq.LineQubit.range(problem_graph.number_of_nodes())
    circuit = get_generic_qaoa_circuit(
        problem_graph=problem_graph,
        qubits=dummy_qubits,
        gammas=gammas,
        betas=betas)
    circuit = compile_problem_unitary_to_arbitrary_zz(circuit)
    circuit = compile_driver_unitary_to_rx(circuit)
    circuit, initial_qubit_map, final_qubit_map = place_on_device(circuit, device)
    initial_qubits = [initial_qubit_map[q] for q in dummy_qubits]
    final_qubits = [final_qubit_map[q] for q in dummy_qubits]
    return initial_qubits, circuit, final_qubits


def get_compiled_3_regular_maxcut_circuit(
        problem: ThreeRegularProblem,
        device: cirq.Device,
        gammas: Sequence[float],
        betas: Sequence[float],
) -> Tuple[List[cirq.Qid], cirq.Circuit, List[cirq.Qid]]:
    """Get a fully-compiled SK model circuit.

    Args:
        problem: A ThreeRegularProblem
        device: The device to target
        gammas: Gamma angles to use as parameters for problem unitaries
        betas: Beta angles to use as parameters for driver unitaries

    Returns:
        initial_qubits: The qubits in their initial order
        circuit: The final routed, gateset-targeted, and placed circuit with
            measurements.
        final_qubits: The qubits in their final logical order.
    """
    # TODO: explicitly compile gates, avoid optimized_for_sycamore, make structured circuits
    initial_qubits, circuit, final_qubits = get_routed_3_regular_maxcut_circuit(
        problem_graph=problem.graph,
        device=device,
        gammas=gammas,
        betas=betas)
    circuit.append(cirq.measure(*final_qubits, key='z'))
    circuit = cg.optimized_for_sycamore(circuit, optimizer_type='sycamore')
    return initial_qubits, circuit, final_qubits
