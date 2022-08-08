from collections import defaultdict
from typing import Sequence, List, Optional, Tuple, Dict, Iterator

import networkx as nx
import numpy as np

import cirq
from recirq.qaoa.problems import _validate_problem_graph

# TODO(mpharrigan): We should change this to use non-private functions
try:
    from cirq_google.optimizers.convert_to_sycamore_gates import swap_rzz, rzz
except ImportError:
    from cirq_google.transformers.analytical_decompositions.two_qubit_to_sycamore import (
        _swap_rzz as swap_rzz,
        _rzz as rzz
    )

from recirq.qaoa.circuit_structure import validate_well_structured


class ZZSwap(cirq.Gate):
    """A composite ZZPowGate followed by a SWAP.

    Used as a building-block for the QAOA linear swap network for
    fully-connected problems."
    """

    def __init__(self, *,
                 zz_exponent: float,
                 zz_global_shift: float = 0):
        self._zz_gate = cirq.ZZPowGate(exponent=zz_exponent,
                                       global_shift=zz_global_shift)
        self._swap_gate = cirq.SWAP

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        assert len(qubits) == 2
        yield self._zz_gate.on(*qubits)
        yield self._swap_gate.on(*qubits)

    @property
    def theta(self):
        return self._zz_gate.exponent * np.pi / 2

    def _circuit_diagram_info_(
            self,
            args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return cirq.CircuitDiagramInfo(
            wire_symbols=('zzswap', f't={self._zz_gate.exponent:.3f}'))


def compile_problem_unitary_to_zzswap(
        problem_graph: nx.Graph,
        gamma: float,
        qubits: Sequence[cirq.Qid],
) -> Iterator[cirq.Operation]:
    """Yield ZZSwap operations to implement the linear swap network.

    ZZ exponents will be set according to 2*gamma*weight/pi where
    weight is the edge weight from problem_graph.
    """
    n_qubits = len(qubits)
    order = list(range(n_qubits))

    for layer_num in range(n_qubits):
        lowest_active_qubit = layer_num % 2
        active_pairs = ((i, i + 1)
                        for i in range(lowest_active_qubit, n_qubits - 1, 2))
        for i, j in active_pairs:
            p, q = order[i], order[j]
            weight = problem_graph[p][q]['weight']
            yield ZZSwap(
                zz_exponent=2 * gamma * weight / np.pi,
                zz_global_shift=-0.5,
            ).on(qubits[i], qubits[j])
            order[i], order[j] = q, p


class ProblemUnitary(cirq.Gate):
    def __init__(self, problem_graph: nx.Graph, gamma: float):
        """An n-qubit gate representing the full problem unitary for
        problem_graph applied with the given gamma value."""

        _validate_problem_graph(problem_graph)
        self.problem_graph = problem_graph
        self.gamma = gamma

    def _num_qubits_(self) -> int:
        return self.problem_graph.number_of_nodes()

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        for i1, i2, weight in self.problem_graph.edges.data('weight'):
            q0 = qubits[i1]
            q1 = qubits[i2]
            yield cirq.ZZPowGate(
                exponent=2 * self.gamma * weight / np.pi, global_shift=-0.5).on(q0, q1)

    def _circuit_diagram_info_(
            self,
            args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        excess_q = self.num_qubits() - 2
        return cirq.CircuitDiagramInfo(
            wire_symbols=('problem', f'g={self.gamma:.3f}') + tuple(
                f'#{i + 2 + 1}' for i in range(excess_q)))


class SwapNetworkProblemUnitary(ProblemUnitary):
    """A ProblemUnitary with classical permutation of indices afterwards."""

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        yield from super()._decompose_(qubits)
        yield cirq.QubitPermutationGate(list(range(len(qubits)))[::-1]).on(*qubits)

    def _circuit_diagram_info_(
            self,
            args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        excess_q = self.num_qubits() - 2
        return cirq.CircuitDiagramInfo(
            wire_symbols=('swap-network', f'g={self.gamma:.3f}') + tuple(
                f'#{i + 2 + 1}' for i in range(excess_q)))


def compile_problem_unitary_to_swap_network(circuit: cirq.Circuit) -> cirq.Circuit:
    """Compile ProblemUnitary's in the input circuit to
    SwapNetworkProblemUnitary's with appropriate bookkeeping for permutation
    that will happen during the swap network."""
    permutation = {q: q for q in circuit.all_qubits()}

    new_moments = []
    for moment in circuit.moments:
        permuted_moment = moment.transform_qubits(lambda q: permutation[q])
        new_ops = []
        for op in permuted_moment.operations:
            if (op.gate is not None and isinstance(op.gate, ProblemUnitary)
                    and not isinstance(op.gate, SwapNetworkProblemUnitary)):
                gate = op.gate  # type: ProblemUnitary
                new_gate = SwapNetworkProblemUnitary(problem_graph=gate.problem_graph,
                                                     gamma=gate.gamma)
                new_op = new_gate.on(*op.qubits)
                new_ops.append(new_op)

                qubits = op.qubits
                nq = len(qubits)
                qs_to_permute = {qubits[i]: qubits[nq - i - 1] for i in range(nq)}
                permutation = {q_from: qs_to_permute.get(q_to, permutation[q_to])
                               for q_from, q_to in permutation.items()}

            else:
                new_ops.append(op)
        new_moments.append(cirq.Moment(new_ops))

    needs_permute = sorted(q_from
                           for q_from, q_to in permutation.items()
                           if q_from != q_to)
    # Gotta find indices
    permute_i = []
    for q_from in needs_permute:
        q_to = permutation[q_from]
        permute_i.append(needs_permute.index(q_to))

    # And tack it on
    if len(permute_i) > 0:
        new_moments.append(cirq.Moment([
            cirq.QubitPermutationGate(permute_i).on(*needs_permute)
        ]))

    return cirq.Circuit(new_moments)


class _SwapNetworkToZZSWAP(cirq.PointOptimizer):
    """Circuit optimizer to turn a high-level swap network object to ZZSwap gates.

    Prefer to use :py:func:`compile_swap_network_to_zzswap`, which wraps this
    object.
    """

    def optimization_at(
            self,
            circuit: 'cirq.Circuit',
            index: int,
            op: 'cirq.Operation'
    ) -> Optional[cirq.PointOptimizationSummary]:
        if isinstance(op.gate, SwapNetworkProblemUnitary):
            gate = op.gate  # type: SwapNetworkProblemUnitary
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=compile_problem_unitary_to_zzswap(
                    gate.problem_graph, gate.gamma, op.qubits)
            )


def compile_swap_network_to_zzswap(circuit: cirq.Circuit, *, mutate=False) -> cirq.Circuit:
    """Compile a circuit containing SwapNetworkProblemUnitary's to one
    using ZZSwap interactions."""
    if mutate:
        c2 = circuit
    else:
        c2 = circuit.copy()
    _SwapNetworkToZZSWAP().optimize_circuit(c2)
    return c2


def _hardware_graph(problem_graph: nx.Graph, gamma: float,
                    node_coordinates: List[Tuple[int, int]],
                    qubits: Sequence[cirq.Qid]) -> Iterator[cirq.Moment]:
    """Used by compile_problem_unitary_to_hardware_graph.

    Activates links according to node_coordinates (and using the weights
    from problem_graph). Yield four moments (corresponding to degree-4 grid).
    """
    row_start = min(r for r, c in node_coordinates)
    row_end = max(r for r, c in node_coordinates) + 1
    col_start = min(c for r, c in node_coordinates)
    col_end = max(c for r, c in node_coordinates) + 1

    coord_to_i = {coord: i for i, coord in enumerate(node_coordinates)}

    def _interaction(row_start_offset=0, row_end_offset=0, row_step=1,
                     col_start_offset=0, col_end_offset=0, col_step=1,
                     get_neighbor=lambda row, col: (row, col)):
        for row in range(row_start + row_start_offset, row_end + row_end_offset, row_step):
            for col in range(col_start + col_start_offset, col_end + col_end_offset, col_step):
                coord1 = (row, col)
                if coord1 not in node_coordinates:
                    continue
                coord2 = get_neighbor(row, col)
                if coord2 not in node_coordinates:
                    continue
                node1 = coord_to_i[coord1]
                node2 = coord_to_i[coord2]
                if (node1, node2) not in problem_graph.edges:
                    continue

                weight = problem_graph.edges[node1, node2]['weight']

                yield cirq.ZZPowGate(exponent=2 * gamma * weight / np.pi, global_shift=-0.5) \
                    .on(qubits[node1], qubits[node2])

    # Horizontal
    yield cirq.Moment(_interaction(
        col_start_offset=0, col_end_offset=-1, col_step=2,
        get_neighbor=lambda row, col: (row, col + 1)))
    yield cirq.Moment(_interaction(
        col_start_offset=1, col_end_offset=-1, col_step=2,
        get_neighbor=lambda row, col: (row, col + 1)))
    # Vertical
    yield cirq.Moment(_interaction(
        row_start_offset=0, row_end_offset=-1, row_step=2,
        get_neighbor=lambda row, col: (row + 1, col)))
    yield cirq.Moment(_interaction(
        row_start_offset=1, row_end_offset=-1, row_step=2,
        get_neighbor=lambda row, col: (row + 1, col)))


class _ProblemUnitaryToHardwareGraph(cirq.PointOptimizer):
    """Optimizer to compile a hardware grid problem to a hardware graph.

    Prefer to use `compile_problem_unitary_to_hardware_graph`, which
    wraps this object.
    """

    def __init__(self, node_coordinates: List[Tuple[int, int]]):
        super().__init__()
        self._node_coordinates = node_coordinates

    def optimize_circuit(self, circuit: cirq.Circuit):
        # Note: this includes a lot of the functionality of cirq.PointOptimizer
        # duplicated so that this object preserves moment structure.
        # https://github.com/quantumlib/Cirq/issues/2406
        frontier: Dict[cirq.Qid, int] = defaultdict(lambda: 0)
        i = 0
        while i < len(circuit):  # Note: circuit may mutate as we go.
            for op in circuit[i].operations:
                # Don't touch stuff inserted by previous optimizations.
                if any(frontier[q] > i for q in op.qubits):
                    continue

                # Skip if an optimization removed the circuit underneath us.
                if i >= len(circuit):
                    continue
                # Skip if an optimization removed the op we're considering.
                if op not in circuit[i].operations:
                    continue
                opt = self.optimization_at(circuit, i, op)
                # Skip if the optimization did nothing.
                if opt is None:
                    continue

                # Clear target area, and insert new operations.
                circuit.clear_operations_touching(
                    opt.clear_qubits,
                    [e for e in range(i, i + opt.clear_span)])

                # Drop empty moments
                e = 0
                e_max = opt.clear_span
                while e < e_max:
                    if len(circuit._moments[i + e]) == 0:
                        circuit._moments.pop(i + e)
                        e_max -= 1
                    else:
                        e += 1

                # Insert
                circuit.insert(i, opt.new_operations)

            i += 1

    def optimization_at(
            self,
            circuit: 'cirq.Circuit',
            index: int,
            op: 'cirq.Operation'
    ) -> Optional[cirq.PointOptimizationSummary]:
        if isinstance(op.gate, ProblemUnitary):
            gate = op.gate  # type: ProblemUnitary
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=_hardware_graph(gate.problem_graph, gate.gamma,
                                               self._node_coordinates, op.qubits),
                preserve_moments=True,
            )


def compile_problem_unitary_to_hardware_graph(
        circuit: cirq.Circuit,
        node_coordinates: List[Tuple[int, int]],
        *,
        mutate=False) -> cirq.Circuit:
    """Compile ProblemUnitary gates to ZZPowGate on a grid

    Args:
        circuit: The circuit
        node_coordinates: A list which maps 0-indexed node indices to
            coordinates on a grid; used for determining the order
            of application of the ZZ operations in ProblemUnitary
        mutate: By default, return a copy of the circuit. Otherwise,
            mutate in place.
    """
    if mutate:
        c2 = circuit
    else:
        c2 = circuit.copy()
    _ProblemUnitaryToHardwareGraph(node_coordinates).optimize_circuit(c2)
    return c2


def _problem_to_zz(problem_graph: nx.Graph, qubits: Sequence[cirq.Qid], gamma: float):
    """Helper function used by `compile_problem_unitary_to_arbitrary_zz`."""
    for i1, i2, weight in problem_graph.edges.data('weight'):
        q0 = qubits[i1]
        q1 = qubits[i2]
        yield cirq.ZZPowGate(
            exponent=2 * gamma * weight / np.pi, global_shift=-0.5).on(q0, q1)


class _ProblemUnitaryToZZ(cirq.PointOptimizer):
    """An optimizer which compiles arbitrary problem graphs to ZZPowGate
    operations without regard for connectivity.

    Prefer using `compile_problem_unitary_to_arbitrary_zz`, which wraps this
    object.
    """

    def optimization_at(
            self,
            circuit: 'cirq.Circuit',
            index: int,
            op: 'cirq.Operation'
    ) -> Optional[cirq.PointOptimizationSummary]:
        if isinstance(op.gate, ProblemUnitary):
            gate = op.gate  # type: ProblemUnitary
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=_problem_to_zz(
                    problem_graph=gate.problem_graph,
                    qubits=op.qubits,
                    gamma=gate.gamma),
            )


def compile_problem_unitary_to_arbitrary_zz(
        circuit: cirq.Circuit,
        *,
        mutate=False) -> cirq.Circuit:
    """Compile ProblemUnitary gates to ZZPowGate without regard for qubit
    connectivity.

    Args:
        circuit: The circuit
        mutate: By default, return a copy of the circuit. Otherwise,
            mutate in place.
    """
    if mutate:
        c2 = circuit
    else:
        c2 = circuit.copy()
    _ProblemUnitaryToZZ().optimize_circuit(c2)
    return c2


def _rx(rads):
    # Cirq > 0.10 introduced bespoke cirq.Rx gate which is a descendant
    # of XPowGate. Unfortunately, this has broken pytket which checks
    # for type equality.
    return cirq.XPowGate(exponent=rads / np.pi, global_shift=-0.5)


class DriverUnitary(cirq.Gate):
    """An N-body gate which applies the QAOA driver unitary with
    parameter `beta` to all qubits."""

    def __init__(self, num_qubits: int, beta: float):
        self._num_qubits = num_qubits
        self.beta = beta

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        yield _rx(2 * self.beta).on_each(*qubits)

    def _circuit_diagram_info_(
            self,
            args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        excess_q = self.num_qubits() - 2
        return cirq.CircuitDiagramInfo(
            wire_symbols=('driver', f'b={self.beta:.3f}') + tuple(
                f'#{i + 2 + 1}' for i in range(excess_q)))


class _DriverToRx(cirq.PointOptimizer):
    """Convert an n-qubit driver unitary to n XPowGates.

    Prefer using `compile_driver_unitary_to_rx`, which wraps this object.
    """

    def optimization_at(
            self,
            circuit: 'cirq.Circuit',
            index: int,
            op: 'cirq.Operation'
    ) -> Optional[cirq.PointOptimizationSummary]:
        if isinstance(op.gate, DriverUnitary):
            gate = op.gate  # type: DriverUnitary
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=_rx(2 * gate.beta).on_each(op.qubits)
            )


def compile_driver_unitary_to_rx(circuit: cirq.Circuit, *, mutate=False):
    """Compile DriverUnitary gates to single-qubit XPowGates.

    Args:
        circuit: The circuit
        mutate: By default, return a copy of the circuit. Otherwise,
            mutate in place.
    """
    if mutate:
        c2 = circuit
    else:
        c2 = circuit.copy()
    _DriverToRx().optimize_circuit(c2)
    return c2


def single_qubit_matrix_to_phased_x_z_const_depth(
        mat: np.ndarray
) -> List[cirq.Gate]:
    """Implements a single-qubit operation with a PhasedX and Z gate.

    If one of the gates isn't needed, it will still be included with
    zero exponent. This always returns two gates, in contrast to the
    function in Cirq.

    Args:
        mat: The 2x2 unitary matrix of the operation to implement.
        atol: A limit on the amount of error introduced by the
            construction.

    Returns:
        A list of gates that, when applied in order, perform the desired
            operation.
    """
    from cirq.transformers.analytical_decompositions.single_qubit_decompositions \
        import _deconstruct_single_qubit_matrix_into_gate_turns
    xy_turn, xy_phase_turn, total_z_turn = _deconstruct_single_qubit_matrix_into_gate_turns(mat)

    return [
        cirq.PhasedXPowGate(exponent=2 * xy_turn,
                            phase_exponent=2 * xy_phase_turn),
        cirq.Z ** (2 * total_z_turn)
    ]


class _SingleQubitGates(cirq.PointOptimizer):
    """Optimizes runs of adjacent unitary 1-qubit operations.

    This uses `single_qubit_matrix_to_phased_x_z_const_depth` to make each
    single qubit layer exactly one PhX and one Z gate.
    """

    def __init__(self):
        super().__init__()

    def _rewrite(self, operations: List[cirq.Operation]
                 ) -> Optional[cirq.OP_TREE]:
        if not operations:
            return None
        q = operations[0].qubits[0]
        unitary = cirq.linalg.dot(*(cirq.unitary(op) for op in operations[::-1]))
        out_gates = single_qubit_matrix_to_phased_x_z_const_depth(unitary)
        return [gate(q) for gate in out_gates]

    def optimization_at(self,
                        circuit: cirq.Circuit,
                        index: int,
                        op: cirq.Operation
                        ) -> Optional[cirq.PointOptimizationSummary]:
        if len(op.qubits) != 1:
            return None

        start = {op.qubits[0]: index}
        op_list = circuit.findall_operations_until_blocked(
            start,
            is_blocker=lambda next_op: len(next_op.qubits) != 1)
        operations = [op for idx, op in op_list]
        indices = [idx for idx, op in op_list]
        rewritten = self._rewrite(operations)
        if rewritten is None:
            return None
        return cirq.PointOptimizationSummary(
            clear_span=max(indices) + 1 - index,
            clear_qubits=op.qubits,
            new_operations=rewritten)


def compile_single_qubit_gates(
        circuit: cirq.Circuit
) -> cirq.Circuit:
    """Compile single qubit gates to constant-depth PhX and Z gates

    Args:
        circuit: The circuit
    """
    c2 = circuit.copy()
    _SingleQubitGates().optimize_circuit(c2)
    c2 = cirq.drop_empty_moments(c2)
    return c2


def zzswap_as_syc(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Return a composite Exp[i theta ZZ] SWAP circuit with three SYC gates."""
    swz = cirq.Circuit(swap_rzz(theta, q0, q1))
    _SingleQubitGates().optimize_circuit(swz)
    swz = cirq.drop_empty_moments(swz)
    return swz


def zz_as_syc(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Return an Exp[i theta ZZ] circuit with two SYC gates."""
    swz = cirq.Circuit(rzz(theta, q0, q1))
    _SingleQubitGates().optimize_circuit(swz)
    swz = cirq.drop_empty_moments(swz)
    return swz


class _TwoQubitOperationsAsSYC(cirq.PointOptimizer):
    """Optimizer to compile ZZSwap and ZZPowGate gates into SYC."""

    def optimization_at(
            self,
            circuit: 'cirq.Circuit',
            index: int,
            op: 'cirq.Operation'
    ) -> Optional[cirq.PointOptimizationSummary]:
        if isinstance(op.gate, ZZSwap):
            gate = op.gate  # type: ZZSwap
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=zzswap_as_syc(gate.theta, *op.qubits)
            )
        if isinstance(op.gate, cirq.ZZPowGate):
            gate = op.gate  # type: cirq.ZZPowGate
            theta = gate.exponent * np.pi / 2
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=zz_as_syc(theta, *op.qubits)
            )


def compile_to_syc(circuit: cirq.Circuit) -> cirq.Circuit:
    """Compile a QAOA circuit to SYC gates.

    Args:
        circuit: The circuit
    """
    c2 = circuit.copy()
    _TwoQubitOperationsAsSYC().optimize_circuit(c2)
    _SingleQubitGates().optimize_circuit(c2)
    c2 = cirq.drop_empty_moments(c2)
    return c2


def measure_with_final_permutation(
        circuit: cirq.Circuit,
        qubits: List[cirq.Qid],
        *,
        mutate=False) -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    """Apply a measurement gate at the end of a circuit and classically
    permute qubit indices.

    If the circuit contains a permutation gate at its end, the input
    argument `qubits` will be permuted and returned as the second return
    value.


    Args:
        circuit: The circuit
        qubits: Which qubits to measure
        mutate: By default, return a copy of the circuit. Otherwise,
            mutate in place.

    Returns:
        circuit: The output circuit with measurement
        final_qubits: The input list of qubits permuted according to the
            final permutation gate.
    """
    if mutate:
        c2 = circuit
    else:
        c2 = circuit.copy()

    mom_classes, stats = validate_well_structured(c2, allow_terminal_permutations=True)
    if stats.has_measurement:
        raise ValueError("Circuit already has measurements")

    mapping = {}
    if stats.has_permutation:
        for op in c2.moments[-1].operations:
            if isinstance(op.gate, cirq.QubitPermutationGate):
                # do something with it
                permuted_qs = op.qubits
                gate = op.gate  # type: cirq.QubitPermutationGate
                for i, q in enumerate(permuted_qs):
                    mapping[q] = permuted_qs[gate.permutation[i]]
        c2.moments.pop(-1)

    final_qubits = [mapping.get(q, q) for q in qubits]
    c2.append(cirq.measure(*qubits, key='z'))
    return c2, final_qubits


def compile_out_virtual_z(
        circuit: cirq.Circuit,
        ) -> cirq.Circuit:
    """Eject Z gates from the circuit.

    This is a wrapper around cirq.EjectZ()

    Args:
        circuit: The circuit
    """
    c2 = circuit
    c2 = cirq.eject_z(c2)
    c2 = cirq.drop_empty_moments(c2)
    return c2


def compile_to_non_negligible(
        circuit: cirq.Circuit,
        *,
        tolerance=1e-5,
        ) -> cirq.Circuit:
    """Remove negligible gates from the circuit.

    This is a wrapper around cirq.DropNegligible(tolerance)

    Args:
        circuit: The circuit
        tolerance: Gates with trace distance below this value will be
            considered negligible.
    """
    c2 = circuit
    c2 = cirq.drop_negligible_operations(c2, atol=tolerance)
    c2 = cirq.drop_empty_moments(c2)
    return c2
