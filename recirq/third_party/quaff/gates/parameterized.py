from typing import Any, Iterable, Optional, Sequence

import cirq
import sympy
from recirq.third_party.quaff import cz_layer
from recirq.third_party.quaff.gates.truncated_clifford import TruncatedCliffordGate


def get_truncated_clifford_resolver(
    gate: TruncatedCliffordGate, suffix: Optional[Any] = None
):
    if suffix is None:
        suffix = ""
    else:
        suffix = "_" + str(suffix)

    resolver = {}
    for i, h in enumerate(gate.H):
        resolver[f"h_{i}{suffix}"] = h

    linear, quadratic = gate._CZ_layer()._get_phases()

    for i, s in enumerate((linear + gate._full_P()) % 4):
        resolver[f"s_{i}{suffix}"] = s
    for i, row in enumerate(quadratic):
        for j, u in enumerate(row):
            resolver[f"u_{i}_{j}{suffix}"] = u

    return resolver


def get_truncated_cliffords_resolver(gates: Iterable[TruncatedCliffordGate]):
    resolver = {}
    for i, gate in enumerate(gates):
        resolver.update(get_truncated_clifford_resolver(gate, i))
    return resolver


def get_parameterized_truncated_clifford_ops(
    qubits, suffix: Optional[Any] = None
) -> cirq.OP_TREE:
    if suffix is None:
        suffix = ""
    else:
        suffix = "_" + str(suffix)
    n = len(qubits)
    for i, q in enumerate(qubits):
        yield cirq.H(q) ** (1 - sympy.Symbol(f"h_{i}{suffix}"))
    for i, q in enumerate(qubits):
        yield cirq.S(q) ** sympy.Symbol(f"s_{i}{suffix}")

    basis_change_op = cz_layer.CZLayerBasisChangeStageGate(n, 2)(*qubits)
    for i in range(n // 2):
        yield basis_change_op
        for j, q in enumerate(qubits):
            yield cirq.S(q) ** sympy.Symbol(f"u_{i}_{j}{suffix}")

    yield cz_layer.CZLayerBasisChangeStageGate(n, 1 + (n % 2))(*qubits)
    for q in qubits:
        yield cirq.H(q)


def get_parameterized_truncated_cliffords_ops(
    partition: Sequence[Sequence[cirq.Qid]],
) -> cirq.OP_TREE:
    for i, part in enumerate(partition):
        yield get_parameterized_truncated_clifford_ops(part, i)
