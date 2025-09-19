from typing import Iterable

import cirq


class SingleQubitLayerGate(cirq.Gate):
    def __init__(self, gates: Iterable[cirq.testing.SingleQubitGate]):
        self.gates = tuple(gates)
        for gate in gates:
            if cirq.num_qubits(gate) != 1:
                raise ValueError(f"{gate} is not a single-qubit gate.")

    def _unitary_(self):
        return cirq.kron(*(cirq.unitary(gate) for gate in self.gates))

    def _decompose_(self, qubits):
        for qubit, gate in zip(qubits, self.gates):
            yield gate(qubit)

    def num_qubits(self):
        return len(self.gates)

    def __pow__(self, exponent):
        return type(self)(gate**exponent for gate in self.gates)

    def __str__(self):
        return str(self.gates)

    def __eq__(self, other):
        if not isinstance(other, cirq.Gate):
            raise TypeError
        if not isinstance(other, type(self)):
            return False
        if self.num_qubits() != other.num_qubits():
            return False
        return all(gate == other_gate for gate, other_gate in zip(self.gates, other.gates))
