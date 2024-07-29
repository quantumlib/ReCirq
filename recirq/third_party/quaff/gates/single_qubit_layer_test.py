import cirq
import numpy as np
import pytest
from recirq.third_party import quaff


def test_single_qubit_layer_gate_bad():
    with pytest.raises(ValueError):
        quaff.SingleQubitLayerGate([cirq.CNOT])


@pytest.mark.parametrize(
    "num_qubits, seed",
    [(n, quaff.random_seed()) for n in range(1, 6) for _ in range(2)],
)
def test_single_qubit_layer_gate_pow(num_qubits, seed):
    rng = np.random.default_rng(seed)
    gates = [cirq.I, cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.T]
    gates = rng.choice(gates, size=num_qubits)
    exponent = 10 * rng.random() - 5

    gate = quaff.SingleQubitLayerGate(gates)

    expected = gate**exponent
    actual = quaff.SingleQubitLayerGate([G**exponent for G in gates])
    assert expected == actual


def test_single_qubit_layer_gate_eq_bad():
    gate = quaff.SingleQubitLayerGate([cirq.X])
    with pytest.raises(TypeError):
        _ = gate == 3
    other = quaff.SingleQubitLayerGate([cirq.X, cirq.Z])
    assert gate != other
    assert gate != cirq.X


def test_single_qubit_layer_gate_str():
    gate = quaff.SingleQubitLayerGate([cirq.H, cirq.X])
    assert str(gate) == "(cirq.H, cirq.X)"
