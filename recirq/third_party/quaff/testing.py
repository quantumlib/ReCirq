from typing import Any, Tuple

import cirq
import numpy as np
from quaff import linalg


def assert_consistent_unitary_round_trip(num_qubits, gate_type, seed):
    gate = gate_type.random(num_qubits, seed=seed)

    U = gate._unitary_()
    assert np.allclose(U @ U.T.conjugate(), np.eye(len(U)))

    reconstructed_gate = gate_type.from_unitary(U)
    reconstructed_U = reconstructed_gate._unitary_()
    assert np.allclose(reconstructed_U @ reconstructed_U.T.conjugate(), np.eye(len(U)))

    assert np.allclose(U, reconstructed_U)


def assert_equivalent_repr(value: Any):
    print("VALUE")
    print(value, type(value))
    return cirq.testing.assert_equivalent_repr(value)


def is_boolean_tuple_of_len(obj: Any, n: int):
    return len(obj) == n and all(0 <= x < 2 for x in obj)


def is_nested_boolean_tuples_of_shape(obj: Any, shape: Tuple[int, ...]):
    array = np.array(obj, dtype=linalg.DTYPE)
    if not is_boolean_array_of_shape(array, shape):
        return False
    return linalg.tuplify(array) == obj


def is_boolean_array_of_shape(obj: Any, shape: Tuple[int, ...]):
    if not isinstance(obj, np.ndarray):
        return False
    if obj.shape != shape:
        return False
    if obj.dtype != linalg.DTYPE:
        return False
    if (obj < 0).any():
        return False
    if (obj > 1).any():
        return False
    return True


def complex_formatter(val) -> str:
    if np.isclose(val, 0):
        return "  0"
    formatted = ""
    for part, suffix in zip((val.real, val.imag), ("", "j")):
        if np.isclose(part, 0):
            continue
        if np.isclose(part, round(part)):
            part = round(part)
        sign = "+" if part > 0 else "-"
        formatted += (sign + str(abs(part)) + suffix).rjust(3)
    if formatted[0] == "+":
        return " " + formatted[1:]
    return formatted
