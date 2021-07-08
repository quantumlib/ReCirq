import dataclasses
import itertools

import cirq
import numpy as np
import pytest
from cirq import TiltedSquareLattice

from recirq.cirqflow.quantum_executable import QuantumExecutable, QuantumExecutableGroup, Bitstrings


def _get_random_circuit(qubits, n_moments=10, op_density=0.8, random_state=52):
    return cirq.testing.random_circuit(qubits, n_moments=n_moments, op_density=op_density,
                                       random_state=random_state)


def get_all_diagonal_rect_topologies(min_side_length=2, max_side_length=8):
    width_heights = np.arange(min_side_length, max_side_length + 1)
    return [TiltedSquareLattice(width, height)
            for width, height in itertools.combinations_with_replacement(width_heights, r=2)]


def test_executable():
    qubits = cirq.LineQubit.range(10)
    exe = QuantumExecutable(
        spec={'name': 'example-program'},
        circuit=_get_random_circuit(qubits),
        measurement=Bitstrings(n_repetitions=10),
    )

    # Check args get turned into immutable fields
    # assert exe.info == (
    #     ('name', 'example-program'),
    # )
    assert isinstance(exe.circuit, cirq.FrozenCircuit)

    # Uses guid field since object is immutable
    assert hash(exe) is not None

    # But you could theoretically use the fields (it's just slower)
    assert hash(dataclasses.astuple(exe)) is not None
    assert hash(dataclasses.astuple(exe)) == exe._hash

    prog2 = QuantumExecutable(
        spec={'name': 'example-program'},
        circuit=_get_random_circuit(qubits),
        measurement=Bitstrings(n_repetitions=10),
    )
    assert exe == prog2
    assert hash(exe) == hash(prog2)

    prog3 = QuantumExecutable(
        spec={'name': 'example-program'},
        circuit=_get_random_circuit(qubits),
        measurement=Bitstrings(n_repetitions=20),  # note: changed n_repetitions
    )
    assert exe != prog3
    assert hash(exe) != hash(prog3)

    with pytest.raises(dataclasses.FrozenInstanceError):
        prog3.measurement.n_repetitions = 10

    # assert str(exe) == "QuantumExecutable(info={'name': 'example-program'})"
    # assert repr(exe) == str(exe)


def test_native_rep():
    rs = np.random.RandomState(52)
    depths = np.arange(2, 8 + 1, 2)
    min_side_length = 1
    max_side_length = 3
    n_instances = 0
    n_repetitions = 10_000

    exegroup1 = QuantumExecutableGroup(
        info=(('name', 'recirq.quantum_executable_test'),),
        executables=tuple(
            QuantumExecutableGroup(
                info=(('topology', topo),),
                executables=tuple(
                    QuantumExecutableGroup(
                        info=(('depth', depth),),
                        executables=tuple(
                            QuantumExecutable(
                                info=(('instance_i', instance_i),),
                                problem_topology=topo,
                                circuit=_get_random_circuit(
                                    topo.qubits(), n_moments=depth, random_state=rs),
                                measurement=Bitstrings(n_repetitions=n_repetitions),
                            )
                            for instance_i in range(n_instances)
                        )
                    )
                    for depth in depths
                )
            )
            for topo in get_all_diagonal_rect_topologies(min_side_length=min_side_length,
                                                         max_side_length=max_side_length)
        )
    )
    assert hash(dataclasses.astuple(exegroup1)) == exegroup1._hash

    rs = np.random.RandomState(52)
    exegroup2 = QuantumExecutableGroup(
        info={'name': 'recirq.quantum_executable_test'},
        executables=[
            QuantumExecutableGroup(
                info={'topology': topo},
                executables=[
                    QuantumExecutableGroup(
                        info={'depth': depth},
                        executables=[
                            QuantumExecutable(
                                info={'instance_i': instance_i},
                                problem_topology=topo,
                                circuit=_get_random_circuit(
                                    topo.qubits(), n_moments=depth, random_state=rs),
                                measurement=Bitstrings(n_repetitions=n_repetitions),
                            )
                            for instance_i in range(n_instances)
                        ]
                    )
                    for depth in depths
                ]
            )
            for topo in get_all_diagonal_rect_topologies(min_side_length=min_side_length,
                                                         max_side_length=max_side_length)
        ]
    )

    assert repr(exegroup1) == repr(exegroup2)
    assert hash(exegroup1) == hash(exegroup2)
    assert exegroup1 == exegroup2
