import random

import networkx as nx
import numpy as np
import pytest

import cirq
from cirq.testing import random_special_unitary
from recirq.qaoa.circuit_structure import validate_well_structured
from recirq.qaoa.gates_and_compilation import ZZSwap, compile_problem_unitary_to_zzswap, \
    ProblemUnitary, DriverUnitary, SwapNetworkProblemUnitary, \
    compile_problem_unitary_to_swap_network, compile_swap_network_to_zzswap, \
    single_qubit_matrix_to_phased_x_z_const_depth, zzswap_as_syc, zz_as_syc, \
    compile_driver_unitary_to_rx, compile_single_qubit_gates, compile_to_syc, \
    measure_with_final_permutation, compile_out_virtual_z, compile_to_non_negligible, \
    _hardware_graph, compile_problem_unitary_to_hardware_graph
from recirq.qaoa.problems import random_plus_minus_1_weights


def test_zz_swap():
    q1, q2 = cirq.LineQubit.range(2)
    circuit1 = cirq.Circuit(ZZSwap(zz_exponent=0.123).on(q1, q2))
    u1 = circuit1.unitary(dtype=np.complex128)
    circuit2 = cirq.Circuit(
        cirq.ZZ(q1, q2) ** 0.123,
        cirq.SWAP(q1, q2)
    )
    u2 = circuit2.unitary(dtype=np.complex128)
    circuit3 = cirq.Circuit(
        cirq.SWAP(q1, q2),
        cirq.ZZ(q1, q2) ** 0.123
    )
    u3 = circuit3.unitary(dtype=np.complex128)

    np.testing.assert_allclose(u1, u2)
    np.testing.assert_allclose(u2, u3)
    np.testing.assert_allclose(u3, u1)


def test_compile_problem_unitary_to_zzswap():
    n = 5
    q = cirq.LineQubit.range(n)
    problem = nx.complete_graph(n=n)
    problem = random_plus_minus_1_weights(problem)

    gamma = 0.123
    circuit1 = cirq.Circuit()
    for i1, i2, w in problem.edges.data('weight'):
        circuit1.append(
            cirq.ZZPowGate(exponent=2 * gamma * w / np.pi, global_shift=-0.5).on(q[i1], q[i2]))
    u1 = circuit1.unitary(dtype=np.complex128)

    circuit2 = cirq.Circuit(
        compile_problem_unitary_to_zzswap(problem_graph=problem, gamma=gamma, qubits=q),
        compile_problem_unitary_to_zzswap(problem_graph=problem, gamma=0, qubits=q))
    u2 = circuit2.unitary(dtype=np.complex128)
    np.testing.assert_allclose(u1, u2)


def test_problem_unitary():
    n = 5
    q = cirq.LineQubit.range(n)
    problem = nx.complete_graph(n=n)
    problem = random_plus_minus_1_weights(problem, rs=np.random.RandomState(52))
    gamma = 0.151
    problem_unitary = ProblemUnitary(problem_graph=problem, gamma=gamma)
    u1 = cirq.Circuit(problem_unitary.on(*q)).unitary(qubit_order=q, dtype=np.complex128)

    circuit = cirq.Circuit()
    for i1, i2, w in problem.edges.data('weight'):
        circuit.append(
            cirq.ZZPowGate(exponent=2 * gamma * w / np.pi, global_shift=-0.5).on(q[i1], q[i2]))
    u2 = circuit.unitary(dtype=np.complex128)

    np.testing.assert_allclose(u1, u2)


def test_swap_network_problem_unitary():
    n = 5
    q = cirq.LineQubit.range(n)
    problem = nx.complete_graph(n=n)
    problem = random_plus_minus_1_weights(problem, rs=np.random.RandomState(52))
    gamma = 0.151
    spu = SwapNetworkProblemUnitary(problem_graph=problem, gamma=gamma)
    u1 = cirq.Circuit(spu.on(*q)).unitary(qubit_order=q, dtype=np.complex128)

    circuit = cirq.Circuit()
    for i1, i2, w in problem.edges.data('weight'):
        circuit.append(
            cirq.ZZPowGate(exponent=2 * gamma * w / np.pi, global_shift=-0.5).on(q[i1], q[i2]))
    circuit += cirq.QubitPermutationGate(list(range(n))[::-1]).on(*q)
    u2 = circuit.unitary(dtype=np.complex128)
    np.testing.assert_allclose(u1, u2)


def test_compile_problem_unitary_to_swap_network_p1():
    n = 5
    q = cirq.LineQubit.range(n)
    problem = nx.complete_graph(n=n)
    problem = random_plus_minus_1_weights(problem)

    c1 = cirq.Circuit(ProblemUnitary(problem_graph=problem, gamma=0.123).on(*q))
    c2 = compile_problem_unitary_to_swap_network(c1)
    assert c1 != c2
    assert isinstance(c2.moments[-1].operations[0].gate, cirq.QubitPermutationGate)

    u1 = c1.unitary(dtype=np.complex128)
    u2 = c2.unitary(dtype=np.complex128)
    np.testing.assert_allclose(u1, u2)


def test_compile_problem_unitary_to_swap_network_p2():
    n = 5
    q = cirq.LineQubit.range(n)
    problem = nx.complete_graph(n=n)
    problem = random_plus_minus_1_weights(problem)

    c1 = cirq.Circuit(
        ProblemUnitary(problem_graph=problem, gamma=0.123).on(*q),
        ProblemUnitary(problem_graph=problem, gamma=0.321).on(*q),
    )
    c2 = compile_problem_unitary_to_swap_network(c1)
    assert c1 != c2
    assert not isinstance(c2.moments[-1].operations[0].gate, cirq.QubitPermutationGate)
    u1 = c1.unitary(dtype=np.complex128)
    u2 = c2.unitary(dtype=np.complex128)
    np.testing.assert_allclose(u1, u2)


def test_compile_swap_network_to_zzswap():
    n = 5
    q = cirq.LineQubit.range(n)
    problem = nx.complete_graph(n=n)
    problem = random_plus_minus_1_weights(problem)

    c1 = cirq.Circuit(ProblemUnitary(problem_graph=problem, gamma=0.123).on(*q))
    c2 = compile_problem_unitary_to_swap_network(c1)
    assert c1 != c2
    c3 = compile_swap_network_to_zzswap(c2)
    assert c2 != c3

    u1 = c1.unitary(dtype=np.complex128)
    u3 = c3.unitary(dtype=np.complex128)
    np.testing.assert_allclose(u1, u3)


def test_compile_sk_problem_unitary_to_zzswap_2():
    n = 5
    q = cirq.LineQubit.range(n)
    problem = nx.complete_graph(n=n)
    problem = random_plus_minus_1_weights(problem)

    c1 = cirq.Circuit(
        ProblemUnitary(problem_graph=problem, gamma=0.123).on(*q),
        DriverUnitary(num_qubits=n, beta=0.456).on(*q),
        ProblemUnitary(problem_graph=problem, gamma=0.789).on(*q),
    )
    c2 = compile_problem_unitary_to_swap_network(c1)
    assert c1 != c2
    c3 = compile_swap_network_to_zzswap(c2)
    assert c2 != c3

    u1 = c1.unitary(dtype=np.complex128)
    u3 = c3.unitary(dtype=np.complex128)
    np.testing.assert_allclose(u1, u3)


def test_hardware_graph():
    coordinates = [
        (10, 10), (10, 11),
        (9, 10), (9, 11),
    ]
    problem = nx.from_edgelist([
        (0, 1),  # 10,10 - 10,11
        (1, 3),  # 10,11 - 9,11
        (0, 2),  # 10,10 - 9,10
        (2, 3),  # 9,10, 9,11
    ])
    nx.set_edge_attributes(problem, 1, name='weight')

    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(_hardware_graph(problem, 0.123, coordinates, qubits))
    assert circuit.to_text_diagram(transpose=True) == """  0  1        2        3
  │  │        │        │
  ZZ─ZZ^0.078 ZZ───────ZZ^0.078
  │  │        │        │
  │  │        │        │
  │  │        │        │
┌╴│  │        │        │       ╶┐
│ ZZ─┼────────ZZ^0.078 │        │
│ │  ZZ───────┼────────ZZ^0.078 │
└╴│  │        │        │       ╶┘
  │  │        │        │
  │  │        │        │
  │  │        │        │"""


def test_compile_problem_unitary_to_hardware_graph():
    problem = nx.grid_2d_graph(3, 3)
    coordinates = sorted(problem.nodes)
    problem = nx.relabel_nodes(problem, {coord: i for i, coord in enumerate(coordinates)})
    problem = random_plus_minus_1_weights(problem)
    qubits = cirq.LineQubit.range(problem.number_of_nodes())

    c1 = cirq.Circuit(ProblemUnitary(problem, gamma=random.random()).on(*qubits))
    c2 = compile_problem_unitary_to_hardware_graph(c1, coordinates)
    assert c1 != c2

    np.testing.assert_allclose(c1.unitary(dtype=np.complex128), c2.unitary(dtype=np.complex128))


def test_driver_unitary():
    n = 5
    q = cirq.LineQubit.range(n)
    du = DriverUnitary(num_qubits=n, beta=0.123).on(*q)
    circuit = cirq.Circuit(cirq.rx(2 * 0.123).on_each(*q))

    u1 = cirq.Circuit(du).unitary(qubit_order=q, dtype=np.complex128)
    u2 = circuit.unitary(dtype=np.complex128)
    np.testing.assert_allclose(u1, u2)


def test_compile_driver_unitary_to_rx():
    n = 5
    q = cirq.LineQubit.range(n)
    c1 = cirq.Circuit(DriverUnitary(num_qubits=n, beta=0.123).on(*q))
    c2 = cirq.Circuit(cirq.rx(2 * 0.123).on_each(*q))
    c3 = compile_driver_unitary_to_rx(c1)

    u1 = c1.unitary(dtype=np.complex128)
    u2 = c2.unitary(dtype=np.complex128)
    u3 = c3.unitary(dtype=np.complex128)
    np.testing.assert_allclose(u1, u2)
    np.testing.assert_allclose(u1, u3)


def test_single_q_const_depth():
    su = random_special_unitary(2)
    ops = single_qubit_matrix_to_phased_x_z_const_depth(su)
    assert len(ops) == 2
    circuit = cirq.Circuit([op.on(cirq.LineQubit(1)) for op in ops])
    u2 = circuit.unitary(dtype=np.complex128)
    cirq.testing.assert_allclose_up_to_global_phase(su, u2, atol=1e-8)


def test_compile_single_qubit_gates():
    q = cirq.LineQubit(0)
    c1 = cirq.Circuit()
    for _ in range(10):
        c1.append(random.choice([cirq.X, cirq.Y, cirq.Z])(q) ** random.random())

    c2 = compile_single_qubit_gates(c1)
    assert c1 != c2
    assert len(c2) == 2
    assert isinstance(c2[0].operations[0].gate, cirq.PhasedXPowGate)
    assert isinstance(c2[1].operations[0].gate, cirq.ZPowGate)

    u1 = c1.unitary(dtype=np.complex128)
    u2 = c2.unitary(dtype=np.complex128)
    cirq.testing.assert_allclose_up_to_global_phase(u1, u2, atol=1e-8)


def test_zzswap_as_syc():
    q1, q2 = cirq.LineQubit.range(2)
    zzs = ZZSwap(zz_exponent=np.random.rand(1))
    circuit = zzswap_as_syc(zzs.theta, q1, q2)
    assert len(circuit) == 3 * 3 + 2

    u1 = cirq.Circuit(zzs.on(q1, q2)).unitary(qubit_order=(q1, q2), dtype=np.complex128)
    u2 = circuit.unitary(dtype=np.complex128)
    cirq.testing.assert_allclose_up_to_global_phase(u1, u2, atol=1e-8)


@pytest.mark.skip("KAK instability")
def test_zzswap_as_syc_2():
    q1, q2 = cirq.LineQubit.range(2)
    zzs = ZZSwap(zz_exponent=0.123)
    circuit = zzswap_as_syc(zzs.theta, q1, q2)
    assert str(circuit) == """\
0: ───PhX(0.145)^(0)───Z^-0.2──────SYC───PhX(0.214)^0.576───Z^-0.131───SYC─────────────────────────SYC───PhX(-0.0833)^0.576───Z^-0.548───
                                   │                                   │                           │
1: ───PhX(0.973)^(0)───Z^(-1/14)───SYC───PhX(-0.369)────────Z^0.869────SYC───PhX(0.0)^0.52───Z^0───SYC───PhX(-0.394)──────────Z^0.036────\
"""


def test_zz_as_syc():
    q1, q2 = cirq.LineQubit.range(2)
    zz = cirq.ZZPowGate(exponent=np.random.rand(1))
    circuit = zz_as_syc(zz.exponent * np.pi / 2, q1, q2)
    assert len(circuit) == 3 * 2 + 2

    u1 = cirq.Circuit(zz.on(q1, q2)).unitary(qubit_order=(q1, q2), dtype=np.complex128)
    u2 = circuit.unitary(dtype=np.complex128)
    cirq.testing.assert_allclose_up_to_global_phase(u1, u2, atol=1e-8)


@pytest.mark.skip("KAK instability")
def test_zz_as_syc_2():
    q1, q2 = cirq.LineQubit.range(2)
    zz = cirq.ZZPowGate(exponent=0.123)
    circuit = zz_as_syc(zz.exponent * np.pi / 2, q1, q2)
    assert len(circuit) == 3 * 2 + 2
    validate_well_structured(circuit)
    cirq.testing.assert_has_diagram(circuit, """
0: ───PhX(1)^0.483───Z^(1/12)─────SYC────────────────────────SYC───PhX(0.917)^0.483───Z^(1/12)───
                                  │                          │
1: ───PhX(-0.583)────Z^(-11/12)───SYC───PhX(0)^0.873───Z^0───SYC───PhX(0)─────────────T^-1───────
""")


@pytest.mark.parametrize("p", [1, 2, 3])
def test_compile_to_syc(p):
    problem = nx.complete_graph(n=5)
    problem = random_plus_minus_1_weights(problem)
    qubits = cirq.LineQubit.range(5)
    c1 = cirq.Circuit(
        cirq.H.on_each(qubits),
        [
            [
                ProblemUnitary(problem, gamma=np.random.random()).on(*qubits),
                DriverUnitary(5, beta=np.random.random()).on(*qubits)
            ]
            for _ in range(p)
        ]
    )
    c2 = compile_problem_unitary_to_swap_network(c1)
    c3 = compile_swap_network_to_zzswap(c2)
    c4 = compile_driver_unitary_to_rx(c3)
    c5 = compile_to_syc(c4)
    validate_well_structured(c5, allow_terminal_permutations=True)

    np.testing.assert_allclose(c1.unitary(dtype=np.complex128), c2.unitary(dtype=np.complex128))
    np.testing.assert_allclose(c1.unitary(dtype=np.complex128), c3.unitary(dtype=np.complex128))
    np.testing.assert_allclose(c1.unitary(dtype=np.complex128), c4.unitary(dtype=np.complex128))
    # Single qubit throws out global phase
    cirq.testing.assert_allclose_up_to_global_phase(
        c1.unitary(dtype=np.complex128), c5.unitary(dtype=np.complex128), atol=1e-8)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_measure_with_final_permutation(p):
    problem = nx.complete_graph(n=5)
    problem = random_plus_minus_1_weights(problem)
    qubits = cirq.LineQubit.range(5)
    c1 = cirq.Circuit(
        cirq.H.on_each(qubits),
        [
            [
                ProblemUnitary(problem, gamma=np.random.random()).on(*qubits),
                DriverUnitary(5, beta=np.random.random()).on(*qubits)
            ]
            for _ in range(p)
        ]
    )
    c2 = compile_problem_unitary_to_swap_network(c1)
    c3 = compile_swap_network_to_zzswap(c2)
    c4 = compile_driver_unitary_to_rx(c3)
    c5 = compile_to_syc(c4)
    validate_well_structured(c5, allow_terminal_permutations=True)
    c6, final_qubits = measure_with_final_permutation(c5, qubits)
    validate_well_structured(c6, allow_terminal_permutations=False)

    if p % 2 == 1:
        assert final_qubits == qubits[::-1]
    else:
        assert final_qubits == qubits

    permutation = []
    for q in qubits:
        permutation.append(final_qubits.index(q))
    c1_prime = (
            c1
            + cirq.QubitPermutationGate(permutation).on(*qubits)
            + cirq.measure(*qubits, key='z')
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c1_prime, c6, atol=1e-5)


def test_compile_out_virtual_z():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(ZZSwap(zz_exponent=0.123).on(*qubits))
    c1 = compile_to_syc(circuit)
    assert len(c1) == 3 * 3 + 2
    c2 = compile_out_virtual_z(c1)
    assert c1 != c2
    assert len(c2) == 3 * 2 + 2


def test_structured():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(ZZSwap(zz_exponent=0.123).on(*qubits))
    circuit = compile_to_syc(circuit)
    validate_well_structured(circuit)


def test_compile_to_non_negligible():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(ZZSwap(zz_exponent=0.123).on(*qubits))
    c1 = compile_to_syc(circuit)
    validate_well_structured(c1)
    assert len(c1) == 3 * 3 + 2

    c2 = compile_to_non_negligible(c1)
    assert c1 != c2
    # KAK instability (https://github.com/quantumlib/Cirq/issues/1647)
    # means the first layer of PhX gets removed on mpharrigan's machine
    # but not in docker / in CI.
    assert len(c2) in [9, 10]
