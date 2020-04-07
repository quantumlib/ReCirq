import networkx as nx
import numpy as np
import pytest

import cirq
from recirq.qaoa.circuit_structure import validate_well_structured
from recirq.qaoa.problems import random_plus_minus_1_weights, HardwareGridProblem
from recirq.qaoa.problem_circuits import get_routed_hardware_grid_circuit, \
    get_compiled_hardware_grid_circuit, get_generic_qaoa_circuit


def test_get_generic_qaoa_circuit():
    problem_graph = nx.gnp_random_graph(n=6, p=0.5)
    nx.set_edge_attributes(problem_graph, 1, name='weight')
    qubits = cirq.GridQubit.rect(2, 3, 10, 10)

    circuit = get_generic_qaoa_circuit(problem_graph=problem_graph,
                                       qubits=qubits,
                                       gammas=[0.1, 0.2],
                                       betas=[0.3, 0.4])
    # Hadamard, then p = 2 problem+driver unitaries
    assert len(circuit) == 1 + (2 * 2)


def _random_grid_model(x, y, rs):
    connectivity_graph = nx.grid_2d_graph(x, y)
    coordinates = [(i, j) for i, j in connectivity_graph.nodes]
    problem_graph = nx.convert_node_labels_to_integers(connectivity_graph)
    problem_graph = random_plus_minus_1_weights(problem_graph, rs)
    return HardwareGridProblem(problem_graph, coordinates)


def test_get_routed_grid_model_circuit():
    problem = _random_grid_model(2, 3, np.random.RandomState(0))
    qubits = cirq.GridQubit.rect(2, 3)
    circuit = get_routed_hardware_grid_circuit(
        problem_graph=problem.graph,
        qubits=qubits,
        coordinates=problem.coordinates,
        gammas=[np.pi / 2, np.pi / 4],
        betas=[np.pi / 2, np.pi / 4],
    )

    cirq.testing.assert_has_diagram(circuit, """
  (0, 0)   (0, 1)   (0, 2)   (1, 0)   (1, 1)   (1, 2)
  │        │        │        │        │        │
  H        H        H        H        H        H
  │        │        │        │        │        │
  ZZ───────ZZ       │        ZZ───────ZZ       │
  │        │        │        │        │        │
  │        ZZ───────ZZ       │        ZZ───────ZZ
  │        │        │        │        │        │
┌╴│        │        │        │        │        │       ╶┐
│ ZZ───────┼────────┼────────ZZ       │        │        │
│ │        ZZ───────┼────────┼────────ZZ       │        │
│ │        │        ZZ───────┼────────┼────────ZZ       │
└╴│        │        │        │        │        │       ╶┘
  │        │        │        │        │        │
  │        │        │        │        │        │
  │        │        │        │        │        │
  Rx(π)    Rx(π)    Rx(π)    Rx(π)    Rx(π)    Rx(π)
  │        │        │        │        │        │
  ZZ───────ZZ^0.5   │        ZZ───────ZZ^0.5   │
  │        │        │        │        │        │
  │        ZZ───────ZZ^-0.5  │        ZZ───────ZZ^0.5
  │        │        │        │        │        │
┌╴│        │        │        │        │        │       ╶┐
│ ZZ───────┼────────┼────────ZZ^-0.5  │        │        │
│ │        ZZ───────┼────────┼────────ZZ^0.5   │        │
│ │        │        ZZ───────┼────────┼────────ZZ^0.5   │
└╴│        │        │        │        │        │       ╶┘
  │        │        │        │        │        │
  │        │        │        │        │        │
  │        │        │        │        │        │
  Rx(0.5π) Rx(0.5π) Rx(0.5π) Rx(0.5π) Rx(0.5π) Rx(0.5π)
  │        │        │        │        │        │
""", transpose=True)


def test_get_compiled_grid_model_circuit():
    problem = _random_grid_model(2, 2, np.random.RandomState(0))
    qubits = cirq.GridQubit.rect(2, 2)
    circuit, final_qubits = get_compiled_hardware_grid_circuit(
        problem, qubits,
        gammas=[np.pi / 2, np.pi / 4],
        betas=[np.pi / 2, np.pi / 4],
    )
    assert final_qubits == qubits
    validate_well_structured(circuit)

    rc = get_routed_hardware_grid_circuit(
        problem.graph, qubits, problem.coordinates,
        gammas=[np.pi / 2, np.pi / 4],
        betas=[np.pi / 2, np.pi / 4],
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        rc + cirq.measure(*qubits), circuit, atol=1e-5)

    cirq.testing.assert_has_diagram(circuit, """
  (0, 0)            (0, 1)             (1, 0)            (1, 1)
  │                 │                  │                 │
  PhX(-0.44)^(4/9)  PhX(-0.5)^(7/12)   PhX(-0.44)^(4/9)  PhX(-0.5)^(7/12)
  │                 │                  │                 │
  SYC───────────────SYC                SYC───────────────SYC
  │                 │                  │                 │
  │                 PhX(-0.59)         │                 PhX(-0.59)
  │                 │                  │                 │
  SYC───────────────SYC                SYC───────────────SYC
  │                 │                  │                 │
  PhX(0.52)^(11/13) PhX(0.73)^(1/16)   PhX(0.88)^(15/16) PhX(0.36)^(2/13)
  │                 │                  │                 │
┌╴│                 │                  │                 │                ╶┐
│ SYC───────────────┼──────────────────SYC               │                 │
│ │                 SYC────────────────┼─────────────────SYC               │
└╴│                 │                  │                 │                ╶┘
  │                 │                  │                 │
  │                 │                  PhX(0.62)         PhX(0.2)
  │                 │                  │                 │
┌╴│                 │                  │                 │                ╶┐
│ SYC───────────────┼──────────────────SYC               │                 │
│ │                 SYC────────────────┼─────────────────SYC               │
└╴│                 │                  │                 │                ╶┘
  │                 │                  │                 │
  PhX(0.97)^(7/15)  PhX(-0.69)^(12/13) PhX(0.33)^(2/3)   PhX(0.19)^(12/13)
  │                 │                  │                 │
  SYC───────────────SYC                SYC───────────────SYC
  │                 │                  │                 │
  │                 PhX(-0.058)^0.48   │                 PhX(0.24)^0.48
  │                 │                  │                 │
  SYC───────────────SYC                SYC───────────────SYC
  │                 │                  │                 │
  PhX(0.44)^0.48    PhX(0.89)^(7/12)   PhX(-0.84)^(5/12) PhX(-0.92)^(1)
  │                 │                  │                 │
┌╴│                 │                  │                 │                ╶┐
│ SYC───────────────┼──────────────────SYC               │                 │
│ │                 SYC────────────────┼─────────────────SYC               │
└╴│                 │                  │                 │                ╶┘
  │                 │                  │                 │
  │                 │                  PhX(0.94)^0.48    PhX(0.81)^0.48
  │                 │                  │                 │
┌╴│                 │                  │                 │                ╶┐
│ SYC───────────────┼──────────────────SYC               │                 │
│ │                 SYC────────────────┼─────────────────SYC               │
└╴│                 │                  │                 │                ╶┘
  │                 │                  │                 │
  PhX(-0.64)^0.5    PhX(0.73)^(1/12)   PhX(0.49)^0.5     PhX(-0.55)^0.5
  │                 │                  │                 │
  M('z')────────────M──────────────────M─────────────────M
  │                 │                  │                 │
""", transpose=True, precision=2)
