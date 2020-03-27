import networkx as nx
import numpy as np
import pytest

import cirq
from recirq.qaoa.circuit_structure import validate_well_structured
from recirq.qaoa.problems import random_plus_minus_1_weights
from recirq.qaoa.problem_circuits import old_grid_model_to_new_problem, \
    get_routed_grid_model_circuit, get_compiled_hardware_grid_circuit

def test_get_generic_qaoa_circuit():



    circuit = get_generic_qaoa_circuit(problem=problem, qubits=qubits, gammas=[0.1, 0.2], betas=[0.3, 0.4])
    print(circuit)


def _random_grid_model_old(x, y, rs):
    connectivity_graph = nx.grid_2d_graph(x, y)
    graph = nx.Graph()
    for (i, j), (k, l) in connectivity_graph.edges:
        graph.add_edge(cirq.GridQubit(i, j), cirq.GridQubit(k, l))
    return random_plus_minus_1_weights(graph, rs)


def test_get_routed_grid_model_circuit():
    graph = _random_grid_model_old(2, 3, np.random.RandomState(0))
    problem, qubits, coordinates = old_grid_model_to_new_problem(graph)
    circuit = get_routed_grid_model_circuit(
        problem, qubits, coordinates,
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
  │        ZZ───────ZZ^0.5   │        ZZ───────ZZ^0.5
  │        │        │        │        │        │
┌╴│        │        │        │        │        │       ╶┐
│ ZZ───────┼────────┼────────ZZ^-0.5  │        │        │
│ │        ZZ───────┼────────┼────────ZZ^-0.5  │        │
│ │        │        ZZ───────┼────────┼────────ZZ^0.5   │
└╴│        │        │        │        │        │       ╶┘
  │        │        │        │        │        │
  │        │        │        │        │        │
  │        │        │        │        │        │
  Rx(0.5π) Rx(0.5π) Rx(0.5π) Rx(0.5π) Rx(0.5π) Rx(0.5π)
  │        │        │        │        │        │
""", transpose=True)


def test_get_compiled_grid_model_circuit():
    graph = _random_grid_model_old(2, 2, np.random.RandomState(0))
    problem, qubits, coordinates = old_grid_model_to_new_problem(graph)
    circuit, final_qubits = get_compiled_hardware_grid_circuit(
        problem, qubits,
        gammas=[np.pi / 2, np.pi / 4],
        betas=[np.pi / 2, np.pi / 4],
    )
    assert final_qubits == qubits
    validate_well_structured(circuit)

    rc = get_routed_grid_model_circuit(
        problem, qubits, coordinates,
        gammas=[np.pi / 2, np.pi / 4],
        betas=[np.pi / 2, np.pi / 4],
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        rc + cirq.measure(*qubits), circuit, atol=1e-5)

    cirq.testing.assert_has_diagram(circuit, """
  (0, 0)            (0, 1)           (1, 0)            (1, 1)
  │                 │                │                 │
  PhX(0.023)^(3/7)  PhX(0.07)^0.48   PhX(0.023)^(3/7)  PhX(0.07)^0.48
  │                 │                │                 │
  SYC───────────────SYC              SYC───────────────SYC
  │                 │                │                 │
  │                 PhX(0.029)       │                 PhX(0.029)
  │                 │                │                 │
  SYC───────────────SYC              SYC───────────────SYC
  │                 │                │                 │
  PhX(0.54)^0.045   PhX(0.25)^0.9    PhX(0.39)^0.1     PhX(0.4)^(6/7)
  │                 │                │                 │
┌╴│                 │                │                 │              ╶┐
│ SYC───────────────┼────────────────SYC               │               │
│ │                 SYC──────────────┼─────────────────SYC             │
└╴│                 │                │                 │              ╶┘
  │                 │                │                 │
  │                 │                PhX(0.051)        PhX(0.61)
  │                 │                │                 │
┌╴│                 │                │                 │              ╶┐
│ SYC───────────────┼────────────────SYC               │               │
│ │                 SYC──────────────┼─────────────────SYC             │
└╴│                 │                │                 │              ╶┘
  │                 │                │                 │
  PhX(0.82)^(10/11) PhX(0.12)^0.48   PhX(0.077)^0.028  PhX(0.23)^(4/7)
  │                 │                │                 │
  SYC───────────────SYC              SYC───────────────SYC
  │                 │                │                 │
  │                 PhX(0.0072)^0.48 │                 PhX(0.3)^0.48
  │                 │                │                 │
  SYC───────────────SYC              SYC───────────────SYC
  │                 │                │                 │
  PhX(0.51)^0.48    PhX(0.38)^(7/12) PhX(-0.78)^(5/12) PhX(-0.7)
  │                 │                │                 │
┌╴│                 │                │                 │              ╶┐
│ SYC───────────────┼────────────────SYC               │               │
│ │                 SYC──────────────┼─────────────────SYC             │
└╴│                 │                │                 │              ╶┘
  │                 │                │                 │
  │                 │                PhX(-0.99)^0.48   PhX(0.3)^0.48
  │                 │                │                 │
┌╴│                 │                │                 │              ╶┐
│ SYC───────────────┼────────────────SYC               │               │
│ │                 SYC──────────────┼─────────────────SYC             │
└╴│                 │                │                 │              ╶┘
  │                 │                │                 │
  PhX(-0.58)^0.5    PhX(0.72)^0.5    PhX(0.049)^0.5    PhX(-0.62)^0.5
  │                 │                │                 │
  M('z')────────────M────────────────M─────────────────M
  │                 │                │                 │
""", transpose=True, precision=2)
