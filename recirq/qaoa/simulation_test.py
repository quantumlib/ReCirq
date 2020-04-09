# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import networkx as nx

import cirq
from recirq.qaoa.problems import random_plus_minus_1_weights
from recirq.qaoa.simulation import multiply_single_spin, evolve_by_HamB, \
    create_ZZ_HamC, ising_qaoa_grad, hamiltonian_objectives, hamiltonian_objective, \
    hamiltonian_objective_avg_and_err


def test_multiply_single_spin():
    N = 6
    # initialize in |000000>
    psi0 = np.zeros(2 ** N)
    psi0[0] = 1

    # apply sigma_y on the first spin to get 1j|100000>
    psi1 = multiply_single_spin(psi0, N, 0, 2)
    assert psi1[1] == 1j

    # apply sigma_z on the first spin
    psi2 = multiply_single_spin(psi1, N, 0, 3)
    assert np.vdot(psi1, psi2) == -1

    # apply sigma_x on spin 2 through 6
    for i in range(1, N):
        psi1 = multiply_single_spin(psi1, N, i, 1)

    # vector norm should still be 1
    assert np.vdot(psi1, psi1) == 1

    # should be 1j|111111>
    assert psi1[-1] == 1j


def test_evolve_by_HamB():
    N = 6

    # initialize in |000000>
    psi0 = np.zeros(2 ** N)
    psi0[0] = 1

    # evolve by e^{-i (\pi/2) \sum_i X_i}
    psi1 = evolve_by_HamB(N, np.pi / 2, psi0)

    # should get (-1j)^N |111111>
    assert np.vdot(psi1, psi1) == 1
    assert psi1[-1] == (-1j) ** 6


def test_ising_qaoa_grad():
    # construct a known graph
    mygraph = nx.Graph()

    mygraph.add_edge(0, 1, weight=1)
    mygraph.add_edge(0, 2, weight=1)
    mygraph.add_edge(2, 3, weight=1)
    mygraph.add_edge(0, 4, weight=1)
    mygraph.add_edge(1, 4, weight=1)
    mygraph.add_edge(3, 4, weight=1)
    mygraph.add_edge(1, 5, weight=1)
    mygraph.add_edge(2, 5, weight=1)
    mygraph.add_edge(3, 5, weight=1)

    N = mygraph.number_of_nodes()
    HamC = create_ZZ_HamC(mygraph, flag_z2_sym=True)

    # test that the calculated objective function and gradients are correct
    F, Fgrad = ising_qaoa_grad(N, HamC, [1, 0.5], flag_z2_sym=True)

    assert np.abs(F - 1.897011131463) <= 1e-10
    assert np.all(np.abs(Fgrad - [14.287009047096, -0.796709998210]) <= 1e-10)


def test_hamiltonian_objectives_consistent():
    n = 10
    graph = nx.complete_graph(n=n)
    graph = random_plus_minus_1_weights(graph)
    bitstrings = np.random.choice([True, False], size=(100, n))
    expected_energies = [hamiltonian_objective(bitstring, graph)
                         for bitstring in bitstrings]
    actual_energies = hamiltonian_objectives(bitstrings, graph)
    np.testing.assert_allclose(expected_energies, actual_energies)


def test_hamiltonian_objective_avg_and_var():
    bitstrings = np.array([[1, 0, 1, 0, 1],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [1, 1, 0, 0, 1],
                           [0, 1, 1, 1, 1],
                           [0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0],
                           [1, 0, 0, 0, 1],
                           [1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1],
                           [1, 0, 0, 0, 1],
                           [1, 1, 0, 1, 0],
                           [0, 0, 1, 1, 0],
                           [1, 0, 0, 0, 1],
                           [1, 1, 0, 0, 1],
                           [0, 1, 1, 1, 1],
                           [0, 1, 1, 1, 0]], dtype=np.uint8)
    g = nx.Graph()
    g.add_edge(cirq.GridQubit(5, 3), cirq.GridQubit(6, 3), weight=1.0)
    g.add_edge(cirq.GridQubit(6, 3), cirq.GridQubit(7, 3), weight=1.0)
    g.add_edge(cirq.GridQubit(6, 3), cirq.GridQubit(6, 2), weight=-1.0)
    g.add_edge(cirq.GridQubit(6, 3), cirq.GridQubit(6, 4), weight=-1.0)
    vals = hamiltonian_objectives(bitstrings, g)
    val1 = np.mean(vals)
    naive_var = np.var(vals)
    val2, std_err = hamiltonian_objective_avg_and_err(bitstrings, g)
    prop_var = std_err ** 2 * len(bitstrings)
    print()
    print(naive_var, prop_var)
    print(std_err)
    assert prop_var > naive_var
    assert np.isclose(val1, val2)
