from typing import List

import networkx as nx
import cirq
import recirq
import numpy as np

from recirq.qaoa.simulation import create_ZZ_HamC


def expectations(graph: nx.Graph, thetas: List[float]):
    q1, q2 = cirq.LineQubit.range(2)
    ZZ = cirq.Z(q1) * cirq.Z(q2)
    sim = cirq.Simulator(dtype=np.complex128)
    for n1, n2, w in graph.edges.data('weight'):
        psi = sim.simulate(cirq.Circuit(
            cirq.XPowGate(exponent=thetas[n1]).on(q1),
            cirq.XPowGate(exponent=thetas[n2]).on(q2),
        ))

        zz_val = w * ZZ.expectation_from_wavefunction(
            psi.final_state, psi.qubit_map, check_preconditions=False)
        assert zz_val.imag < 1e-6, zz_val
        zz_val = zz_val.real
        yield n1, n2, zz_val


def expectation(graph: nx.Graph, thetas: List[float]):
    return sum(zz_val for _, _, zz_val in expectations(graph, thetas))


def get_answer_inds(graph):
    dense_ham = create_ZZ_HamC(graph, flag_z2_sym=False,
                               node_to_index_map={i: i for i in range(graph.number_of_nodes())})
    return np.where(dense_ham == np.min(dense_ham))[0]


def get_answer_bitstrings(graph):
    answer_inds = get_answer_inds(graph)
    return np.asarray([cirq.big_endian_int_to_bits(answer_ind, bit_count=graph.number_of_nodes())
                       for answer_ind in answer_inds])


def get_p_single_bit(theta, should_be):
    q1 = cirq.LineQubit(0)
    Z = cirq.Z(q1) * 1
    sim = cirq.Simulator()

    psi = sim.simulate(cirq.Circuit(
        cirq.XPowGate(exponent=theta).on(q1),
    ))
    z_val = Z.expectation_from_wavefunction(psi.final_state, psi.qubit_map)
    assert z_val.imag < 1e-6, z_val
    z_val = z_val.real

    if should_be == 0:
        return (z_val + 1) / 2
    if should_be == 1:
        return (1 - z_val) / 2
    raise ValueError(f'{should_be}')


def get_pzstar(thetas: List[float], answer_bitstring):
    assert answer_bitstring.ndim == 1
    p = 1.0
    for theta, bx in zip(thetas, answer_bitstring):
        p *= get_p_single_bit(theta, bx)
    return p


def get_pzstars(thetas: List[float], answer_bitstrings: np.array):
    assert answer_bitstrings.ndim == 2
    return sum(get_pzstar(thetas, answer_bitstring) for answer_bitstring in answer_bitstrings)
