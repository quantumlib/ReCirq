from typing import cast

import cirq
import cirq_google
import numpy as np
import pytest

import recirq.contextuality.ghz_game_2d as ghz_2d

graph = cirq_google.Sycamore.metadata.nx_graph
center_qubit = cirq.GridQubit(4, 5)


@pytest.mark.parametrize("num_qubits", list(range(1, len(graph.nodes) + 1)))
@pytest.mark.parametrize("randomized", [True, False])
@pytest.mark.parametrize("add_dd_and_align_right", [True, False])
def test_ghz_circuits_size(num_qubits: int, randomized: bool, add_dd_and_align_right: bool) -> None:
    """Tests the size of the GHZ circuits."""
    circuit = ghz_2d.generate_2d_ghz_circuit(
        center_qubit,
        graph,
        num_qubits=num_qubits,
        randomized=randomized,
        add_dd_and_align_right=add_dd_and_align_right,
    )
    assert len(circuit.all_qubits()) == num_qubits


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6, 8, 10])
@pytest.mark.parametrize("randomized", [True, False])
@pytest.mark.parametrize("add_dd_and_align_right", [True, False])
def test_ghz_circuits_state(
    num_qubits: int, randomized: bool, add_dd_and_align_right: bool
) -> None:
    """Tests the state vector form of the GHZ circuits."""

    circuit = ghz_2d.generate_2d_ghz_circuit(
        center_qubit,
        graph,
        num_qubits=num_qubits,
        randomized=randomized,
        add_dd_and_align_right=add_dd_and_align_right,
    )

    # Check 2: State vector form
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    state = result.final_state_vector

    # The first and the last elements have to be 1/sqrt(2)
    np.testing.assert_allclose(np.abs(state[0]), 1 / np.sqrt(2), atol=1e-7)
    np.testing.assert_allclose(np.abs(state[-1]), 1 / np.sqrt(2), atol=1e-7)
    # For more than one qubits, everything else is 0
    if num_qubits > 1:
        np.testing.assert_allclose(state[1:-1], 0)


def test_transform_circuit_properties() -> None:
    """Tests that _transform_circuit preserves circuit properties."""
    circuit = ghz_2d.generate_2d_ghz_circuit(
        center_qubit, graph, num_qubits=9, randomized=False, add_dd_and_align_right=False
    )
    transformed_circuit = ghz_2d._transform_circuit(circuit)

    # Check all the qubits are the same
    assert transformed_circuit.all_qubits() == circuit.all_qubits()

    # Stratifying might increase the number of moments
    assert len(transformed_circuit) >= len(circuit)

    # Check there are no measurement gates in the final moment
    final_moment = transformed_circuit[-1]
    assert not any(isinstance(op.gate, cirq.MeasurementGate) for op in final_moment)

    # Check the two circuits are equivalent
    assert cirq.equal_up_to_global_phase(circuit.unitary(), transformed_circuit.unitary())


def manhattan_distance(q1: cirq.GridQubit, q2: cirq.GridQubit) -> int:
    """Calculates the Manhattan distance between two GridQubits."""
    return abs(q1.row - q2.row) + abs(q1.col - q2.col)


@pytest.mark.parametrize("num_qubits", [2, 4, 9, 15, 20])
def test_ghz_circuits_bfs_order(num_qubits: int) -> None:
    """
    Verifies that the circuit construction maintains BFS order by ensuring
    the maximum Manhattan distance of entangled qubits is non-decreasing
    across the circuit moments.
    """

    circuit = ghz_2d.generate_2d_ghz_circuit(
        center_qubit,
        graph,
        num_qubits=num_qubits,
        randomized=False,  # Test must run on the deterministic BFS order
        add_dd_and_align_right=False,  # Test must run on the raw circuit
    )

    # Initialize the maximum distance seen so far to 0 (for the center qubit)
    max_dist_seen = 0

    # Iterate through the circuit moments (starting from the first H at moment 0)
    for moment in circuit:
        for op in moment:
            if isinstance(op.gate, cirq.CZPowGate):
                qubits = op.qubits

                # Identify the distance of the two qubits involved in the CZ
                dist_q0 = manhattan_distance(center_qubit, cast(cirq.GridQubit, qubits[0]))
                dist_q1 = manhattan_distance(center_qubit, cast(cirq.GridQubit, qubits[1]))

                # The 'child' is the qubit farther away from the center.
                child_qubit_distance = max(dist_q0, dist_q1)

                # Assertion: The distance of the newly entangled qubit must be
                # less than or equal to the maximum distance seen so far OR
                # exactly one greater than the max distance seen so far.

                # If the distance is NEW (i.e., greater than max_dist_seen),
                # it must be exactly max_dist_seen + 1 (the next layer of BFS).
                if child_qubit_distance > max_dist_seen:
                    assert child_qubit_distance == max_dist_seen + 1
                    # Update the maximum distance seen
                    max_dist_seen = child_qubit_distance

                # If the distance is NOT new, it must be less than or equal to max_dist_seen
                assert child_qubit_distance <= max_dist_seen

    # Final check: Ensure we entangled up to the maximum distance required by num_qubits
    included_qubits = circuit.all_qubits()
    if included_qubits:
        max_dist_required = max(
            manhattan_distance(center_qubit, cast(cirq.GridQubit, q)) for q in included_qubits
        )
        assert max_dist_seen == max_dist_required


@pytest.mark.parametrize("randomized", [True, False])
@pytest.mark.parametrize("add_dd_and_align_right", [True, False])
def test_ghz_game(randomized: bool, add_dd_and_align_right: bool):
    """
    Tests the full 2D GHZ game workflow (Generation, Execution, Analysis)
    using the Cirq Simulator and standard asserts. Verifies correct circuit
    counts and perfect fidelity results (expected for noise-free simulation).
    """

    num_trials_per_circuit = 20
    repetitions_for_batch = 120
    target_qubit_counts = [3, 4, 5, 6]

    experiment = ghz_2d.GHZ2dExperiment(graph=graph, center=center_qubit)

    analyzed_results = experiment.run(
        target_qubit_counts=target_qubit_counts,
        num_trials_per_circuit=num_trials_per_circuit,
        sampler=cirq.Simulator(),
        repetitions_for_batch=repetitions_for_batch,
        randomize_growth=randomized,
        add_dd_and_align_right=add_dd_and_align_right,
    )

    # Check that we have results for all target qubit counts
    assert set(analyzed_results.win_mean.keys()) == set(target_qubit_counts)
    assert set(analyzed_results.x_fidelity.keys()) == set(target_qubit_counts)
    assert set(analyzed_results.z_fidelity.keys()) == set(target_qubit_counts)

    for num_qubits in target_qubit_counts:
        # Get metrics for the current qubit size
        x_fidelity = analyzed_results.x_fidelity.get(num_qubits, [])
        z_fidelity = analyzed_results.z_fidelity.get(num_qubits, [])
        win_mean = analyzed_results.win_mean.get(num_qubits, [])

        # Check that the number of trials is correct
        assert len(x_fidelity) == num_trials_per_circuit
        assert len(z_fidelity) == num_trials_per_circuit
        assert len(win_mean) == num_trials_per_circuit

        # For a noise-free simulator, all fidelities and win probabilities should be 1.0
        np.testing.assert_allclose(np.mean(x_fidelity), 1.0)
        np.testing.assert_allclose(np.mean(z_fidelity), 1.0)
        np.testing.assert_allclose(np.mean(win_mean), 1.0)
