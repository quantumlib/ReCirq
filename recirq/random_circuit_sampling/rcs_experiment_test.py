# Copyright 2026 Google
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

import pytest
import numpy as np
import cirq
import recirq.random_circuit_sampling.rcs_experiment as rcs


def test_rcs_multi_depth_regression():
    """Regression test for multi-depth, multi-instance RCS fidelity."""
    patch_1 = cirq.GridQubit.rect(4, 3, top=0, left=0)
    patch_2 = cirq.GridQubit.rect(4, 3, top=0, left=3)
    patch_3 = cirq.GridQubit.rect(4, 3, top=4, left=0)
    patches = [patch_1, patch_2, patch_3]

    DEPTHS = [30, 50]
    NUM_INSTANCES = 3
    N_REPETITIONS = 10000
    FIXED_SEED = 2026

    # Benchmark values from the deterministic run
    expected_benchmarks = {
        (0, 30): 0.9846,
        (0, 50): 0.9753,
        (1, 30): 0.9896,
        (1, 50): 0.9967,
        (2, 30): 1.0030,
        (2, 50): 0.9927,
    }

    experiment = rcs.RCSExperiment(
        patches=patches,
        depths=DEPTHS,
        num_instances=NUM_INSTANCES,
        pattern_name="staggered",
        seed=FIXED_SEED
    )

    simulator = cirq.Simulator(seed=FIXED_SEED)
    results = experiment.run(sampler=simulator, n_repetitions=N_REPETITIONS, characterize=False)

    fidelities = results.fidelities_lin()

    for (patch_idx, depth), expected_val in expected_benchmarks.items():
        actual_val = np.mean(fidelities[(patch_idx, depth)])
        assert actual_val == pytest.approx(expected_val, abs=1e-4)


def test_rcs_validation_logic():
    """Verifies that the experiment catches invalid patch configurations."""
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    q_far = cirq.GridQubit(5, 5)

    # Test Overlapping Patches
    with pytest.raises(ValueError, match="disjoint"):
        rcs.RCSExperiment(patches=[[q0, q1], [q1]], depths=[5], num_instances=1)

    # Test Isolated Qubits
    with pytest.raises(ValueError, match="isolated"):
        rcs.RCSExperiment(patches=[[q0, q1, q_far]], depths=[5], num_instances=1)

    # Test Disconnected Islands
    q_far_neighbor = cirq.GridQubit(5, 6)
    with pytest.raises(ValueError, match="connected"):
        rcs.RCSExperiment(patches=[[q0, q1, q_far, q_far_neighbor]], depths=[5], num_instances=1)


def test_rcs_data_consistency():
    """Checks that the internal data structures maintain correct shapes and qubit sets."""
    p1 = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
    p2 = [cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]
    depths = [5, 10, 20]
    num_instances = 4

    exp = rcs.RCSExperiment(patches=[p1, p2], depths=depths, num_instances=num_instances)
    results = exp.run(sampler=cirq.Simulator(), n_repetitions=10)

    # Check total result count (Patches * Depths * Instances)
    expected_total = 2 * 3 * 4
    assert len(results.circuits) == expected_total
    assert len(results.measurements) == expected_total
    assert len(results.metadata) == expected_total

    # Check qubit isolation (ensure parallel zipping didn't leak qubits)
    for i, meta in enumerate(results.metadata):
        patch_idx = meta["patch_idx"]
        expected_qubits = set(exp.patches[patch_idx])
        actual_qubits = results.circuits[i].all_qubits()
        assert set(actual_qubits) == expected_qubits


def test_rcs_analysis_evaluation():
    """Ensures fidelities_lin triggers analysis automatically via property."""
    p = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
    exp = rcs.RCSExperiment(patches=[p], depths=[2], num_instances=1)
    results = exp.run(sampler=cirq.Simulator(), n_repetitions=100)

    # Check that analysis hasn't run yet
    assert results._fidelities_lin is None
    # Accessing the property triggers rcs.RCSresults._analyze()
    fids = results.fidelities_lin()
    assert fids is not None
    assert (0, 2) in fids


def test_get_calibrated_circuit():
    """Verifies that 2-qubit measurements are not replaced by calibrated gates."""
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)


    circuit = cirq.Circuit(
        cirq.CZ(q0, q1),
        cirq.measure(q0, q1, key='m')
    )

    # Mock characterization
    calibrated_gate = cirq.PhasedFSimGate(theta=0.1)
    characterization = {(q0, q1): calibrated_gate}

    calibrated_circuit = rcs.get_calibrated_circuit(circuit, characterization)


    assert any(isinstance(op.gate, cirq.PhasedFSimGate) for op in
               calibrated_circuit.all_operations())

    measurements = [op for op in calibrated_circuit.all_operations() if
                    cirq.is_measurement(op)]
    assert len(measurements) == 1
    assert measurements[0].qubits == (q0, q1)


def test_characterize_pairs_ideal():
    """Verifies that characterization on a noiseless simulator returns ideal angles."""
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    qubits = [q0, q1]
    sampler = cirq.Simulator()

    theta = np.pi / 2
    phi = np.pi / 6
    base_gate = cirq.FSimGate(theta=theta, phi=phi)

    char_results = rcs.characterize_pairs(
        sampler=sampler,
        qubits=qubits,
        gate=base_gate,
        theta=False,
        phi=False,
        zeta=True,
        chi=True,
        gamma=True
    )


    pair = tuple(sorted((q0, q1)))
    assert pair in char_results

    actual_gate = char_results[pair]
    assert isinstance(actual_gate, cirq.PhasedFSimGate)

    # Theta and phi angles should be close to the ideal values in ideal sim
    assert actual_gate.theta == pytest.approx(theta, abs=1e-2)
    assert actual_gate.phi == pytest.approx(phi, abs=1e-2)

    # Z-phases (zeta, chi, gamma) should be effectively zero in ideal sim
    assert actual_gate.zeta == pytest.approx(0, abs=1e-1)
    assert actual_gate.chi == pytest.approx(0, abs=1e-1)
    assert actual_gate.gamma == pytest.approx(0, abs=1e-1)
