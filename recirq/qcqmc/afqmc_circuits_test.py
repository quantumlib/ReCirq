import cirq
import numpy as np
import pytest

from qc_afqmc.afqmc_circuits import (
    ControlledSlaterDeterminantPreparationGate,
    GeminalStatePreparationGate,
    get_computational_qubits_and_ancilla,
    get_geminal_and_slater_det_overlap,
    get_geminal_and_slater_det_overlap_circuit,
    get_geminal_and_slater_det_overlap_via_simulation,
    get_geminal_state,
    get_jordan_wigner_string,
    get_phase_state_preparation_gate,
    jw_slater_determinant,
    SlaterDeterminantPreparationGate,
)

RNG = np.random.default_rng(5345346)


@pytest.mark.parametrize(
    'angle', (tuple(np.arange(20) * np.pi / 8) + tuple(5 * np.pi * (RNG.random(size=10) - 0.5)))
)
def test_geminal_state_preparation_gate(angle):
    for indicator in (False, True):
        gate = GeminalStatePreparationGate(angle, indicator=indicator)
        unitary = cirq.unitary(gate)

        actual_state = unitary[:, 0b0010 if indicator else 0]
        expected_state = get_geminal_state(angle)
        assert np.allclose(actual_state, expected_state)

        actual_state = unitary[:, 0 if indicator else 0b0010]
        expected_state = cirq.one_hot(index=0, shape=16, dtype=np.complex128)
        assert np.allclose(actual_state, expected_state)


@pytest.mark.parametrize(
    'n_elec, n_orb, seed',
    [
        (n_elec, n_orb, int(seed))
        for seed in RNG.integers(2 << 30, size=5)
        for n_orb in range(1, 7)
        for n_elec in range(n_orb + 1)
    ],
)
def test_slater_determinant_preparation_gate(n_elec, n_orb, seed):
    orbitals = cirq.testing.random_unitary(n_orb, random_state=seed)[:n_elec]
    gate = SlaterDeterminantPreparationGate(orbitals)
    unitary = cirq.unitary(gate)

    actual_state = unitary[:, 0]
    expected_state = jw_slater_determinant(orbitals)
    assert np.allclose(actual_state, expected_state)

    ref = (2**n_orb - 1) ^ (2 ** (n_orb - n_elec) - 1)
    actual_state = unitary[:, ref]
    expected_state = np.eye(2**n_orb, 1).flatten()
    assert np.allclose(actual_state, expected_state)


@pytest.mark.parametrize(
    'angle',
    tuple(np.arange(-np.pi, 3 * np.pi, np.pi / 8)) + tuple(5 * np.pi * (RNG.random(size=5) - 0.5)),
)
def test_get_phase_state_preparation_gate(angle):
    gate = get_phase_state_preparation_gate(angle)
    unitary = cirq.unitary(gate)
    state = np.sqrt(1 / 2) * np.exp(1j * angle * np.arange(2))
    assert np.allclose(unitary[:, 0], state)


@pytest.mark.parametrize('n_pairs', [1, 2, 3, 4, 5])
def test_jordan_wigner_string(n_pairs):
    jw_indices = list(get_jordan_wigner_string(n_pairs))
    reverse_jw_indices = list(get_jordan_wigner_string(n_pairs, True))
    assert sorted(jw_indices) == sorted(reverse_jw_indices) == list(range(4 * n_pairs))
    assert all(reverse_jw_indices[j] == i for i, j in enumerate(jw_indices))


@pytest.mark.parametrize(
    'n_orbs, n_elec',
    [
        (n_orbs, n_elec)
        for n_orbs in range(13)
        for n_elec in RNG.integers(n_orbs + 1, size=min(1, n_orbs // 3))
    ],
)
def test_controlled_slater_determinant_preparation_gate(n_orbs, n_elec):
    orbitals = np.eye(n_elec, n_orbs)
    gate = ControlledSlaterDeterminantPreparationGate(orbitals)
    unitary = cirq.unitary(gate)

    actual_state = unitary[:, 0]
    expected_state = np.eye(2 ** (n_orbs + 1), 1).flatten()
    assert np.allclose(actual_state, expected_state)

    actual_state = unitary[:, 1 << n_orbs]
    expected_state = cirq.one_hot(
        index=(1,) * (n_elec + 1) + (0,) * (n_orbs - n_elec), shape=(2,) * (n_orbs + 1), dtype=int
    ).flatten()
    assert np.allclose(actual_state, expected_state)


@pytest.mark.parametrize(
    'n_pairs, n_elec, seed',
    [
        (n_pairs, n_elec, int(seed))
        for n_pairs in (1, 2, 3)
        for n_elec in (2 * n_pairs, RNG.integers(4 * n_pairs + 1))
        for seed in [11111]
    ],
)
def test_overlap_circuit(n_pairs, n_elec, seed):
    n_orbs = 4 * n_pairs
    rng = np.random.default_rng(seed)

    orbitals = cirq.testing.random_unitary(n_orbs, random_state=seed)[:n_elec]
    assert orbitals.shape == (n_elec, n_orbs)
    angles = 2 * np.pi * rng.random(n_pairs)
    phase = 2 * np.pi * rng.random()

    kwargs = {'angles': angles, 'orbitals': orbitals}

    calculated_overlap = get_geminal_and_slater_det_overlap(angles=angles, orbitals=orbitals)
    atol = 10e-6

    for fold_in_measurement in (False, True):
        kwargs['fold_in_measurement'] = fold_in_measurement
        overlap_phase = get_geminal_and_slater_det_overlap_via_simulation(phase=phase, **kwargs)
        overlap_re = get_geminal_and_slater_det_overlap_via_simulation(phase=0, **kwargs)
        overlap_im = get_geminal_and_slater_det_overlap_via_simulation(phase=-np.pi / 2, **kwargs)
        actual_overlap = overlap_re + 1j * overlap_im

        assert np.isclose(overlap_phase, (np.exp(1j * phase) * calculated_overlap).real, atol=atol)
        assert np.isclose(overlap_re, calculated_overlap.real, atol=atol)
        assert np.isclose(overlap_im, calculated_overlap.imag, atol=atol)
        assert np.isclose(actual_overlap, calculated_overlap, atol=atol)


@pytest.mark.parametrize('n_pairs', [1, 2, 3, 4, 5, 6])
def test_get_computational_qubits_and_ancilla(n_pairs):
    comp_qubits, _ = get_computational_qubits_and_ancilla(4 * n_pairs)
    for i in range(n_pairs):
        edges = [
            (4 * i, 4 * i + 1),
            (4 * i, 4 * i + 2),
            (4 * i + 2, 4 * i + 3),
            (4 * i + 1, 4 * i + 3),
        ]
        if i + 1 < n_pairs:
            edges.append((4 * i + 2, 4 * i + 4))
            edges.append((4 * i + 3, 4 * i + 5))
        for qubit, other_qubit in edges:
            assert comp_qubits[qubit].is_adjacent(comp_qubits[other_qubit])


@pytest.mark.parametrize(
    'n_pairs, n_elec, seed',
    [
        (n_pairs, n_elec, int(seed))
        for n_pairs in (1, 2, 3)
        for n_elec in (2 * n_pairs, RNG.integers(4 * n_pairs + 1))
        for seed in RNG.integers(2 << 30, size=(5 if n_elec == 2 * n_elec else 1))
    ],
)
def test_overlap_circuit_connectivity(n_pairs, n_elec, seed):
    rng = np.random.default_rng(seed)
    n_orbs = 4 * n_pairs
    orbitals = cirq.testing.random_unitary(n_orbs, random_state=seed)[:n_elec]
    assert orbitals.shape == (n_elec, n_orbs)
    angles = 2 * np.pi * rng.random(n_pairs)
    phase = 2 * np.pi * rng.random()

    comp_qubits, ancilla = get_computational_qubits_and_ancilla(n_orbs)

    for fold_in_measurement in (False, True):
        circuit = cirq.Circuit(
            get_geminal_and_slater_det_overlap_circuit(
                ancilla=ancilla,
                comp_qubits=comp_qubits,
                orbitals=orbitals,
                angles=angles,
                phase=phase,
                decompose=True,
                fold_in_measurement=fold_in_measurement,
            )
        )
        for operation in circuit.all_operations():
            if len(operation.qubits) == 1:
                continue
            qubit, other_qubit = operation.qubits
            assert isinstance(qubit, cirq.GridQubit)
            assert qubit.is_adjacent(other_qubit)


@pytest.mark.parametrize(
    'n_orbs, seed',
    [(n_orbs, int(seed)) for n_orbs in range(1, 10) for seed in RNG.integers(2 << 30, size=2)],
)
def test_slater_determinant(n_orbs, seed):
    rng = np.random.default_rng(seed)
    n_elec = rng.integers(1, n_orbs + 1)
    orbitals = cirq.testing.random_unitary(n_orbs, random_state=seed)[:n_elec]

    rotation = cirq.testing.random_unitary(n_elec, random_state=seed)
    det = np.linalg.det(rotation)
    rotated_orbitals = np.dot(rotation, orbitals)
    assert orbitals.shape == rotated_orbitals.shape

    sd_state = jw_slater_determinant(orbitals)
    rotated_sd_state = jw_slater_determinant(rotated_orbitals)
    assert np.allclose(det * sd_state, rotated_sd_state)

    eye = np.eye(n_elec, n_orbs)
    ref_state = jw_slater_determinant(eye)
    n_extra_orbs = n_orbs - n_elec
    assert np.allclose(
        ref_state,
        cirq.one_hot(
            index=(1,) * n_elec + (0,) * n_extra_orbs, shape=(2,) * n_orbs, dtype=float
        ).flatten(),
    )
    rotated_orbitals = np.dot(rotation, eye)
    if n_extra_orbs:
        extra_zeros = np.zeros((n_elec, n_extra_orbs))
        assert np.allclose(np.hstack((rotation, extra_zeros)), rotated_orbitals)
    else:
        assert np.allclose(rotation, rotated_orbitals)
    assert orbitals.shape == rotated_orbitals.shape
    rotated_sd_state = jw_slater_determinant(rotated_orbitals)
    assert np.allclose(det * ref_state, rotated_sd_state)


@pytest.mark.parametrize(
    'n_orbs, n_elec, orbitals, angles, expected_overlap',
    [
        (
            8,
            4,
            np.asarray(
                [
                    [0.695178, 0.0, 0.129336, 0.0, 0.695178, 0.0, -0.129336, 0.0],
                    [0.0, 0.695179, 0.0, 0.129331, 0.0, -0.695179, 0.0, 0.129331],
                    [0.544096, 0.0, 0.451619, 0.0, -0.544096, 0.0, 0.451619, 0.0],
                    [0.0, 0.544074, 0.0, 0.451646, 0.0, 0.544074, 0.0, -0.451646],
                    [-0.451619, 0.0, 0.544096, 0.0, 0.451619, 0.0, 0.544096, 0.0],
                    [0.0, 0.451646, 0.0, -0.544074, 0.0, 0.451646, 0.0, 0.544074],
                    [0.129336, 0.0, -0.695178, 0.0, 0.129336, 0.0, 0.695178, 0.0],
                    [0.0, -0.129331, 0.0, 0.695179, 0.0, 0.129331, 0.0, 0.695179],
                ]
            ),
            np.asarray([0.123044, 0.123044]),
            -0.5998193275459747,
        ),
        (4, 2, np.eye(4), np.zeros(1), 1.0),
        (
            8,
            4,
            np.asarray(
                [
                    [0.695178, 0.0, 0.129336, 0.0, -0.695178, 0.0, -0.129336, 0.0],
                    [0.0, 0.695179, 0.0, -0.129331, 0.0, -0.695179, 0.0, 0.129331],
                    [-0.451646, 0.0, 0.544074, 0.0, -0.451646, 0.0, 0.544074, 0.0],
                    [0.0, 0.451619, 0.0, 0.544096, 0.0, 0.451619, 0.0, 0.544096],
                    [0.544074, 0.0, 0.451646, 0.0, 0.544074, 0.0, 0.451646, 0.0],
                    [0.0, 0.544096, 0.0, -0.451619, 0.0, 0.544096, 0.0, -0.451619],
                    [0.129336, 0.0, -0.695178, 0.0, -0.129336, 0.0, 0.695178, 0.0],
                    [0.0, -0.129331, 0.0, -0.695179, 0.0, 0.129331, 0.0, 0.695179],
                ]
            ),
            np.asarray([-0.123044, -0.123044]),
            -0.4135738636386284,
        ),
    ],
)
def test_overlap_circuit_julia_vals(n_orbs, n_elec, orbitals, angles, expected_overlap):
    orbitals = orbitals[:n_elec]

    assert orbitals.shape == (n_elec, n_orbs)
    phase = 0.0

    kwargs = {'angles': angles, 'orbitals': orbitals}

    calculated_overlap = get_geminal_and_slater_det_overlap(angles=angles, orbitals=orbitals)
    atol = 10e-6

    for fold_in_measurement in (False, True):
        kwargs['fold_in_measurement'] = fold_in_measurement
        overlap_phase = get_geminal_and_slater_det_overlap_via_simulation(phase=phase, **kwargs)
        overlap_re = get_geminal_and_slater_det_overlap_via_simulation(phase=0, **kwargs)
        overlap_im = get_geminal_and_slater_det_overlap_via_simulation(phase=-np.pi / 2, **kwargs)
        actual_overlap = overlap_re + 1j * overlap_im

        assert np.isclose(overlap_phase, (np.exp(1j * phase) * calculated_overlap).real, atol=atol)
        assert np.isclose(overlap_re, calculated_overlap.real, atol=atol)
        assert np.isclose(overlap_im, calculated_overlap.imag, atol=atol)
        assert np.isclose(actual_overlap, calculated_overlap, atol=atol)
        assert np.isclose(expected_overlap, calculated_overlap, atol=atol)
