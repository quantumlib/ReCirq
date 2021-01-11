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
"""Construct Fermi-Hubbard circuits embedded on two parallel lines or zig-zag
like layout.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import cirq
from collections import defaultdict
import numpy as np
import openfermion

from recirq.fermi_hubbard.decomposition import CPhaseEchoGate
from recirq.fermi_hubbard.fermionic_circuits import create_one_particle_circuit
from recirq.fermi_hubbard.parameters import (
    ChainInitialState,
    FermiHubbardParameters,
    IndependentChainsInitialState,
    SingleParticle,
    TrappingPotential
)


def create_line_circuits(parameters: FermiHubbardParameters,
                         trotter_steps: int
                         ) -> Tuple[cirq.Circuit, cirq.Circuit, cirq.Circuit]:
    """Creates circuit mapped on a LineLayout.

    See LineLayout docstring for details.

    Args:
        parameters: Fermi-Hubbard problem parameters.
        trotter_steps: Number to Trotter steps to include.

    Returns:
        Tuple of:
         - initial state preparation circuit,
         - trotter steps circuit,
         - measurement circuit.
    """
    return (create_initial_circuit(parameters),
            create_line_trotter_circuit(parameters, trotter_steps),
            create_measurement_circuit(parameters))


def create_zigzag_circuits(parameters: FermiHubbardParameters,
                           trotter_steps: int
                           ) -> Tuple[cirq.Circuit, cirq.Circuit, cirq.Circuit]:
    """Creates circuit mapped on a ZigZagLayout.

    See ZigZagLayout docstring for details.

    Args:
        parameters: Fermi-Hubbard problem parameters.
        trotter_steps: Number to Trotter steps to include.

    Returns:
        Tuple of:
         - initial state preparation circuit,
         - trotter steps circuit,
         - measurement circuit.
    """
    return (create_initial_circuit(parameters),
            create_zigzag_trotter_circuit(parameters, trotter_steps),
            create_measurement_circuit(parameters))


def create_initial_circuit(parameters: FermiHubbardParameters
                           ) -> cirq.Circuit:
    """Creates initial state circuit mapped on a LineLayout or ZigZagLayout."""
    if isinstance(parameters.initial_state, IndependentChainsInitialState):
        return _create_independent_chains_initial_circuit(parameters)
    else:
        raise ValueError(f'Unsupported initial state '
                         f'{parameters.initial_state}')


def create_measurement_circuit(parameters: FermiHubbardParameters
                               ) -> cirq.Circuit:
    """Creates measurement circuit mapped on a LineLayout or ZigZagLayout."""
    layout = parameters.layout
    return cirq.Circuit(
        cirq.measure(*(layout.up_qubits + layout.down_qubits), key='z'))


def create_line_trotter_circuit(parameters: FermiHubbardParameters,
                                trotter_steps: int
                                ) -> cirq.Circuit:
    """Creates Trotter steps circuit mapped on a LineLayout."""
    circuit = cirq.Circuit()
    for _ in range(trotter_steps):
        circuit += create_line_trotter_step_circuit(parameters)
    return circuit


def create_line_trotter_step_circuit(parameters: FermiHubbardParameters
                                     ) -> cirq.Circuit:
    """Creates a single Trotter step circuit mapped on a LineLayout."""

    layout = parameters.layout
    hamiltonian = parameters.hamiltonian
    dt = parameters.dt

    j_theta = dt * hamiltonian.j_array
    j_theta_even, j_theta_odd = j_theta[0::2], j_theta[1::2]

    u_phi = -dt * hamiltonian.u_array

    v_phi = -dt * hamiltonian.v_array
    v_phi_even, v_phi_odd = v_phi[0::2], v_phi[1::2]

    local_up = -dt * hamiltonian.local_up_array
    local_down = -dt * hamiltonian.local_down_array

    circuit = cirq.Circuit()

    # Nearest-neighbor hopping terms.
    circuit += cirq.Circuit((
        _create_hopping_ops(j_theta_even, layout.up_even_pairs),
        _create_hopping_ops(j_theta_even, layout.down_even_pairs)))
    circuit += cirq.Circuit((
        _create_hopping_ops(j_theta_odd, layout.up_odd_pairs),
        _create_hopping_ops(j_theta_odd, layout.down_odd_pairs)))

    # On-site interaction terms.
    if not np.allclose(u_phi, 0.0):
        circuit += cirq.Circuit(
            _create_interaction_ops(u_phi, layout.interaction_pairs))

    # Nearest-neighbor interaction terms.
    if not np.allclose(v_phi_even, 0.0):
        circuit += cirq.Circuit((
            _create_interaction_ops(v_phi_even, layout.up_even_pairs),
            _create_interaction_ops(v_phi_even, layout.down_even_pairs)))
    if not np.allclose(v_phi_odd, 0.0):
        circuit += cirq.Circuit((
            _create_interaction_ops(v_phi_odd, layout.up_odd_pairs),
            _create_interaction_ops(v_phi_odd, layout.down_odd_pairs)))

    # Local fields.
    if not np.allclose(local_up, 0.0) or not np.allclose(local_down, 0.0):
        circuit += cirq.Circuit((
            _create_local_field_ops(local_up, layout.up_qubits),
            _create_local_field_ops(local_down, layout.down_qubits)))

    return circuit


def create_zigzag_trotter_circuit(parameters: FermiHubbardParameters,
                                  trotter_steps: int
                                  ) -> cirq.Circuit:
    """Creates Trotter steps circuit mapped on a ZigZagLayout."""
    circuit = cirq.Circuit()
    for _ in range(trotter_steps):
        circuit += create_zigzag_trotter_step_circuit(parameters)
    return circuit


def create_zigzag_trotter_step_circuit(parameters: FermiHubbardParameters,
                                       add_cphase_echo: bool = True
                                       ) -> cirq.Circuit:
    """Creates a single Trotter step circuit mapped on a ZigZagLayout."""

    layout = parameters.layout
    hamiltonian = parameters.hamiltonian
    dt = parameters.dt

    j_theta = dt * hamiltonian.j_array
    j_theta_even, j_theta_odd = j_theta[0::2], j_theta[1::2]

    iswap_theta = np.full(layout.size // 2 - 1, np.pi / 2)

    u_phi = -dt * hamiltonian.u_array
    u_phi_even, u_phi_odd = u_phi[0::2], u_phi[1::2]

    v_phi = -dt * hamiltonian.v_array
    v_phi_even, v_phi_odd = v_phi[0::2], v_phi[1::2]

    local_up = -dt * hamiltonian.local_up_array
    local_down = -dt * hamiltonian.local_down_array

    circuit = cirq.Circuit()

    # Even nearest-neighbor hopping terms.
    circuit += cirq.Circuit((
        _create_hopping_ops(j_theta_even, layout.up_even_pairs),
        _create_hopping_ops(j_theta_even, layout.down_even_pairs)))

    # Even nearest-neighbor interaction terms.
    if not np.allclose(v_phi_even, 0.0):
        circuit += cirq.Circuit((
            _create_interaction_ops(v_phi_even, layout.up_even_pairs),
            _create_interaction_ops(v_phi_even, layout.down_even_pairs)))

    # Even on-site interaction terms.
    if not np.allclose(u_phi_even, 0.0):
        circuit += cirq.Circuit((
            _create_interaction_ops(u_phi_even, layout.interaction_even_pairs),
            _create_cphase_echo_ops(
                layout.interaction_even_other_qubits) if add_cphase_echo else ()
        ))

    # Odd on-site interaction terms.
    if not np.allclose(u_phi_odd, 0.0):
        # Odd terms are present, an iSWAP on inner qubits is required.

        # iSWAP of inner pairs.
        circuit += cirq.Circuit((
            _create_hopping_ops(iswap_theta, layout.up_odd_pairs),
            _create_hopping_ops(iswap_theta, layout.down_odd_pairs)))

        # First half of the odd nearest-neighbor interaction terms.
        if not np.allclose(v_phi_odd, 0.0):
            circuit += cirq.Circuit((
                _create_interaction_ops(v_phi_odd / 2, layout.up_odd_pairs),
                _create_interaction_ops(v_phi_odd / 2, layout.down_odd_pairs)))

        # Actual odd on-site interaction terms.
        circuit += cirq.Circuit((
            _create_interaction_ops(u_phi_odd, layout.interaction_odd_pairs),
            _create_cphase_echo_ops(
                layout.interaction_odd_other_qubits) if add_cphase_echo else ()
        ))

        # Odd nearest-neighbor hopping terms with iSWAP on inner pairs.
        circuit += cirq.Circuit((
            _create_hopping_ops(j_theta_odd + iswap_theta, layout.up_odd_pairs),
            _create_hopping_ops(j_theta_odd + iswap_theta,
                                layout.down_odd_pairs)))

        # Second half of the odd nearest-neighbor interaction terms.
        if not np.allclose(v_phi_odd, 0.0):
            circuit += cirq.Circuit((
                _create_interaction_ops(v_phi_odd / 2, layout.up_odd_pairs),
                _create_interaction_ops(v_phi_odd / 2, layout.down_odd_pairs)))

        # Fix the extra Z rotations introduced by the two iSWAPs. This rotations
        # together with the pi/2 term in iswap_theta effectively implement the
        # iSWAP^dagger gate.
        circuit += cirq.Circuit((
            (cirq.Z(qubit) for qubit in layout.up_qubits[1:-1]),
            (cirq.Z(qubit) for qubit in layout.down_qubits[1:-1])))
    else:
        # Odd on-site interaction terms are not present, do a shorter circuit.

        # Odd nearest-neighbor hopping terms.
        circuit += cirq.Circuit((
            _create_hopping_ops(j_theta_odd, layout.up_odd_pairs),
            _create_hopping_ops(j_theta_odd, layout.down_odd_pairs)))

        # Odd nearest-neighbor interaction terms.
        if not np.allclose(v_phi_odd, 0.0):
            circuit += cirq.Circuit((
                _create_interaction_ops(v_phi_odd, layout.up_odd_pairs),
                _create_interaction_ops(v_phi_odd, layout.down_odd_pairs)))

    # Local fields.
    if not np.allclose(local_up, 0.0) or not np.allclose(local_down, 0.0):
        circuit += cirq.Circuit((
            _create_local_field_ops(local_up, layout.up_qubits),
            _create_local_field_ops(local_down, layout.down_qubits)))

    return circuit


# TODO: Candidate for move to cirq package.
def run_in_parallel(first: cirq.Circuit, second: cirq.Circuit) -> cirq.Circuit:
    """Merges two circuits so that consecutive moments of two sub-circuits run
    in parallel.
    """
    min_size = min(len(first), len(second))
    return cirq.Circuit((cirq.Moment([a.operations, b.operations])
                         for a, b in zip(first, second)) +
                        first[min_size:] +
                        second[min_size:])


# TODO: Candidate for move to cirq package.
def align_givens_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """Re-aligns the Z gates of a circuit generated by givens rotation."""

    # Operations which are followed by a list of Z operations.
    following_zs: Dict[Optional[cirq.Operation],
                      List[cirq.Operation]] = defaultdict(lambda: [])

    # Make a copy of a circuit without Z operations.
    frontier: Dict[cirq.Qid, cirq.Operation] = {}
    aligned = cirq.Circuit()
    for index, moment in enumerate(circuit):
        for operation in moment:
            if (isinstance(operation, cirq.GateOperation) and
                    isinstance(operation.gate, cirq.ZPowGate)):
                qubit, = operation.qubits
                following_zs[frontier.get(qubit)].append(operation)
            else:
                aligned.append(operation)

            for qubit in operation.qubits:
                frontier[qubit] = operation

    # Restore the removed Z operations back.
    circuit = cirq.Circuit()
    if following_zs[None]:
        circuit += cirq.Moment(following_zs[None])
    for moment in aligned:
        circuit += moment
        zs = sum((following_zs[op] for op in moment), [])
        while zs:
            circuit += cirq.Moment(zs)
            zs = sum((following_zs[op] for op in zs), [])

    return circuit


def _create_independent_chains_initial_circuit(
        parameters: FermiHubbardParameters
) -> cirq.Circuit:
    """Creates circuit that realizes IndependentChainsInitialState initial
    state.
    """
    layout = parameters.layout
    initial = cast(IndependentChainsInitialState, parameters.initial_state)

    up_circuit = _create_chain_initial_circuit(
        parameters, layout.up_qubits, initial.up)
    down_circuit = _create_chain_initial_circuit(
        parameters, layout.down_qubits, initial.down)

    circuit = run_in_parallel(up_circuit, down_circuit)
    circuit = align_givens_circuit(circuit)
    return circuit


def _create_chain_initial_circuit(parameters: FermiHubbardParameters,
                                  qubits: List[cirq.Qid],
                                  chain: ChainInitialState) -> cirq.Circuit:
    """Creates circuit that realizes ChainInitialState for given chain. """
    if isinstance(chain, SingleParticle):
        return create_one_particle_circuit(
            qubits, chain.get_amplitudes(len(qubits)))
    elif isinstance(chain, TrappingPotential):
        return _create_quadratic_hamiltonian_circuit(
            qubits,
            chain.particles,
            chain.as_quadratic_hamiltonian(len(qubits),
                                           parameters.hamiltonian.j))
    else:
        raise ValueError(f'Unsupported chain initial state {chain}')


def _create_quadratic_hamiltonian_circuit(
        qubits: List[cirq.Qid],
        particles: int,
        hamiltonian: openfermion.QuadraticHamiltonian) -> cirq.Circuit:
    """Creates circuit that realizes network of givens rotation."""

    circuit = cirq.Circuit()

    # Prepare ground state in diagonal basis and add initial circuits.
    circuit.append([cirq.X.on(q) for q in qubits[:particles]])

    # Create Givens network circuit.
    _, transform, _ = hamiltonian.diagonalizing_bogoliubov_transform()
    circuit += cirq.Circuit(openfermion.bogoliubov_transform(
        qubits, transform, range(particles)))

    return circuit


def _create_hopping_ops(angles: np.ndarray,
                        pairs: Sequence[Tuple[cirq.Qid, cirq.Qid]]
                        ) -> Iterable[cirq.Operation]:
    """Generator of operations that realize hopping terms with given angles."""
    assert len(angles) == len(pairs), ("Length of angles and qubit pairs must "
                                       "be equal")
    for angle, qubits in zip(angles, pairs):
        yield cirq.FSimGate(-angle, 0.0).on(*qubits)


def _create_interaction_ops(angles: np.ndarray,
                            pairs: Sequence[Tuple[cirq.Qid, cirq.Qid]]
                            ) -> Iterable[cirq.Operation]:
    """Generator of operations that realize interaction terms with given angles.
    """
    assert len(angles) == len(pairs), ("Length of angles and qubit pairs must "
                                       "be equal")
    for angle, qubits in zip(angles, pairs):
        yield cirq.CZ(*qubits) ** (angle / np.pi)


def _create_local_field_ops(angles: np.ndarray,
                            qubits: Sequence[cirq.Qid]
                            ) -> Iterable[cirq.Operation]:
    """Generator of operations that realize local fields with given strengths.
    """
    assert len(angles) == len(qubits), ("Length of angles and qubits must be "
                                        "equal")
    for angle, qubit in zip(angles, qubits):
        yield cirq.Z(qubit) ** (angle / np.pi)


def _create_cphase_echo_ops(qubits: Iterable[cirq.Qid]
                            ) -> Iterable[cirq.Operation]:
    """Generator of operations that realize cphase echo gates."""
    return (CPhaseEchoGate().on(qubit) for qubit in qubits)
