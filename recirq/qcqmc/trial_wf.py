# Copyright 2024 Google
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
"""Specification of a trial wavefunction."""

import abc
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import attrs
import cirq
import fqe
import numpy as np
import openfermion as of
import scipy.sparse

from recirq.qcqmc import (
    bitstrings,
    config,
    data,
    fermion_mode,
    fqe_conversion,
    hamiltonian,
    layer_spec,
    qubit_maps,
)


@attrs.frozen
class TrialWavefunctionParams(data.Params, metaclass=abc.ABCMeta):
    """Parameters specifying a trial wavefunction.

    Args:
        name: A descriptive name for the wavefunction parameters.
        hamiltonian_params: Hamiltonian parameters specifying the molecule.
    """

    name: str
    hamiltonian_params: hamiltonian.HamiltonianParams

    @property
    def bitstrings(self) -> Iterable[Tuple[bool, ...]]:
        raise NotImplementedError(
            "TrialWavefunctionParams should be subclassed and this method should be overwritten."
        )

    @property
    def qubits_jordan_wigner_ordered(self) -> Tuple[cirq.GridQubit, ...]:
        raise NotImplementedError(
            "TrialWavefunctionParams should be subclassed and this method should be overwritten."
        )

    @property
    def qubits_linearly_connected(self) -> Tuple[cirq.GridQubit, ...]:
        raise NotImplementedError(
            "TrialWavefunctionParams should be subclassed and this method should be overwritten."
        )


def _to_numpy(x: Optional[Iterable] = None) -> Optional[np.ndarray]:
    return np.asarray(x)


def _to_tuple(x: Iterable[layer_spec.LayerSpec]) -> Sequence[layer_spec.LayerSpec]:
    return tuple(x)


@attrs.frozen(repr=False, eq=False)
class PerfectPairingPlusTrialWavefunctionParams(TrialWavefunctionParams):
    """Class for storing the parameters that specify the trial wavefunction.

    This class specifically stores the parameters for a trial wavefunction that
    is a combination of a perfect pairing wavefunction with some number of
    hardware-efficient layers appended.

    Args:
        name: A name for the trial wavefunction.
        hamiltonian_params: The hamiltonian parameters specifying the molecule.
        heuristic_layers: A tuple of circuit layers to append to the perfect pairing circuit.
        do_pp: Implement the perfect pairing circuit along with the heuristic
            layers. Defaults to true.
        restricted: Use a restricted perfect pairing ansatz. Defaults to false,
            i.e. allow spin-symmetry breaking.
        random_parameter_scale: A float to scale the random parameters by.
        n_optimization_restarts: The number of times to restart the optimization
            from a random guess in an attempt at global optimization.
        seed: The random number seed to initialize the RNG with.
        initial_orbital_rotation: An optional initial orbital rotation matrix,
            which will be implmented as a givens circuit.
        initial_two_body_qchem_amplitudes: Initial perfect pairing two-body
            amplitudes using a qchem convention.
        do_optimization: Optimize the ansatz using BFGS.
        use_fast_gradients: Compute the parameter gradients using an anlytic
            form. Default to false (use finite difference gradients).
    """

    name: str
    hamiltonian_params: hamiltonian.HamiltonianParams
    heuristic_layers: Tuple[layer_spec.LayerSpec, ...] = attrs.field(
        converter=_to_tuple
    )
    do_pp: bool = True
    restricted: bool = False
    random_parameter_scale: float = 1.0
    n_optimization_restarts: int = 1
    seed: int = 0
    initial_orbital_rotation: Optional[np.ndarray] = attrs.field(
        default=None,
        converter=lambda v: _to_numpy(v) if v is not None else None,
    )
    initial_two_body_qchem_amplitudes: Optional[np.ndarray] = attrs.field(
        default=None,
        converter=lambda v: _to_numpy(v) if v is not None else None,
    )
    do_optimization: bool = True
    use_fast_gradients: bool = False
    path_prefix: str = ""

    @property
    def n_orb(self) -> int:
        return self.hamiltonian_params.n_orb

    @property
    def n_elec(self) -> int:
        return self.hamiltonian_params.n_elec

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_orb

    @property
    def n_pairs(self) -> int:
        return self.n_elec // 2

    @property
    def path_string(self) -> str:
        if self.path_prefix:
            return (
                self.path_prefix
                + config.OUTDIRS.DEFAULT_TRIAL_WAVEFUNCTION_DIRECTORY.strip(".")
                + self.name
            )
        else:
            return config.OUTDIRS.DEFAULT_TRIAL_WAVEFUNCTION_DIRECTORY + self.name

    @property
    def bitstrings(self) -> Iterable[Tuple[bool, ...]]:
        """The full set of bitstrings (determinants) for this wavefunction."""
        return bitstrings.get_bitstrings_a_b(n_orb=self.n_orb, n_elec=self.n_elec)

    def _json_dict_(self):
        simple_dict = attrs.asdict(self)
        simple_dict["hamiltonian_params"] = self.hamiltonian_params
        return simple_dict

    @property
    def qubits_jordan_wigner_ordered(self) -> Tuple[cirq.GridQubit, ...]:
        """Get the cirq qubits assuming a Jordan-Wigner ordering."""
        return qubit_maps.get_qubits_a_b(n_orb=self.n_orb)

    @property
    def qubits_linearly_connected(self) -> Tuple[cirq.GridQubit, ...]:
        """Get the cirq qubits assuming a linear connected qubit ordering."""
        return qubit_maps.get_qubits_a_b_reversed(n_orb=self.n_orb)

    @property
    def mode_qubit_map(self) -> Dict[fermion_mode.FermionicMode, cirq.GridQubit]:
        """Get the mapping between fermionic modes and cirq qubits."""
        return qubit_maps.get_mode_qubit_map_pp_plus(n_qubits=self.n_qubits)


@attrs.frozen
class TrialWavefunctionData(data.Data):
    """Class for storing a trial wavefunction's data.

    Args:
        params: The trial wavefunction parameters.
        ansatz_circuit: The circuit specifying the (pp) ansatz.
        superposition_circuit: The superposition circuit.
        hf_energy: The Hartree--Fock energy for the underlying molecule.
        ansatze_energy: The expected energy optimized wavefunction.
        fci_energy: The exact ground state energy of the underlying molecule.
        one_body_basis_change_mat: The one-body basis change matrix.
        one_body_params: The one-body variational parameters.
        two_body_params: The two-body variational parameters.
    """

    params: PerfectPairingPlusTrialWavefunctionParams
    ansatz_circuit: cirq.Circuit
    superposition_circuit: cirq.Circuit
    hf_energy: float
    ansatz_energy: float
    fci_energy: float
    one_body_basis_change_mat: np.ndarray = attrs.field(
        converter=_to_numpy, eq=attrs.cmp_using(eq=np.array_equal)
    )
    one_body_params: np.ndarray = attrs.field(
        converter=_to_numpy, eq=attrs.cmp_using(eq=np.array_equal)
    )
    two_body_params: np.ndarray = attrs.field(
        converter=_to_numpy, eq=attrs.cmp_using(eq=np.array_equal)
    )

    def _json_dict_(self):
        simple_dict = attrs.asdict(self)
        simple_dict["params"] = self.params
        return simple_dict


def get_rotated_hamiltonians(
    *,
    hamiltonian_data: hamiltonian.HamiltonianData,
    one_body_basis_change_mat: np.ndarray,
    mode_qubit_map: Mapping[fermion_mode.FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> Tuple[fqe.hamiltonians.hamiltonian.Hamiltonian, float, scipy.sparse.csc_matrix]:
    """A helper method that gets the hamiltonians in the basis of the trial_wf.

    Returns:
        The hamiltonian in FQE form, minus a constant energy shift.
        The constant part of the Hamiltonian missing from the FQE Hamiltonian.
        The qubit Hamiltonian as a sparse matrix.
    """
    n_qubits = len(mode_qubit_map)

    fqe_ham = hamiltonian_data.get_restricted_fqe_hamiltonian()
    e_core = hamiltonian_data.e_core

    mol_ham = hamiltonian_data.get_molecular_hamiltonian()
    mol_ham.rotate_basis(one_body_basis_change_mat)
    fermion_operator_ham = of.get_fermion_operator(mol_ham)

    reorder_func = fqe_conversion.get_reorder_func(
        mode_qubit_map=mode_qubit_map, ordered_qubits=ordered_qubits
    )
    fermion_operator_ham_qubit_ordered = of.reorder(
        fermion_operator_ham, reorder_func, num_modes=n_qubits
    )

    sparse_qubit_ham = of.get_sparse_operator(fermion_operator_ham_qubit_ordered)

    return fqe_ham, e_core, sparse_qubit_ham
