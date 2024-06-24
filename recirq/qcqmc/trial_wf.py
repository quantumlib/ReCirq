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

import abc
import copy
import itertools
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import attrs
import cirq
import fqe
import fqe.algorithm.low_rank
import fqe.hamiltonians.restricted_hamiltonian as fqe_hams
import fqe.openfermion_utils
import fqe.wavefunction as fqe_wfn
import numpy as np
import numpy.typing as npt
import openfermion as of
import scipy.optimize
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.sparse import csc_matrix

from recirq.qcqmc import afqmc_circuits, bitstrings, config, data, hamiltonian


@attrs.frozen
class FermionicMode:
    """A specification of a fermionic mode.

    Args:
        orb_ind: The spatial orbital index.
        spin: The spin state of the fermion mode (up or down (alpha or beta)).
    """

    orb_ind: int
    spin: str

    def __attrs_post_init__(self):
        if self.spin not in ["a", "b"]:
            raise ValueError(
                "Spin must be either a or b for spin alpha(up) or beta(down) respectively."
            )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"

    def _json_dict_(self):
        return attrs.asdict(self)

    @property
    def openfermion_standard_index(self) -> int:
        return 2 * self.orb_ind + (0 if self.spin == "a" else 1)


@attrs.frozen
class LayerSpec:
    """A specification of a hardware-efficient layer of gates.

    Args:
        base_gate: 'charge_charge' for the e^{i n_i n_j} gate and 'givens' for a givens rotation.
        layout: Should be 'in_pair', 'cross_pair', or 'cross_spin' only.
    """

    base_gate: str
    layout: str

    def __post_init__(self):
        if self.base_gate not in ["charge_charge", "givens"]:
            raise ValueError(
                f'base_gate is set to {self.base_gate}, it should be either "charge_charge or "givens".'
            )
        if self.layout not in ["in_pair", "cross_pair", "cross_spin"]:
            raise ValueError(
                f'layout is set to {self.layout}, it should be either "cross_pair", "in_pair", or "cross_spin".'
            )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"

    def _json_dict_(self):
        return attrs.asdict(self)


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


def _to_tuple(x: Iterable[LayerSpec]) -> Sequence[LayerSpec]:
    return tuple(x)


@attrs.frozen(repr=False)
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
    heuristic_layers: Tuple[LayerSpec, ...] = attrs.field(converter=_to_tuple)
    do_pp: bool = True
    restricted: bool = False
    random_parameter_scale: float = 1.0
    n_optimization_restarts: int = 1
    seed: int = 0
    initial_orbital_rotation: Optional[np.ndarray] = attrs.field(
        default=None, converter=lambda v: _to_numpy(v) if v is not None else None
    )
    initial_two_body_qchem_amplitudes: Optional[np.ndarray] = attrs.field(
        default=None, converter=lambda v: _to_numpy(v) if v is not None else None
    )
    do_optimization: bool = True
    use_fast_gradients: bool = False

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
        return config.OUTDIRS.DEFAULT_TRIAL_WAVEFUNCTION_DIRECTORY + self.name

    @property
    def bitstrings(self) -> Iterable[Tuple[bool, ...]]:
        return bitstrings.get_bitstrings_a_b(n_orb=self.n_orb, n_elec=self.n_elec)

    def _json_dict_(self):
        return attrs.asdict(self)

    @property
    def qubits_jordan_wigner_ordered(self) -> Tuple[cirq.GridQubit, ...]:
        return _get_qubits_a_b(n_orb=self.n_orb)

    @property
    def qubits_linearly_connected(self) -> Tuple[cirq.GridQubit, ...]:
        return _get_qubits_a_b_reversed(n_orb=self.n_orb)

    @property
    def mode_qubit_map(self) -> Dict[FermionicMode, cirq.GridQubit]:
        return _get_mode_qubit_map_pp_plus(n_qubits=self.n_qubits)


def _get_qubits_a_b(*, n_orb: int) -> Tuple[cirq.GridQubit, ...]:
    """Get grid alpha/beta grid qubits in ascending order.

    Args:
        n_orb: The number of spatial orbitals.

    This ordering creates qubits to facilitate a Jordan Wigner string
    threading through a row of alpha orbitals in ascending order followed by a
    row of beta orbitals in ascending order.
    """
    return tuple(
        [cirq.GridQubit(0, i) for i in range(n_orb)]
        + [cirq.GridQubit(1, i) for i in range(n_orb)]
    )


def _get_qubits_a_b_reversed(*, n_orb: int) -> Tuple[cirq.GridQubit, ...]:
    """Get grid quibts with correct spin ordering.

    This ordering creates qubits to facilitate operations that need a linearly
    connected array of qubits with the order threading through a row of alpha
    orbitals in ascending order followed by a row of beta orbitals in
    descending order.

    Args:
        n_orb: The number of spatial orbitals.
    """
    return tuple(
        [cirq.GridQubit(0, i) for i in range(n_orb)]
        + [cirq.GridQubit(1, i) for i in reversed(range(n_orb))]
    )


def _get_fermion_qubit_map_pp_plus(*, n_qubits: int) -> Dict[int, cirq.GridQubit]:
    """Dispatcher for qubit mappings."""
    if n_qubits == 4:
        return get_4_qubit_fermion_qubit_map()
    elif n_qubits == 8:
        return get_8_qubit_fermion_qubit_map()
    elif n_qubits == 12:
        return get_12_qubit_fermion_qubit_map()
    elif n_qubits == 16:
        return get_16_qubit_fermion_qubit_map()
    else:
        raise NotImplementedError()


def _get_mode_qubit_map_pp_plus(
    *, n_qubits: int
) -> Dict[FermionicMode, cirq.GridQubit]:
    """A map from Fermionic modes to qubits for our particular circuits.

    This function dispatches to _get_fermion_qubit_map_pp_plus, and ultimately
    to get_X_qubit_fermion_qubit_map for specific values of X but it translates
    this logic to a new system that uses the FermionicMode class rather than
    opaque combinations of integers and strings.

    Args:
        n_qubits: The number of qubits.
    """
    old_fermion_qubit_map = _get_fermion_qubit_map_pp_plus(n_qubits=n_qubits)

    n_orb = n_qubits // 2

    mode_qubit_map = {}

    for i in range(n_orb):
        mode_qubit_map[FermionicMode(i, "a")] = old_fermion_qubit_map[2 * i]
        mode_qubit_map[FermionicMode(i, "b")] = old_fermion_qubit_map[2 * i + 1]

    return mode_qubit_map


def _get_reorder_func(
    *,
    mode_qubit_map: Mapping[FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> Callable[[int, int], int]:
    """This is a helper function that allows us to reorder fermionic modes.

    Under the Jordan-Wigner transform, each fermionic mode is assigned to a
    qubit. If we are provided an openfermion FermionOperator with the modes
    assigned to qubits as described by mode_qubit_map this function gives us a
    reorder_func that we can use to reorder the modes (with
    openfermion.reorder(...)) so that they match the order of the qubits in
    ordered_qubits. This is necessary to make a correspondence between
    fermionic operators / wavefunctions and their qubit counterparts.

    Args:
        mode_qubit_map: A dict that shows how each FermionicMode is mapped to a qubit.
        ordered_qubits: An ordered sequence of qubits.
    """
    qubits = list(mode_qubit_map.values())
    assert len(qubits) == len(ordered_qubits)

    # We sort the key: value pairs by the order of the values (qubits) in
    # ordered_qubits.
    sorted_mapping = list(mode_qubit_map.items())
    sorted_mapping.sort(key=lambda x: ordered_qubits.index(x[1]))

    remapping_map = {}
    for i, (mode, _) in enumerate(sorted_mapping):
        openfermion_index = 2 * mode.orb_ind + (0 if mode.spin == "a" else 1)
        remapping_map[openfermion_index] = i

    def remapper(index: int, _: int) -> int:
        """A function that maps from the old index to the new one.

        The _ argument is because it's expected by openfermion.reorder"""
        return remapping_map[index]

    return remapper


def _get_pp_plus_gate_generators(
    *, n_elec: int, heuristic_layers: Tuple[LayerSpec, ...], do_pp: bool = True
) -> List[of.FermionOperator]:
    heuristic_gate_generators = get_heuristic_gate_generators(n_elec, heuristic_layers)
    if not do_pp:
        return heuristic_gate_generators

    n_pairs = n_elec // 2
    pair_gate_generators = get_pair_hopping_gate_generators(n_pairs, n_elec)
    return pair_gate_generators + heuristic_gate_generators


def _get_ansatz_qubit_wf(
    *, ansatz_circuit: cirq.Circuit, ordered_qubits: Sequence[cirq.Qid]
):
    return cirq.final_state_vector(
        ansatz_circuit, qubit_order=list(ordered_qubits), dtype=np.complex128
    )


def _get_superposition_wf(
    *, superposition_circuit: cirq.Circuit, ordered_qubits: Sequence[cirq.Qid]
) -> np.ndarray:
    return cirq.final_state_vector(
        superposition_circuit, qubit_order=list(ordered_qubits), dtype=np.complex128
    )


@attrs.frozen
class TrialWavefunctionData(data.Data):
    """Class for storing a trial wavefunction's data."""

    params: PerfectPairingPlusTrialWavefunctionParams
    ansatz_circuit: cirq.Circuit
    superposition_circuit: cirq.Circuit
    hf_energy: float
    ansatz_energy: float
    fci_energy: float
    one_body_basis_change_mat: np.ndarray = attrs.field(converter=_to_numpy)
    one_body_params: np.ndarray = attrs.field(converter=_to_numpy)
    two_body_params: np.ndarray = attrs.field(converter=_to_numpy)

    def _json_dict_(self):
        # return cirq.dataclass_json_dict(self)
        return attrs.asdict(self)


def get_and_check_energy(
    *,
    hamiltonian_data: hamiltonian.HamiltonianData,
    ansatz_circuit: cirq.Circuit,
    one_body_params: np.ndarray,
    two_body_params: np.ndarray,
    one_body_basis_change_mat: np.ndarray,
    params: PerfectPairingPlusTrialWavefunctionParams,
) -> Tuple[float, float]:
    ansatz_qubit_wf = _get_ansatz_qubit_wf(
        ansatz_circuit=ansatz_circuit,
        ordered_qubits=params.qubits_jordan_wigner_ordered,
    )

    fqe_ham, e_core, sparse_ham = get_rotated_hamiltonians(
        hamiltonian_data=hamiltonian_data,
        one_body_basis_change_mat=one_body_basis_change_mat,
        mode_qubit_map=params.mode_qubit_map,
        ordered_qubits=params.qubits_jordan_wigner_ordered,
    )

    initial_wf = fqe.Wavefunction([[params.n_elec, 0, params.n_orb]])
    initial_wf.set_wfn(strategy="hartree-fock")

    hf_energy = initial_wf.expectationValue(fqe_ham) + e_core

    fqe_wf, unrotated_fqe_wf = get_evolved_wf(
        one_body_params=one_body_params,
        two_body_params=two_body_params,
        wf=initial_wf,
        gate_generators=_get_pp_plus_gate_generators(
            n_elec=params.n_elec,
            heuristic_layers=params.heuristic_layers,
            do_pp=params.do_pp,
        ),
        n_orb=params.n_orb,
        restricted=params.restricted,
        initial_orbital_rotation=params.initial_orbital_rotation,
    )

    ansatz_energy = get_energy_and_check_sanity(
        circuit_wf=ansatz_qubit_wf,
        fqe_wf=fqe_wf,
        unrotated_fqe_wf=unrotated_fqe_wf,
        fqe_ham=fqe_ham,
        sparse_ham=sparse_ham,
        e_core=e_core,
        mode_qubit_map=params.mode_qubit_map,
        ordered_qubits=params.qubits_jordan_wigner_ordered,
    )

    return ansatz_energy, hf_energy


def build_pp_plus_trial_wavefunction(
    params: PerfectPairingPlusTrialWavefunctionParams,
    *,
    dependencies: Dict[data.Params, data.Data],
    do_print: bool = False,
) -> TrialWavefunctionData:
    """Builds a TrialWavefunctionData from a TrialWavefunctionParams"""

    if do_print:
        print("Building Trial Wavefunction")
    np.random.seed(params.seed)
    hamiltonian_data = dependencies[params.hamiltonian_params]
    assert isinstance(hamiltonian_data, hamiltonian.HamiltonianData)

    assert (
        params.n_orb == params.n_elec
    )  ## Necessary for perfect pairing wavefunction to make sense.

    if params.do_optimization:
        (
            one_body_params,
            two_body_params,
            one_body_basis_change_mat,
        ) = get_pp_plus_params(
            hamiltonian_data=hamiltonian_data,
            restricted=params.restricted,
            random_parameter_scale=params.random_parameter_scale,
            initial_orbital_rotation=params.initial_orbital_rotation,
            heuristic_layers=params.heuristic_layers,
            do_pp=params.do_pp,
            n_optimization_restarts=params.n_optimization_restarts,
            do_print=do_print,
            use_fast_gradients=params.use_fast_gradients,
        )
    else:
        if (
            params.initial_two_body_qchem_amplitudes is None
            or params.initial_orbital_rotation is not None
        ):
            raise NotImplementedError("TODO: Implement whatever isn't finished here.")

        n_one_body_params = params.n_orb * (params.n_orb - 1)
        one_body_params = np.zeros(n_one_body_params)
        one_body_basis_change_mat = np.diag(np.ones(params.n_orb * 2))
        two_body_params = get_two_body_params_from_qchem_amplitudes(
            params.initial_two_body_qchem_amplitudes
        )

    (superposition_circuit, ansatz_circuit) = get_circuits(
        two_body_params=two_body_params,
        n_orb=params.n_orb,
        n_elec=params.n_elec,
        heuristic_layers=params.heuristic_layers,
    )

    ansatz_energy, hf_energy = get_and_check_energy(
        hamiltonian_data=hamiltonian_data,
        ansatz_circuit=ansatz_circuit,
        params=params,
        one_body_params=one_body_params,
        two_body_params=two_body_params,
        one_body_basis_change_mat=one_body_basis_change_mat,
    )

    return TrialWavefunctionData(
        params=params,
        ansatz_circuit=ansatz_circuit,
        superposition_circuit=superposition_circuit,
        hf_energy=hf_energy,
        ansatz_energy=ansatz_energy,
        fci_energy=hamiltonian_data.e_fci,
        one_body_basis_change_mat=one_body_basis_change_mat,
        one_body_params=one_body_params,
        two_body_params=two_body_params,
    )


def get_rotated_hamiltonians(
    *,
    hamiltonian_data: hamiltonian.HamiltonianData,
    one_body_basis_change_mat: np.ndarray,
    mode_qubit_map: Mapping[FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> Tuple[fqe.hamiltonians.hamiltonian.Hamiltonian, float, csc_matrix]:
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

    reorder_func = _get_reorder_func(
        mode_qubit_map=mode_qubit_map, ordered_qubits=ordered_qubits
    )
    fermion_operator_ham_qubit_ordered = of.reorder(
        fermion_operator_ham, reorder_func, num_modes=n_qubits
    )

    sparse_qubit_ham = of.get_sparse_operator(fermion_operator_ham_qubit_ordered)

    return fqe_ham, e_core, sparse_qubit_ham


def get_energy_and_check_sanity(
    *,
    circuit_wf: np.ndarray,
    fqe_wf: fqe_wfn.Wavefunction,
    unrotated_fqe_wf: fqe_wfn.Wavefunction,
    fqe_ham: fqe.hamiltonians.hamiltonian.Hamiltonian,
    sparse_ham: csc_matrix,
    e_core: float,
    mode_qubit_map: Mapping[FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> float:
    """A method that checks for consistency and returns the ansatz energy."""

    unrotated_fqe_wf_as_cirq = convert_fqe_wf_to_cirq(
        fqe_wf=unrotated_fqe_wf,
        mode_qubit_map=mode_qubit_map,
        ordered_qubits=ordered_qubits,
    )
    ansatz_energy = np.real_if_close(
        (np.conj(circuit_wf) @ sparse_ham @ circuit_wf)
    ).item()
    assert isinstance(ansatz_energy, float)

    fqe_energy = np.real(fqe_wf.expectationValue(fqe_ham) + e_core)
    print(csc_matrix(circuit_wf))
    print(csc_matrix(unrotated_fqe_wf_as_cirq))
    np.testing.assert_array_almost_equal(ansatz_energy, fqe_energy)
    np.testing.assert_array_almost_equal(
        circuit_wf, unrotated_fqe_wf_as_cirq, decimal=5
    )
    return ansatz_energy


def get_4_qubit_fermion_qubit_map() -> Dict[int, cirq.GridQubit]:
    """A helper function that provides the fermion qubit map for 4 qubits.

    We map the fermionic orbitals to grid qubits like so:
    3 1
    2 0
    """
    fermion_index_to_qubit_map = {
        2: cirq.GridQubit(0, 0),
        3: cirq.GridQubit(1, 0),
        0: cirq.GridQubit(0, 1),
        1: cirq.GridQubit(1, 1),
    }

    return fermion_index_to_qubit_map


def get_4_qubit_pp_circuits(
    *, two_body_params: np.ndarray, n_elec: int, heuristic_layers: Tuple[LayerSpec, ...]
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A helper function that builds the circuits for the four qubit ansatz.

    We map the fermionic orbitals to grid qubits like so:
    3 1
    2 0
    """
    assert n_elec == 2

    fermion_index_to_qubit_map = get_4_qubit_fermion_qubit_map()
    geminal_gate = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[0], inline_control=True
    )

    ansatz_circuit = cirq.Circuit(
        cirq.decompose(
            geminal_gate.on(
                fermion_index_to_qubit_map[0],
                fermion_index_to_qubit_map[1],
                fermion_index_to_qubit_map[2],
                fermion_index_to_qubit_map[3],
            )
        )
    )

    heuristic_layer_circuit = get_heuristic_circuit(
        heuristic_layers, n_elec, two_body_params[1:], fermion_index_to_qubit_map
    )

    ansatz_circuit += heuristic_layer_circuit

    indicator = fermion_index_to_qubit_map[2]
    superposition_circuit = cirq.Circuit([cirq.H(indicator) + ansatz_circuit])
    ansatz_circuit = cirq.Circuit([cirq.X(indicator) + ansatz_circuit])

    return superposition_circuit, ansatz_circuit


def get_8_qubit_fermion_qubit_map():
    """A helper function that provides the fermion qubit map for 8 qubits.

    We map the fermionic orbitals to grid qubits like so:
    3 5 1 7
    2 4 0 6
    """

    # Linear connectivity is fine.
    # This ordering is dictated by the way we specify perfect pairing (i)
    # Elsewhere we generate the perfect pairing parameters using a specific
    # convention for how we index the FermionOperators (which is itself)
    # partly dictated by the OpenFermion conventions. Here we choose a
    # mapping between the indices of these FermionOperators and the qubits in our
    # grid that allows for the perfect pairing pairs to be in squares of four.
    fermion_index_to_qubit_map = {
        2: cirq.GridQubit(0, 0),
        3: cirq.GridQubit(1, 0),
        4: cirq.GridQubit(0, 1),
        5: cirq.GridQubit(1, 1),
        0: cirq.GridQubit(0, 2),
        1: cirq.GridQubit(1, 2),
        6: cirq.GridQubit(0, 3),
        7: cirq.GridQubit(1, 3),
    }

    return fermion_index_to_qubit_map


def get_8_qubit_circuits(
    *, two_body_params: np.ndarray, n_elec: int, heuristic_layers: Tuple[LayerSpec, ...]
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A helper function that builds the circuits for the four qubit ansatz.

    We map the fermionic orbitals to grid qubits like so:
    3 5 1 7
    2 4 0 6
    """
    fermion_index_to_qubit_map = get_8_qubit_fermion_qubit_map()

    geminal_gate_1 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[0], inline_control=True
    )
    geminal_gate_2 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[1], inline_control=True
    )

    # We'll add the initial bit flips later.
    ansatz_circuit = cirq.Circuit(
        cirq.decompose(
            geminal_gate_1.on(
                fermion_index_to_qubit_map[2],
                fermion_index_to_qubit_map[3],
                fermion_index_to_qubit_map[4],
                fermion_index_to_qubit_map[5],
            )
        ),
        cirq.decompose(
            geminal_gate_2.on(
                fermion_index_to_qubit_map[0],
                fermion_index_to_qubit_map[1],
                fermion_index_to_qubit_map[6],
                fermion_index_to_qubit_map[7],
            )
        ),
    )

    heuristic_layer_circuit = get_heuristic_circuit(
        heuristic_layers, n_elec, two_body_params[2:], fermion_index_to_qubit_map
    )

    ansatz_circuit += heuristic_layer_circuit

    superposition_circuit = (
        cirq.Circuit(
            [
                cirq.H(fermion_index_to_qubit_map[0]),
                cirq.CNOT(fermion_index_to_qubit_map[0], fermion_index_to_qubit_map[6]),
                cirq.SWAP(fermion_index_to_qubit_map[0], fermion_index_to_qubit_map[4]),
            ]
        )
        + ansatz_circuit
    )

    ansatz_circuit = (
        cirq.Circuit(
            [
                cirq.X(fermion_index_to_qubit_map[4]),
                cirq.X(fermion_index_to_qubit_map[6]),
            ]
        )
        + ansatz_circuit
    )

    return superposition_circuit, ansatz_circuit


def get_12_qubit_fermion_qubit_map():
    """A helper function that provides the fermion qubit map for 12 qubits.

    We map the fermionic orbitals to grid qubits like so:
    5 7 3 9 1 11
    4 6 2 8 0 10
    """

    # Linear connectivity is fine.
    # This ordering is dictated by the way we specify perfect pairing (i)
    # Elsewhere we generate the perfect pairing parameters using a specific
    # convention for how we index the FermionOperators (which is itself)
    # partly dictated by the OpenFermion conventions. Here we choose a
    # mapping between the indices of these FermionOperators and the qubits in our
    # grid that allows for the perfect pairing pairs to be in squares of four.
    fermion_index_to_qubit_map = {
        4: cirq.GridQubit(0, 0),
        5: cirq.GridQubit(1, 0),
        6: cirq.GridQubit(0, 1),
        7: cirq.GridQubit(1, 1),
        2: cirq.GridQubit(0, 2),
        3: cirq.GridQubit(1, 2),
        8: cirq.GridQubit(0, 3),
        9: cirq.GridQubit(1, 3),
        0: cirq.GridQubit(0, 4),
        1: cirq.GridQubit(1, 4),
        10: cirq.GridQubit(0, 5),
        11: cirq.GridQubit(1, 5),
    }

    return fermion_index_to_qubit_map


def get_12_qubit_circuits(
    *, two_body_params: np.ndarray, n_elec: int, heuristic_layers: Tuple[LayerSpec, ...]
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A helper function that builds the circuits for the four qubit ansatz.

    We map the fermionic orbitals to grid qubits like so:
    5 7 3 9 1 11
    4 6 2 8 0 10
    """

    fermion_index_to_qubit_map = get_12_qubit_fermion_qubit_map()

    geminal_gate_1 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[0], inline_control=True
    )
    geminal_gate_2 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[1], inline_control=True
    )
    geminal_gate_3 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[2], inline_control=True
    )

    # We'll add the initial bit flips later.
    ansatz_circuit = cirq.Circuit(
        cirq.decompose(
            geminal_gate_1.on(
                fermion_index_to_qubit_map[4],
                fermion_index_to_qubit_map[5],
                fermion_index_to_qubit_map[6],
                fermion_index_to_qubit_map[7],
            )
        ),
        cirq.decompose(
            geminal_gate_2.on(
                fermion_index_to_qubit_map[2],
                fermion_index_to_qubit_map[3],
                fermion_index_to_qubit_map[8],
                fermion_index_to_qubit_map[9],
            )
        ),
        cirq.decompose(
            geminal_gate_3.on(
                fermion_index_to_qubit_map[0],
                fermion_index_to_qubit_map[1],
                fermion_index_to_qubit_map[10],
                fermion_index_to_qubit_map[11],
            )
        ),
    )

    heuristic_layer_circuit = get_heuristic_circuit(
        heuristic_layers, n_elec, two_body_params[3:], fermion_index_to_qubit_map
    )

    ansatz_circuit += heuristic_layer_circuit

    superposition_circuit = (
        cirq.Circuit(
            [
                cirq.H(fermion_index_to_qubit_map[8]),
                cirq.CNOT(fermion_index_to_qubit_map[8], fermion_index_to_qubit_map[0]),
                cirq.CNOT(fermion_index_to_qubit_map[8], fermion_index_to_qubit_map[2]),
                cirq.SWAP(
                    fermion_index_to_qubit_map[0], fermion_index_to_qubit_map[10]
                ),
                cirq.SWAP(fermion_index_to_qubit_map[2], fermion_index_to_qubit_map[6]),
            ]
        )
        + ansatz_circuit
    )

    ansatz_circuit = (
        cirq.Circuit(
            [
                cirq.X(fermion_index_to_qubit_map[6]),
                cirq.X(fermion_index_to_qubit_map[8]),
                cirq.X(fermion_index_to_qubit_map[10]),
            ]
        )
        + ansatz_circuit
    )

    return superposition_circuit, ansatz_circuit


def get_16_qubit_fermion_qubit_map():
    """A helper function that provides the fermion qubit map for 16 qubits.

    We map the fermionic orbitals to grid qubits like so:
    7 9 5 11 3 13 1 15
    6 8 4 10 2 12 0 14
    """

    # Linear connectivity is fine.
    # This ordering is dictated by the way we specify perfect pairing (i)
    # Elsewhere we generate the perfect pairing parameters using a specific
    # convention for how we index the FermionOperators (which is itself)
    # partly dictated by the OpenFermion conventions. Here we choose a
    # mapping between the indices of these FermionOperators and the qubits in our
    # grid that allows for the perfect pairing pairs to be in squares of four.
    fermion_index_to_qubit_map = {
        6: cirq.GridQubit(0, 0),
        7: cirq.GridQubit(1, 0),
        8: cirq.GridQubit(0, 1),
        9: cirq.GridQubit(1, 1),
        4: cirq.GridQubit(0, 2),
        5: cirq.GridQubit(1, 2),
        10: cirq.GridQubit(0, 3),
        11: cirq.GridQubit(1, 3),
        2: cirq.GridQubit(0, 4),
        3: cirq.GridQubit(1, 4),
        12: cirq.GridQubit(0, 5),
        13: cirq.GridQubit(1, 5),
        0: cirq.GridQubit(0, 6),
        1: cirq.GridQubit(1, 6),
        14: cirq.GridQubit(0, 7),
        15: cirq.GridQubit(1, 7),
    }

    return fermion_index_to_qubit_map


def get_16_qubit_circuits(
    *, two_body_params: np.ndarray, n_elec: int, heuristic_layers: Tuple[LayerSpec, ...]
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A helper function that builds the circuits for the four qubit ansatz.

    We map the fermionic orbitals to grid qubits like so:
    7 9 5 11 3 13 1 15
    6 8 4 10 2 12 0 14
    """
    fermion_index_to_qubit_map = get_16_qubit_fermion_qubit_map()

    geminal_gate_1 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[0], inline_control=True
    )
    geminal_gate_2 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[1], inline_control=True
    )
    geminal_gate_3 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[2], inline_control=True
    )
    geminal_gate_4 = afqmc_circuits.GeminalStatePreparationGate(
        two_body_params[3], inline_control=True
    )

    # We'll add the initial bit flips later.
    ansatz_circuit = cirq.Circuit(
        cirq.decompose(
            geminal_gate_1.on(
                fermion_index_to_qubit_map[6],
                fermion_index_to_qubit_map[7],
                fermion_index_to_qubit_map[8],
                fermion_index_to_qubit_map[9],
            )
        ),
        cirq.decompose(
            geminal_gate_2.on(
                fermion_index_to_qubit_map[4],
                fermion_index_to_qubit_map[5],
                fermion_index_to_qubit_map[10],
                fermion_index_to_qubit_map[11],
            )
        ),
        cirq.decompose(
            geminal_gate_3.on(
                fermion_index_to_qubit_map[2],
                fermion_index_to_qubit_map[3],
                fermion_index_to_qubit_map[12],
                fermion_index_to_qubit_map[13],
            )
        ),
        cirq.decompose(
            geminal_gate_4.on(
                fermion_index_to_qubit_map[0],
                fermion_index_to_qubit_map[1],
                fermion_index_to_qubit_map[14],
                fermion_index_to_qubit_map[15],
            )
        ),
    )

    heuristic_layer_circuit = get_heuristic_circuit(
        heuristic_layers, n_elec, two_body_params[4:], fermion_index_to_qubit_map
    )

    ansatz_circuit += heuristic_layer_circuit

    superposition_circuit = (
        cirq.Circuit(
            [
                cirq.H(fermion_index_to_qubit_map[10]),
                cirq.CNOT(
                    fermion_index_to_qubit_map[10], fermion_index_to_qubit_map[2]
                ),
                cirq.SWAP(
                    fermion_index_to_qubit_map[2], fermion_index_to_qubit_map[12]
                ),
                cirq.CNOT(
                    fermion_index_to_qubit_map[10], fermion_index_to_qubit_map[4]
                ),
                cirq.CNOT(
                    fermion_index_to_qubit_map[12], fermion_index_to_qubit_map[0]
                ),
                cirq.SWAP(fermion_index_to_qubit_map[4], fermion_index_to_qubit_map[8]),
                cirq.SWAP(
                    fermion_index_to_qubit_map[0], fermion_index_to_qubit_map[14]
                ),
            ]
        )
        + ansatz_circuit
    )

    ansatz_circuit = (
        cirq.Circuit(
            [
                cirq.X(fermion_index_to_qubit_map[8]),
                cirq.X(fermion_index_to_qubit_map[10]),
                cirq.X(fermion_index_to_qubit_map[12]),
                cirq.X(fermion_index_to_qubit_map[14]),
            ]
        )
        + ansatz_circuit
    )

    return superposition_circuit, ansatz_circuit


def get_circuits(
    *,
    two_body_params: np.ndarray,
    # from wf_params:
    n_orb: int,
    n_elec: int,
    heuristic_layers: Tuple[LayerSpec, ...],
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A function that runs a specialized method to get the ansatz circuits."""

    # TODO(?): Just input one of these quantities.
    if n_orb != n_elec:
        raise ValueError("n_orb must equal n_elec.")

    circ_funcs = {
        2: get_4_qubit_pp_circuits,
        4: get_8_qubit_circuits,
        6: get_12_qubit_circuits,
        8: get_16_qubit_circuits,
    }
    try:
        circ_func = circ_funcs[n_orb]
    except KeyError:
        raise NotImplementedError(f"No circuits for n_orb = {n_orb}")

    return circ_func(
        two_body_params=two_body_params,
        n_elec=n_elec,
        heuristic_layers=heuristic_layers,
    )


def get_two_body_params_from_qchem_amplitudes(
    qchem_amplitudes: np.ndarray,
) -> np.ndarray:
    """Translates perfect pairing amplitudes from qchem to rotation angles.

    qchem style: 1 |1100> + t_i |0011>
    our style: cos(\theta_i) |1100> + sin(\theta_i) |0011>
    """

    two_body_params = np.arccos(1 / np.sqrt(1 + qchem_amplitudes**2)) * np.sign(
        qchem_amplitudes
    )

    # Numpy casts the array improperly to a float when we only have one parameter.
    two_body_params = np.atleast_1d(two_body_params)

    return two_body_params


#################### Here be dragons.###########################################


def convert_fqe_wf_to_cirq(
    fqe_wf: fqe_wfn.Wavefunction,
    mode_qubit_map: Mapping[FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> np.ndarray:
    """Converts an FQE wavefunction to one on qubits with a particular ordering.

    Args:
        fqe_wf: The FQE wavefunction.
        mode_qubit_map: A mapping from fermion modes to cirq qubits.
        ordered_qubits:
    """
    n_qubits = len(mode_qubit_map)
    fermion_op = fqe.openfermion_utils.fqe_to_fermion_operator(fqe_wf)

    reorder_func = _get_reorder_func(
        mode_qubit_map=mode_qubit_map, ordered_qubits=ordered_qubits
    )
    fermion_op = of.reorder(fermion_op, reorder_func, num_modes=n_qubits)

    qubit_op = of.jordan_wigner(fermion_op)

    return fqe.qubit_wavefunction_from_vacuum(
        qubit_op, list(cirq.LineQubit.range(n_qubits))
    )


def get_one_body_cluster_coef(
    params: np.ndarray, n_orb: int, restricted: bool
) -> npt.NDArray[np.complex128]:
    if restricted:
        one_body_cluster_op = np.zeros((n_orb, n_orb), dtype=np.complex128)
    else:
        one_body_cluster_op = np.zeros((2 * n_orb, 2 * n_orb), dtype=np.complex128)
    param_num = 0

    for i in range(n_orb):
        for j in range(i):
            one_body_cluster_op[i, j] = params[param_num]
            one_body_cluster_op[j, i] = -params[param_num]
            param_num += 1

    if not restricted:
        for i in range(n_orb, 2 * n_orb):
            for j in range(n_orb, i):
                one_body_cluster_op[i, j] = params[param_num]
                one_body_cluster_op[j, i] = -params[param_num]
                param_num += 1

    return one_body_cluster_op


def get_evolved_wf(
    one_body_params: np.ndarray,
    two_body_params: np.ndarray,
    wf: fqe.Wavefunction,
    gate_generators: List[of.FermionOperator],
    n_orb: int,
    restricted: bool = True,
    initial_orbital_rotation: Optional[np.ndarray] = None,
) -> Tuple[fqe.Wavefunction, fqe.Wavefunction]:
    """Get the wavefunction evaluated for this set of variational parameters.

    Args:
        one_body_params: The variational parameters for the one-body terms in the ansatz.
        two_body_params: The variational parameters for the two-body terms in the ansatz.
        wf: The FQE wavefunction to evolve.
        gate_generators: The generators of the two-body interaction terms.
        n_orb: The number of spatial orbitals.
        restricted: Whether the ansatz is restricted or not.
        initial_orbital_rotation: Any initial orbital rotation to prepend to the circuit.

    Returs:
        rotated_wf: the evolved wavefunction
        wf: The original wavefunction
    """
    param_num = 0
    for gate_generator in gate_generators:
        wf = wf.time_evolve(two_body_params[param_num], gate_generator)
        param_num += 1

    one_body_cluster_op = get_one_body_cluster_coef(
        one_body_params, n_orb, restricted=restricted
    )

    if restricted:
        one_body_ham = fqe.get_restricted_hamiltonian((1j * one_body_cluster_op,))
    else:
        one_body_ham = fqe.get_sso_hamiltonian((1j * one_body_cluster_op,))

    rotated_wf = wf.time_evolve(1.0, one_body_ham)

    if initial_orbital_rotation is not None:
        rotated_wf = fqe.algorithm.low_rank.evolve_fqe_givens(
            rotated_wf, initial_orbital_rotation
        )

    return rotated_wf, wf


def get_pair_hopping_gate_generators(
    n_pairs: int, n_elec: int
) -> List[of.FermionOperator]:
    """Get the generators of the pair-hopping unitaries.

    Args:
        n_pairs: The number of pair coupling terms.
        n_elec: The total number of electrons.

    Returns:
        A list of gate generators
    """
    gate_generators = []
    for pair in range(n_pairs):
        to_a = n_elec + 2 * pair
        to_b = n_elec + 2 * pair + 1
        from_a = n_elec - 2 * pair - 2
        from_b = n_elec - 2 * pair - 1

        fop_string = f"{to_b} {to_a} {from_b}^ {from_a}^"

        gate_generator = of.FermionOperator(fop_string, 1.0)
        gate_generator = 1j * (gate_generator - of.hermitian_conjugated(gate_generator))
        gate_generators.append(gate_generator)

    return gate_generators


def get_indices_heuristic_layer_in_pair(n_elec: int) -> Iterator[Tuple[int, int]]:
    """Get the indicies for the heuristic layers.

    Args:
        n_elec: The number of electrons

    Returns:
        An iterator of the indices
    """
    n_pairs = n_elec // 2

    for pair in range(n_pairs):
        to_a = n_elec + 2 * pair
        to_b = n_elec + 2 * pair + 1
        from_a = n_elec - 2 * pair - 2
        from_b = n_elec - 2 * pair - 1
        yield (from_a, to_a)
        yield (from_b, to_b)


def get_indices_heuristic_layer_cross_pair(n_elec) -> Iterator[Tuple[int, int]]:
    """Indices that couple adjacent pairs.

    Args:
        n_elec: The number of electrons

    Returns:
        An iterator of the indices
    """
    n_pairs = n_elec // 2

    for pair in range(n_pairs - 1):
        to_a = n_elec + 2 * pair
        to_b = n_elec + 2 * pair + 1
        from_next_a = n_elec - 2 * (pair + 1) - 2
        from_next_b = n_elec - 2 * (pair + 1) - 1
        yield (to_a, from_next_a)
        yield (to_b, from_next_b)


def get_indices_heuristic_layer_cross_spin(n_elec) -> Iterator[Tuple[int, int]]:
    """Get indices that couple the two spin sectors.

    Args:
        n_elec: The number of electrons

    Returns:
        An iterator of the indices that couple spin sectors.
    """
    n_pairs = n_elec // 2

    for pair in range(n_pairs):
        to_a = n_elec + 2 * pair
        to_b = n_elec + 2 * pair + 1
        from_a = n_elec - 2 * pair - 2
        from_b = n_elec - 2 * pair - 1
        yield (to_a, to_b)
        yield (from_a, from_b)


def get_charge_charge_generator(indices: Tuple[int, int]) -> of.FermionOperator:
    """Returns the generator for density evolution between the indices

    Args:
        indices: The indices to for charge-charge terms.:w

    Returns:
        The generator for density evolution for this pair of electrons.
    """

    fop_string = "{:d}^ {:d} {:d}^ {:d}".format(
        indices[0], indices[0], indices[1], indices[1]
    )
    gate_generator = of.FermionOperator(fop_string, 1.0)

    return gate_generator


def get_charge_charge_gate(
    qubits: Tuple[cirq.Qid, ...], param: float
) -> cirq.Operation:
    """Get the cirq charge-charge gate.

    Args:
        qubits: Two qubits you want to apply the gate to.
        param: The parameter for the charge-charge interaction.

    Returns:
        The charge-charge gate.
    """
    return cirq.CZ(qubits[0], qubits[1]) ** (-param / np.pi)


def get_givens_generator(indices: Tuple[int, int]) -> of.FermionOperator:
    """Returns the generator for givens rotation between two orbitals.

    Args:
        indices: The two indices for the givens rotation.

    Returns:
        The givens generator for evolution for this pair of electrons.
    """

    fop_string = "{:d}^ {:d}".format(indices[0], indices[1])
    gate_generator = of.FermionOperator(fop_string, 1.0)
    gate_generator = 1j * (gate_generator - of.hermitian_conjugated(gate_generator))

    return gate_generator


def get_givens_gate(qubits: Tuple[cirq.Qid, ...], param: float) -> cirq.Operation:
    """Get a the givens rotation gate on two qubits.

    Args:
        qubits: The two qubits to apply the gate to.
        param: The parameter for the givens rotation.

    Returns:
        The givens rotation gate.
    """
    return cirq.givens(param).on(qubits[0], qubits[1])


def get_layer_indices(layer_spec: LayerSpec, n_elec: int) -> List[Tuple[int, int]]:
    """Get the indices for the heuristic layers.

    Args:
        layer_spec: The layer specification.
        n_elec: The number of electrons.

    Returns:
        A list of indices for the layer.
    """
    indices_generators = {
        "in_pair": get_indices_heuristic_layer_in_pair(n_elec),
        "cross_pair": get_indices_heuristic_layer_cross_pair(n_elec),
        "cross_spin": get_indices_heuristic_layer_cross_spin(n_elec),
    }
    indices_generator = indices_generators[layer_spec.layout]

    return [indices for indices in indices_generator]


def get_layer_gates(
    layer_spec: LayerSpec,
    n_elec: int,
    params: np.ndarray,
    fermion_index_to_qubit_map: Dict[int, cirq.GridQubit],
) -> List[cirq.Operation]:
    """Gets the gates for a hardware efficient layer of the ansatz.

    Args:
        layer_spec: The layer specification.
        n_elec: The number of electrons.
        params: The variational parameters for the hardware efficient gate layer.
        fermion_index_to_qubit_map: A mapping between fermion mode indices and qubits.

    Returns:
        A list of gates for the layer.
    """

    indices_list = get_layer_indices(layer_spec, n_elec)

    gate_funcs = {"givens": get_givens_gate, "charge_charge": get_charge_charge_gate}
    gate_func = gate_funcs[layer_spec.base_gate]

    gates = []
    for indices, param in zip(indices_list, params):
        qubits = tuple(fermion_index_to_qubit_map[ind] for ind in indices)
        gates.append(gate_func(qubits, param))

    return gates


def get_layer_generators(
    layer_spec: LayerSpec, n_elec: int
) -> List[of.FermionOperator]:
    """Gets the generators for rotations in a hardware efficient layer of the ansatz.

    Args:
        layer_spec: The layer specification.
        n_elec: The number of electrons.

    Returns:
        A list of generators for the layers.
    """

    indices_list = get_layer_indices(layer_spec, n_elec)

    gate_funcs = {
        "givens": get_givens_generator,
        "charge_charge": get_charge_charge_generator,
    }
    gate_func = gate_funcs[layer_spec.base_gate]

    return [gate_func(indices) for indices in indices_list]


def get_heuristic_gate_generators(
    n_elec: int, layer_specs: Sequence[LayerSpec]
) -> List[of.FermionOperator]:
    """Get gate generators for the heuristic ansatz.

    Args:
        n_elec: The number of electrons.
        layer_specs: The layer specifications.

    Returns:
        A list of generators for the layers.
    """
    gate_generators = []

    for layer_spec in layer_specs:
        gate_generators += get_layer_generators(layer_spec, n_elec)

    return gate_generators


def get_heuristic_circuit(
    layer_specs: Sequence[LayerSpec],
    n_elec: int,
    params: np.ndarray,
    fermion_index_to_qubit_map: Dict[int, cirq.GridQubit],
) -> cirq.Circuit:
    """Get a circuit for the heuristic ansatz.

    Args:
        layer_specs: The layer specs for the heuristic layers.
        n_elec: The number of electrons.
        params: The variational parameters for the circuit.
        fermion_index_to_qubit_map: A mapping between fermion mode indices and qubits.

    Returns:
        A circuit for the heuristic ansatz.
    """
    gates: List[cirq.Operation] = []

    for layer_spec in layer_specs:
        params_slice = params[len(gates) :]
        gates += get_layer_gates(
            layer_spec, n_elec, params_slice, fermion_index_to_qubit_map
        )

    return cirq.Circuit(gates)


def orbital_rotation_gradient_matrix(
    generator_mat: np.ndarray, a: int, b: int
) -> np.ndarray:
    """The gradient of the orbital rotation unitary with respect to its parameters.

    Args:
        generator_mat: The orbital rotation one-body generator matrix.
        a, b: row and column indices corresponding to the location in the matrix
            of the parameter we wish to find the gradient with respect to.

    Returns:
        The orbital rotation matrix gradient wrt theta_{a, b}. Corresponds to
            expression in G15 of https://arxiv.org/abs/2004.04174.
    """
    w_full, v_full = np.linalg.eigh(-1j * generator_mat)
    eigs_diff = np.zeros((w_full.shape[0], w_full.shape[0]), dtype=np.complex128)
    for i, j in itertools.product(range(w_full.shape[0]), repeat=2):
        if np.isclose(abs(w_full[i] - w_full[j]), 0):
            eigs_diff[i, j] = 1
        else:
            eigs_diff[i, j] = (np.exp(1j * (w_full[i] - w_full[j])) - 1) / (
                1j * (w_full[i] - w_full[j])
            )

    Y_full = np.zeros_like(v_full, dtype=np.complex128)
    if a == b:
        Y_full[a, b] = 0
    else:
        Y_full[a, b] = 1.0
        Y_full[b, a] = -1.0

    Y_kl_full = v_full.conj().T @ Y_full @ v_full
    # now rotate Y_{kl} * (exp(i(l_{k} - l_{l})) - 1) / (i(l_{k} - l_{l}))
    # into the original basis
    pre_matrix_full = v_full @ (eigs_diff * Y_kl_full) @ v_full.conj().T

    return pre_matrix_full


def evaluate_gradient_and_cost_function(
    initial_wf: fqe.Wavefunction,
    fqe_ham: fqe_hams.RestrictedHamiltonian,
    n_orb: int,
    one_body_params: npt.NDArray[np.float64],
    two_body_params: npt.NDArray[np.float64],
    gate_generators: List[of.FermionOperator],
    restricted: bool,
    e_core: float,
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Evaluate gradient and cost function for optimization.

    Args:
        initial_wf: Initial state (typically Hartree--Fock).
        fqe_ham: The restricted Hamiltonian in FQE format.
        n_orb: The number of spatial orbitals.
        one_body_params: The parameters of the single-particle rotations.
        two_body_params: The parameters for the two-particle terms.
        gate_generators: The generators for the two-particle terms.
        retricted: Whether the single-particle rotations are restricted (True)
            or unrestricted (False). Unrestricted implies different parameters
            for the alpha- and beta-spin rotations.

    Returns:
        cost_val: The cost function (total energy) evaluated for the input wavefunction parameters.
        grad: An array of gradients with respect to the one- and two-body
            parameters. The first n_orb * (n_orb + 1) // 2 parameters correspond to
            the one-body gradients.
    """
    phi = get_evolved_wf(
        one_body_params,
        two_body_params,
        initial_wf,
        gate_generators,
        n_orb,
        restricted=restricted,
    )[0]
    lam = copy.deepcopy(phi)
    lam = lam.apply(fqe_ham)
    cost_val = fqe.vdot(lam, phi) + e_core

    # 1body
    one_body_cluster_op = get_one_body_cluster_coef(
        one_body_params, n_orb, restricted=restricted
    )
    tril = np.tril_indices(n_orb, k=-1)
    if restricted:
        one_body_ham = fqe.get_restricted_hamiltonian((-1j * one_body_cluster_op,))
    else:
        one_body_ham = fqe.get_sso_hamiltonian((-1j * one_body_cluster_op,))
    # Apply U1b^{dag}
    phi.time_evolve(1, one_body_ham, inplace=True)
    lam.time_evolve(1, one_body_ham, inplace=True)
    one_body_grad = np.zeros_like(one_body_params)
    n_one_body_params = len(one_body_params)
    grad_position = n_one_body_params - 1
    for iparam in range(len(one_body_params)):
        mu_state = copy.deepcopy(phi)
        pidx = n_one_body_params - iparam - 1
        pidx_spin = 0 if restricted else pidx // (n_one_body_params // 2)
        pidx_spat = pidx if restricted else pidx - (n_one_body_params // 2) * pidx_spin
        p, q = (tril[0][pidx_spat], tril[1][pidx_spat])
        p += n_orb * pidx_spin
        q += n_orb * pidx_spin
        pre_matrix = orbital_rotation_gradient_matrix(-one_body_cluster_op, p, q)
        assert of.is_hermitian(1j * pre_matrix)
        if restricted:
            fqe_quad_ham_pre = fqe.get_restricted_hamiltonian((pre_matrix,))
        else:
            fqe_quad_ham_pre = fqe.get_sso_hamiltonian((pre_matrix,))
        mu_state = mu_state.apply(fqe_quad_ham_pre)
        one_body_grad[grad_position] = 2 * fqe.vdot(lam, mu_state).real
        grad_position -= 1
    # Get two-body contributions
    two_body_grad = np.zeros(len(two_body_params))
    for pidx in reversed(range(len(gate_generators))):
        mu = copy.deepcopy(phi)
        mu = mu.apply(gate_generators[pidx])
        two_body_grad[pidx] = -np.real(2 * 1j * (fqe.vdot(lam, mu)))
        phi = phi.time_evolve(-two_body_params[pidx], gate_generators[pidx])
        lam = lam.time_evolve(-two_body_params[pidx], gate_generators[pidx])

    return cost_val, np.concatenate((two_body_grad, one_body_grad))


def get_pp_plus_params(
    *,
    hamiltonian_data: hamiltonian.HamiltonianData,
    restricted: bool = False,
    random_parameter_scale: float = 1.0,
    initial_orbital_rotation: Optional[np.ndarray] = None,
    heuristic_layers: Tuple[LayerSpec, ...],
    do_pp: bool = True,
    n_optimization_restarts: int = 1,
    do_print: bool = True,
    use_fast_gradients: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize the PP + Hardware layer ansatz.

    Args:
        hamiltonian_data: Hamiltonian (molecular) specification.
        restricted: Whether to use a spin-restricted ansatz or not.
        random_parameter_scale: A float to scale the random parameters by.
        initial_orbital_rotation: An optional initial orbital rotation matrix,
            which will be implmented as a givens circuit.
        heuristic_layers: A tuple of circuit layers to append to the perfect pairing circuit.
        do_pp: Implement the perfect pairing circuit along with the heuristic
            layers. Defaults to true.
        n_optimization_restarts: The number of times to restart the optimization
            from a random guess in an attempt at global optimization.
        do_print: Whether to print optimization progress to stdout.
        use_fast_gradients: Compute the parameter gradients anlytically using Wilcox formula.
            Default to false (use finite difference gradients).

    Returns:
        one_body_params: Optimized one-body parameters.
        two_body_params: Optimized two-body parameters
        one_body_basis_change_mat: The basis change matrix including any initial orbital rotation.
    """
    n_elec = hamiltonian_data.params.n_elec
    n_orb = hamiltonian_data.params.n_orb
    sz = 0

    initial_wf = fqe.Wavefunction([[n_elec, sz, n_orb]])
    initial_wf.set_wfn(strategy="hartree-fock")

    fqe_ham = hamiltonian_data.get_restricted_fqe_hamiltonian()
    e_core = hamiltonian_data.e_core

    hf_energy = initial_wf.expectationValue(fqe_ham) + e_core

    # We're only supporting closed shell stuff here.
    assert n_elec % 2 == 0
    assert n_elec <= n_orb
    if use_fast_gradients:
        err_msg = "use_fast_gradients does not work with initial orbital rotation."
        assert initial_orbital_rotation is None, err_msg

    gate_generators = _get_pp_plus_gate_generators(
        n_elec=n_elec, heuristic_layers=heuristic_layers, do_pp=do_pp
    )

    n_two_body_params = len(gate_generators)

    if restricted:
        n_one_body_params = n_orb * (n_orb - 1) // 2
    else:
        n_one_body_params = n_orb * (n_orb - 1)

    best = np.inf
    best_res: Union[None, scipy.optimize.OptimizeResult] = None
    for i in range(n_optimization_restarts):
        if do_print:
            print(f"Optimization restart {i}", flush=True)

            def progress_cb(_):
                print(".", end="", flush=True)

        else:

            def progress_cb(_):
                pass

        params = random_parameter_scale * np.random.normal(
            size=(n_two_body_params + n_one_body_params)
        )

        def objective(params):
            one_body_params = params[-n_one_body_params:]
            two_body_params = params[:n_two_body_params]

            wf, _ = get_evolved_wf(
                one_body_params,
                two_body_params,
                initial_wf,
                gate_generators,
                n_orb,
                restricted=restricted,
                initial_orbital_rotation=initial_orbital_rotation,
            )

            energy = wf.expectationValue(fqe_ham) + e_core
            if do_print:
                print(f"energy {energy}")
            if np.abs(energy.imag) < 1e-6:
                return energy.real
            else:
                return 1e6

        def fast_obj_grad(params):
            one_body_params = params[-n_one_body_params:]
            two_body_params = params[:n_two_body_params]
            energy, grad = evaluate_gradient_and_cost_function(
                initial_wf,
                fqe_ham,
                n_orb,
                one_body_params,
                two_body_params,
                gate_generators,
                restricted,
                e_core,
            )
            if do_print:
                print(f"energy {energy}, max|grad| {np.max(np.abs(grad))}")
            if np.abs(energy.imag) < 1e-6:
                return energy.real, grad
            else:
                return 1e6, 1e6

        if use_fast_gradients:
            res = minimize(
                fast_obj_grad, params, jac=True, method="BFGS", callback=progress_cb
            )
        else:
            res = minimize(objective, params, callback=progress_cb)
        if res.fun < best:
            best = res.fun
            best_res = res

        if do_print:
            print(res, flush=True)

    assert best_res is not None
    params = best_res.x
    one_body_params = params[-n_one_body_params:]
    two_body_params = params[:n_two_body_params]

    wf, _ = get_evolved_wf(
        one_body_params,
        two_body_params,
        initial_wf,
        gate_generators,
        n_orb,
        restricted=restricted,
        initial_orbital_rotation=initial_orbital_rotation,
    )

    one_body_cluster_mat = get_one_body_cluster_coef(
        one_body_params, n_orb, restricted=restricted
    )
    # We need to change the ordering to match OpenFermion's abababab ordering
    if not restricted:
        index_rearrangement = np.asarray(
            [i // 2 % (n_orb) + (i % 2) * n_orb for i in range(2 * n_orb)]
        )
        one_body_cluster_mat = one_body_cluster_mat[:, index_rearrangement]
        one_body_cluster_mat = one_body_cluster_mat[index_rearrangement, :]

    one_body_basis_change_mat = expm(one_body_cluster_mat)

    if initial_orbital_rotation is not None:
        if restricted:
            one_body_basis_change_mat = (
                initial_orbital_rotation @ one_body_basis_change_mat
            )
        else:
            big_initial_orbital_rotation = np.zeros_like(one_body_basis_change_mat)

            for i in range(len(initial_orbital_rotation)):
                for j in range(len(initial_orbital_rotation)):
                    big_initial_orbital_rotation[2 * i, 2 * j] = (
                        initial_orbital_rotation[i, j]
                    )
                    big_initial_orbital_rotation[2 * i + 1, 2 * j + 1] = (
                        initial_orbital_rotation[i, j]
                    )

            one_body_basis_change_mat = (
                big_initial_orbital_rotation @ one_body_basis_change_mat
            )

    if do_print:
        print("Hartree-Fock Energy:")
        print(hf_energy)
        initial_wf.print_wfn()
        print("-" * 80)
        print("FCI Energy:")
        print(hamiltonian_data.e_fci)
        print("-" * 80)
        print(best_res)

        print("-" * 80)
        print("Ansatz Energy:")
        print(np.real_if_close(wf.expectationValue(fqe_ham) + e_core))
        wf.print_wfn()
        print("Basis Rotation Matrix:")
        print(one_body_basis_change_mat)
        print("Two Body Rotation Parameters:")
        print(two_body_params)

    return one_body_params, two_body_params, one_body_basis_change_mat
