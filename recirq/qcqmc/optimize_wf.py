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

import copy
import itertools
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cirq
import fqe
import fqe.hamiltonians.restricted_hamiltonian as fqe_hams
import fqe.wavefunction as fqe_wfn
import numpy as np
import openfermion as of
import scipy.linalg
import scipy.optimize
import scipy.sparse

from recirq.qcqmc import (afqmc_circuits, afqmc_generators, converters, data,
                          fermion_mode, hamiltonian, layer_spec, trial_wf)


def get_and_check_energy(
    *,
    hamiltonian_data: hamiltonian.HamiltonianData,
    ansatz_circuit: cirq.Circuit,
    one_body_params: np.ndarray,
    two_body_params: np.ndarray,
    one_body_basis_change_mat: np.ndarray,
    params: trial_wf.PerfectPairingPlusTrialWavefunctionParams,
) -> Tuple[float, float]:
    """Compute the energy of the ansatz circuit and check against known values where possible.

    Args:
        hamiltonian_data: The Hamiltonian data.
        ansatz_circuit: The ansatz circuit.
        one_body_params: The one-body variational parameters.
        two_body_params: The two-body variational parameters.
        one_body_basis_change_mat: The one-body basis change matrix.
        params: The trial wavefunction parameters.

    Returns:
        ansatz_energy: The total energy of the ansatz circuit.
        hf_energy: The hartree-fock energy (initial energy of the ansatz circuit).
    """
    ansatz_qubit_wf = converters.get_ansatz_qubit_wf(
        ansatz_circuit=ansatz_circuit,
        ordered_qubits=params.qubits_jordan_wigner_ordered,
    )

    fqe_ham, e_core, sparse_ham = get_rotated_hamiltonians(
        hamiltonian_data=hamiltonian_data,
        one_body_basis_change_mat=one_body_basis_change_mat,
        mode_qubit_map=params.mode_qubit_map,
        ordered_qubits=params.qubits_jordan_wigner_ordered,
    )

    initial_wf = fqe_wfn.Wavefunction([[params.n_elec, 0, params.n_orb]])
    initial_wf.set_wfn(strategy="hartree-fock")

    hf_energy = initial_wf.expectationValue(fqe_ham) + e_core

    fqe_wf, unrotated_fqe_wf = get_evolved_wf(
        one_body_params=one_body_params,
        two_body_params=two_body_params,
        wf=initial_wf,
        gate_generators=afqmc_generators.get_pp_plus_gate_generators(
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
    params: trial_wf.PerfectPairingPlusTrialWavefunctionParams,
    *,
    dependencies: Dict[data.Params, data.Data],
    do_print: bool = False,
) -> trial_wf.TrialWavefunctionData:
    """Builds a TrialWavefunctionData from a TrialWavefunctionParams

    Args:
        params: The parameters specifying the PP+ trial wavefunction.
        dependencies: Data dependencies
        do_print: Print debugging information to stdout

    Returns:
        The constructed TrialWavefunctionData object.
    """

    if do_print:
        print("Building Trial Wavefunction")
    np.random.seed(params.seed)
    hamiltonian_data = dependencies[params.hamiltonian_params]
    assert isinstance(hamiltonian_data, hamiltonian.HamiltonianData)

    if params.n_orb != params.n_elec:
        raise ValueError("PP wavefunction must have n_orb = n_elec")

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
        two_body_params = converters.get_two_body_params_from_qchem_amplitudes(
            params.initial_two_body_qchem_amplitudes
        )

    (superposition_circuit, ansatz_circuit) = afqmc_circuits.get_circuits(
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

    return trial_wf.TrialWavefunctionData(
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
    mode_qubit_map: Mapping[fermion_mode.FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> Tuple[fqe_hams.RestrictedHamiltonian, float, scipy.sparse.csc_matrix]:
    """A helper method that gets the hamiltonians in the basis of the trial_wf.

    Args:
        hamiltonian_data: The specification of the Hamiltonian.
        one_body_basis_change_mat: The basis change matrix for the one body hamiltonian.
        mode_qubit_map: A mapping between fermionic modes and cirq qubits.
        ordered_qubits: An ordered set of qubits.

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

    reorder_func = converters.get_reorder_func(
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
    fqe_ham: fqe_hams.RestrictedHamiltonian,
    sparse_ham: scipy.sparse.csc_matrix,
    e_core: float,
    mode_qubit_map: Mapping[fermion_mode.FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> float:
    """A method that checks for consistency and returns the ansatz energy.

    Args:
        circuit_wf: The cirq statevector constructed from the wavefunction ansatz circuit.
        fqe_wf: The FQE wavefunction used for optimization.
        unrotated_fqe_wf: The (unoptimized) unrotated FQE wavefunction.
        fqe_ham: The restricted FQE hamiltonian.
        sparse_ham: The qubit Hamiltonian as a sparse matrix.
        e_core: The core energy (nuclear-repulision + any frozen core energy.)
        mode_qubit_map: A mapping between fermionic modes and cirq qubits.
        ordered_qubits: An ordered set of qubits.

    Returns:
        The ansatz energy.
    """

    unrotated_fqe_wf_as_cirq = converters.convert_fqe_wf_to_cirq(
        fqe_wf=unrotated_fqe_wf,
        mode_qubit_map=mode_qubit_map,
        ordered_qubits=ordered_qubits,
    )
    ansatz_energy = np.real_if_close(
        (np.conj(circuit_wf) @ sparse_ham @ circuit_wf)
    ).item()
    assert isinstance(ansatz_energy, float)

    fqe_energy = np.real(fqe_wf.expectationValue(fqe_ham) + e_core)
    np.testing.assert_array_almost_equal(ansatz_energy, fqe_energy)
    np.testing.assert_array_almost_equal(
        circuit_wf, unrotated_fqe_wf_as_cirq, decimal=5
    )
    return ansatz_energy


def get_one_body_cluster_coef(
    params: np.ndarray, n_orb: int, restricted: bool
) -> np.ndarray:
    """Get the matrix elements associated with the one-body cluster operator.

    Args:
        params: The variational parameters of the one-body cluster operator.
        n_orb: The number of spatial orbitals.
        restricted: Whether a spin-restricted cluster operator is used.

    Returns:
        The one-body cluster operator matrix.
    """
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


def evaluate_energy_and_gradient(
    initial_wf: fqe.Wavefunction,
    fqe_ham: fqe_hams.RestrictedHamiltonian,
    n_orb: int,
    one_body_params: np.ndarray,
    two_body_params: np.ndarray,
    gate_generators: List[of.FermionOperator],
    restricted: bool,
    e_core: float,
) -> Tuple[float, np.ndarray]:
    """Evaluate gradient and cost function for optimization.

    Uses the linear scaling algorithm (see algo 2 from example:
    https://arxiv.org/pdf/2009.02823) at the expense of three copies of the
    wavefunction.

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
        energy: The cost function (total energy) evaluated for the input wavefunction parameters.
        grad: An array of gradients with respect to the one- and two-body
            parameters. The first n_orb * (n_orb + 1) // 2 parameters correspond to
            the one-body gradients.
    """
    # Build |phi> = U(theta)|phi_0>
    phi = get_evolved_wf(
        one_body_params,
        two_body_params,
        initial_wf,
        gate_generators,
        n_orb,
        restricted=restricted,
    )[0]
    # Set |lambda> = |phi> initially
    lam = copy.deepcopy(phi)
    # H|lambda>
    lam = lam.apply(fqe_ham)
    # E = <lambda |phi>
    cost_val = fqe.vdot(lam, phi) + e_core

    # First build the 1body cluster op as a matrix
    one_body_cluster_op = get_one_body_cluster_coef(
        one_body_params, n_orb, restricted=restricted
    )
    # Build the one-body FQE hamiltonian
    if restricted:
        one_body_ham = fqe.get_restricted_hamiltonian((-1j * one_body_cluster_op,))
    else:
        one_body_ham = fqe.get_sso_hamiltonian((-1j * one_body_cluster_op,))
    # |phi> = U1b U2b |phi_0>
    # 1. Remove U1b from |phi> by U1b^dagger |phi>
    phi.time_evolve(1, one_body_ham, inplace=True)
    lam.time_evolve(1, one_body_ham, inplace=True)
    one_body_grad = np.zeros_like(one_body_params)
    n_one_body_params = len(one_body_params)
    grad_position = n_one_body_params - 1
    # The parameters correspond to the lower triangular part of the matrix.
    # we need the row and column indices corresponding to each flattened lower triangular index.
    tril = np.tril_indices(n_orb, k=-1)
    # Now compute the gradient of the one-body orbital rotation operator for each parameter.
    # If we write E(theta) = <phi_0| U(theta)^ H U(theta)|phi_0>
    # Then d E(theta)/ d theta_p = -2 i Im <phi_0 | U(theta)^ H dU/dtheta_p |phi_0>
    for iparam in range(len(one_body_params)):
        mu_state = copy.deepcopy(phi)
        # get the parameter index starting from the end and working backwards.
        pidx = n_one_body_params - iparam - 1
        pidx_spin = 0 if restricted else pidx // (n_one_body_params // 2)
        pidx_spat = pidx if restricted else pidx - (n_one_body_params // 2) * pidx_spin
        # Get the actual row and column indicies corresponding to this parameter index.
        p, q = (tril[0][pidx_spat], tril[1][pidx_spat])
        p += n_orb * pidx_spin
        q += n_orb * pidx_spin
        # Get the orbital rotation gradient "pre" matrix and apply it the |mu>.
        # For the orbital rotation part we compute dU(theta)/dtheta_p using the
        # wilcox identity (see e.g.:  https://arxiv.org/abs/2004.04174.)
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
    # Here we already have the generators so the gradient is simple to evaluate
    # as the derivative just brings down a generator which we need to apply to
    # the state before computing the overlap.
    two_body_grad = np.zeros(len(two_body_params))
    for pidx in reversed(range(len(gate_generators))):
        mu = copy.deepcopy(phi)
        mu = mu.apply(gate_generators[pidx])
        two_body_grad[pidx] = -np.real(2 * 1j * (fqe.vdot(lam, mu)))
        phi = phi.time_evolve(-two_body_params[pidx], gate_generators[pidx])
        lam = lam.time_evolve(-two_body_params[pidx], gate_generators[pidx])

    return cost_val, np.concatenate((two_body_grad, one_body_grad))


def objective(
    params: np.ndarray,
    n_one_body_params: int,
    n_two_body_params: int,
    initial_wf: fqe_wfn.Wavefunction,
    fqe_ham: fqe_hams.RestrictedHamiltonian,
    gate_generators: List[of.FermionOperator],
    n_orb: int,
    restricted: bool,
    initial_orbital_rotation: np.ndarray,
    e_core: float,
    do_print: bool = False,
) -> float:
    """Helper function to compute energy from the variational parameters.

    Args:
        params: A packed array containing the one and two-body parameters.
        n_one_body_params: The number of variational parameters for the one-body terms.
        n_two_body_params: The number of variational parameters for the two-body terms.
        initial_wf: The initial wavefunction the circuit unitary is applied to.
        fqe_ham: The restricted FQE hamiltonian.
        gate_generators: The list of gate generators.
        n_orb: The number of spatial orbitals.
        restricted: Whether to use a spin-restricted ansatz or not.
        e_core: The Hamiltonian core (all the constants) energy.
        do_print: Whether to print optimization progress to stdout.

    Returns:
        The energy capped at 1e6 if the energy is imaginary.
    """
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


def objective_and_gradient(
    params: np.ndarray,
    n_one_body_params: int,
    n_two_body_params: int,
    initial_wf: fqe_wfn.Wavefunction,
    fqe_ham: fqe_hams.RestrictedHamiltonian,
    gate_generators: List[of.FermionOperator],
    n_orb: int,
    restricted: bool,
    e_core: float,
    do_print: bool = False,
) -> Tuple[float, np.array]:
    """Helper function to compute energy and gradient from the variational parameters

    Args:
        params: A packed array containing the one and two-body parameters.
        n_one_body_params: The number of variational parameters for the one-body terms.
        n_two_body_params: The number of variational parameters for the two-body terms.
        initial_wf: The initial wavefunction the circuit unitary is applied to.
        fqe_ham: The restricted FQE hamiltonian.
        gate_generators: The list of gate generators.
        n_orb: The number of spatial orbitals.
        restricted: Whether to use a spin-restricted ansatz or not.
        e_core: The Hamiltonian core (all the constants) energy.
        do_print: Whether to print optimization progress to stdout.

    Returns:
        A tuple containing the energy and gradient.  These are capped at 1e6 if
        the energy is imaginary.
    """
    one_body_params = params[-n_one_body_params:]
    two_body_params = params[:n_two_body_params]
    energy, grad = evaluate_energy_and_gradient(
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
        return energy.real, grad.real
    else:
        return 1e6, np.array([1e6]) * len(grad)


def optimize_parameters(
    initial_wf: fqe_wfn.Wavefunction,
    gate_generators: List[of.FermionOperator],
    n_orb: int,
    n_one_body_params: int,
    n_two_body_params: int,
    fqe_ham: fqe_hams.RestrictedHamiltonian,
    e_core: float,
    initial_orbital_rotation: Optional[np.ndarray] = None,
    restricted: bool = False,
    use_fast_gradients: bool = False,
    n_optimization_restarts: int = 1,
    random_parameter_scale: float = 1.0,
    do_print: bool = True,
) -> Optional[scipy.optimize.OptimizeResult]:
    """Optimize the cost function (total energy) for the PP+ ansatz.

    Loops over n_optimization_restarts to try to find a good minimum value of the total energy.

    Args:
        initial_wf: The initial wavefunction the circuit unitary is applied to.
        gate_generators: The generators of the two-body interaction terms.
        n_orb: The number of orbitals.
        n_one_body_params: The number of variational parameters for the one-body terms.
        n_two_body_params: The number of variational parameters for the two-body terms.
        initial_orbital_rotation: An optional initial orbital rotation matrix,
            which will be implmented as a givens circuit.
        fqe_ham: The restricted FQE hamiltonian.
        e_core: The Hamiltonian core (all the constants) energy.
        use_fast_gradients: Compute the parameter gradients anlytically using Wilcox formula.
            Default to false (use finite difference gradients).
        n_optimization_restarts: The number of times to restart the optimization
            from a random guess in an attempt at global optimization.
        restricted: Whether to use a spin-restricted ansatz or not.
        random_parameter_scale: A float to scale the random parameters by.
        do_print: Whether to print optimization progress to stdout.

    Returns:
        The optimization result.
    """
    best = np.inf
    best_res: Optional[scipy.optimize.OptimizeResult] = None
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

        if use_fast_gradients:
            # Use analytic gradient rather than finite differences.
            res = scipy.optimize.minimize(
                objective_and_gradient,
                params,
                jac=True,
                method="BFGS",
                callback=progress_cb,
                args=(
                    n_one_body_params,
                    n_two_body_params,
                    initial_wf,
                    fqe_ham,
                    gate_generators,
                    n_orb,
                    restricted,
                    e_core,
                    do_print,
                ),
            )
        else:
            res = scipy.optimize.minimize(
                objective,
                params,
                callback=progress_cb,
                args=(
                    n_one_body_params,
                    n_two_body_params,
                    initial_wf,
                    fqe_ham,
                    gate_generators,
                    n_orb,
                    restricted,
                    initial_orbital_rotation,
                    e_core,
                    do_print,
                ),
            )
        if res.fun < best:
            best = res.fun
            best_res = res

        if do_print:
            print(res, flush=True)
    return best_res


def get_pp_plus_params(
    *,
    hamiltonian_data: hamiltonian.HamiltonianData,
    restricted: bool = False,
    random_parameter_scale: float = 1.0,
    initial_orbital_rotation: Optional[np.ndarray] = None,
    heuristic_layers: Tuple[layer_spec.LayerSpec, ...],
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

    gate_generators = afqmc_generators.get_pp_plus_gate_generators(
        n_elec=n_elec, heuristic_layers=heuristic_layers, do_pp=do_pp
    )

    n_two_body_params = len(gate_generators)

    if restricted:
        n_one_body_params = n_orb * (n_orb - 1) // 2
    else:
        n_one_body_params = n_orb * (n_orb - 1)

    best_res = optimize_parameters(
        initial_wf,
        gate_generators,
        n_orb,
        n_one_body_params,
        n_two_body_params,
        fqe_ham,
        e_core,
        restricted=restricted,
        initial_orbital_rotation=initial_orbital_rotation,
        use_fast_gradients=use_fast_gradients,
        n_optimization_restarts=n_optimization_restarts,
        random_parameter_scale=random_parameter_scale,
        do_print=do_print,
    )

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

    one_body_basis_change_mat = scipy.linalg.expm(one_body_cluster_mat)

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
