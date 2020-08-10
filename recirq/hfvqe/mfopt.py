# pylint: disable=C
# coverage: ignore
from typing import Callable, List, Optional, Tuple
from itertools import product
import time
import numpy as np
import scipy as sp
from scipy.optimize import OptimizeResult

from openfermion.transforms import get_interaction_operator
from openfermion.ops import FermionOperator, InteractionRDM

from joblib import Parallel, delayed  # type: ignore

from openfermioncirq.experiments.hfvqe.objective import \
    RestrictedHartreeFockObjective
from openfermioncirq.experiments.hfvqe.circuits import rhf_params_to_matrix


def get_one_body_fermion_operator(coeff_matrix):  # testpragma: no cover
    # coverage: ignore
    operator = FermionOperator()
    for i, j in product(range(coeff_matrix.shape[0]), repeat=2):
        operator += coeff_matrix[i, j] * FermionOperator(((i, 1), (j, 0)))
    return operator


def kdelta(i: int, j: int) -> float:  # testpragma: no cover
    # coverage: ignore
    """Delta function function"""
    return 1.0 if i == j else 0.0


def group_action(old_unitary: np.ndarray,
                 new_parameters: np.ndarray,
                 occ: List[int],
                 virt: List[int]) -> np.ndarray:  # testpragma: no cover
    # coverage: ignore
    """
    U(e^{kappa}) . U(e^{kappa'}) = U(e^{kappa}.e^{kappa'})

    :param old_unitary: unitary that we update--left multiply
    :param new_parameters: parameters fornew unitary
    :param occ: list of occupied indices
    :param virt: list of virtual indices
    :return: updated unitary
    """
    kappa_new = rhf_params_to_matrix(new_parameters,
                                     len(occ) + len(virt),
                                     occ, virt)
    assert kappa_new.shape == (len(occ) + len(virt), len(occ) + len(virt))
    return sp.linalg.expm(kappa_new) @ old_unitary


def non_redundant_rotation_generators(
        rhf_objective: RestrictedHartreeFockObjective) -> List[FermionOperator]:  # testpragma: no cover
    # coverage: ignore
    """
    Generate the fermionic representation of all non-redundant rotation
    generators for restricted Hartree-fock

    :param rhf_objective: openfermioncirq.experiments.hfvqe.RestrictedHartreeFock object
    :return: list of fermionic generators.
    """
    rotation_generators = []
    for p in range(rhf_objective.nocc * rhf_objective.nvirt):
        grad_params = np.zeros(rhf_objective.nocc * rhf_objective.nvirt)
        grad_params[p] = 1
        kappa_spatial_orbital = rhf_params_to_matrix(grad_params,
                                                     len(rhf_objective.occ) + len(rhf_objective.virt),
                                                     rhf_objective.occ,
                                                     rhf_objective.virt)
        p0 = np.array([[1, 0], [0, 1]])
        kappa_spin_orbital = np.kron(kappa_spatial_orbital, p0)
        fermion_op = get_one_body_fermion_operator(kappa_spin_orbital)
        rotation_generators.append(fermion_op)
    return rotation_generators


def get_dvec_hmat(rotation_generators: List[FermionOperator],
                  rhf_objective: RestrictedHartreeFockObjective,
                  rdms: InteractionRDM,
                  diagonal_hessian=False) -> (np.ndarray, np.ndarray):  # testpragma: no cover
    # coverage: ignore
    """
    Generate first and second terms of the BCH expansion

    :param rotation_generators: List FermionOperators corresponding to
                                non-redundant rotation generators
    :param rhf_objective: openfermioncirq.experiments.hfvqe.RestrictedHartreeFockObject
    :param rdms: openfermion.InteractionRDMs where the 2-RDM is generated
                 from the 1-RDM as of.wedge(opdm, opdm)
    :param diagonal_hessian: Boolean indicator for what type of Hessian
                             construction should be used.
    :return:
    """
    dvec = np.zeros(len(rotation_generators), dtype=np.complex128)
    hmat = np.zeros((len(rotation_generators), len(rotation_generators)),
                    dtype=np.complex128)
    num_qubits = rhf_objective.num_qubits
    kdelta_mat = np.eye(rhf_objective.hamiltonian.one_body_tensor.shape[0])

    def single_commutator_einsum(idx: int,
                                 rot_gen: FermionOperator) -> Tuple[int, float]:  # testpragma: no cover
        # coverage: ignore
        """
        Evaluate <psi|[H, p^q - q^p]|psi>

        :param idx: integer index of p^q - q^p in the ordered set
        :param rot_gen: Rotation generator p^q - q^p as a FermionOperator
        :return: index and value for the commutator
        """
        rot_gen_tensor = get_interaction_operator(rot_gen,
                                                  n_qubits=num_qubits).one_body_tensor
        opdm = rdms.n_body_tensors[(1, 0)].copy()
        tpdm = rdms.n_body_tensors[(1, 1, 0, 0)].copy()
        commutator_expectation = 0
        #   (  -1.00000) kdelta(i,q) cre(p) des(j)
        commutator_expectation += -1.0 * np.einsum('ij,pq,iq,pj',
                                                   rhf_objective.hamiltonian.one_body_tensor,
                                                   rot_gen_tensor, kdelta_mat,
                                                   opdm, optimize=True)
        #   (   1.00000) kdelta(j,p) cre(i) des(q)
        commutator_expectation += 1.0 * np.einsum('ij,pq,jp,iq',
                                                  rhf_objective.hamiltonian.one_body_tensor,
                                                  rot_gen_tensor, kdelta_mat,
                                                  opdm, optimize=True)
        #   (   1.00000) kdelta(i,q) cre(j) cre(p) des(k) des(l)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,iq,jpkl',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rot_gen_tensor, kdelta_mat,
                                                  tpdm, optimize=True)
        #   (  -1.00000) kdelta(j,q) cre(i) cre(p) des(k) des(l)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,jq,ipkl',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rot_gen_tensor, kdelta_mat,
                                                   tpdm, optimize=True)
        #   (  -1.00000) kdelta(k,p) cre(i) cre(j) des(l) des(q)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,kp,ijlq',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rot_gen_tensor, kdelta_mat,
                                                   tpdm, optimize=True)
        #   (   1.00000) kdelta(l,p) cre(i) cre(j) des(k) des(q)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,lp,ijkq',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rot_gen_tensor, kdelta_mat,
                                                  tpdm, optimize=True)

        return idx, commutator_expectation

    def double_commutator_einsum(ridx: int, rgen: FermionOperator,
                                 sidx: int, sgen: FermionOperator) -> Tuple[int, int, float]:  # testpragma: no cover
        # coverage: ignore
        """
        Evaluate <psi|[[H, p^q - q^p], r^s - s^r]|psi>

        :param ridx: index of p^q - q^p operator in ordered list of operators
        :param rgen: FermionOperator of p^q - q^p
        :param sidx: ndex of r^s - s^r operator in ordered list of operators
        :param sgen: FermionOperator of r^s - s^r
        :return: index of p^q-q^p, index of r^s - s^r, and the commutator value
        """
        rgen_tensor = get_interaction_operator(rgen,
                                               n_qubits=num_qubits).one_body_tensor
        sgen_tensor = get_interaction_operator(sgen,
                                               n_qubits=num_qubits).one_body_tensor
        opdm = rdms.n_body_tensors[(1, 0)].copy()
        tpdm = rdms.n_body_tensors[(1, 1, 0, 0)].copy()
        commutator_expectation = 0
        #   (  -1.00000) kdelta(i,q) kdelta(j,r) cre(p) des(s)
        commutator_expectation += -1.0 * np.einsum('ij,pq,rs,iq,jr,ps',
                                                   rhf_objective.hamiltonian.one_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, opdm,
                                                   optimize=True)
        #   (   1.00000) kdelta(i,q) kdelta(p,s) cre(r) des(j)
        commutator_expectation += 1.0 * np.einsum('ij,pq,rs,iq,ps,rj',
                                                  rhf_objective.hamiltonian.one_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, opdm,
                                                  optimize=True)
        #   (  -1.00000) kdelta(i,s) kdelta(j,p) cre(r) des(q)
        commutator_expectation += -1.0 * np.einsum('ij,pq,rs,is,jp,rq',
                                                   rhf_objective.hamiltonian.one_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, opdm,
                                                   optimize=True)
        #   (   1.00000) kdelta(j,p) kdelta(q,r) cre(i) des(s)
        commutator_expectation += 1.0 * np.einsum('ij,pq,rs,jp,qr,is',
                                                  rhf_objective.hamiltonian.one_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, opdm,
                                                  optimize=True)

        #   (   1.00000) kdelta(i,q) kdelta(j,s) cre(p) cre(r) des(k) des(l)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,rs,iq,js,prkl',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, tpdm,
                                                  optimize=True)
        #   (  -1.00000) kdelta(i,q) kdelta(k,r) cre(j) cre(p) des(l) des(s)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,rs,iq,kr,jpls',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, tpdm,
                                                   optimize=True)
        #   (   1.00000) kdelta(i,q) kdelta(l,r) cre(j) cre(p) des(k) des(s)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,rs,iq,lr,jpks',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, tpdm,
                                                  optimize=True)
        #   (  -1.00000) kdelta(i,q) kdelta(p,s) cre(j) cre(r) des(k) des(l)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,rs,iq,ps,jrkl',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, tpdm,
                                                   optimize=True)
        #   (  -1.00000) kdelta(i,s) kdelta(j,q) cre(p) cre(r) des(k) des(l)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,rs,is,jq,prkl',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, tpdm,
                                                   optimize=True)
        #   (  -1.00000) kdelta(i,s) kdelta(k,p) cre(j) cre(r) des(l) des(q)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,rs,is,kp,jrlq',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, tpdm,
                                                   optimize=True)
        #   (   1.00000) kdelta(i,s) kdelta(l,p) cre(j) cre(r) des(k) des(q)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,rs,is,lp,jrkq',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, tpdm,
                                                  optimize=True)
        #   (   1.00000) kdelta(j,q) kdelta(k,r) cre(i) cre(p) des(l) des(s)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,rs,jq,kr,ipls',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, tpdm,
                                                  optimize=True)
        #   (  -1.00000) kdelta(j,q) kdelta(l,r) cre(i) cre(p) des(k) des(s)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,rs,jq,lr,ipks',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, tpdm,
                                                   optimize=True)
        #   (   1.00000) kdelta(j,q) kdelta(p,s) cre(i) cre(r) des(k) des(l)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,rs,jq,ps,irkl',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, tpdm,
                                                  optimize=True)
        #   (   1.00000) kdelta(j,s) kdelta(k,p) cre(i) cre(r) des(l) des(q)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,rs,js,kp,irlq',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, tpdm,
                                                  optimize=True)
        #   (  -1.00000) kdelta(j,s) kdelta(l,p) cre(i) cre(r) des(k) des(q)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,rs,js,lp,irkq',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, tpdm,
                                                   optimize=True)
        #   (   1.00000) kdelta(k,p) kdelta(l,r) cre(i) cre(j) des(q) des(s)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,rs,kp,lr,ijqs',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, tpdm,
                                                  optimize=True)
        #   (  -1.00000) kdelta(k,p) kdelta(q,r) cre(i) cre(j) des(l) des(s)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,rs,kp,qr,ijls',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, tpdm,
                                                   optimize=True)
        #   (  -1.00000) kdelta(k,r) kdelta(l,p) cre(i) cre(j) des(q) des(s)
        commutator_expectation += -1.0 * np.einsum('ijkl,pq,rs,kr,lp,ijqs',
                                                   rhf_objective.hamiltonian.two_body_tensor,
                                                   rgen_tensor, sgen_tensor,
                                                   kdelta_mat, kdelta_mat, tpdm,
                                                   optimize=True)
        #   (   1.00000) kdelta(l,p) kdelta(q,r) cre(i) cre(j) des(k) des(s)
        commutator_expectation += 1.0 * np.einsum('ijkl,pq,rs,lp,qr,ijks',
                                                  rhf_objective.hamiltonian.two_body_tensor,
                                                  rgen_tensor, sgen_tensor,
                                                  kdelta_mat, kdelta_mat, tpdm,
                                                  optimize=True)
        return ridx, sidx, commutator_expectation

    with Parallel(n_jobs=-1, backend='threading') as parallel:
        dk_res = parallel(delayed(single_commutator_einsum)(*x) for x in
                          enumerate(rotation_generators))

    if diagonal_hessian:
        doubles_generator = zip(enumerate(rotation_generators),
                                enumerate(rotation_generators))
    else:
        doubles_generator = product(enumerate(rotation_generators), repeat=2)
    with Parallel(n_jobs=-1, backend='threading') as parallel:
        hrs_res = parallel(delayed(double_commutator_einsum)(*x) for x in
                           [(z[0][0], z[0][1], z[1][0], z[1][1])
                            for z in doubles_generator]
                           )
    for idx, val in dk_res:
        dvec[idx] = val

    for ridx, sidx, val in hrs_res:
        hmat[ridx, sidx] = val

    return dvec, hmat


def moving_frame_augmented_hessian_optimizer(rhf_objective: RestrictedHartreeFockObjective,
                                             initial_parameters: np.ndarray,
                                             opdm_aa_measurement_func: Callable,
                                             max_iter: Optional[int]=15,
                                             rtol: Optional[float]=0.2E-2,
                                             delta: Optional[float]=0.03,
                                             verbose: Optional[bool]=True,
                                             hessian_update: Optional[bool]='diagonal'):  # testpragma: no cover
    # coverage: ignore
    """
    The moving frame optimizer

    Determine an optimal basis rotation by continuously updating the
    coordinate system and asking if stationarity is achieved.

    :param rhf_objective: openfermioncirq.experiments.hfvqe.RestrictedHartreeFockObjective
    :param initial_parameters: parameters to start the optimization
    :param opdm_aa_measurement_func: callable functioon that takes the parameter
                                     vector and returns the opdm
    :param max_iter: maximum number of iterations to take
    :param rtol: Terminate the optimization with the norm of the update angles
                 falls below this threshold
    :param verbose: Allow printing of intermediate optimization information
    :param hessian_update: Optional argument if diagonal or full Hessian is used
    :return:
    """
    if delta > 1 or delta < 0:
        raise ValueError("Delta must be in the domain [0, 1]")
    if hessian_update not in ['diagonal', 'energy']:
        raise ValueError("hessian_update parameter not valid.")

    res = OptimizeResult()
    res.fr_vals = []
    res.opdms = []
    res.x_iters = []
    res.func_vals = []
    res.f = None
    res.iter_times = []

    fr_vals = initial_parameters
    current_unitary = np.eye(rhf_objective.nocc + rhf_objective.nvirt)
    break_at_count = max_iter
    current_count = 0
    energies = []
    fval_norms = []
    # for debugging
    opdm_initial = np.diag([1] * rhf_objective.nocc + [0] * rhf_objective.nvirt)
    start_time = time.time()
    while current_count < break_at_count:
        # Iterate of algorithm has a unitary and parameters
        # first step is to generate new unitary
        u_new = group_action(old_unitary=current_unitary,
                             new_parameters=fr_vals,
                             occ=rhf_objective.occ,
                             virt=rhf_objective.virt)

        # get initial opdm from starting parameters
        opdm = opdm_aa_measurement_func(u_new.copy())
        # opdm = u_new @ opdm_initial @ u_new.conj().T

        # Calculate energy, residual, and hessian terms
        rdms: InteractionRDM = rhf_objective.rdms_from_opdm_aa(opdm)
        current_energy: float = rdms.expectation(rhf_objective.hamiltonian).real
        energies.append(current_energy)

        res.x_iters.append(u_new)
        res.func_vals.append(current_energy)
        res.fr_vals.append(fr_vals)
        res.opdms.append(opdm)
        res.iter_times.append(time.time() - start_time)

        rot_gens = non_redundant_rotation_generators(rhf_objective)
        dvec, hmat = get_dvec_hmat(rotation_generators=rot_gens,
                                   rhf_objective=rhf_objective,
                                   rdms=rdms,
                                   diagonal_hessian=True if hessian_update == 'diagonal' else False)
        # talk if talking is allowed
        if verbose:
            print("\nITERATION NUMBER : ", current_count)
            print("\n unitary")
            print(current_unitary)
            test_opdm_aa = u_new @ opdm_initial @ u_new.conj().T
            true_energy = rhf_objective.energy_from_opdm(test_opdm_aa)
            print("Current Energy: ", current_energy)
            print("true energy ", true_energy)
            print("dvec")
            print(list(zip(dvec, rot_gens)))

        # build augmented Hessian
        dvec = dvec.reshape((-1, 1))
        aug_hess = np.hstack((np.array([[0]]), dvec.conj().T))
        aug_hess = np.vstack(
            (aug_hess, np.hstack((dvec, hmat))))

        w, v = np.linalg.eig(aug_hess)
        sort_idx = np.argsort(w)
        w = w[sort_idx]
        v = v[:, sort_idx]
        new_fr_vals = v[1:, [0]].flatten() / v[0, 0]

        assert new_fr_vals.shape[0] == initial_parameters.shape[0]
        assert np.isclose(w[0], dvec.T @ new_fr_vals)

        # Qiming's algorithm for no learning rate rescaling
        if np.max(abs(new_fr_vals)) >= delta:
            new_fr_vals = delta * new_fr_vals / np.max(abs(new_fr_vals))

        # keep track of the norm
        fval_norms.append(np.linalg.norm(new_fr_vals))
        # allow a stopping condition
        if verbose:
            print("New fr values norm")
            print(np.linalg.norm(new_fr_vals))
        if np.linalg.norm(new_fr_vals) < rtol:
            if verbose:
                print("Finished Optimization")
            break

        # assign new values to the things being evaluated next iteration
        fr_vals = new_fr_vals.copy()
        current_unitary = u_new.copy()

        current_count += 1

    return res
