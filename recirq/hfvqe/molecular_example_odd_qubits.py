import os
from typing import Tuple

import numpy as np
import openfermion as of
import scipy as sp
import openfermioncirq.experiments.hfvqe as hfvqe

from openfermioncirq.experiments.hfvqe.gradient_hf import rhf_minimization
from openfermioncirq.experiments.hfvqe.objective import (
    RestrictedHartreeFockObjective, generate_hamiltonian)


def make_h3_2_5() -> Tuple[RestrictedHartreeFockObjective, of.MolecularData, np.
                           ndarray, np.ndarray, np.ndarray]:
    # load the molecule from moelcular data
    h3_2_5_path = os.path.join(
        hfvqe.__path__[0],
        'molecular_data/hydrogen_chains/h_3_p_sto-3g/bond_distance_2.5')

    molfile = os.path.join(h3_2_5_path,
                           'H3_plus_sto-3g_singlet_linear_r-2.5.hdf5')
    molecule = of.MolecularData(filename=molfile)
    molecule.load()

    S = np.load(os.path.join(h3_2_5_path, 'overlap.npy'))
    Hcore = np.load(os.path.join(h3_2_5_path, 'h_core.npy'))
    TEI = np.load(os.path.join(h3_2_5_path, 'tei.npy'))

    _, X = sp.linalg.eigh(Hcore, S)
    obi = of.general_basis_change(Hcore, X, (1, 0))
    tbi = np.einsum('psqr', of.general_basis_change(TEI, X, (1, 0, 1, 0)))
    molecular_hamiltonian = generate_hamiltonian(obi, tbi,
                                                 molecule.nuclear_repulsion)

    rhf_objective = RestrictedHartreeFockObjective(molecular_hamiltonian,
                                                   molecule.n_electrons)

    scipy_result = rhf_minimization(rhf_objective)
    return rhf_objective, molecule, scipy_result.x, obi, tbi
