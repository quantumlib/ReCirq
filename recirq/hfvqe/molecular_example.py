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

import os
from typing import Tuple

import numpy as np
import openfermion as of
import scipy as sp

from recirq.hfvqe.gradient_hf import rhf_minimization
from recirq.hfvqe.objective import (RestrictedHartreeFockObjective,
                                    generate_hamiltonian)

_MOLECULAR_DATA_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + '/molecular_data'


def make_h6_1_3(molecular_data_directory=None) \
        -> Tuple[RestrictedHartreeFockObjective, of.MolecularData,
                 np.ndarray, np.ndarray, np.ndarray]:
    if molecular_data_directory is None:
        molecular_data_directory = _MOLECULAR_DATA_DIRECTORY

    h6_1_3_path = f'{molecular_data_directory}/hydrogen_chains/h_6_sto-3g/bond_distance_1.3'
    molfile = f'{h6_1_3_path}/H6_sto-3g_singlet_linear_r-1.3.hdf5'
    molecule = of.MolecularData(filename=molfile)
    molecule.load()

    S = np.load(os.path.join(h6_1_3_path, 'overlap.npy'))
    Hcore = np.load(os.path.join(h6_1_3_path, 'h_core.npy'))
    TEI = np.load(os.path.join(h6_1_3_path, 'tei.npy'))

    _, X = sp.linalg.eigh(Hcore, S)
    obi = of.general_basis_change(Hcore, X, (1, 0))
    tbi = np.einsum('psqr', of.general_basis_change(TEI, X, (1, 0, 1, 0)))
    molecular_hamiltonian = generate_hamiltonian(obi, tbi,
                                                 molecule.nuclear_repulsion)

    rhf_objective = RestrictedHartreeFockObjective(molecular_hamiltonian,
                                                   molecule.n_electrons)

    scipy_result = rhf_minimization(rhf_objective)

    return rhf_objective, molecule, scipy_result.x, obi, tbi


def make_h3_2_5(molecular_data_directory=None) \
        -> Tuple[RestrictedHartreeFockObjective, of.MolecularData,
                 np.ndarray, np.ndarray, np.ndarray]:
    if molecular_data_directory is None:
        molecular_data_directory = _MOLECULAR_DATA_DIRECTORY

    h3_2_5_path = f'{molecular_data_directory}/hydrogen_chains/h_3_p_sto-3g/bond_distance_2.5'
    molfile = f'{h3_2_5_path}/H3_plus_sto-3g_singlet_linear_r-2.5.hdf5'
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
