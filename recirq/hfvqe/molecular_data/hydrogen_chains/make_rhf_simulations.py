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

# coverage: ignore
"""
Implement the H2-experiment with OpenFermion
"""
import os
import numpy
import scipy

from openfermion.ops import general_basis_change

from recirq.hfvqe.molecular_data.molecular_data_construction import (
    h6_linear_molecule, h8_linear_molecule, h10_linear_molecule,
    h12_linear_molecule, get_ao_integrals)
from recirq.hfvqe.gradient_hf import rhf_minimization, rhf_func_generator
from recirq.hfvqe.objective import \
    RestrictedHartreeFockObjective, generate_hamiltonian


def make_rhf_objective(molecule):
    # coverage: ignore
    S, Hcore, TEI = get_ao_integrals(molecule)
    _, X = scipy.linalg.eigh(Hcore, S)

    molecular_hamiltonian = generate_hamiltonian(
        general_basis_change(Hcore, X, (1, 0)),
        numpy.einsum('psqr', general_basis_change(TEI, X, (1, 0, 1, 0)),
                     molecule.nuclear_repulsion))

    rhf_objective = RestrictedHartreeFockObjective(molecular_hamiltonian,
                                                   molecule.n_electrons)
    return rhf_objective, S, Hcore, TEI


if __name__ == "__main__":
    # coverage: ignore
    # make simulations
    molecule_generator = {
        6: h6_linear_molecule,
        8: h8_linear_molecule,
        10: h10_linear_molecule,
        12: h12_linear_molecule
    }

    for n in range(6, 13, 2):
        name = "h_{}_sto-3g".format(n)
        print(name)
        # now make a dirctory with the name
        os.mkdir(name)
        os.chdir(name)
        # # now make a separate folder for each of 50 points along a line
        bond_distances = numpy.linspace(0.5, 2.5, 6)
        for bb in bond_distances:
            print(bb)
            local_dir = 'bond_distance_{:.1f}'.format(bb)
            os.mkdir(local_dir)
            os.chdir(local_dir)
            molecule = molecule_generator[n](bb)

            rhf_objective, S, HCore, TEI = make_rhf_objective(molecule)

            numpy.save("overlap.npy", S)
            numpy.save("h_core.npy", HCore)
            numpy.save("tei.npy", TEI)

            ansatz, energy, gradient = rhf_func_generator(rhf_objective)
            scipy_result = rhf_minimization(rhf_objective)
            print(molecule.hf_energy)
            print(scipy_result.fun)
            assert numpy.isclose(molecule.hf_energy, scipy_result.fun)

            numpy.save("parameters.npy", numpy.asarray(scipy_result.x))
            initial_opdm = numpy.diag([1] * rhf_objective.nocc +
                                      [0] * rhf_objective.nvirt)
            unitary = ansatz(scipy_result.x)
            final_opdm = unitary @ initial_opdm @ numpy.conjugate(unitary).T
            assert numpy.isclose(rhf_objective.energy_from_opdm(final_opdm),
                                 scipy_result.fun)
            numpy.save("true_opdm.npy", numpy.asarray(final_opdm))

            molecule.filename = os.path.join(os.getcwd(), molecule.name)
            molecule.save()

            os.chdir('../')
        os.chdir('../')
