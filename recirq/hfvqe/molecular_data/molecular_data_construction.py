# pylint: disable=C
# coverage: ignore
"""
Construct MolecularData objects for various molecules
"""
from typing import Optional

import openfermion as of

NO_PSI4 = False
try:
    import psi4  # type: ignore
except ImportError:
    NO_PSI4 = True

import numpy as np

import scipy as sp

NO_OFPSI4 = False
try:
    from openfermionpsi4 import run_psi4  # type: ignore
    from openfermionpsi4._run_psi4 import create_geometry_string  # type: ignore
except ImportError:
    NO_OFPSI4 = True

from openfermion.hamiltonians import MolecularData

from openfermioncirq.experiments.hfvqe.objective import generate_hamiltonian, \
    RestrictedHartreeFockObjective


class NOOFPsi4Error(Exception):
    pass


class NOPsi4Error(Exception):
    pass


def _h_n_linear_geometry(bond_distance: float, n_hydrogens: int):
    # coverage: ignore
    """Create a geometry of evenly-spaced hydrogen atoms along the Z-axis
    appropriate for consumption by MolecularData."""
    return [('H', (0, 0, i * bond_distance)) for i in range(n_hydrogens)]


def h_n_linear_molecule(bond_distance: float, n_hydrogens: int,
                        basis: str = 'sto-3g'):
    # coverage: ignore
    if n_hydrogens < 1 or n_hydrogens % 2 != 0:
        raise ValueError('Must specify a positive, even number of hydrogens.')
    molecule = MolecularData(
        geometry=_h_n_linear_geometry(bond_distance, n_hydrogens),
        charge=0,
        basis=basis,
        multiplicity=1,
        description=f"linear_r-{bond_distance}",
    )
    if NO_OFPSI4:
        raise NOOFPsi4Error("openfermion-psi4 is not installed")

    molecule = run_psi4(molecule, run_fci=False, run_mp2=False, run_cisd=False,
                        run_ccsd=False, delete_input=False, delete_output=False)

    return molecule


def h2_molecule(bond_distance: float,
                basis: Optional[str]='sto-3g'):
    # coverage: ignore
    return h_n_linear_molecule(bond_distance, n_hydrogens=2, basis=basis)  # testpragma: no cover


def h4_linear_molecule(bond_distance: float,
                       basis: Optional[str]='sto-3g'):
    return h_n_linear_molecule(bond_distance, n_hydrogens=4, basis=basis)  # testpragma: no cover


def h6_linear_molecule(bond_distance: float,
                       basis: Optional[str]='sto-3g'):
    return h_n_linear_molecule(bond_distance, n_hydrogens=6, basis=basis)  # testpragma: no cover


def h8_linear_molecule(bond_distance: float,
                       basis: Optional[str]='sto-3g'):
    return h_n_linear_molecule(bond_distance, n_hydrogens=8, basis=basis)  # testpragma: no cover


def h10_linear_molecule(bond_distance: float,
                        basis: Optional[str]='sto-3g'):
    return h_n_linear_molecule(bond_distance, n_hydrogens=10, basis=basis)  # testpragma: no cover


def h12_linear_molecule(bond_distance: float,
                        basis: Optional[str]='sto-3g'):
    return h_n_linear_molecule(bond_distance, n_hydrogens=12, basis=basis)  # testpragma: no cover


def get_ao_integrals(molecule: MolecularData,
                     e_convergence: Optional[float]=1e-8):
    """
    Use psi4numpy to grab the atomic orbital integrals

    Modified from https://github.com/psi4/psi4numpy/blob/master/Tutorials/01_Psi4NumPy-Basics/1e_mints-helper.ipynb

    :param molecule:
    :return:
    """
    if NO_PSI4:
        raise NOPsi4Error("Psi4 is not installed")

    psi4.core.be_quiet()
    allocated_ram_gb = 0.5
    psi4.set_memory(f"{allocated_ram_gb} GB")
    psi4.core.be_quiet()

    mol = psi4.geometry("""
    {} {}
    {}
    symmetry c1
    """.format(molecule.charge, molecule.multiplicity,
               create_geometry_string(molecule.geometry)))

    psi4.set_options({'basis': molecule.basis,
                      'scf_type': 'pk',
                      'e_convergence': e_convergence})
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
    mints = psi4.core.MintsHelper(wfn.basisset())

    ao_overlap = np.asarray(mints.ao_overlap())
    ao_eri = np.asarray(mints.ao_eri())  # [Gives integrals in Chemist form [(𝜇𝜈∣𝜆𝜎)]
    ao_kinetic = np.asarray(mints.ao_kinetic())
    ao_potential = np.asarray(mints.ao_potential())
    ao_core_hamiltonian = ao_kinetic + ao_potential

    return ao_overlap, ao_core_hamiltonian, ao_eri


def make_rhf_objective(molecule: of.MolecularData):
    S, Hcore, TEI = get_ao_integrals(molecule)
    _, X = sp.linalg.eigh(Hcore, S)

    obi = of.general_basis_change(Hcore, X, (1, 0))
    tbi = np.einsum('psqr', of.general_basis_change(TEI, X, (1, 0, 1, 0)))
    molecular_hamiltonian = generate_hamiltonian(obi, tbi,
                                                 molecule.nuclear_repulsion)

    rhf_objective = RestrictedHartreeFockObjective(molecular_hamiltonian,
                                                   molecule.n_electrons)
    return rhf_objective, S, Hcore, TEI, obi, tbi