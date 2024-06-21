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

import functools
from pathlib import Path
from typing import Tuple, Union

import attrs
import fqe
import h5py
import numpy as np
import openfermion as of
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.openfermion_utils import integrals_to_fqe_restricted
from pyscf import ao2mo, fci, gto, scf

from recirq.qcqmc import config, data

# xyz coordinates of an atom
PositionT = Tuple[float, float, float]
# [atom name, xyz]
GeomT = Tuple[str, PositionT]


def compute_integrals(
    pyscf_molecule: gto.Mole, pyscf_scf: Union[scf.RHF, scf.ROHF]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the (restricted) 1-electron and 2-electron integrals in the spatial orbital basis.

    From: openfermionpyscf _run_pyscf.

    Args:
        pyscf_molecule: A pyscf Mole instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: A 4-D tensor storing h_{pqrs} in using the openfermion convention.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = functools.reduce(
        np.dot, (pyscf_scf.mo_coeff.T, pyscf_scf.get_hcore(), pyscf_scf.mo_coeff)
    )
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals
    ).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule, pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, two_electron_compressed, n_orbitals  # no permutational symmetry
    )
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order="C"
    )

    return one_electron_integrals, two_electron_integrals


@attrs.frozen
class HamiltonianFileParams(data.Params):
    """Class for storing the parameters that specify loading hamiltonian integrals from a file.

    Args:
        name: A descriptive name for the hamiltonian.
        integral_key: A string specifying the directory in data/integrals where
            the integrals should live.
        n_orb: The number of spatial orbitals.
        n_elec: The total number of electrons.
        do_eri_restore: Whether to restore 8-fold symmetry to the integrals.
        path_prefix: An optional path string to prepend to the default
            hamiltonian data directory (by default the integrals will be written /
            read to ./data/hamiltonians.)
    """

    name: str
    integral_key: str
    n_orb: int
    n_elec: int
    do_eri_restore: bool = False
    path_prefix: str = ""

    def _json_dict_(self):
        return attrs.asdict(self)

    @property
    def path_string(self) -> str:
        return (
            self.path_prefix + config.OUTDIRS.DEFAULT_HAMILTONIAN_DIRECTORY + self.name
        )


@attrs.frozen
class PyscfHamiltonianParams(data.Params):
    """Class for describing the parameters to get a Hamiltonian from Pyscf.

    name: A name for the Hamiltonian
    n_orb: The number of orbitals (redundant with later parameters but used
        for validation).
    n_elec: The number of electrons (redundant with later parameters but
        used for validation).
    geometry: The coordinates of each atom. An example is [('H', (0, 0, 0)),
        ('H', (0, 0, 0.7414))]. Distances are in angstrom. Use atomic
        symbols to specify atoms.
    basis: The basis set, e.g., 'cc-pvtz'.
    multiplicity: The spin multiplicity
    charge: The total molecular charge.
    rhf: Whether to use RHF (True) or ROHF (False).
    verbose_scf: Setting passed to pyscf's scf verbose attribute.
    save_chkfile: If True, then pyscf will save a chk file for this Hamiltonian.
    overwrite_chk_file: If save_chkfile and overwrite_chk_file are both true then Pyscf
        will be allowed to overwrite the previously saved chk file. Otherwise, if save_chkfile
        is True and overwrite_chk_file is False, then we raise a FileExistsError if the
        chk file already exists.
    path_prefix: A prefix to prepend to the path where the data will be saved (if any)
    """

    name: str
    n_orb: int
    n_elec: int
    geometry: Tuple[GeomT, ...] = attrs.field(
        converter=lambda x: tuple((at[0], tuple(at[1])) for at in x)
    )
    basis: str
    multiplicity: int
    charge: int = 0
    rhf: bool = True
    verbose_scf: int = 0
    save_chkfile: bool = False
    overwrite_chk_file: bool = False
    path_prefix: str = ""

    def _json_dict_(self):
        return attrs.asdict(self)

    @property
    def path_string(self) -> str:
        """Output path string"""
        return (
            self.path_prefix + config.OUTDIRS.DEFAULT_HAMILTONIAN_DIRECTORY + self.name
        )

    @property
    def chk_path(self) -> Path:
        """Path string to any checkpoint file is saved."""
        parent = self.base_path.parent
        if not parent.is_dir():
            parent.mkdir(exist_ok=True, parents=True)
        return self.base_path.with_suffix(".chk")

    @property
    def pyscf_molecule(self) -> gto.Mole:
        """Underlying pyscf.gto.Mole. instance."""
        molecule = gto.Mole()

        molecule.atom = self.geometry
        molecule.basis = self.basis
        molecule.spin = self.multiplicity - 1
        molecule.charge = self.charge
        molecule.symmetry = False

        molecule.build()

        assert int(molecule.nao_nr()) == self.n_orb
        assert molecule.nelectron == self.n_elec

        return molecule


HamiltonianParams = Union[HamiltonianFileParams, PyscfHamiltonianParams]


@attrs.frozen
class HamiltonianData(data.Data):
    """Data class that contains information about a Hamiltonian.

    Args:
        params: parameters defining the hamiltonian.
        e_core: The nuclear-nuclear repulsion energy (or any constant term from the Hamiltonian.)
        one_body_integrals: The one-electron integrals as a matrix.
        two_body_integrals: The two-electron integrals using the openfermion convention.
        e_hf: The restricted hartree fock energy of the molecule.
        e_fci: The FCI energy for the molecule / hamiltonian.
    """

    params: HamiltonianParams
    e_core: float
    one_body_integrals: np.ndarray = attrs.field(converter=np.asarray)
    two_body_integrals_pqrs: np.ndarray = attrs.field(converter=np.asarray)
    e_hf: float
    e_fci: float

    def _json_dict_(self):
        return attrs.asdict(self)

    def get_molecular_hamiltonian(self) -> of.InteractionOperator:
        """Gets the OpenFermion Hamiltonian from a HamiltonianData object."""
        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
            one_body_integrals=self.one_body_integrals,
            two_body_integrals=self.two_body_integrals_pqrs,
        )

        molecular_hamiltonian = of.InteractionOperator(
            constant=self.e_core,
            one_body_tensor=one_body_coefficients,
            two_body_tensor=1 / 2 * two_body_coefficients,
        )

        return molecular_hamiltonian

    def get_restricted_fqe_hamiltonian(self) -> RestrictedHamiltonian:
        """Gets an Fqe RestrictedHamiltonian from a HamiltonianData object."""

        return integrals_to_fqe_restricted(
            h1e=self.one_body_integrals, h2e=self.two_body_integrals_pqrs
        )


def load_integrals(
    *, filepath: Union[str, Path]
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Load integrals from a checkpoint file.

    Args:
        filepath: The path of the .h5 file containing the integrals.
    """
    with h5py.File(filepath, "r") as f:
        ecore = float(f["ecore"][()])
        one_body_integrals = np.asarray(f["h1"][:, :])
        efci = float(f["efci"][()])
        two_body_integrals = np.asarray(f["h2"][:, :])

    return ecore, one_body_integrals, two_body_integrals, efci


def spinorb_from_spatial(
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    truncation_tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert integrals from spatial to spin-orbital form.

    Args:
        one_body_integrals: The one-electron integrals in spatial orbital format.
        two_body_integrals: The two-electron integrals in spatial oribtal format.
        truncation_tol: The value below which integrals are considered zero.

    Returns:
        one_body_coefficients: The one-body coefficients in spin-orbital format.
        two_body_coefficients: The two-body coefficients in spin-orbital format.
    """

    n_qubits = 2 * one_body_integrals.shape[0]
    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                        two_body_integrals[p, q, r, s]
                    )
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                        two_body_integrals[p, q, r, s]
                    )

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = (
                        two_body_integrals[p, q, r, s]
                    )
                    two_body_coefficients[
                        2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                    ] = two_body_integrals[p, q, r, s]

    # Truncate.
    one_body_coefficients[np.absolute(one_body_coefficients) < truncation_tol] = 0.0
    two_body_coefficients[np.absolute(two_body_coefficients) < truncation_tol] = 0.0

    return one_body_coefficients, two_body_coefficients


def _cast_to_float(x: np.complex_, tol=1e-8) -> float:
    assert np.abs(x.imag) < tol, "Large imaginary component found."
    return float(x.real)


def build_hamiltonian_from_file(
    params: HamiltonianFileParams,
) -> HamiltonianData:
    """Function for loading a Hamiltonian from a file.

    Args:
        params: parameters defining the hamiltonian.

    Returns:
        properly constructed HamiltonianFileParams object.
    """
    filepath = data.get_integrals_path(name=params.integral_key)
    e_core, one_body_integrals, two_body_integrals_psqr, e_fci = load_integrals(
        filepath=filepath
    )

    n_orb = params.n_orb

    if params.do_eri_restore:
        two_body_integrals_psqr = ao2mo.restore(1, two_body_integrals_psqr, n_orb)
        # This step may be necessary depending on the format in which the integrals were stored.

    # We have to reshape to a four index tensor and reorder the indices from
    # chemist's ordering to physicist's.
    two_body_integrals_psqr = two_body_integrals_psqr.reshape(
        (n_orb, n_orb, n_orb, n_orb)
    )
    two_body_integrals_pqrs = np.einsum("psqr->pqrs", two_body_integrals_psqr)

    fqe_ham = integrals_to_fqe_restricted(
        h1e=one_body_integrals, h2e=two_body_integrals_pqrs
    )

    initial_wf = fqe.Wavefunction([[params.n_elec, 0, params.n_orb]])
    initial_wf.set_wfn(strategy="hartree-fock")
    e_hf = _cast_to_float(initial_wf.expectationValue(fqe_ham) + e_core)

    return HamiltonianData(
        params=params,
        e_core=e_core,
        one_body_integrals=one_body_integrals,
        two_body_integrals_pqrs=two_body_integrals_pqrs,
        e_hf=e_hf,
        e_fci=e_fci,
    )


def build_hamiltonian_from_pyscf(params: PyscfHamiltonianParams) -> HamiltonianData:
    """Construct a HamiltonianData object from pyscf molecule.

    Runs a RHF or ROHF simulation, performs an integral transformation and computes the FCI energy.

    Args:
        params: A PyscfHamiltonianParams object specifying the molecule. pyscf_molecule is required.

    Returns:
        A HamiltonianData object.
    """
    molecule = params.pyscf_molecule

    if params.rhf:
        pyscf_scf = scf.RHF(molecule)
    else:
        pyscf_scf = scf.ROHF(molecule)

    pyscf_scf.verbose = params.verbose_scf
    if params.save_chkfile:
        if not params.overwrite_chk_file:
            if params.chk_path.exists():
                raise FileExistsError(f"A chk file already exists at {params.chk_path}")
        pyscf_scf.chkfile = str(params.chk_path.resolve())
    pyscf_scf = pyscf_scf.newton()
    pyscf_scf.run()

    e_hf = float(pyscf_scf.e_tot)

    one_body_integrals, two_body_integrals = compute_integrals(
        pyscf_molecule=molecule, pyscf_scf=pyscf_scf
    )

    e_core = float(molecule.energy_nuc())

    pyscf_fci = fci.FCI(molecule, pyscf_scf.mo_coeff)
    pyscf_fci.verbose = 0
    e_fci = pyscf_fci.kernel()[0]

    return HamiltonianData(
        params=params,
        e_core=e_core,
        one_body_integrals=one_body_integrals,
        two_body_integrals_pqrs=two_body_integrals,
        e_hf=e_hf,
        e_fci=e_fci,
    )
