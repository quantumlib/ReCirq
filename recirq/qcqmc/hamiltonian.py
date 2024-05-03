from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cirq
import fqe
import h5py
import numpy as np
import openfermion as of
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.openfermion_utils import integrals_to_fqe_restricted
from openfermionpyscf._run_pyscf import compute_integrals  # type: ignore
from pyscf import ao2mo, fci, gto, scf

from qc_afqmc.data import get_integrals_path
from qc_afqmc.utilities import Data, OUTDIRS, Params


@dataclass(frozen=True, repr=False)
class LoadFromFileHamiltonianParams(Params):
    """Class for storing the parameters that specify loading integrals from a file."""

    name: str
    integral_key: str
    n_orb: int
    n_elec: int
    do_eri_restore: bool = False

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @property
    def path_string(self) -> str:
        return OUTDIRS.DEFAULT_HAMILTONIAN_DIRECTORY + self.name


@dataclass(frozen=True, repr=False)
class PyscfHamiltonianParams(Params):
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
    """

    name: str
    n_orb: int
    n_elec: int
    geometry: Tuple[Tuple[str, Tuple[float, float, float]], ...]
    basis: str
    multiplicity: int
    charge: int = 0
    rhf: bool = True
    verbose_scf: int = 0
    save_chkfile: bool = False
    overwrite_chk_file: bool = False

    # TODO: Add support for active space selection here. See #69

    def __post_init__(self):
        """A little special sauce to make sure that this ends up as a tuple."""
        object.__setattr__(self, 'geometry', tuple((i[0], tuple(i[1])) for i in self.geometry))

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @property
    def path_string(self) -> str:
        return OUTDIRS.DEFAULT_HAMILTONIAN_DIRECTORY + self.name

    @property
    def chk_path(self) -> Path:
        return self.base_path.with_suffix(".chk")

    @property
    def pyscf_molecule(self) -> gto.Mole:
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


HamiltonianParams = Union[LoadFromFileHamiltonianParams, PyscfHamiltonianParams]


@dataclass(frozen=True, eq=False)
class HamiltonianData(Data):
    """Data class that contains information about a Hamiltonian."""

    params: HamiltonianParams
    e_core: float
    one_body_integrals: np.ndarray
    two_body_integrals_pqrs: np.ndarray
    e_hf: float
    e_fci: float

    def __post_init__(self):
        """We need to make some inputs into np.ndarrays if aren't provided that way."""
        array_like = ["one_body_integrals", "two_body_integrals_pqrs"]

        for field in array_like:
            object.__setattr__(self, field, np.asarray(getattr(self, field)))

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

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


def load_integrals(*, filepath: Union[str, Path]) -> Tuple[float, np.ndarray, np.ndarray, float]:
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
    one_body_integrals: np.ndarray, two_body_integrals: np.ndarray, EQ_TOLERANCE: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
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
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = two_body_integrals[
                        p, q, r, s
                    ]
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = two_body_integrals[
                        p, q, r, s
                    ]

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = two_body_integrals[
                        p, q, r, s
                    ]
                    two_body_coefficients[
                        2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                    ] = two_body_integrals[p, q, r, s]

    # Truncate.
    one_body_coefficients[np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.0
    two_body_coefficients[np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.0

    return one_body_coefficients, two_body_coefficients


def _assert_real(x: np.complex_, tol=1e-8) -> float:
    assert np.abs(x.imag) < tol
    return float(x.real)


def build_hamiltonian_from_file(params: LoadFromFileHamiltonianParams) -> HamiltonianData:
    """Function for loading a Hamiltonian from a file."""
    filepath = get_integrals_path(name=params.integral_key)
    e_core, one_body_integrals, two_body_integrals_psqr, e_fci = load_integrals(filepath=filepath)

    n_orb = params.n_orb

    if params.do_eri_restore:
        two_body_integrals_psqr = ao2mo.restore(1, two_body_integrals_psqr, n_orb)
        # This step may be necessary depending on the format in which the integrals were stored.

    # We have to reshape to a four index tensor and reorder the indices from
    # chemist's ordering to physicist's.
    two_body_integrals_psqr = two_body_integrals_psqr.reshape((n_orb, n_orb, n_orb, n_orb))
    two_body_integrals_pqrs = np.einsum("psqr->pqrs", two_body_integrals_psqr)

    fqe_ham = integrals_to_fqe_restricted(h1e=one_body_integrals, h2e=two_body_integrals_pqrs)

    initial_wf = fqe.Wavefunction([[params.n_elec, 0, params.n_orb]])
    initial_wf.set_wfn(strategy="hartree-fock")
    e_hf = _assert_real(initial_wf.expectationValue(fqe_ham) + e_core)

    return HamiltonianData(
        params=params,
        e_core=e_core,
        one_body_integrals=one_body_integrals,
        two_body_integrals_pqrs=two_body_integrals_pqrs,
        e_hf=e_hf,
        e_fci=e_fci,
    )


def build_hamiltonian_from_pyscf(params: PyscfHamiltonianParams) -> HamiltonianData:
    """Function for generating a Hamiltonian using pyscf."""
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
