import numpy as np
import pytest
from openfermion import (
    get_fermion_operator,
    get_ground_state,
    get_number_preserving_sparse_operator,
)

from qc_afqmc.hamiltonian import (
    build_hamiltonian_from_file,
    build_hamiltonian_from_pyscf,
    LoadFromFileHamiltonianParams,
    PyscfHamiltonianParams,
)


def test_load_from_file_hamiltonian_runs():
    params = LoadFromFileHamiltonianParams(
        name="test hamiltonian", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    hamiltonian_data = build_hamiltonian_from_file(params)

    assert hamiltonian_data.one_body_integrals.shape == (2, 2)
    assert hamiltonian_data.two_body_integrals_pqrs.shape == (2, 2, 2, 2)


@pytest.mark.parametrize(
    'integral_key, n_orb, n_elec, do_eri_restore',
    [
        ('fh_sto3g', 2, 2, False),
        ('pabi_def2svp', 2, 2, False),
        ('h4_sto3g', 4, 4, False),
        ('diamond_dzvp/cas66', 6, 6, True),
    ],
)
def test_hamiltonian_energy_consistent(
    integral_key: str, n_orb: int, n_elec: int, do_eri_restore: bool
):
    params = LoadFromFileHamiltonianParams(
        name="test hamiltonian",
        integral_key=integral_key,
        n_orb=n_orb,
        n_elec=n_elec,
        do_eri_restore=do_eri_restore,
    )

    hamiltonian_data = build_hamiltonian_from_file(params)

    molecular_hamiltonian = hamiltonian_data.get_molecular_hamiltonian()

    ham_matrix = get_number_preserving_sparse_operator(
        get_fermion_operator(molecular_hamiltonian), n_orb * 2, n_elec, True
    )
    energy, _ = get_ground_state(ham_matrix)

    np.testing.assert_almost_equal(energy, hamiltonian_data.e_fci)


def test_pyscf_h4_consistent_with_file():
    pyscf_params = PyscfHamiltonianParams(
        name="TEST_Square H4",
        n_orb=4,
        n_elec=4,
        geometry=(
            ('H', (0, 0, 0)),
            ('H', (0, 0, 1.23)),
            ('H', (1.23, 0, 0)),
            ('H', (1.23, 0, 1.23)),
        ),
        basis="sto3g",
        multiplicity=1,
        charge=0,
    )

    pyscf_hamiltonian = build_hamiltonian_from_pyscf(pyscf_params)

    from_file_params = LoadFromFileHamiltonianParams(
        name="test hamiltonian", integral_key='h4_sto3g', n_orb=4, n_elec=4, do_eri_restore=False
    )

    from_file_hamiltonian = build_hamiltonian_from_file(from_file_params)

    np.testing.assert_almost_equal(
        pyscf_hamiltonian.e_core, from_file_hamiltonian.e_core, decimal=10
    )
    np.testing.assert_almost_equal(pyscf_hamiltonian.e_hf, from_file_hamiltonian.e_hf, decimal=10)
    np.testing.assert_almost_equal(pyscf_hamiltonian.e_fci, from_file_hamiltonian.e_fci, decimal=10)


def test_pyscf_saves_chk_without_overwrite():
    pyscf_params = PyscfHamiltonianParams(
        name="TEST Square H4 chk save test",
        n_orb=4,
        n_elec=4,
        geometry=(
            ('H', (0, 0, 0)),
            ('H', (0, 0, 1.23)),
            ('H', (1.23, 0, 0)),
            ('H', (1.23, 0, 1.23)),
        ),
        basis="sto3g",
        multiplicity=1,
        charge=0,
        save_chkfile=True,
    )

    chk_path = pyscf_params.base_path.with_suffix(".chk")
    chk_path.unlink(missing_ok=True)

    pyscf_hamiltonian = build_hamiltonian_from_pyscf(pyscf_params)
    assert chk_path.exists()

    with pytest.raises(FileExistsError):
        pyscf_hamiltonian = build_hamiltonian_from_pyscf(pyscf_params)

    chk_path.unlink()


def test_pyscf_saves_chk_with_overwrite():
    pyscf_params = PyscfHamiltonianParams(
        name="TEST Square H4 chk save test",
        n_orb=4,
        n_elec=4,
        geometry=(
            ('H', (0, 0, 0)),
            ('H', (0, 0, 1.23)),
            ('H', (1.23, 0, 0)),
            ('H', (1.23, 0, 1.23)),
        ),
        basis="sto3g",
        multiplicity=1,
        charge=0,
        save_chkfile=True,
        overwrite_chk_file=True,
    )

    chk_path = pyscf_params.base_path.with_suffix(".chk")
    chk_path.unlink(missing_ok=True)

    pyscf_hamiltonian = build_hamiltonian_from_pyscf(pyscf_params)
    pyscf_hamiltonian = build_hamiltonian_from_pyscf(pyscf_params)

    assert chk_path.exists()

    chk_path.unlink()
