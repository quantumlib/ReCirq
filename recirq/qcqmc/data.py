from pathlib import Path

DEFAULT_BASE_DATA_DIR = (Path(__file__).parent / '../data').resolve()


def get_integrals_path(name: str, *, base_data_dir: Path = DEFAULT_BASE_DATA_DIR) -> Path:
    """Find integral data file by name.

    Args:
        name: The molecule name.
    """
    return Path(base_data_dir) / 'integrals' / name / 'hamiltonian.chk'


_MOLECULE_INFO = {'fh_sto3g'}
