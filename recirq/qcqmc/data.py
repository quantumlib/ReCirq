from pathlib import Path

from recirq.qcqmc.config import OUTDIRS


def get_integrals_path(
    name: str, *, integrals_dir: str = OUTDIRS.DEFAULT_INTEGRALS_DIR
) -> Path:
    """Find integral data file by name.

    Args:
        name: The molecule name.
    """
    return Path(integrals_dir) / name / "hamiltonian.chk"


_MOLECULE_INFO = {"fh_sto3g"}
