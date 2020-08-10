# pylint: disable=C
# coverage: ignore
import pytest
from openfermioncirq.experiments.hfvqe.molecular_data.molecular_data_construction import h_n_linear_molecule


def test_negative_n_hydrogen_chain():
    # coverage: ignore
    with pytest.raises(ValueError):
        h_n_linear_molecule(1.3, 0)