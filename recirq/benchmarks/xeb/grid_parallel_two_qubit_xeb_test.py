# pylint: disable=wrong-or-nonexistent-copyright-notice
import os
import numpy as np
import cirq
from recirq.benchmarks.xeb import (
    collect_grid_parallel_two_qubit_xeb_data,
    compute_grid_parallel_two_qubit_xeb_results,
)
from recirq.benchmarks.xeb.grid_parallel_two_qubit_xeb import (
    GridParallelXEBMetadata,
    LAYER_A,
    LAYER_B,
)

ALIGNED_HORIZONTAL = cirq.experiments.GridInteractionLayer(
    col_offset=0, vertical=False, stagger=False
)
ALIGNED_VERTICAL = cirq.experiments.GridInteractionLayer(
    col_offset=0, vertical=True, stagger=False
)


def test_estimate_parallel_two_qubit_xeb_fidelity_on_grid_no_noise(tmpdir):
    # No noise, fidelities should be close to 1
    base_dir = os.path.abspath(tmpdir)
    qubits = cirq.GridQubit.square(2)
    two_qubit_gate = cirq.ISWAP**0.5
    cycles = [5, 10, 15]
    data_collection_id = collect_grid_parallel_two_qubit_xeb_data(
        sampler=cirq.Simulator(seed=34310, split_untangled_states=False),
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=2,
        repetitions=1_000,
        cycles=cycles,
        layers=(ALIGNED_HORIZONTAL, ALIGNED_VERTICAL),
        seed=43435,
        num_workers=1,
        base_dir=base_dir,
    )
    results = compute_grid_parallel_two_qubit_xeb_results(
        data_collection_id, base_dir=base_dir
    )

    assert len(results) == 4


def test_estimate_parallel_two_qubit_xeb_fidelity_on_grid_depolarizing(tmpdir):
    # With depolarizing probability e
    base_dir = os.path.abspath(tmpdir)
    qubits = cirq.GridQubit.square(2)
    two_qubit_gate = cirq.ISWAP**0.5
    cycles = [5, 10, 15]
    e = 0.01
    data_collection_id = collect_grid_parallel_two_qubit_xeb_data(
        sampler=cirq.DensityMatrixSimulator(
            noise=cirq.depolarize(e), seed=65008, split_untangled_states=False
        ),
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=2,
        repetitions=1_000,
        cycles=cycles,
        layers=(ALIGNED_HORIZONTAL, ALIGNED_VERTICAL),
        seed=np.random.RandomState(14948),
        num_workers=1,
        base_dir=base_dir,
    )
    results = compute_grid_parallel_two_qubit_xeb_results(
        data_collection_id, num_processors=4, base_dir=base_dir
    )

    assert len(results) == 4


def test_estimate_parallel_two_qubit_xeb_fidelity_on_grid_concurrent(tmpdir):
    # Use multiple threads during data collection
    base_dir = os.path.abspath(tmpdir)
    qubits = cirq.GridQubit.square(2)
    two_qubit_gate = cirq.ISWAP**0.5
    cycles = [5, 10, 15]
    data_collection_id = collect_grid_parallel_two_qubit_xeb_data(
        sampler=cirq.Simulator(seed=34310),
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=2,
        repetitions=1_000,
        cycles=cycles,
        layers=(ALIGNED_HORIZONTAL, ALIGNED_VERTICAL),
        seed=43435,
        num_workers=4,
        base_dir=base_dir,
    )
    results = compute_grid_parallel_two_qubit_xeb_results(
        data_collection_id, base_dir=base_dir
    )

    assert len(results) == 4


def test_grid_parallel_xeb_metadata_repr():
    metadata = GridParallelXEBMetadata(
        qubits=cirq.GridQubit.square(2),
        two_qubit_gate=cirq.ISWAP,
        num_circuits=10,
        repetitions=10_000,
        cycles=[2, 4, 6, 8, 10],
        layers=[LAYER_A, LAYER_B],
        seed=1234,
    )
    cirq.testing.assert_equivalent_repr(metadata,
                                        setup_code="import recirq.benchmarks.xeb\nimport cirq")

