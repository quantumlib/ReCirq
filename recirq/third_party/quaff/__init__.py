from quaff import indexing, json_serialization, linalg, random, testing

from .basis_change import BasisChangeGate, get_clearing_network, get_reversal_network
from .comb import binom, get_inversion_number, get_num_perms_with_inversion_number
from .cz_layer import CZLayerBasisChangeStageGate, CZLayerGate
from .gates import (
    FHFGate,
    HadamardFreeGate,
    SingleQubitLayerGate,
    TruncatedCliffordGate,
    get_parameterized_truncated_clifford_ops,
    get_parameterized_truncated_cliffords_ops,
    get_truncated_clifford_resolver,
    get_truncated_cliffords_resolver,
)
from .indexing import (
    bitstring_to_index,
    bitstrings_to_indices,
    get_all_bitstrings,
    get_unit_vector_index,
    get_unit_vector_indices,
    index_to_bitstring,
    log2,
    offdiag_indices,
    tril_indices,
    triu_indices,
)
from .json_serialization import DEFAULT_RESOLVERS, read_json
from .linalg import (
    DTYPE,
    apply_affine_transform_to_bitstrings,
    apply_affine_transform_to_state_vector,
    dot,
    get_coordinates,
    get_inverse,
    get_lexicographic_basis,
    get_min_in_span,
    invert_permutation,
    is_invertible,
    is_nw_tri,
    is_tril_and_unit_diag,
    row_reduce,
    tuple_of_tuples,
    tuplify,
    with_qubits_reversed,
)
from .random import (
    RNG,
    random_invertible_matrix,
    random_invertible_nw_tri_matrix,
    random_nw_tri_matrix,
    random_permutation,
    random_seed,
    random_state,
    random_symmetric_matrix,
    random_tril_and_unit_diag_matrix,
)
from .sampling import (
    CliffordRandomness,
    CliffordSample,
    CliffordSampler,
    InvertibleMatrixSampler,
    MallowsSampler,
    QuantumMallowsSampler,
)


def _register_resolver() -> None:
    from quaff.json_resolver import json_resolver
    from quaff.json_serialization import _internal_register_resolver

    _internal_register_resolver(json_resolver)


_register_resolver()
