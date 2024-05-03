from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, Iterator, List, Protocol, Sequence, Tuple, Union

import cirq
import numpy as np

from qc_afqmc.quaff import (
    get_parameterized_truncated_cliffords_ops,
    get_truncated_cliffords_resolver,
    TruncatedCliffordGate,
)
from qc_afqmc.trial_wf import TrialWavefunctionData, TrialWavefunctionParams
from qc_afqmc.utilities import Data, is_expected_elementary_cirq_op, OUTDIRS, Params


class _CirqOptimizer(Protocol):
    def optimize_circuit(self, circuit: cirq.Circuit) -> None:
        ...


def apply_optimizer_suite_0(circuit: cirq.Circuit) -> cirq.Circuit:
    """A circuit optimization routine that tries to merge gates."""

    circuit = cirq.expand_composite(circuit)
    circuit = cirq.align_left(circuit)

    circuit = cirq.drop_empty_moments(circuit)

    return circuit


@dataclass(frozen=True, repr=False)
class BlueprintParamsTrialWf(Params):
    """Class for storing the parameters that specify a BlueprintData.

    This stage of the experiment concerns itself with the Hardware-specific concerns
    of compilation and shadow tomography implementation.

    Args:
        name: `Params` name for this experiment.
        trial_wf_params: A back-reference to the `TrialWavefunctionParams`
            used in this experiment.
        n_cliffords: The number of random cliffords to use during shadow tomography.
        qubit_partition: For shadow tomography, we partition the qubits into these
            disjoint partitions. For example, we can partition into single-qubit partitions
            and sample from random single-qubit cliffords or put all qubits in one partition
            and sample from random n-qubit cliffords.
        seed: The random seed used for clifford generation.
        optimizer_suite: How to compile/optimize circuits for running on real devices. Can
            be `0` or `1` corresponding to the functions `apply_optimizer_suite_x`.
    """

    name: str
    trial_wf_params: TrialWavefunctionParams
    n_cliffords: int
    qubit_partition: Tuple[Tuple[cirq.Qid, ...], ...]
    seed: int = 0
    optimizer_suite: int = 0

    def __post_init__(self):
        """A little helper to ensure that tuples end up as tuples after loading."""
        object.__setattr__(
            self, 'qubit_partition', tuple(tuple(thing) for thing in self.qubit_partition)
        )

    @property
    def path_string(self) -> str:
        return OUTDIRS.DEFAULT_BLUEPRINT_DIRECTORY + self.name

    @property
    def qubits_jordan_wigner_order(self) -> Tuple[cirq.GridQubit, ...]:
        """A helper that gets the qubits for this Blueprint."""
        return self.trial_wf_params.qubits_jordan_wigner_ordered

    @property
    def qubits_linearly_connected(self) -> Tuple[cirq.GridQubit, ...]:
        """A helper that gets the qubits for this Blueprint."""
        return self.trial_wf_params.qubits_linearly_connected

    @property
    def qubits(self) -> Tuple[cirq.Qid, ...]:
        return self.trial_wf_params.qubits_linearly_connected

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass(frozen=True, repr=False)
class BlueprintParamsRobustShadow(Params):
    """Class for storing the parameters that specify a BlueprintData."""

    name: str
    n_cliffords: int
    qubit_partition: Tuple[Tuple[cirq.Qid, ...], ...]
    seed: int = 0
    optimizer_suite: int = 0

    def __post_init__(self):
        """A little helper to ensure that tuples end up as tuples after loading."""
        object.__setattr__(
            self,
            'qubit_partition',
            tuple(tuple(inner for inner in thing) for thing in self.qubit_partition),
        )

    @property
    def path_string(self) -> str:
        return OUTDIRS.DEFAULT_BLUEPRINT_DIRECTORY + self.name

    @property
    def qubits(self) -> Tuple[cirq.Qid, ...]:
        return tuple(chain(*self.qubit_partition))

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


BlueprintParams = Union[BlueprintParamsTrialWf, BlueprintParamsRobustShadow]


def _get_truncated_cliffords(
    n_cliffords: int, qubit_partition: Sequence[Sequence[cirq.Qid]], seed: int
) -> Iterator[List[TruncatedCliffordGate]]:
    """Gets the gates (not the circuits) for applying the random circuit for shadow tomography"""
    rng = np.random.default_rng(seed)

    for _ in range(n_cliffords):
        yield [TruncatedCliffordGate.random(len(part), rng) for part in qubit_partition]


def _get_resolvers(
    n_cliffords: int, qubit_partition: Sequence[Sequence[cirq.Qid]], seed: int
) -> Iterator[Dict[str, np.integer]]:
    """Gets the resolvers for a parameterized shadow tomography circuit.

    These are used in running the experiment / simulation.
    """
    truncated_cliffords = _get_truncated_cliffords(
        n_cliffords=n_cliffords, qubit_partition=qubit_partition, seed=seed
    )

    for clifford_set in truncated_cliffords:
        yield get_truncated_cliffords_resolver(clifford_set)


@dataclass(frozen=True)
class BlueprintData(Data):
    """Data resulting from the "Blueprint" phase of the experiment.

    This stage of the experiment concerns itself with the Hardware-specific concerns
    of compilation and shadow tomography implementation.

    Args:
        params: A back-reference to the `BlueprintParams` used to create this `Data`.
        compiled_circuit: A circuit suitable for running on the hardware including the
            ansatz preparation segment and shadow-tomography rotations (i.e. layers of
            cliffords). Its clifford layers are parameterized for efficient execution,
            so you must combine this with `resolvers`.
        parameterized_clifford_circuits: A parameterized circuit that corresponds to
            just the Clifford part of the shadow tomography circuit. Useful for
            inverting the channel when combined with resolvers.
        resolvers: A list of `cirq.ParamResolver` corresponding to the (outer) list of
            random cliffords. When combined with the parameterized `compiled_circuit` and
            `cirq.Sampler.run_sweep`, this will execute all the different random clifford
            circuits.
    """

    params: BlueprintParams
    compiled_circuit: cirq.Circuit
    parameterized_clifford_circuits: Sequence[cirq.Circuit]
    resolvers: List[cirq.ParamResolverOrSimilarType]

    def __post_init__(self):
        """A little helper to ensure that tuples end up as tuples after loading."""
        object.__setattr__(
            self, 'parameterized_clifford_circuits', tuple(self.parameterized_clifford_circuits)
        )

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @property
    def resolved_clifford_circuits(self) -> Iterator[Tuple[cirq.Circuit, ...]]:
        for resolver in self.resolvers:
            yield tuple(
                cirq.resolve_parameters(clifford, resolver)
                for clifford in self.parameterized_clifford_circuits
            )


def build_blueprint_from_base_circuit(
    params: BlueprintParams, *, base_circuit: cirq.AbstractCircuit
) -> BlueprintData:
    """Builds a BlueprintData from BlueprintParams.

    Args:
        base_circuit: The circuit to shadow tomographize.
    """
    resolvers = list(_get_resolvers(params.n_cliffords, params.qubit_partition, params.seed))

    parameterized_clifford_ops: Iterable[cirq.OP_TREE] = get_parameterized_truncated_cliffords_ops(params.qubit_partition)  # type: ignore

    parameterized_clifford_circuits = tuple(
        cirq.expand_composite(cirq.Circuit(ops), no_decomp=is_expected_elementary_cirq_op)
        for ops in parameterized_clifford_ops
    )

    parameterized_clifford_circuit = sum(parameterized_clifford_circuits, cirq.Circuit())

    compiled_circuit = cirq.Circuit([base_circuit, parameterized_clifford_circuit])

    circuit_with_measurement = compiled_circuit + cirq.Circuit(
        cirq.measure(*params.qubits, key='all')
    )

    apply_optimizer_suite = {0: apply_optimizer_suite_0}[params.optimizer_suite]

    optimized_circuit = apply_optimizer_suite(circuit_with_measurement)

    return BlueprintData(
        params=params,
        compiled_circuit=optimized_circuit,
        parameterized_clifford_circuits=parameterized_clifford_circuits,
        resolvers=resolvers,  # type: ignore
    )


def build_blueprint(
    params: BlueprintParams, dependencies: Dict[Params, Data] = None  # type: ignore
) -> BlueprintData:
    """Builds a BlueprintData from BlueprintParams using the dependency-injection workflow system."""
    if isinstance(params, BlueprintParamsRobustShadow):
        base_circuit = cirq.Circuit()
    elif isinstance(params, BlueprintParamsTrialWf):
        assert dependencies is not None, 'Provide trial_wf'
        assert params.trial_wf_params in dependencies, 'trial_wf dependency'
        trial_wf = dependencies[params.trial_wf_params]
        assert isinstance(trial_wf, TrialWavefunctionData)
        base_circuit = trial_wf.superposition_circuit
    else:
        raise ValueError(f"Bad param type {type(params)}")

    return build_blueprint_from_base_circuit(params=params, base_circuit=base_circuit)
