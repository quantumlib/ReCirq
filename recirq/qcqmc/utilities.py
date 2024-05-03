import abc
import dataclasses
import hashlib
import itertools
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Union

import cirq
import numpy as np
from scipy.sparse import coo_matrix


@dataclass
class OutputDirectories:
    DEFAULT_HAMILTONIAN_DIRECTORY: str = "./data/hamiltonians/"
    DEFAULT_TRIAL_WAVEFUNCTION_DIRECTORY: str = "./data/trial_wfs/"
    DEFAULT_QMC_DIRECTORY: str = "./data/afqmc/"
    DEFAULT_BLUEPRINT_DIRECTORY: str = "./data/blueprints/"
    DEFAULT_EXPERIMENT_DIRECTORY: str = "./data/experiments/"
    DEFAULT_ANALYSIS_DIRECTORY: str = "./data/analyses/"


OUTDIRS = OutputDirectories()

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
DEVICE_LOCK_PATH = pathlib.Path(ROOT_DIR + '/data/data_taking.lock')

SINGLE_PRECISION_DEFAULT = True
DO_INVERSE_SIMULATION_QUBIT_NUMBER_CUTOFF = 6

# Controls a variety of print statement.
VERBOSE_EXECUTION = True


def make_output_directories():
    """Make the output directories given in OUTDIRS"""
    for _, dirpath in asdict(OUTDIRS).items():
        try:
            os.makedirs(f"{dirpath}")
        except FileExistsError:
            pass


@dataclass(frozen=True, repr=False)
class Params(abc.ABC):
    name: str

    def __eq__(self, other):
        """A helper method to compare two dataclasses which might contain arrays."""
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            return NotImplemented

        return all(
            array_compatible_eq(thing1, thing2)
            for thing1, thing2 in zip(dataclasses.astuple(self), dataclasses.astuple(other))
        )

    @property
    def path_string(self) -> str:
        raise NotImplementedError()

    @property
    def base_path(self) -> pathlib.Path:
        return pathlib.Path(self.path_string + "_" + self.hash_key)

    @property
    def hash_key(self) -> str:
        """Gets the hash key for a set of params."""
        return hashlib.md5(str(self).encode("utf-8")).hexdigest()[0:16]

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        # This custom repr lets us add fields with default values without changing
        # the repr. This, in turn, lets us use the hash_key reliably even when
        # we add new fields with a default value.
        fields = dataclasses.fields(self)
        # adjusted_fields = [f for f in fields if getattr(self, f.name) != f.default]
        adjusted_fields = [
            f for f in fields if not array_compatible_eq(getattr(self, f.name), f.default)
        ]

        return (
            self.__class__.__qualname__
            + "("
            + ", ".join([f"{f.name}={getattr(self, f.name)}" for f in adjusted_fields])
            + ")"
        )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"


@dataclass(frozen=True, eq=False)
class Data(abc.ABC):
    params: Params

    def __eq__(self, other):
        """A helper method to compare two dataclasses which might contain arrays."""
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            return NotImplemented

        return all(
            array_compatible_eq(thing1, thing2)
            for thing1, thing2 in zip(dataclasses.astuple(self), dataclasses.astuple(other))
        )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"


def array_compatible_eq(thing1, thing2):
    """A check for equality which can handle arrays."""
    if thing1 is thing2:
        return True

    # Here we handle dicts because they might have arrays in them.
    if isinstance(thing1, dict) and isinstance(thing2, dict):
        return all(
            array_compatible_eq(k_1, k_2) and array_compatible_eq(v_1, v_2)
            for (k_1, v_1), (k_2, v_2) in zip(thing1.items(), thing2.items())
        )
    if isinstance(thing1, np.ndarray) and isinstance(thing2, np.ndarray):
        return np.array_equal(thing1, thing2)
    if isinstance(thing1, np.ndarray) + isinstance(thing2, np.ndarray) == 1:
        return False
    try:
        return thing1 == thing2
    except TypeError:
        return NotImplemented


class ConstantTwoQubitGateDepolarizingNoiseModel(cirq.NoiseModel):
    """Applies noise to each two qubit gate individually at the end of every moment."""

    def __init__(self, depolarizing_probability):
        self.noise_gate = cirq.DepolarizingChannel(depolarizing_probability)

    @property
    def name(self):
        return "noise_gate :" + repr(self.noise_gate) + "after each two qubit gate"

    def noisy_operation(self, operation: "cirq.Operation"):
        not_measurement = not isinstance(operation.gate, cirq.ops.MeasurementGate)
        if len(operation.qubits) > 1 and not_measurement:
            return [operation] + [self.noise_gate(q) for q in operation.qubits]
        else:
            return operation


def get_noise_model(name: str, params: Optional[Sequence[float]]) -> Union[None, cirq.NoiseModel]:
    if name == "ConstantTwoQubitGateDepolarizingNoiseModel":
        assert params is not None
        assert len(params) == 1
        return ConstantTwoQubitGateDepolarizingNoiseModel(params[0])
    elif name == "None":
        return None
    else:
        raise NotImplementedError("Noise model not implemented")


def count_circuit_elements(circuit: cirq.Circuit) -> Dict[str, int]:
    """A helper method to count the number of gates and moments in a circuit."""
    num_one_qubit_gates = 0
    num_two_qubit_gates = 0
    for moment in circuit:
        for op in moment:
            if isinstance(op, cirq.ops.GateOperation):
                if len(op.qubits) == 2:
                    num_two_qubit_gates += 1
                if len(op.qubits) == 1:
                    num_one_qubit_gates += 1

    return {
        "Moments": len(circuit),
        "Single Qubit Gates": num_one_qubit_gates,
        "Two Qubit Gates": num_two_qubit_gates,
    }


def iterate_permutation_matrices(dim: int):
    """A helper function that iterates over dim x dim permutation matrices."""
    for perm in itertools.permutations(range(dim)):
        # We generate the indices of the non-zero entries.
        data = np.asarray([1.0] * dim)
        xs = []
        ys = []
        for i in range(dim):
            xs.append(i)
            ys.append(perm[i])

        xs = np.asarray(xs)
        ys = np.asarray(ys)

        mat = coo_matrix((data, (xs, ys)), shape=(dim, dim))

        yield mat.todense()


def iterate_virtual_permutation_matrices(n_orb, n_elec):
    # TODO: Test to see if this is useful once orbital optimization code is improved.
    n_virt = n_orb - n_elec // 2

    for mat in iterate_permutation_matrices(n_virt):
        base_mat = np.eye(n_orb)

        base_mat[n_elec // 2 : n_orb, n_elec // 2 : n_orb] = mat

        yield base_mat


def reorder_qubit_wavefunction(
    *, wf: np.ndarray, qubits_old_order: Sequence[cirq.Qid], qubits_new_order: Sequence[cirq.Qid]
) -> np.ndarray:
    """Reorders the amplitudes of wf based on permutation of qubits.

    May or may not destructively affect the original argument wf.
    """
    assert set(qubits_new_order) == set(qubits_old_order)
    n_qubits = len(qubits_new_order)
    assert wf.shape == (2**n_qubits,)

    wf = np.reshape(wf, (2,) * n_qubits)
    axes = tuple(qubits_old_order.index(qubit) for qubit in qubits_new_order)
    wf = np.transpose(wf, axes)
    wf = np.reshape(wf, (2**n_qubits,))

    return wf


def is_expected_elementary_cirq_op(op: cirq.Operation) -> bool:
    """Checks whether op is one of the operations that we expect when decomposing a quaff op."""
    to_keep = isinstance(
        op.gate,
        (
            cirq.HPowGate,
            cirq.CXPowGate,
            cirq.ZPowGate,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.CZPowGate,
        ),
    )
    return to_keep
