from typing import Callable, Mapping, Sequence

import cirq
import fqe
import fqe.wavefunction as fqe_wfn
import numpy as np
import openfermion as of

from recirq.qcqmc import fermion_mode


def get_reorder_func(
    *,
    mode_qubit_map: Mapping[fermion_mode.FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> Callable[[int, int], int]:
    """This is a helper function that allows us to reorder fermionic modes.

    Under the Jordan-Wigner transform, each fermionic mode is assigned to a
    qubit. If we are provided an openfermion FermionOperator with the modes
    assigned to qubits as described by mode_qubit_map this function gives us a
    reorder_func that we can use to reorder the modes (with
    openfermion.reorder(...)) so that they match the order of the qubits in
    ordered_qubits. This is necessary to make a correspondence between
    fermionic operators / wavefunctions and their qubit counterparts.

    Args:
        mode_qubit_map: A dict that shows how each FermionicMode is mapped to a qubit.
        ordered_qubits: An ordered sequence of qubits.
    """
    qubits = list(mode_qubit_map.values())
    assert len(qubits) == len(ordered_qubits)

    # We sort the key: value pairs by the order of the values (qubits) in
    # ordered_qubits.
    sorted_mapping = list(mode_qubit_map.items())
    sorted_mapping.sort(key=lambda x: ordered_qubits.index(x[1]))

    remapping_map = {}
    for i, (mode, _) in enumerate(sorted_mapping):
        openfermion_index = 2 * mode.orb_ind + (0 if mode.spin == "a" else 1)
        remapping_map[openfermion_index] = i

    def remapper(index: int, _: int) -> int:
        """A function that maps from the old index to the new one.

        The _ argument is because it's expected by openfermion.reorder"""
        return remapping_map[index]

    return remapper


def get_ansatz_qubit_wf(
    *, ansatz_circuit: cirq.Circuit, ordered_qubits: Sequence[cirq.Qid]
):
    """Get the cirq statevector from the ansatz circuit."""
    return cirq.final_state_vector(
        ansatz_circuit, qubit_order=list(ordered_qubits), dtype=np.complex128
    )


def get_two_body_params_from_qchem_amplitudes(
    qchem_amplitudes: np.ndarray,
) -> np.ndarray:
    """Translates perfect pairing amplitudes from qchem to rotation angles.

    qchem style: 1 |1100> + t_i |0011>
    our style: cos(\theta_i) |1100> + sin(\theta_i) |0011>
    """

    two_body_params = np.arccos(1 / np.sqrt(1 + qchem_amplitudes**2)) * np.sign(
        qchem_amplitudes
    )

    # Numpy casts the array improperly to a float when we only have one parameter.
    two_body_params = np.atleast_1d(two_body_params)

    return two_body_params


def convert_fqe_wf_to_cirq(
    fqe_wf: fqe_wfn.Wavefunction,
    mode_qubit_map: Mapping[fermion_mode.FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> np.ndarray:
    """Converts an FQE wavefunction to one on qubits with a particular ordering.

    Args:
        fqe_wf: The FQE wavefunction.
        mode_qubit_map: A mapping from fermion modes to cirq qubits.
        ordered_qubits:
    """
    n_qubits = len(mode_qubit_map)
    fermion_op = fqe.openfermion_utils.fqe_to_fermion_operator(fqe_wf)

    reorder_func = get_reorder_func(
        mode_qubit_map=mode_qubit_map, ordered_qubits=ordered_qubits
    )
    fermion_op = of.reorder(fermion_op, reorder_func, num_modes=n_qubits)

    qubit_op = of.jordan_wigner(fermion_op)

    return fqe.qubit_wavefunction_from_vacuum(
        qubit_op, list(cirq.LineQubit.range(n_qubits))
    )
