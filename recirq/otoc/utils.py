# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions used for the OTOC experiment."""

import pickle
from typing import Any, Optional, Tuple, Dict, List, Sequence

import cirq
import numpy as np
from scipy.optimize import curve_fit


def save_data(data: Any, file_name: str) -> None:
    """Function for saving data to a pickle file.

    Args:
        data: Data to save to a pickle file.
        file_name: Path to save the file.
    """
    with open(file_name, "wb") as output:
        pickle.dump(data, output)


def load_data(file_name: str) -> Optional[Any]:
    """Function for reading data from a pickle file.

    Args:
        file_name: Path for the file to be read.

    Returns:
        Data that the file contains.
    """
    with open(file_name, "rb") as input_data:
        data = pickle.load(input_data)
    return data


def pauli_error_fit(
    num_cycle_range: np.ndarray, data: np.ndarray, *, num_qubits: int = 2, add_offset: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Obtaining a Pauli error rate from an empirical decay curve.

    Args:
        num_cycle_range: The number of cycles in an error-benchmarking experiment (e.g. XEB).
        data: Fidelity at each given cycle. Must have the same dimension as num_cycle_range.
        num_qubits: Number of qubits involved in the experiment.
        add_offset: Whether the fitted exponential decay is forced to asymptote to 0 or a
            non-zero offset.

    Returns:
        pauli_error: The average Pauli error per cycle from the fit.
        x_vals: A list of 200 evenly spaced values between 0 and max(num_cycle_range).
        y_vals: The fitted fidelity for each value in x_vals.
    """
    f_s = 1.0e2

    def _exp_decay_with_offset(
        length: np.ndarray, err: float, a_coeff: float, b_coeff: float
    ) -> np.ndarray:
        p = 1.0 - err / (1.0 - 4 ** (-num_qubits)) / f_s
        return a_coeff * np.power(p, length) + b_coeff

    def _exp_decay_no_offset(length: np.ndarray, err: float, a_coeff: float) -> np.ndarray:
        return _exp_decay_with_offset(length, err, a_coeff, 0.0)

    fit_fun = _exp_decay_with_offset if add_offset else _exp_decay_no_offset
    params_fitted, _ = curve_fit(fit_fun, num_cycle_range, data, ftol=1e-2)
    pauli_error = float(params_fitted[0] / f_s)
    x_vals = np.linspace(0.0, max(num_cycle_range), 200)
    y_vals = fit_fun(x_vals, *params_fitted)
    return pauli_error, x_vals, y_vals


def bits_to_probabilities(
    all_qubits: Sequence[Tuple[int, int]],
    subsys_qubits: Sequence[Tuple[int, int]],
    bits: np.ndarray,
) -> np.ndarray:
    """Converts a collection of bit-strings into a probability distribution.

    Args:
        all_qubits: A list of Tuples representing the (row, column) locations of all qubits that
            are measured. The qubits are assumed to be on a 2D grid.
        subsys_qubits: The qubits whose joint probabilities are to be computed. They should be a
            subset of all_qubits.
        bits: A 2D np.array where each row represents a bit-string (with the same length as
            all_qubits) and each column represents a trial.

    Returns:
        The probabilities of all possible bit-strings for subsys_qubits, ordered according to the
        big-endian notation.
    """
    num_qubits = len(subsys_qubits)
    indices = [all_qubits.index(q) for q in subsys_qubits]
    bits_red = np.asarray(bits[:, indices], dtype=int)
    for i in range(num_qubits):
        bits_red[:, i] *= 2 ** (num_qubits - i - 1)
    bits = list(np.sum(bits_red, axis=1))
    p_reduced = np.bincount(bits, minlength=2 ** num_qubits)
    num_trials, _ = bits_red.shape
    return p_reduced / num_trials


def angles_to_fsim(
    theta: float,
    phi: float,
    delta_plus: float,
    delta_minus_diag: float,
    delta_minus_off_diag: float,
) -> np.ndarray:
    """Converts five phases to a 2x2 FSIM unitary.

    Args:
        theta: The swap angle.
        phi: The conditional phase.
        delta_plus: A single-qubit phase.
        delta_minus_diag: A single-qubit phase.
        delta_minus_off_diag: A single-qubit phase.

    Returns:
        A 2x2 np.array corresponding to the FSIM unitary with the five angles.
    """
    c, s = np.cos(theta), np.sin(theta)
    u11 = c * np.exp(1j * (delta_plus + delta_minus_diag) / 2.0)
    u12 = -1j * s * np.exp(1j * (delta_plus - delta_minus_off_diag) / 2.0)
    u21 = -1j * s * np.exp(1j * (delta_plus + delta_minus_off_diag) / 2.0)
    u22 = c * np.exp(1j * (delta_plus - delta_minus_diag) / 2.0)
    u33 = np.exp(1j * (delta_plus - phi))
    return np.array(
        [[1, 0, 0, 0], [0, u11, u12, 0], [0, u21, u22, 0], [0, 0, 0, u33]],
        dtype=complex,
    )


def fsim_to_angles(u_fsim: np.ndarray) -> Dict[str, float]:
    """Converts a 2x2 FSIM unitary to five phases.

    Args:
        u_fsim: A 2x2 np.array corresponding to an FSIM unitary

    Returns:
        A dictionary with the keys being the name of a unitary phase and value being the value of
        the phase.
    """
    u_fsim = u_fsim * np.exp(-1j * np.angle(u_fsim[0, 0]))
    theta = np.arctan(np.abs(u_fsim[1, 2] / u_fsim[1, 1]))
    delta_plus = np.angle(-u_fsim[1, 2] * u_fsim[2, 1])
    phi = -np.angle(u_fsim[3, 3]) + delta_plus
    delta_minus_off_diag = np.angle(u_fsim[2, 1] / (-1j) / np.exp(1j * delta_plus / 2.0)) * 2.0
    delta_minus_diag = np.angle(u_fsim[1, 1] / np.exp(1j * delta_plus / 2.0)) * 2.0

    angles = {
        "theta": theta,
        "delta_plus": delta_plus,
        "delta_minus_off_diag": delta_minus_off_diag,
        "delta_minus_diag": delta_minus_diag,
        "phi": phi,
    }  # type: Dict[str, float]

    return angles


def generic_fsim_gate(
    fsim_angles: Dict[str, float], qubits: Tuple[cirq.GridQubit, cirq.GridQubit]
) -> List[cirq.OP_TREE]:
    """Converts five FSIM phases to a list of Cirq ops.

    Args:
        fsim_angles:A dictionary with the keys being the name of a unitary phase and value being
            the value of the phase.
        qubits: The pair of cirq.GridQubits to which the gates are applied.

    Returns:
        A list of gates (equivalent to an FSIM gate) acting on the two qubits.
    """
    q_0, q_1 = qubits
    g_f = [
        cirq.Z(q_0)
        ** (
            -(
                fsim_angles["delta_minus_off_diag"]
                + fsim_angles["delta_minus_diag"]
                - 2 * fsim_angles["delta_plus"]
            )
            / np.pi
            / 4.0
        ),
        cirq.Z(q_1)
        ** (
            (
                fsim_angles["delta_minus_off_diag"]
                + fsim_angles["delta_minus_diag"]
                + 2 * fsim_angles["delta_plus"]
            )
            / np.pi
            / 4.0
        ),
    ]  # type: List[cirq.OP_TREE]

    if not np.isclose(fsim_angles["phi"], 0):
        g_f.append(cirq.CZ(q_0, q_1) ** (-fsim_angles["phi"] / np.pi))

    if not np.isclose(fsim_angles["theta"], 0):
        g_f.append(cirq.ISWAP(q_0, q_1) ** (-fsim_angles["theta"] / (np.pi / 2.0)))

    g_f.append(
        cirq.Z(q_0)
        ** (-(fsim_angles["delta_minus_diag"] - fsim_angles["delta_minus_off_diag"]) / np.pi / 4.0)
    )
    g_f.append(
        cirq.Z(q_1)
        ** ((fsim_angles["delta_minus_diag"] - fsim_angles["delta_minus_off_diag"]) / np.pi / 4.0)
    )
    return g_f


def cz_to_sqrt_iswap(
    qubit_0: cirq.GridQubit,
    qubit_1: cirq.GridQubit,
) -> cirq.Circuit:
    """Generates a composite CZ gate with sqrt-iSWAP and single-qubit gates.

    Args:
        qubit_0: The first qubit the CZ acts upon.
        qubit_1: The second qubit the CZ acts upon.

    Returns:
        A circuit equivalent to a CZ gate between the two qubits.
    """

    op_list = [
        cirq.Z(qubit_0) ** 0.5,
        cirq.Z(qubit_1) ** 0.5,
        cirq.X(qubit_0) ** 0.5,
        cirq.X(qubit_1) ** -0.5,
        cirq.ISWAP(qubit_0, qubit_1) ** -0.5,
        cirq.X(qubit_0) ** -1,
        cirq.ISWAP(qubit_0, qubit_1) ** 0.5,
        cirq.X(qubit_0) ** 0.5,
        cirq.X(qubit_1) ** 0.5,
    ]

    return cirq.Circuit(op_list)
