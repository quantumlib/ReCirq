import os
import pickle
from collections.abc import Sequence
from typing import List

import cirq
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def trotter_circuit(
    grid: Sequence[cirq.GridQubit],
    n_cycles: int,
    dt: float,
    h: float,
    mu: float,
    two_qubit_gate="cz",
) -> cirq.Circuit:
    if two_qubit_gate == "cz":
        return _layer_floquet_cz(grid, dt, h, mu) * n_cycles

    elif two_qubit_gate == "cphase":
        if n_cycles == 0:
            return cirq.Circuit()
        if n_cycles == 1:
            return _layer_floquet_cphase_first(
                grid, dt, h, mu
            ) + _layer_floquet_cphase_last_missing_piece(grid, dt, h, mu)
        else:
            return (
                _layer_floquet_cphase_first(grid, dt, h, mu)
                + (n_cycles - 1) * _layer_floquet_cphase_middle(grid, dt, h, mu)
                + _layer_floquet_cphase_last_missing_piece(grid, dt, h, mu)
            )
    else:
        raise ValueError("Two-qubit gate can only be cz or cphase")


def _layer_floquet_cz(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    n = len(grid)
    moment_rz = []
    moment_rx = []
    moment_h = []
    for i in range(n):
        q = grid[i]
        if i % 2 == 1:
            moment_rz.append(cirq.rz(dt).on(q))
            moment_h.append(cirq.H(q))
            moment_rx.append(cirq.rx(2 * h * dt).on(q))
        else:
            moment_rx.append(cirq.rx(2 * mu * dt).on(q))
    return (
        cirq.Circuit.from_moments(cirq.Moment(moment_rz))
        + _change_basis(grid)
        + cirq.Circuit.from_moments(cirq.Moment(moment_rx))
        + _change_basis(grid)
        + cirq.Circuit.from_moments(cirq.Moment(moment_rz))
    )


def _layer_floquet_cphase_middle(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    n = len(grid)
    moment_0 = []
    moment_1 = []
    moment_h = []
    moment_rz = []
    moment_rx = []

    for i in range(n // 2):
        q0 = grid[(2 * i)]
        q = grid[2 * i + 1]
        q1 = grid[(2 * i + 2) % n]
        moment_0.append(cirq.CZ(q0, q))  # left to right
        moment_1.append(cirq.cphase(-4 * dt).on(q1, q))  # right to left

        moment_h.append(cirq.H(q))

        moment_rz.append(cirq.rz(2 * dt).on(q))
        moment_rz.append(cirq.rz(2 * dt).on(q1))

    for i in range(n):
        q = grid[i]
        if i % 2 == 1:
            moment_rx.append(cirq.rx(2 * h * dt).on(q))
        else:
            moment_rx.append(cirq.rx(2 * mu * dt).on(q))

    moment_rz_new = []
    qubits_covered = []
    for gate in moment_rz:
        count = moment_rz.count(gate)
        qubit = gate.qubits[0]
        if qubit not in qubits_covered:
            moment_rz_new.append(cirq.rz(2 * count * dt).on(qubit))
            qubits_covered.append(qubit)

    return cirq.Circuit.from_moments(
        cirq.Moment(moment_h),
        cirq.Moment(moment_0),
        cirq.Moment(moment_h),
        cirq.Moment(moment_rz_new),
        cirq.Moment(moment_1),
        cirq.Moment(moment_h),
        cirq.Moment(moment_0),
        cirq.Moment(moment_h),
        cirq.Moment(moment_rx),
    )


def _layer_floquet_cphase_first(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    n = len(grid)
    moment_rz = []
    moment_rx = []

    for i in range(n):
        q = grid[i]
        if i % 2 == 1:
            moment_rz.append(cirq.rz(dt).on(q))
            moment_rx.append(cirq.rx(2 * h * dt).on(q))
        else:
            moment_rx.append(cirq.rx(2 * mu * dt).on(q))

    return (
        cirq.Circuit.from_moments(cirq.Moment(moment_rz))
        + _change_basis(grid)
        + cirq.Circuit.from_moments(cirq.Moment(moment_rx))
    )


def _layer_floquet_cphase_last_missing_piece(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    n = len(grid)
    moment_rz = []
    for i in range(n):
        q = grid[i]
        if i % 2 == 1:
            moment_rz.append(cirq.rz(dt).on(q))

    return _change_basis(grid) + cirq.Circuit.from_moments(cirq.Moment(moment_rz))


def _layer_hadamard(grid: Sequence[cirq.GridQubit], which_qubits="all") -> cirq.Circuit:
    moment = []
    n = len(grid)
    for i in range(n):
        q = grid[i]

        if i % 2 == 1 and (which_qubits == "gauge" or which_qubits == "all"):
            moment.append(cirq.H(q))

        elif i % 2 == 0 and (which_qubits == "matter" or which_qubits == "all"):
            moment.append(cirq.H(q))
    return cirq.Circuit.from_moments(cirq.Moment(moment))


def _layer_measure(grid: Sequence[cirq.GridQubit]) -> cirq.Circuit:
    n = len(grid)
    moment = []
    for i in range(n // 2):
        moment.append(cirq.measure(grid[2 * i], key="m"))
    return cirq.Circuit.from_moments(cirq.Moment(moment))


def _change_basis(grid: Sequence[cirq.GridQubit]) -> cirq.Circuit:
    """change the basis so that the matter sites encode the gauge charge."""
    n = len(grid)
    moment_1 = []
    moment_2 = []
    moment_h = []
    for i in range(0, n // 2):
        q1 = grid[(2 * i)]
        q2 = grid[2 * i + 1]
        q3 = grid[(2 * i + 2) % n]
        moment_1.append(cirq.CZ(q1, q2))
        moment_2.append(cirq.CZ(q3, q2))
        moment_h.append(cirq.H(q2))
    return cirq.Circuit.from_moments(
        cirq.Moment(moment_h), cirq.Moment(moment_1), cirq.Moment(moment_2), cirq.Moment(moment_h)
    )


def _energy_bump_initial_state(
    grid, h: float, matter_config: str, excited_qubits: Sequence[cirq.GridQubit]
) -> cirq.Circuit:
    """Circuit for energy bump initial state.
    It typically consists of single qubit gates and the basis change circuit U_B.
    But in this second order implementation, I am removing U_B since
    it cancels out with the U_B in the second order trotter circuit."""
    n = len(grid)
    theta = np.arctan(h)
    moment = []
    for i in range(n):
        q = grid[i]
        if i % 2 == 1:
            if q in excited_qubits:
                moment.append(cirq.ry(np.pi + theta).on(q))
            else:
                moment.append(cirq.ry(theta).on(q))

        else:
            if matter_config == "single_sector":
                moment.append(cirq.H(q))
    return cirq.Circuit.from_moments(moment)


def get_1d_dfl_experiment_circuits(
    grid: Sequence[cirq.GridQubit],
    initial_state: str,
    n_cycles: Sequence[int] | npt.NDArray,
    excited_qubits: Sequence[cirq.GridQubit],
    dt: float,
    h: float,
    mu: float,
    n_instances: int = 10,
    two_qubit_gate: str = "cz",
    basis="dual",
):
    if initial_state == "single_sector":
        initial_circuit = _energy_bump_initial_state(grid, h, "single_sector", excited_qubits)
    elif initial_state == "superposition":
        initial_circuit = _energy_bump_initial_state(grid, h, "superposition", excited_qubits)
    else:
        raise ValueError("Invalid initial state")
    circuits = []
    for n_cycle in tqdm(n_cycles):
        print(int(np.max([0, n_cycle - 1])))
        circ = initial_circuit + trotter_circuit(grid, n_cycle, dt, h, mu, two_qubit_gate)
        if basis == "lgt":
            circ += _change_basis(grid)
        elif basis == "dual":
            pass
        else:
            raise ValueError("Invalid option for basis")
        for _ in range(n_instances):

            if basis == "lgt":
                circ_z = circ + cirq.measure([q for q in grid], key="m")
            elif basis == "dual":
                circ_z = (
                    circ
                    + _layer_hadamard(grid, "matter")
                    + cirq.measure([q for q in grid], key="m")
                )
            else:
                raise ValueError("Invalid option for basis")
            circ_x = circ + _layer_hadamard(grid, "all") + cirq.measure([q for q in grid], key="m")
            circuits.append(circ_z)
            circuits.append(circ_x)
    return circuits
