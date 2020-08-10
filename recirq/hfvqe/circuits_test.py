import cirq
import numpy as np
import scipy as sp
import pytest

from openfermioncirq.experiments.hfvqe.circuits import (
                            rhf_params_to_matrix,
                            ryxxy,
                            ryxxy2,
                            ryxxy3,
                            ryxxy4,
                            xxyy_basis_rotation,
                            prepare_slater_determinant)
# pylint: disable=C


def test_rhf_params_to_matrix():
    test_kappa = rhf_params_to_matrix(np.array([0.1, 0.2, 0.3, 0.4]), 4)
    kappa = np.zeros((4, 4))
    kappa[0, 2] = -0.1
    kappa[0, 3] = -0.3
    kappa[1, 2] = -0.2
    kappa[1, 3] = -0.4
    kappa -= kappa.T
    assert np.allclose(test_kappa, kappa)

    with pytest.raises(ValueError):
        rhf_params_to_matrix(np.array([1.0 + 1j]), 2)


def test_ryxxy():
    a, b = cirq.LineQubit.range(2)
    test_circuit = list(ryxxy(a, b, np.pi / 3))
    true_circuit = [cirq.ISWAP.on(a, b)**0.5,
                    cirq.rz(-np.pi / 3 + np.pi).on(a),
                    cirq.rz(np.pi / 3).on(b),
                    cirq.ISWAP.on(a, b) ** 0.5,
                    cirq.rz(np.pi).on(a)
                    ]
    assert test_circuit == true_circuit


def test_ryxxy2():
    a, b = cirq.LineQubit.range(2)
    test_circuit = list(ryxxy2(a, b, np.pi / 3))
    true_circuit = [cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b),
                    cirq.rz(-np.pi / 3 + np.pi).on(a),
                    cirq.rz(np.pi / 3).on(b),
                    cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b),
                    cirq.rz(np.pi).on(a)
                    ]
    assert test_circuit == true_circuit


def test_ryxxy3():
    a, b = cirq.LineQubit.range(2)
    test_circuit = list(ryxxy3(a, b, np.pi / 3))
    true_circuit = [cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b),
                    cirq.rz(-np.pi / 3 + np.pi + np.pi / 48).on(a),
                    cirq.rz(np.pi / 3 + np.pi / 48).on(b),
                    cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b),
                    cirq.rz(np.pi + np.pi/48).on(a),
                    cirq.rz(np.pi / 48).on(b)
                    ]
    assert test_circuit == true_circuit


def test_ryxxy4():
    a, b = cirq.LineQubit.range(2)
    test_circuit = list(ryxxy4(a, b, np.pi / 3))
    true_circuit = [cirq.FSimGate(-np.pi / 4, 0).on(a, b),
                    cirq.rz(-np.pi / 3 + np.pi + np.pi / 48).on(a),
                    cirq.rz(np.pi / 3 + np.pi / 48).on(b),
                    cirq.FSimGate(-np.pi / 4, 0).on(a, b),
                    cirq.rz(np.pi + np.pi/48).on(a),
                    cirq.rz(np.pi / 48).on(b)
                    ]
    assert test_circuit == true_circuit


def test_xxyy_basis_rotation():
    qubits = cirq.LineQubit.range(4)
    pairs = [(qubits[0], qubits[1]), (qubits[2], qubits[3])]
    test_rotation = xxyy_basis_rotation(pairs, clean_xxyy=True)
    true_rotation = [cirq.rz(-np.pi * 0.25).on(qubits[0]),
                     cirq.rz(np.pi * 0.25).on(qubits[1]),
                     cirq.ISWAP.on(qubits[0], qubits[1]) ** 0.5,
                     cirq.rz(-np.pi * 0.25).on(qubits[2]),
                     cirq.rz(np.pi * 0.25).on(qubits[3]),
                     cirq.ISWAP.on(qubits[2], qubits[3]) ** 0.5]
    assert test_rotation == true_rotation

    test_rotation = xxyy_basis_rotation(pairs, clean_xxyy=False)
    true_rotation = [cirq.rz(-np.pi * 0.25).on(qubits[0]),
                     cirq.rz(np.pi * 0.25).on(qubits[1]),
                     cirq.FSimGate(-np.pi / 4, np.pi / 24).on(qubits[0], qubits[1]),
                     cirq.rz(-np.pi * 0.25).on(qubits[2]),
                     cirq.rz(np.pi * 0.25).on(qubits[3]),
                     cirq.FSimGate(-np.pi / 4, np.pi / 24).on(qubits[2], qubits[3])]
    assert test_rotation == true_rotation


def test_prepare_slater():
    qubits = cirq.LineQubit.range(4)
    kappa = rhf_params_to_matrix(np.array([0.1, 0.2, 0.3, 0.4]), 4)
    u = sp.linalg.expm(kappa)

    with pytest.raises(ValueError):
        list(prepare_slater_determinant(qubits, u[:, :2].T, clean_ryxxy=5))

    test_circuit = cirq.Circuit(prepare_slater_determinant(qubits, u[:, :2].T))
    true_moments = [cirq.Moment(operations=[
        cirq.X.on(cirq.LineQubit(0)),
        cirq.X.on(cirq.LineQubit(1)),
    ]), cirq.Moment(operations=[
        (cirq.ISWAP ** 0.5).on(cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.1676697243144354).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * -0.1676697243144355).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        (cirq.ISWAP ** 0.5).on(cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0).on(cirq.LineQubit(1)),
        (cirq.ISWAP ** 0.5).on(cirq.LineQubit(2), cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        (cirq.ISWAP ** 0.5).on(cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.rz(np.pi * 1.3947664179536838).on(cirq.LineQubit(2)),
        cirq.rz(np.pi * -0.3947664179536837).on(cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 0.795779308536894).on(cirq.LineQubit(0)),
        cirq.rz(np.pi * 0.20422069146310598).on(cirq.LineQubit(1)),
        (cirq.ISWAP ** 0.5).on(cirq.LineQubit(2), cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        (cirq.ISWAP ** 0.5).on(cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.rz(np.pi * 1.0).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0).on(cirq.LineQubit(0)),
        (cirq.ISWAP ** 0.5).on(cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0212853739870422).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * -0.02128537398704223).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        (cirq.ISWAP ** 0.5).on(cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0).on(cirq.LineQubit(1)),
    ])]
    assert cirq.approx_eq(true_moments, test_circuit._moments, atol=1e-8)

    test_circuit = cirq.Circuit(prepare_slater_determinant(qubits, u[:, :2].T,
                                                           clean_ryxxy=2))
    true_circuit = [cirq.Moment(operations=[
        cirq.X.on(cirq.LineQubit(0)),
        cirq.X.on(cirq.LineQubit(1)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.1676697243144354).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * -0.1676697243144355).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0).on(cirq.LineQubit(1)),
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(2), cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.rz(np.pi * 1.3947664179536838).on(cirq.LineQubit(2)),
        cirq.rz(np.pi * -0.3947664179536837).on(cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 0.795779308536894).on(cirq.LineQubit(0)),
        cirq.rz(np.pi * 0.20422069146310598).on(cirq.LineQubit(1)),
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(2), cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.rz(np.pi * 1.0).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0).on(cirq.LineQubit(0)),
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0212853739870422).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * -0.02128537398704223).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0).on(cirq.LineQubit(1)),
    ])]
    assert cirq.approx_eq(true_circuit, test_circuit._moments, atol=1e-8)

    test_circuit = cirq.Circuit(prepare_slater_determinant(qubits, u[:, :2].T,
                                                           clean_ryxxy=3))
    true_circuit = [cirq.Moment(operations=[
        cirq.X.on(cirq.LineQubit(0)),
        cirq.X.on(cirq.LineQubit(1)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.1885030576477686).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * -0.14683639098110218).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0208333333333333).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * 0.020833333333333332).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(2), cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 0.8166126418702274).on(cirq.LineQubit(0)),
        cirq.rz(np.pi * 0.22505402479643932).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * 1.4155997512870173).on(cirq.LineQubit(2)),
        cirq.rz(np.pi * -0.3739330846203504).on(cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(2), cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0208333333333333).on(cirq.LineQubit(0)),
        cirq.rz(np.pi * 0.020833333333333332).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * 1.0208333333333333).on(cirq.LineQubit(2)),
        cirq.rz(np.pi * 0.020833333333333332).on(cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0421187073203755).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * -0.000452040653708897).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0208333333333333).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * 0.020833333333333332).on(cirq.LineQubit(2)),
    ])]
    assert cirq.approx_eq(true_circuit, test_circuit._moments, atol=1e-8)


    test_circuit = cirq.Circuit(prepare_slater_determinant(qubits, u[:, :2].T,
                                                           clean_ryxxy=4))
    true_circuit = [cirq.Moment(operations=[
        cirq.X.on(cirq.LineQubit(0)),
        cirq.X.on(cirq.LineQubit(1)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.1885030576477686).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * -0.14683639098110218).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0208333333333333).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * 0.020833333333333332).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(2), cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 0.8166126418702274).on(cirq.LineQubit(0)),
        cirq.rz(np.pi * 0.22505402479643932).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * 1.4155997512870173).on(cirq.LineQubit(2)),
        cirq.rz(np.pi * -0.3739330846203504).on(cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(2), cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0208333333333333).on(cirq.LineQubit(0)),
        cirq.rz(np.pi * 0.020833333333333332).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * 1.0208333333333333).on(cirq.LineQubit(2)),
        cirq.rz(np.pi * 0.020833333333333332).on(cirq.LineQubit(3)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0421187073203755).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * -0.000452040653708897).on(cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.FSimGate(theta=-0.7853981633974483, phi=0.1308996938995747).on(
            cirq.LineQubit(1), cirq.LineQubit(2)),
    ]), cirq.Moment(operations=[
        cirq.rz(np.pi * 1.0208333333333333).on(cirq.LineQubit(1)),
        cirq.rz(np.pi * 0.020833333333333332).on(cirq.LineQubit(2)),
    ])]
    assert cirq.approx_eq(true_circuit, test_circuit._moments, atol=1e-8)

