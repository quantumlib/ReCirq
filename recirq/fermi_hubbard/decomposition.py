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
"""Circuit and gate decompositions."""

from typing import (
    Callable, Iterable, Mapping, Optional, Tuple, Union
)

import cirq
from dataclasses import dataclass
from functools import lru_cache
from itertools import zip_longest
import numpy as np

Decomposition = Iterable[Iterable[cirq.Operation]]
DecomposeCallable = Callable[[cirq.Operation], Optional[Decomposition]]


@dataclass(frozen=True)
class ParticleConservingParameters:
    """Parameters of the particle-conserving two-qubit gate.

    [[1,                        0,                        0,               0],
     [0,     cos(θ) e^{-i(γ + δ)},  -i sin(θ) e^{-i(γ - χ)},               0],
     [0,  -i sin(θ) e^{-i(γ + χ)},     cos(θ) e^{-i(γ - δ)},               0],
     [0,                        0,                       0,  e^{-i (2 γ + φ}]]

    Args:
        theta: the iSWAP angle θ
        delta: the phase difference δ
        chi: the gauge associated to the phase difference χ
        gamma: the common phase factor γ
        phi: the CPHASE angle φ
    """
    theta: Optional[float] = None
    delta: Optional[float] = None
    chi: Optional[float] = None
    gamma: Optional[float] = None
    phi: Optional[float] = None

    @lru_cache(maxsize=None)
    def for_qubits_swapped(self) -> 'ParticleConservingParameters':
        """Instance of particle-conserving parameters when qubits are swapped.
        """
        return ParticleConservingParameters(
            theta=self.theta,
            delta=-self.delta if self.delta is not None else None,
            chi=-self.chi if self.chi is not None else None,
            gamma=self.gamma,
            phi=self.phi
        )

    def get_theta(self, qubits: Tuple[cirq.Qid, cirq.Qid]) -> float:
        if self.theta is None:
            raise ValueError(f'Missing θ parameter for qubits pair {qubits}')
        return self.theta

    def get_delta(self, qubits: Tuple[cirq.Qid, cirq.Qid]) -> float:
        if self.delta is None:
            raise ValueError(f'Missing δ parameter for qubits pair {qubits}')
        return self.delta

    def get_chi(self, qubits: Tuple[cirq.Qid, cirq.Qid]) -> float:
        if self.chi is None:
            raise ValueError(f'Missing χ parameter for qubits pair {qubits}')
        return self.chi

    def get_gamma(self, qubits: Tuple[cirq.Qid, cirq.Qid]) -> float:
        if self.gamma is None:
            raise ValueError(f'Missing γ parameter for qubits pair {qubits}')
        return self.gamma

    def get_phi(self, qubits: Tuple[cirq.Qid, cirq.Qid]) -> float:
        if self.phi is None:
            raise ValueError(f'Missing φ parameter for qubits pair {qubits}')
        return self.phi


class ConvertToNonUniformSqrtIswapGates:
    """Converter that decomposes circuits to √iSWAP gate set.

    The converter supports cases where √iSWAP deviate slightly from gate to gate
    on a chip.

    The converter does not support all the Cirq gates but a subset of them
    necessary for Fermi-Hubbard problem. See decompose_gk_gate,
    decompose_cphase_gate and decompose_cphase_echo_gate for supported gates.
    """

    def __init__(self,
                 parameters: Mapping[Tuple[cirq.Qid, cirq.Qid],
                                     ParticleConservingParameters],
                 *,
                 parameters_when_missing: Optional[
                     ParticleConservingParameters] = None,
                 sin_alpha_tolerance: Optional[float] = 0.15,
                 eject_z_gates: bool = True) -> None:
        """Initializes converter.

        Args:
            parameters: Qubir pair dependent √iSWAP gate parameters.
            parameters_when_missing: Fallback parameters to use when either the
                needed qubit pair is not present in parameters set, or when the
                parameters are present but a necessary angle value is missing.
                This object might specify only a subset of angles. If a
                necessary angle value is not resolved, then exception is raised
                during decomposition.
            sin_alpha_tolerance: Tolerance for sin(alpha) value as passed to
                corrected_cphase_ops function.
            eject_z_gates: Whether to apply the cirq.eject_z transformer
                on the decomposed circuit. The effect of this optimization is a
                circuit with virtual Z gates removed; only Z gates which are
                actually realized on the Google hardware are kept.
        """

        if parameters_when_missing:
            self._missing = parameters_when_missing
        else:
            self._missing = ParticleConservingParameters()

        self._parameters = {
            pair: _merge_parameters(value, self._missing)
            for pair, value in parameters.items()
        }

        if sin_alpha_tolerance:
            self._sin_alpha_tolerance = sin_alpha_tolerance
        else:
            self._sin_alpha_tolerance = 0.0

        self._eject_z_gates = eject_z_gates

    def convert(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Decomposes circuit to √iSWAP gate set.

        Gates with no known decompositions are left untouched.
        """
        decomposed = decompose_preserving_moments(
            circuit,
            decompose_func=(self._decompose_gk_gate,
                            self._decompose_cphase_gate,
                            self._decompose_cphase_echo_gate))

        if self._eject_z_gates:
            decomposed = cirq.eject_z(decomposed)
            decomposed = cirq.drop_empty_moments(decomposed)

        return decomposed

    def _decompose_gk_gate(self, operation: cirq.Operation
                           ) -> Optional[Decomposition]:
        """Decomposes √iSWAP with arbitrary exponent to two √iSWAP gates.

        The cirq.PhasedISwapPowGate, cirq.ISwapPowGate and cirq.FSimGate with
        zero phi angle are supported by this decomposition.

        The decomposition compensates for delta, chi and gamma deviations from
        the perfect unitary and keeps theta and phi errors for each √iSWAP gate.

        Args:
            operation: Operation to decompose.

        Returns:
            The decomposition of the supported gate to moments with single-qubit
            Z rotations and two √iSWAP gates. None if operation is not supported
            by this gate.
        """

        if not isinstance(operation, cirq.GateOperation):
            return None

        if len(operation.qubits) != 2:
            return None

        a, b = operation.qubits
        qubits = a, b
        gate = operation.gate

        if isinstance(gate, cirq.PhasedISwapPowGate):
            angle = gate.exponent * np.pi / 2
            phase_exponent = -gate.phase_exponent - 0.5
        elif isinstance(gate, cirq.FSimGate) and np.isclose(gate.phi, 0.0):
            angle = gate.theta
            phase_exponent = 0.0
        elif isinstance(gate, cirq.ISwapPowGate):
            angle = -gate.exponent * np.pi / 2
            phase_exponent = 0.0
        else:
            return None

        return _corrected_gk_ops(qubits,
                                 angle,
                                 phase_exponent,
                                 self._get_parameters(qubits))

    def _decompose_cphase_gate(self, operation: cirq.Operation
                               ) -> Optional[Decomposition]:
        """Decomposes CZ with arbitary exponent to two √iSWAP gates.

        The cirq.CZPowGate and cirq.FSimGate with zero theta angle are supported
        by this decomposition.

        The decomposition compensates for all unitary parameter deviations from
        the perfect √iSWAP unitary.

        Args:
            operation: Operation to decompose.

        Returns:
            The decomposition of the supported gate to moments with single-qubit
            rotations and two √iSWAP gates. None if operation is not supported
            by this gate.
        """

        if not isinstance(operation, cirq.GateOperation):
            return None

        if len(operation.qubits) != 2:
            return None

        a, b = operation.qubits
        qubits = a, b
        gate = operation.gate

        if isinstance(gate, cirq.CZPowGate):
            angle = -gate.exponent * np.pi
        elif isinstance(gate, cirq.FSimGate) and np.isclose(gate.theta, 0.0):
            angle = gate.phi
        else:
            return None

        return _corrected_cphase_ops(qubits,
                                     angle,
                                     self._get_parameters(qubits),
                                     self._sin_alpha_tolerance)

    def _decompose_cphase_echo_gate(self, operation: cirq.Operation
                                    ) -> Optional[Decomposition]:
        """Decomposes CPhaseEchoGate into two pi microwave pulses.

        This effect of this gate is a global phase shift. The decomposition puts
        the microwave pulses at the matching moments in of the
        decompose_cphase_gate results.

        Args:
            operation: Operation to decompose.

        Returns:
            The decomposition of the CPhaseEchoGate gate to two X pulses which
            cancel each other.
        """
        if not isinstance(operation, cirq.GateOperation):
            return None

        if isinstance(operation.gate, CPhaseEchoGate):
            qubit, = operation.qubits
            return _corrected_cphase_echo_ops(qubit)
        else:
            return None

    def _get_parameters(self, qubits: Tuple[cirq.Qid, cirq.Qid]
                        ) -> ParticleConservingParameters:
        a, b = qubits
        if (a, b) in self._parameters:
            return self._parameters[(a, b)]
        elif (b, a) in self._parameters:
            return self._parameters[(b, a)].for_qubits_swapped()
        else:
            return self._missing


@cirq.value_equality()
class CPhaseEchoGate(cirq.Gate):
    """Dummy gate that could be substituted by spin-echo pulses.

    This gate decomposes to nothing by default.
    """

    def _num_qubits_(self) -> int:
        return 1

    def _value_equality_values_(self):
        return ()

    def _decompose_(self, _) -> cirq.OP_TREE:
        return ()

    def _circuit_diagram_info_(self, _: cirq.CircuitDiagramInfoArgs) -> str:
        return 'CPhaseEcho'


class GateDecompositionError(Exception):
    """Raised when gate decomposition is infeasible."""
    pass


def _corrected_gk_ops(
        qubits: Tuple[cirq.Qid, cirq.Qid],
        angle: float,
        phase_exponent: float,
        parameters: ParticleConservingParameters
) -> Tuple[Tuple[cirq.Operation, ...], ...]:
    """Decomposition of cirq.FSim(angle, 0) into two non-ideal √iSWAP gates.

    The target unitary is:

     [[1,      0,       0,   0],
      [0,      c, -i·s·f*,   0],
      [0, -i·s·f,       c,   0],
      [0,      0,       0,   1]]

    where c = cos(angle), s = sin(angle) and f = exp(2πi·phase + πi/2).

    This target unitary might not be always possible to realize. The
    decomposition compensates for delta, chi and gamma deviations from the
    perfect unitary but keeps theta and phi errors for each √iSWAP gate.

    Args:
        qubits: Two qubits to act on. Note the that ordering of qubits is
            relevant because true √iSWAP is not symmetric (see
            ParticleConservingParameters.for_qubits_swapped)
        angle: The desired rotation angle.
        phase_exponent: Single-qubit phase freedom. The effect of this rotation
            can be cancelled out by applying Z rotations before or after this
            gate.
        parameters: True unitary parameters of the √iSWAP gate into which this
            function decomposes to.

    Returns:
        Tuple of tuples of operations. The outer tuple should be mapped directly
        to the circuit moments, and each inner tuple consists of operations that
        should be executed in parallel.
    """

    phase_exponent += 0.25
    sqrt_iswap = _corrected_sqrt_iswap_ops(qubits, parameters)
    a, b = qubits

    return (
        (cirq.Z(a) ** (1.0 - phase_exponent), cirq.Z(b) ** phase_exponent),
    ) + sqrt_iswap + (
        (cirq.rz(angle + np.pi).on(a), cirq.rz(-angle).on(b)),
    ) + sqrt_iswap + (
        (cirq.Z(a) ** phase_exponent, cirq.Z(b) ** -phase_exponent),
    )


def _corrected_cphase_ops(
        qubits: Tuple[cirq.Qid, cirq.Qid],
        angle: float,
        parameters: ParticleConservingParameters,
        sin_alpha_tolerance: float = 0.0
) -> Tuple[Tuple[cirq.Operation, ...], ...]:
    """Decomposition of cirq.FSim(0, angle) into two non-ideal √iSWAP gates.

    The target unitary is:

     [[1   0   0               0],
      [0   1   0               0],
      [0   0   1               0],
      [0   0   0   exp(-i·angle)]]

    The decomposition compensates for all unitary parameter deviations from
    the perfect √iSWAP unitary and utilizes three moments of microwave pulses to
    achieve this. See corrected_cphase_echo_ops for compatible spin echo moment
    structure.

    Args:
        qubits: Two qubits to act on. Note the that ordering of qubits is
            relevant because true √iSWAP is not symmetric (see
            ParticleConservingParameters.for_qubits_swapped)
        angle: The desired rotation angle.
        parameters: True unitary parameters of the √iSWAP gate into which this
            function decomposes to.
        sin_alpha_tolerance: Threshold that controls the magnitude of
            approximation of the decomposition. When set to 0, the gate is
            always exact. Positive value controls the amount of noise this
            decomposition tolerates.
    Returns:
        Tuple of tuples of operations. The outer tuple should be mapped directly
        to the circuit moments, and each inner tuple consists of operations that
        should be executed in parallel.

    Raises:
        GateDecompositionError: When deviations from true √iSWAP make the
            decomposition infeasible.
    """

    theta = parameters.get_theta(qubits)
    phi = parameters.get_phi(qubits)

    sin_alpha = ((np.sin(angle / 4) ** 2 - np.sin(phi / 2) ** 2) /
                 (np.sin(theta) ** 2 - np.sin(phi / 2) ** 2))

    if sin_alpha < 0.0 and np.isclose(sin_alpha, 0.0, atol=1e-3):
        sin_alpha = 0.0
    elif 1.0 < sin_alpha < 1.0 + sin_alpha_tolerance:
        sin_alpha = 1.0

    if 0 <= sin_alpha <= 1:
        alpha = np.arcsin(sin_alpha ** 0.5)
    else:
        raise GateDecompositionError(
            f'Cannot decompose the C-phase gate on qubits {qubits} into the '
            f'given fSim gates (angle={angle}, sin(alpha)={sin_alpha}, '
            f'parameters={parameters})')

    beta = 0.5 * np.pi * (1 - np.sign(np.cos(0.5 * phi)))
    gamma = 0.5 * np.pi * (1 - np.sign(np.sin(0.5 * phi)))

    xi = np.arctan(np.tan(alpha) * np.cos(theta) / np.cos(0.5 * phi)) + beta
    if angle < 0:
        xi += np.pi

    if np.isclose(phi, 0.0):
        eta = 0.5 * np.sign(np.tan(alpha) * np.sin(theta)) * np.pi
    else:
        eta = np.arctan(
            np.tan(alpha) * np.sin(theta) / np.sin(0.5 * phi)) + gamma

    sqrt_iswap = _corrected_sqrt_iswap_ops(qubits, parameters)
    a, b = qubits

    return (
        (cirq.rx(xi).on(a), cirq.rx(eta).on(b)),
        (cirq.rz(0.5 * phi).on(a), cirq.rz(0.5 * phi).on(b))
    ) + sqrt_iswap + (
        (cirq.rx(-2 * alpha).on(a),),
        (cirq.rz(0.5 * phi + np.pi).on(a), cirq.rz(0.5 * phi).on(b))
    ) + sqrt_iswap + (
        (cirq.rx(-xi).on(a), cirq.rx(-eta).on(b)),
        (cirq.rz(-0.5 * angle + np.pi).on(a), cirq.rz(-0.5 * angle).on(b))
    )


def _corrected_cphase_echo_ops(qubit: cirq.Qid
                               ) -> Tuple[Tuple[cirq.Operation, ...], ...]:
    return ((),) * 5 + (cirq.X(qubit),) + ((),) * 4 + (cirq.X(qubit),)


def _corrected_sqrt_iswap_ops(
        qubits: Tuple[cirq.Qid, cirq.Qid],
        parameters: ParticleConservingParameters
) -> Tuple[Tuple[cirq.Operation, ...], ...]:
    """Decomposition of √iSWAP into one non-ideal √iSWAP gate.

    Target unitary is:

     [[1,     0,     0,   0],
      [0,  1/√2, -i/√2,   0],
      [0, -i/√2,  1/√2,   0],
      [0,     0,     0,   1]]

    This target unitary might not be always possible to realize. The
    decomposition compensates for delta, chi and gamma deviations from the
    perfect unitary but keeps theta and phi errors of the non-ideal √iSWAP gate.

    Args:
        qubits: Two qubits to act on. Note the that ordering of qubits is
            relevant because true √iSWAP is not symmetric (see
            ParticleConservingParameters.for_qubits_swapped)
        parameters: True unitary parameters of the √iSWAP gate into which this
            function decomposes to.
    Returns:
        Tuple of tuples of operations. The outer tuple should be mapped directly
        to the circuit moments, and each inner tuple consists of operations that
        should be executed in parallel.
    """

    delta = parameters.get_delta(qubits)
    gamma = parameters.get_gamma(qubits)
    chi = parameters.get_chi(qubits)

    a, b = qubits
    alpha = 0.5 * (delta + chi)
    beta = 0.5 * (delta - chi)
    return (
        (cirq.rz(0.5 * gamma - alpha).on(a),
         cirq.rz(0.5 * gamma + alpha).on(b)),
        (cirq.ISWAP(a, b) ** (-0.5),),
        (cirq.rz(0.5 * gamma - beta).on(a), cirq.rz(0.5 * gamma + beta).on(b))
    )


def _merge_parameters(parameters: ParticleConservingParameters,
                      fallback: ParticleConservingParameters
                      ) -> ParticleConservingParameters:
    """Merges two instances of ParticleConservingParameters together.

    Uses values from parameters argument if they are set and fallbacks to values
    from fallback argument otherwise.
    """
    return ParticleConservingParameters(
        theta=fallback.theta if parameters.theta is None else parameters.theta,
        delta=fallback.delta if parameters.delta is None else parameters.delta,
        chi=fallback.chi if parameters.chi is None else parameters.chi,
        gamma=fallback.gamma if parameters.gamma is None else parameters.gamma,
        phi=fallback.phi if parameters.phi is None else parameters.phi,
    )


def decompose_preserving_moments(
        circuit: cirq.Circuit,
        decompose_func: Union[DecomposeCallable, Iterable[DecomposeCallable]]
) -> cirq.Circuit:
    """Decomposes circuit moment by moment.

    This function decomposes each operation within every moment simultaneously
    and expands the moment into the longest operation that was decomposed. It
    never mixes operation from two different input moments together.

    Args:
        circuit: Circuit to decompose.
        decompose_func: Function or iterable of functions that decomposes
            operation into iterable of moments of simultaneously executed
            operations. If many functions are provided, all off them are tried
            until decomposition is not None. When no decomposition is found,
            input gate is copied as is.

    Returns:
        New cirq.Circuit instance which is a decomposed version of circuit.
    """

    def decompose(operation: cirq.Operation) -> Decomposition:
        for func in decompose_func:
            decomposition = func(operation)
            if decomposition is not None:
                return decomposition
        return (operation,),

    if not isinstance(decompose_func, Iterable):
        decompose_func = decompose_func,

    decomposed = cirq.Circuit()
    for moment in circuit:
        decompositions = (decompose(operation) for operation in moment)
        for operations in zip_longest(*decompositions, fillvalue=()):
            decomposed += cirq.Moment(operations)

    return decomposed
