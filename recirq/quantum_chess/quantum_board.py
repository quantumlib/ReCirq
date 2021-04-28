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
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cirq

from recirq.quantum_chess.bit_utils import (
    bit_to_qubit,
    nth_bit_of,
    num_ones,
    qubit_to_bit,
    set_nth_bit,
    square_to_bit,
    xy_to_bit,
    bit_ones,
)
import recirq.quantum_chess.circuit_transformer as ct
import recirq.quantum_chess.enums as enums
import recirq.quantum_chess.move as move
import recirq.quantum_chess.quantum_moves as qm


class CirqBoard:
    """Implementation of Quantum Board API using cirq sampler.

    This class implements a quantum chess board by representing
    it as a 64-bit integer that represents the initial state as
    well as the classical components of the board.  It also
    contains a `cirq.Circuit` that represents a circuit that
    can be executed to generate samples for the quantum portions
    of the board.

    This board implements all measurements of intermediate moves
    through post-selection.  Measurements are moved to ancilla
    qubits and measured as part of a final measurement.  Results
    are then post-selected to retrieve samples that match board
    positions with the same measurements.

    Args:
        init_basis_state: 64-bit bitboard defining the initial
            classical state of the board.
        sampler: a Cirq.sampler that can simulate or execute
            quantum circuits to run.  Defaults to cirq simulator.
        device: a cirq.Device that should run the circuit.  This
            device should use GridQubits.  If specified, qubit
            mapping and decomposition will be performed to transform
            the circuit to run on the device.
        error_mitigation:  If enabled, detects when pieces have
            unexplainably disappeared or appeared and either throws
            an error or post-selects the result away.
        noise_mitigation: Threshold of samples to overcome in order
            to be considered not noise.
        transformer: The CircuitTransformer to use to convert the board's
            NamedQubit circuit into a GridQubit circuit.
    """

    def __init__(self,
                 init_basis_state: int,
                 sampler: cirq.Sampler = cirq.Simulator(),
                 device: Optional[cirq.Device] = None,
                 error_mitigation: Optional[
                     enums.ErrorMitigation] = enums.ErrorMitigation.Nothing,
                 noise_mitigation: Optional[float] = 0.0,
                 transformer: Optional[ct.CircuitTransformer] = None):
        self.device = device
        self.sampler = sampler
        if device is not None:
            self.transformer = (
                transformer
                or ct.ConnectivityHeuristicCircuitTransformer(device))
        self.with_state(init_basis_state)
        self.error_mitigation = error_mitigation
        self.noise_mitigation = noise_mitigation

        # None if there is no cache, stores the repetition number if there is a cache.
        self.accumulations_repetitions = None

    def with_state(self, basis_state: int) -> 'CirqBoard':
        """Resets the board with a specific classical state."""
        self.accumulations_repetitions = None
        self.state = basis_state
        self.allowed_pieces = set()
        self.allowed_pieces.add(num_ones(self.state))
        self.entangled_squares = set()
        self.post_selection = {}
        self.circuit = cirq.Circuit()
        self.ancilla_count = 0
        self.move_history = []
        # Store the initial basis state so that we can use it for replaying
        # the move-history when undoing moves
        self.init_basis_state = basis_state
        self.clear_debug_log()
        self.timing_stats = defaultdict(list)
        return self

    def clear_debug_log(self) -> None:
        """Clears debug log."""
        self.debug_log = ''

    def print_debug_log(self, clear_log: bool = True) -> None:
        """Prints debug log. Clears debug log if clear_log is enabled."""
        print(self.debug_log)
        if clear_log:
            self.clear_debug_log()

    def clear_timing_stats(self) -> None:
        """Clears timing stats."""
        self.timing_stats = defaultdict(list)

    def print_timing_stats(self, clear_stats: bool = False) -> None:
        """Prints timing stats. Clears timing stats if clears_stats is enabled."""
        print(self.timing_stats)
        if clear_stats:
            self.clear_timing_stats()

    def perform_moves(self, *moves) -> bool:
        """Performs a list of moves, specified as strings.

        Useful for short-hand versions of tests.

        Returns the measurement for the final move
        """
        did_it_move = False
        for m in moves:
            did_it_move = self.do_move(move.Move.from_string(m))
        return did_it_move

    def undo_last_move(self) -> bool:
        """Undoes the last move.

        Instead of undoing the last move, this function resets
        the board to the initial state and replays the move history

        """
        # Store current move history...
        current_move_history = self.move_history.copy()
        # ...because we'll be resetting it here
        self.with_state(self.init_basis_state)

        # Repeat history up to last move
        for m in range(len(current_move_history) - 1):
            if not self.do_move(current_move_history[m]):
                return False
        return True

    def record_time(self, action: str, t0: float, t1: Optional[float] = None) -> None:
        """Writes time span from t0 to t1 (if specified, otherwise the current time is used)
        into the debug log and timing stats.
        """
        if t1 is None:
            t1 = time.perf_counter()
        self.debug_log += (f"{action} takes {t1 - t0:0.4f} seconds.\n")
        self.timing_stats[action].append(t1 - t0)

    def sample_with_ancilla(self, num_samples: int
                            ) -> Tuple[List[int], List[Dict[str, int]]]:
        """Samples the board and returns square and ancilla measurements.

        Sends the current circuit to the sampler then retrieves the results.
        May return less samples than num_samples due to post-selection.

        Returns the results as a tuple.  The first entry is the list of
        measured squares, as represented by a 64-bit int bitboard.
        The second value is a list of ancilla values, as represented as a
        dictionary from ancilla name to value (0 or 1).
        """
        t0 = time.perf_counter()
        measure_circuit = self.circuit.copy()
        ancilla = []
        error_count = 0
        noise_count = 0
        post_count = 0
        if self.entangled_squares:
            qubits = sorted(self.entangled_squares)
            measure_moment = cirq.Moment(
                cirq.measure(q, key=q.name) for q in qubits)
            measure_circuit.append(measure_moment)

            # Try to guess the appropriate number of repetitions needed
            # Assume that each post_selection is about 50/50
            # Noise and error mitigation will discard reps, so increase
            # the total number of repetitions to compensate
            if len(self.post_selection) > 1:
                num_reps = num_samples * (2 ** (len(self.post_selection) + 1))
            else:
                num_reps = num_samples
            if self.error_mitigation == enums.ErrorMitigation.Correct:
                num_reps *= 2
            noise_threshold = self.noise_mitigation * num_samples
            if self.noise_mitigation > 0:
                num_reps *= 3
            if num_reps < 100:
                num_reps = 100

            self.debug_log += (f'Running circuit with {num_reps} reps '
                               f'to get {num_samples} samples:\n'
                               f'{str(measure_circuit)}\n')

            # Translate circuit to grid qubits and sqrtISWAP gates
            if self.device is not None:
                # Decompose 3-qubit operations
                ct.SycamoreDecomposer().optimize_circuit(measure_circuit)
                # Create NamedQubit to GridQubit mapping and transform
                measure_circuit = self.transformer.transform(measure_circuit)

                # For debug, ensure that the circuit correctly validates
                self.device.validate_circuit(measure_circuit)

            # Run the circuit using the provided sampler (simulator or hardware)
            results = self.sampler.run(measure_circuit, repetitions=num_reps)

            # Parse the results
            rtn = []
            noise_buffer = {}
            data = results.data
            for rep in range(num_reps):
                new_sample = self.state
                new_ancilla = {}

                # Go through the results and discard any results
                # that disagree with our pre-defined post-selection criteria
                post_selected = True
                for qubit in self.post_selection.keys():
                    key = qubit.name
                    if key in data.columns:
                        result = data.at[rep, key]
                        if result != self.post_selection[qubit]:
                            post_selected = False
                            break
                if not post_selected:
                    post_count += 1
                    continue

                # Translate qubit results into a 64-bit chess board
                for qubit in qubits:
                    key = qubit.name
                    result = data.at[rep, key]
                    # Ancilla bits should not be part of the chess board
                    if 'anc' not in key:
                        bit = qubit_to_bit(qubit)
                        new_sample = set_nth_bit(bit, new_sample, result)
                    else:
                        new_ancilla[key] = result

                # Perform Error Mitigation
                if self.error_mitigation != enums.ErrorMitigation.Nothing:
                    # Discard boards that have the wrong number of pieces
                    if num_ones(new_sample) not in self.allowed_pieces:
                        if self.error_mitigation == enums.ErrorMitigation.Error:
                            raise ValueError(
                                'Error detected, '
                                f'pieces allowed = {self.allowed_pieces}'
                                f'but got {num_ones(new_sample)}')
                        if self.error_mitigation == enums.ErrorMitigation.Correct:
                            error_count += 1
                            continue

                # Noise mitigation
                if self.noise_mitigation > 0.0:
                    # Ignore samples up to a threshold
                    if new_sample not in noise_buffer:
                        noise_buffer[new_sample] = 0
                    noise_buffer[new_sample] += 1
                    if noise_buffer[new_sample] < noise_threshold:
                        noise_count += 1
                        continue

                # This sample has passed noise and error mitigation
                # Record it as a proper sample
                rtn.append(new_sample)
                ancilla.append(new_ancilla)
                if len(rtn) >= num_samples:
                    self.debug_log += (
                        f'Discarded {error_count} from error mitigation '
                        f'{noise_count} from noise and '
                        f'{post_count} from post-selection\n')
                    self.record_time('sample_with_ancilla', t0)
                    return (rtn, ancilla)
        else:
            rtn = [self.state] * num_samples
            self.debug_log += (
                f'Discarded {error_count} from error mitigation '
                f'{noise_count} from noise and {post_count} from post-selection\n'
            )
        self.record_time("sample_with_ancilla", t0)
        return (rtn, ancilla)

    def sample(self, num_samples: int) -> List[int]:
        """Samples the board and returns square and ancilla measurements.

        Sends the current circuit to the sampler then retrieves the results.

        Returns the results as a list of bitboards.
        """
        rtn = []
        while len(rtn) < num_samples:
            samples, _ = self.sample_with_ancilla(num_samples)
            rtn = rtn + samples
        return rtn[:num_samples]

    def _generate_accumulations(self, repetitions: int = 1000) -> None:
        """ Samples the state and generates the accumulated 
        probabilities, empty_squares, and full_squares.
        """
        self.probabilities = [0] * 64
        self.full_squares = (1 << 64) - 1
        self.empty_squares = (1 << 64) - 1

        samples = self.sample(repetitions)
        for sample in samples:
            self.full_squares &= sample
            self.empty_squares &= ~sample
            for bit in bit_ones(sample):
                self.probabilities[bit] += 1

        for bit in range(64):
            self.probabilities[bit] = float(self.probabilities[bit]) / float(repetitions)

        self.accumulations_repetitions = repetitions

    def get_probability_distribution(self,
                                     repetitions: int = 1000) -> List[float]:
        """Returns the probability of a piece being in each square.

        The values are returned as a list in the same ordering as a
        bitboard.
        """
        if self.accumulations_repetitions != repetitions:
            self._generate_accumulations(repetitions)

        return self.probabilities

    def get_full_squares_bitboard(self, repetitions: int = 1000) -> int:
        """Retrieves which squares are marked as full.

        This information is created using a representative set of
        samples (defined by the repetitions argument) to determine
        which squares are occupied on all boards.

        Returns a bitboard.
        """
        if self.accumulations_repetitions != repetitions:
            self._generate_accumulations(repetitions)

        return self.full_squares

    def get_empty_squares_bitboard(self, repetitions: int = 1000) -> int:
        """Retrieves which squares are marked as empty.

        This information is created using a representative set of
        samples (defined by the repetitions argument) to determine
        which squares are empty on all boards.

        Returns a bitboard.
        """
        if self.accumulations_repetitions != repetitions:
            self._generate_accumulations(repetitions)

        return self.empty_squares

    def add_entangled(self, *qubits):
        """Adds squares as entangled.

        This enables measuring of the square by the quantum circuit
        and also adds a piece in the square to the circuit if the
        classical register is currently set to one.
        """
        for qubit in qubits:
            if qubit not in self.entangled_squares:
                self.entangled_squares.add(qubit)
                if nth_bit_of(qubit_to_bit(qubit), self.state):
                    self.circuit.append(qm.place_piece(qubit))

    def new_ancilla(self) -> cirq.Qid:
        """Adds a new ancilla to the circuit and returns its value."""
        new_name = f'anc{self.ancilla_count}'
        new_qubit = cirq.NamedQubit(new_name)
        self.ancilla_count += 1
        return new_qubit

    def unhook(self, qubit: cirq.Qid) -> cirq.Qid:
        """Removes a qubit from the quantum portion of the board.

        This exchanges all mentions of the qubit in the circuit
        with a new ancilla qubit and removes it from the set of
        entangled squares.
        """
        if qubit not in self.entangled_squares:
            return

        # Create a new ancilla qubit to replace the qubit with
        new_qubit = self.new_ancilla()

        # Replace operations using the qubit with the ancilla instead
        self.circuit = self.circuit.transform_qubits(lambda q: new_qubit if q == qubit else q)

        # Remove the qubit from the list of active qubits
        self.entangled_squares.remove(qubit)
        self.entangled_squares.add(new_qubit)
        return new_qubit

    def path_qubits(self, source: str, target: str) -> List[cirq.Qid]:
        """Returns all entangled qubits (or classical pieces)
        between source and target.

        Source and target should be in the same line, i.e. same row, 
        same column, or same diagonal.

        Source and target should be specified in algebraic notation,
        such as 'f4'.
        """
        rtn = []
        xs = move.x_of(source)
        ys = move.y_of(source)
        xt = move.x_of(target)
        yt = move.y_of(target)
        if xt > xs:
            dx = 1
        elif xt < xs:
            dx = -1
        else:
            dx = 0
        if yt > ys:
            dy = 1
        elif yt < ys:
            dy = -1
        else:
            dy = 0
        x_slide = abs(xt - xs)
        y_slide = abs(yt - ys)
        # Souce and target should always be in the same line.
        if x_slide != y_slide and x_slide * y_slide:
            raise ValueError('Wrong inputs for path_qubits: source and target are not in the same line.')
        max_slide = max(x_slide, y_slide)
        # Only calculates path when max_slide > 1.
        for t in range(1, max_slide):
            path_bit = xy_to_bit(xs + dx * t, ys + dy * t)
            path_qubit = bit_to_qubit(path_bit)
            if (path_qubit in self.entangled_squares or
                nth_bit_of(path_bit, self.state)):
                rtn.append(path_qubit)
        return rtn

    def create_path_ancilla(self, path_qubits: List[cirq.Qid]) -> cirq.Qid:
        """Creates an ancilla that is anti-controlled by the qubits
        in the path."""
        path_ancilla = self.new_ancilla()
        self.circuit.append(
            qm.controlled_operation(cirq.X, [path_ancilla], [], path_qubits))
        return path_ancilla

    def set_castle(self, sbit: int, rook_sbit: int, tbit: int,
                   rook_tbit: int) -> None:
        """Adjusts classical bits for a castling operation."""
        self.state = set_nth_bit(sbit, self.state, False)
        self.state = set_nth_bit(rook_sbit, self.state, False)
        self.state = set_nth_bit(tbit, self.state, True)
        self.state = set_nth_bit(rook_tbit, self.state, True)

    def queenside_castle(self, squbit: int, rook_squbit: int, tqubit: int,
                         rook_tqubit: int, b_qubit: int) -> None:
        """Performs a queenside castling operation."""
        self.add_entangled(squbit, tqubit, rook_squbit, rook_tqubit)
        self.circuit.append(
            qm.queenside_castle(squbit, rook_squbit, tqubit, rook_tqubit,
                                b_qubit))

    def post_select_on(self, qubit: cirq.Qid,
                       measurement_outcome: Optional[int] = None) -> bool:
        """Adds a post-selection requirement to the circuit.

        If no measurement_outcome is provided, performs a single sample of the
        qubit to get a value for the post-selection condition.
        Adjusts the post-selection requirements dictionary to this value.
        If this qubit is a square qubit, it adjusts the classical register
        to match the sample result.

        Args:
            qubit: the qubit to post-select on
            measurement_outcome: the optional measurement outcome. If present,
                post-selection is conditioned on the qubit having the given
                outcome. If absent, a single measurement is performed instead.

        Returns: the measurement outcome or sample result as 1 or 0.
        """
        result = measurement_outcome
        if 'anc' in qubit.name:
            if result is None:
                ancilla_result = []
                while len(ancilla_result) == 0:
                    _, ancilla_result = self.sample_with_ancilla(10)
                result = ancilla_result[0][qubit.name]
            self.post_selection[qubit] = result
        else:
            bit = qubit_to_bit(qubit)
            if result is None:
                result = nth_bit_of(bit, self.sample(1)[0])
            if qubit in self.entangled_squares:
                ancillary = self.unhook(qubit)
                self.post_selection[ancillary] = result
            self.state = set_nth_bit(bit, self.state, result)
        return result

    def do_move(self, m: move.Move) -> int:
        """Performs a move on the quantum board.

        Based on the type and variant of the move requested,
        this function augments the circuit, classical registers,
        and post-selection criteria to perform the board.

        Returns:  The measurement that was performed, or 1 if
            no measurement was required.
        """
        if not m.move_type:
            raise ValueError('No Move defined')
        if m.move_type == enums.MoveType.NULL_TYPE:
            raise ValueError('Move has null type')
        if m.move_type == enums.MoveType.UNSPECIFIED_STANDARD:
            raise ValueError('Move type is unspecified')

        # Reset accumulations here because function has conditional return branches
        self.accumulations_repetitions = None

        # Add move to move_history
        self.move_history.append(m)

        sbit = square_to_bit(m.source)
        tbit = square_to_bit(m.target)
        squbit = bit_to_qubit(sbit)
        tqubit = bit_to_qubit(tbit)

        if (m.move_variant == enums.MoveVariant.CAPTURE or
                m.move_type == enums.MoveType.PAWN_EP or
                m.move_type == enums.MoveType.PAWN_CAPTURE):
            # TODO: figure out if it is a deterministic capture.
            for val in list(self.allowed_pieces):
                self.allowed_pieces.add(val - 1)

        if m.move_type == enums.MoveType.PAWN_EP:
            # For en passant, first determine the square of the pawn being
            # captured, which should be next to the target.
            if m.target[1] == '6':
                epbit = square_to_bit(m.target[0] + '5')
            elif m.target[1] == '3':
                epbit = square_to_bit(m.target[0] + '4')
            else:
                raise ValueError(f'Invalid en passant target {m.target}')
            epqubit = bit_to_qubit(epbit)

            # For the classical version, set the bits appropriately
            if (epqubit not in self.entangled_squares and
                    squbit not in self.entangled_squares and
                    tqubit not in self.entangled_squares):
                if (not nth_bit_of(epbit, self.state) or
                        not nth_bit_of(sbit, self.state) or
                        nth_bit_of(tbit, self.state)):
                    raise ValueError('Invalid classical e.p. move')

                self.state = set_nth_bit(epbit, self.state, False)
                self.state = set_nth_bit(sbit, self.state, False)
                self.state = set_nth_bit(tbit, self.state, True)
                return 1

            # If any squares are quantum, it's a quantum move
            self.add_entangled(squbit, tqubit, epqubit)

            # Capture e.p. post-select on the source
            if m.move_variant == enums.MoveVariant.CAPTURE:
                is_there = self.post_select_on(squbit, m.measurement)
                if not is_there:
                    return 0
                self.add_entangled(squbit)
                # capture e.p. has a special circuit
                self.circuit.append(
                    qm.capture_ep(squbit, tqubit, epqubit, self.new_ancilla(),
                                  self.new_ancilla(), self.new_ancilla()))
                return 1

            # Blocked/excluded e.p. post-select on the target
            if m.move_variant == enums.MoveVariant.EXCLUDED:
                is_there = self.post_select_on(tqubit, m.measurement)
                if is_there:
                    return 0
                self.add_entangled(tqubit)
            self.circuit.append(
                qm.en_passant(squbit, tqubit, epqubit, self.new_ancilla(),
                              self.new_ancilla()))
            return 1

        if m.move_type == enums.MoveType.PAWN_CAPTURE:
            # For pawn capture, first measure source.
            is_there = self.post_select_on(squbit, m.measurement)
            if not is_there:
                return 0
            if tqubit in self.entangled_squares:
                old_tqubit = self.unhook(tqubit)
                self.add_entangled(squbit, tqubit)

                self.circuit.append(
                    qm.controlled_operation(cirq.ISWAP, [squbit, tqubit],
                                            [old_tqubit], []))
            else:
                # Classical case
                self.state = set_nth_bit(sbit, self.state, False)
                self.state = set_nth_bit(tbit, self.state, True)
            return 1

        if m.move_type == enums.MoveType.SPLIT_SLIDE:
            tbit2 = square_to_bit(m.target2)
            tqubit2 = bit_to_qubit(tbit2)

            # Find all the squares on both paths
            path_qubits = self.path_qubits(m.source, m.target)
            path_qubits2 = self.path_qubits(m.source, m.target2)

            if len(path_qubits) == 0 and len(path_qubits2) == 0:
                # No interposing squares, just jump.
                m.move_type = enums.MoveType.SPLIT_JUMP
            else:
                self.add_entangled(squbit, tqubit, tqubit2)
                path1 = self.create_path_ancilla(path_qubits)
                path2 = self.create_path_ancilla(path_qubits2)
                ancilla = self.new_ancilla()
                self.circuit.append(
                    qm.split_slide(squbit, tqubit, tqubit2, path1, path2,
                                   ancilla))
                return 1

        if m.move_type == enums.MoveType.MERGE_SLIDE:
            sbit2 = square_to_bit(m.source2)
            squbit2 = bit_to_qubit(sbit2)
            self.add_entangled(squbit, squbit2, tqubit)

            # Find all the squares on both paths
            path_qubits = self.path_qubits(m.source, m.target)
            path_qubits2 = self.path_qubits(m.source2, m.target)
            if len(path_qubits) == 0 and len(path_qubits2) == 0:
                # No interposing squares, just jump.
                m.move_type = enums.MoveType.MERGE_JUMP
            else:
                path1 = self.create_path_ancilla(path_qubits)
                path2 = self.create_path_ancilla(path_qubits2)
                ancilla = self.new_ancilla()
                self.circuit.append(
                    qm.merge_slide(squbit, tqubit, squbit2, path1, path2,
                                   ancilla))
                return 1

        if (m.move_type == enums.MoveType.SLIDE or
                m.move_type == enums.MoveType.PAWN_TWO_STEP):
            path_qubits = self.path_qubits(m.source, m.target)
            if len(path_qubits) == 0:
                # No path, change to jump
                m.move_type = enums.MoveType.JUMP

        if (m.move_type == enums.MoveType.SLIDE or
                m.move_type == enums.MoveType.PAWN_TWO_STEP):
            for p in path_qubits:
                if (p not in self.entangled_squares and
                        nth_bit_of(qubit_to_bit(p), self.state)):
                    # Classical piece in the way
                    return 0

            # For excluded case, measure target
            if m.move_variant == enums.MoveVariant.EXCLUDED:
                is_there = self.post_select_on(tqubit, m.measurement)
                if is_there:
                    return 0

            self.add_entangled(squbit, tqubit)
            if m.move_variant == enums.MoveVariant.CAPTURE:
                capture_ancilla = self.new_ancilla()
                self.circuit.append(
                    qm.controlled_operation(cirq.X, [capture_ancilla], [squbit],
                                            path_qubits))

                # We need to add the captured_ancilla to entangled squares
                # So that we measure it
                self.entangled_squares.add(capture_ancilla)
                capture_allowed = self.post_select_on(capture_ancilla, m.measurement)

                if not capture_allowed:
                    return 0
                else:
                    # Perform the captured slide
                    self.add_entangled(squbit)
                    # Remove the target from the board into an ancilla
                    # and set bit to zero
                    self.unhook(tqubit)
                    self.state = set_nth_bit(tbit, self.state, False)

                    # Re-add target since we need to swap into the square
                    self.add_entangled(tqubit)

                    # Perform the actual move
                    self.circuit.append(qm.normal_move(squbit, tqubit))

                    # Set source to empty
                    self.unhook(squbit)
                    self.state = set_nth_bit(sbit, self.state, False)

                    # Now set the whole path to empty
                    for p in path_qubits:
                        self.state = set_nth_bit(qubit_to_bit(p), self.state, False)
                        self.unhook(p)
                    return 1
            # Basic slide (or successful excluded slide)

            # Add all involved squares into entanglement
            self.add_entangled(squbit, tqubit, *path_qubits)

            if len(path_qubits) == 1:
                # For path of one, no ancilla needed
                self.circuit.append(qm.slide_move(squbit, tqubit, path_qubits))
                return 1
            # Longer paths require a path ancilla
            ancilla = self.new_ancilla()
            self.circuit.append(
                qm.slide_move(squbit, tqubit, path_qubits, ancilla))
            return 1

        if (m.move_type == enums.MoveType.JUMP or
                m.move_type == enums.MoveType.PAWN_STEP):
            if (squbit not in self.entangled_squares and
                    tqubit not in self.entangled_squares):
                # Classical version
                self.state = set_nth_bit(sbit, self.state, False)
                self.state = set_nth_bit(tbit, self.state, True)
                return 1

            # Measure source for capture
            if m.move_variant == enums.MoveVariant.CAPTURE:
                is_there = self.post_select_on(squbit, m.measurement)
                if not is_there:
                    return 0
                self.unhook(tqubit)

            # Measure target for excluded
            if m.move_variant == enums.MoveVariant.EXCLUDED:
                is_there = self.post_select_on(tqubit, m.measurement)
                if is_there:
                    return 0

            # Only convert source qubit to ancilla if target
            # is empty
            unhook = tqubit not in self.entangled_squares
            self.add_entangled(squbit, tqubit)

            # Execute jump
            self.circuit.append(qm.normal_move(squbit, tqubit))

            if unhook or m.move_variant != enums.MoveVariant.BASIC:
                # The source is empty.
                # Change source qubit to be an ancilla
                # and set classical bit to zero
                self.state = set_nth_bit(sbit, self.state, False)
                self.unhook(squbit)

            return 1

        if m.move_type == enums.MoveType.SPLIT_JUMP:
            tbit2 = square_to_bit(m.target2)
            tqubit2 = bit_to_qubit(tbit2)
            self.add_entangled(squbit, tqubit, tqubit2)
            self.circuit.append(qm.split_move(squbit, tqubit, tqubit2))
            self.state = set_nth_bit(sbit, self.state, False)
            self.unhook(squbit)
            return 1

        if m.move_type == enums.MoveType.MERGE_JUMP:
            sbit2 = square_to_bit(m.source2)
            squbit2 = bit_to_qubit(sbit2)
            self.add_entangled(squbit, squbit2, tqubit)
            self.circuit.append(qm.merge_move(squbit, squbit2, tqubit))
            # TODO: should the source qubit be 'unhooked'?
            return 1

        if m.move_type == enums.MoveType.KS_CASTLE:
            # Figure out the rook squares
            if sbit == square_to_bit('e1') and tbit == square_to_bit('g1'):
                rook_sbit = square_to_bit('h1')
                rook_tbit = square_to_bit('f1')
            elif sbit == square_to_bit('e8') and tbit == square_to_bit('g8'):
                rook_sbit = square_to_bit('h8')
                rook_tbit = square_to_bit('f8')
            else:
                raise ValueError(f'Invalid kingside castling move')
            rook_squbit = bit_to_qubit(rook_sbit)
            rook_tqubit = bit_to_qubit(rook_tbit)

            # Piece in non-superposition in the way, not legal
            if (nth_bit_of(rook_tbit, self.state) and
                    rook_tqubit not in self.entangled_squares):
                return 0
            if (nth_bit_of(tbit, self.state) and
                    tqubit not in self.entangled_squares):
                return 0

            # Not in superposition, just castle
            if (rook_tqubit not in self.entangled_squares and
                    tqubit not in self.entangled_squares):
                self.set_castle(sbit, rook_sbit, tbit, rook_tbit)
                return 1

            # Both intervening squares in superposition
            if (rook_tqubit in self.entangled_squares and
                    tqubit in self.entangled_squares):
                castle_ancilla = self.create_path_ancilla([rook_tqubit, tqubit])
                self.entangled_squares.add(castle_ancilla)
                castle_allowed = self.post_select_on(castle_ancilla, m.measurement)
                if castle_allowed:
                    self.unhook(rook_tqubit)
                    self.unhook(tqubit)
                    self.set_castle(sbit, rook_sbit, tbit, rook_tbit)
                    return 1
                else:
                    self.post_selection[castle_ancilla] = castle_allowed
                    return 0

            # One intervening square in superposition
            if rook_tqubit in self.entangled_squares:
                measure_qubit = rook_tqubit
                measure_bit = rook_tbit
            else:
                measure_qubit = tqubit
                measure_bit = tbit
            is_there = self.post_select_on(measure_qubit, m.measurement)
            if is_there:
                return 0
            self.set_castle(sbit, rook_sbit, tbit, rook_tbit)
            return 1

        if m.move_type == enums.MoveType.QS_CASTLE:

            # Figure out the rook squares and the b-file square involved
            if sbit == square_to_bit('e1') and tbit == square_to_bit('c1'):
                rook_sbit = square_to_bit('a1')
                rook_tbit = square_to_bit('d1')
                b_bit = square_to_bit('b1')
            elif sbit == square_to_bit('e8') and tbit == square_to_bit('c8'):
                rook_sbit = square_to_bit('a8')
                rook_tbit = square_to_bit('d8')
                b_bit = square_to_bit('b8')
            else:
                raise ValueError(f'Invalid queenside castling move')
            rook_squbit = bit_to_qubit(rook_sbit)
            rook_tqubit = bit_to_qubit(rook_tbit)
            b_qubit = bit_to_qubit(b_bit)

            # Piece in non-superposition in the way, not legal
            if (nth_bit_of(rook_tbit, self.state) and
                    rook_tqubit not in self.entangled_squares):
                return 0
            if (nth_bit_of(tbit, self.state) and
                    tqubit not in self.entangled_squares):
                return 0
            if (b_bit is not None and nth_bit_of(b_bit, self.state) and
                    b_qubit not in self.entangled_squares):
                return 0

            # Not in superposition, just castle
            if (rook_tqubit not in self.entangled_squares and
                    tqubit not in self.entangled_squares and
                    b_qubit not in self.entangled_squares):
                self.set_castle(sbit, rook_sbit, tbit, rook_tbit)
                return 1

            # Neither intervening squares in superposition
            if (rook_tqubit not in self.entangled_squares and
                    tqubit not in self.entangled_squares):
                if b_qubit not in self.entangled_squares:
                    self.set_castle(sbit, rook_sbit, tbit, rook_tbit)
                else:
                    self.queenside_castle(squbit, rook_squbit, tqubit,
                                          rook_tqubit, b_qubit)
                return 1

            # Both intervening squares in superposition
            if (rook_tqubit in self.entangled_squares and
                    tqubit in self.entangled_squares):
                castle_ancilla = self.create_path_ancilla([rook_tqubit, tqubit])
                self.entangled_squares.add(castle_ancilla)
                castle_allowed = self.post_select_on(castle_ancilla, m.measurement)
                if castle_allowed:
                    self.unhook(rook_tqubit)
                    self.unhook(tqubit)
                    if b_qubit not in self.entangled_squares:
                        self.set_castle(sbit, rook_sbit, tbit, rook_tbit)
                    else:
                        self.queenside_castle(squbit, rook_squbit, tqubit,
                                              rook_tqubit, b_qubit)
                    return 1
                else:
                    self.post_selection[castle_ancilla] = castle_allowed
                    return 0

            # One intervening square in superposition
            if rook_tqubit in self.entangled_squares:
                measure_qubit = rook_tqubit
                measure_bit = rook_tbit
            else:
                measure_qubit = tqubit
                measure_bit = tbit
            is_there = self.post_select_on(measure_qubit, m.measurement)
            if is_there:
                return 0
            if b_qubit not in self.entangled_squares:
                self.set_castle(sbit, rook_sbit, tbit, rook_tbit)
            else:
                self.queenside_castle(squbit, rook_squbit, tqubit, rook_tqubit,
                                      b_qubit)
            return 1

        raise ValueError(f'Move type {m.move_type} not supported')

    def __str__(self):
        """Renders a ASCII diagram showing the board probabilities."""
        probs = self.get_probability_distribution()
        s = ''
        s += ' +----------------------------------+\n'
        for y in reversed(range(8)):
            s += str(y + 1) + '| '
            for x in range(8):
                bit = xy_to_bit(x, y)
                prob = str(int(100 * probs[bit]))
                if len(prob) <= 2:
                    s += ' '
                if prob == '0':
                    s += '.'
                else:
                    s += prob
                if len(prob) < 2:
                    s += ' '
                s += ' '
            s += ' |\n'
        s += ' +----------------------------------+\n    '
        for x in range(8):
            s += move.to_rank(x) + '   '
        return s
