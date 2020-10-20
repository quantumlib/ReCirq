import enums
import cirq
from typing import Dict, List, Optional, Tuple

from bit_utils import (
    bit_to_qubit,
    nth_bit_of,
    num_ones,
    qubit_to_bit,
    set_nth_bit,
    square_to_bit,
    xy_to_bit,
)
import circuit_transformer
import move
import quantum_moves as qm

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
    """

    def __init__(self,
                 init_basis_state: int,
                 sampler: cirq.Sampler = cirq.Simulator(),
                 device: Optional[cirq.Device] = None,
                 error_mitigation: Optional[
                     enums.ErrorMitigation] = enums.ErrorMitigation.Nothing,
                 noise_mitigation: Optional[float] = 0.0):
        self.device = device
        self.sampler = sampler
        if device is not None:
            self.transformer = circuit_transformer.CircuitTransformer(device)
        self.with_state(init_basis_state)
        self.error_mitigation = error_mitigation
        self.noise_mitigation = noise_mitigation
        self.accumulations_valid = False

    def with_state(self, basis_state: int):
        """Resets the board with a specific classical state."""
        self.accumulations_valid = False
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
        return self

    def clear_debug_log(self):
        self.debug_log = ''

    def print_debug_log(self, clear_log: bool = True):
        print(self.debug_log)
        if clear_log:
            self.clear_debug_log()

    def perform_moves(self, *moves):
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
        self.accumulations_valid = False
        
        # Store current move history...
        current_move_history = self.move_history.copy()
        # ...because we'll be resetting it here
        self.with_state(self.init_basis_state)

        # Repeat history up to last move
        undid_move = False
        for m in range(len(current_move_history)-1):
            undid_move = self.do_move(current_move_history[m])
        return undid_move

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

            # Try to guess the appropriate number of reps
            # Assume that each post_selection is about 50/50
            if len(self.post_selection) > 1:
                num_reps = num_samples * (2**(len(self.post_selection) + 1))
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
                circuit_transformer.SycamoreDecomposer().optimize_circuit(
                    measure_circuit)
                # Create NamedQubit to GridQubit mapping and transform
                self.transformer.qubit_mapping(measure_circuit)
                self.transformer.optimize_circuit(measure_circuit)
                #debug
                self.device.validate_circuit(measure_circuit)

            results = self.sampler.run(measure_circuit, repetitions=num_reps)
            rtn = []
            noise_buffer = {}
            data = results.data
            for rep in range(num_reps):
                new_sample = self.state
                new_ancilla = {}
                post_selected = True
                for qubit in self.post_selection.keys():
                    key = qubit.name
                    if key in data.columns:
                        result = data.at[rep, key]
                        if result != self.post_selection[qubit]:
                            post_selected = False
                if not post_selected:
                    post_count += 1
                    continue
                for qubit in qubits:
                    key = qubit.name
                    result = data.at[rep, key]
                    if 'anc' not in key:
                        bit = qubit_to_bit(qubit)
                        new_sample = set_nth_bit(bit, new_sample, result)
                    else:
                        new_ancilla[key] = result
                if self.error_mitigation != enums.ErrorMitigation.Nothing:
                    if num_ones(new_sample) not in self.allowed_pieces:
                        if self.error_mitigation == enums.ErrorMitigation.Error:
                            raise ValueError(
                                'Error detected, '
                                f'pieces allowed = {self.allowed_pieces}'
                                f'but got {num_ones(new_sample)}')
                        if self.error_mitigation == enums.ErrorMitigation.Correct:
                            error_count += 1
                            continue
                if self.noise_mitigation > 0.0:
                    # Ignore samples up to a threshold
                    if new_sample not in noise_buffer:
                        noise_buffer[new_sample] = 0
                    noise_buffer[new_sample] += 1
                    if noise_buffer[new_sample] < noise_threshold:
                        noise_count += 1
                        continue
                rtn.append(new_sample)
                ancilla.append(new_ancilla)
                if len(rtn) >= num_samples:
                    self.debug_log += (
                        f'Discarded {error_count} from error mitigation '
                        f'{noise_count} from noise and '
                        f'{post_count} from post-selection\n')
                    return (rtn, ancilla)
        else:
            rtn = [self.state] * num_samples
            self.debug_log += (
                f'Discarded {error_count} from error mitigation '
                f'{noise_count} from noise and {post_count} from post-selection\n'
            )
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

    def generate_accumulations(self, repetitions: int = 1000):
        """ Samples the state and generates the accumulated 
        probabilities, empty_squares, and full_squares
        """
        self.probabilities = [0] * 64
        self.full_squares = (1 << 64) - 1
        self.empty_squares = (1 << 64) - 1

        samples = self.sample(repetitions)
        for sample in samples:
            for bit in range(64):
                if nth_bit_of(bit, sample):
                    self.probabilities[bit] += 1
                    self.empty_squares = set_nth_bit(bit,self.empty_squares,0)
                else:
                    self.full_squares = set_nth_bit(bit,self.full_squares,0)

        for bit in range(64):
            self.probabilities[bit] = float(self.probabilities[bit]) / float(repetitions)
        
        self.accumulations_valid = True

    def get_probability_distribution(self,
                                     repetitions: int = 1000) -> List[float]:
        """Returns the probability of a piece being in each square.

        The values are returned as a list in the same ordering as a
        bitboard.
        """
        if not self.accumulations_valid:
            self.generate_accumulations(repetitions)
        
        return self.probabilities

    def get_full_squares_bitboard(self, repetitions: int = 1000) -> int:
        """Retrieves which squares are marked as full.

        This information is created using a representative set of
        samples (defined by the repetitions argument) to determine
        which squares are occupied on all boards.

        Returns a bitboard.
        """
        if not self.accumulations_valid:
            self.generate_accumulations(repetitions)

        return self.full_squares

    def get_empty_squares_bitboard(self, repetitions: int = 1000) -> int:
        """Retrieves which squares are marked as full.

        This information is created using a representative set of
        samples (defined by the repetitions argument) to determine
        which squares are empty on all boards.

        Returns a bitboard.
        """
        if not self.accumulations_valid:
            self.generate_accumulations(repetitions)

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

    def unhook(self, qubit) -> cirq.Qid:
        """Removes a qubit from the quantum portion of the board.

        This exchanges all mentions of the qubit in the circuit
        with a new ancilla qubit and removes it from the set of
        entangled squares.i
        """
        if qubit not in self.entangled_squares:
            return
        new_qubit = self.new_ancilla()
        new_circuit = cirq.Circuit()
        for moment in self.circuit:
            for op in moment:
                if qubit in op.qubits:
                    new_op_qubits = []
                    for q in op.qubits:
                        if q == qubit:
                            new_op_qubits.append(new_qubit)
                        else:
                            new_op_qubits.append(q)
                    new_circuit.append(op.with_qubits(*new_op_qubits))
                else:
                    new_circuit.append(op)
        self.circuit = new_circuit
        self.entangled_squares.remove(qubit)
        self.entangled_squares.add(new_qubit)
        return new_qubit

    def path_qubits(self, source, target) -> List[cirq.Qid]:
        """Returns all entangled qubits (or classical pieces)
        between source and target.

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
        max_slide = max(abs(xs - xt), abs(ys - yt))
        if max_slide > 1:
            for t in range(1, max_slide):
                path_bit = xy_to_bit(xs + dx * t, ys + dy * t)
                path_qubit = bit_to_qubit(path_bit)
                if (path_qubit in self.entangled_squares or
                        nth_bit_of(path_bit, self.state)):
                    rtn.append(path_qubit)
        return rtn

    def create_path_ancilla(self, path_qubits):
        """Creates an ancilla that is anti-controlled by the qubits
        in the path."""
        path_ancilla = self.new_ancilla()
        self.circuit.append(
            qm.controlled_operation(cirq.X, [path_ancilla], [], path_qubits))
        return path_ancilla

    def set_castle(self, sbit, rook_sbit, tbit, rook_tbit):
        """Adjusts classical bits for a castling operation."""
        self.state = set_nth_bit(sbit, self.state, 0)
        self.state = set_nth_bit(rook_sbit, self.state, 0)
        self.state = set_nth_bit(tbit, self.state, 1)
        self.state = set_nth_bit(rook_tbit, self.state, 1)

    def queenside_castle(self, squbit, rook_squbit, tqubit, rook_tqubit,
                         b_qubit):
        """Performs a queenside castling operation."""
        self.add_entangled(squbit, tqubit, rook_squbit, rook_tqubit)
        self.circuit.append(
            qm.queenside_castle(squbit, rook_squbit, tqubit, rook_tqubit,
                                b_qubit))

    def post_select_on(self, qubit):
        """Adds a post-selection requirement to the circuit,

        Performs a single sample of the qubit to get a value.
        Adjusts the post-selection requirements dictionary to this value.
        If this qubit is a square qubit, it adjusts the classical register
        to match the sample result.

        Returns: the sample result as 1 or 0.
        """
        if 'anc' in qubit.name:
            ancilla_result = []
            while len(ancilla_result) == 0:
                _, ancilla_result = self.sample_with_ancilla(10)
            result = ancilla_result[0][qubit.name]
            self.post_selection[qubit] = result
        else:
            bit = qubit_to_bit(qubit)
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
        self.accumulations_valid = False

        if not m.move_type:
            raise ValueError('No Move defined')
        if m.move_type == enums.MoveType.NULL_TYPE:
            raise ValueError('Move has null type')
        if m.move_type == enums.MoveType.UNSPECIFIED_STANDARD:
            raise ValueError('Move type is unspecified')

        # Add move to the move move_history
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
            elif m.target[1] == '2':
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
                    return 0
                self.state = set_nth_bit(epbit, self.state, 0)
                self.state = set_nth_bit(sbit, self.state, 0)
                self.state = set_nth_bit(tbit, self.state, 1)
                return 1

            # If any squares are quantum, it's a quantum move
            self.add_entangled(squbit, tqubit, epqubit)

            # Capture e.p. post-select on the source
            if m.move_variant == enums.MoveVariant.CAPTURE:
                is_there = self.post_select_on(squbit)
                if not is_there:
                    return 0
                self.add_entangled(squbit)
                path_ancilla = self.new_ancilla()
                captured_ancilla = self.new_ancilla()
                captured_ancilla2 = self.new_ancilla()
                # capture e.p. has a special circuit
                self.circuit.append(
                    qm.capture_ep(squbit, tqubit, epqubit, self.new_ancilla(),
                                  self.new_ancilla(), self.new_ancilla()))
                return 1

            # Blocked/excluded e.p. post-select on the target
            if m.move_variant == enums.MoveVariant.EXCLUDED:
                is_there = self.post_select_on(tqubit)
                if is_there:
                    return 0
                self.add_entangled(tqubit)
            self.circuit.append(
                qm.en_passant(squbit, tqubit, epqubit, self.new_ancilla(),
                              self.new_ancilla()))
            return 1

        if m.move_type == enums.MoveType.PAWN_CAPTURE:
            # For pawn capture, first measure source.
            is_there = self.post_select_on(squbit)
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
                self.state = set_nth_bit(sbit, self.state, 0)
                self.state = set_nth_bit(tbit, self.state, 1)
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
                is_there = self.post_select_on(tqubit)
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
                capture_allowed = self.post_select_on(capture_ancilla)

                if not capture_allowed:
                    return 0
                else:
                    # Perform the captured slide
                    self.add_entangled(squbit)
                    # Remove the target from the board into an ancilla
                    # and set bit to zero
                    self.unhook(tqubit)
                    self.state = set_nth_bit(tbit, self.state, 0)

                    # Re-add target since we need to swap into the square
                    self.add_entangled(tqubit)

                    # Perform the actual move
                    self.circuit.append(qm.normal_move(squbit, tqubit))

                    # Set source to empty
                    self.unhook(squbit)
                    self.state = set_nth_bit(sbit, self.state, 0)

                    # Now set the whole path to empty
                    for p in path_qubits:
                        self.state = set_nth_bit(qubit_to_bit(p), self.state, 0)
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
                self.state = set_nth_bit(sbit, self.state, 0)
                self.state = set_nth_bit(tbit, self.state, 1)
                return 1

            # Measure source for capture
            if m.move_variant == enums.MoveVariant.CAPTURE:
                is_there = self.post_select_on(squbit)
                if not is_there:
                    return 0
                self.unhook(tqubit)

            # Measure target for excluded
            if m.move_variant == enums.MoveVariant.EXCLUDED:
                is_there = self.post_select_on(tqubit)
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
                self.state = set_nth_bit(sbit, self.state, 0)
                self.unhook(squbit)

            return 1

        if m.move_type == enums.MoveType.SPLIT_JUMP:
            tbit2 = square_to_bit(m.target2)
            tqubit2 = bit_to_qubit(tbit2)
            self.add_entangled(squbit, tqubit, tqubit2)
            self.circuit.append(qm.split_move(squbit, tqubit, tqubit2))
            self.state = set_nth_bit(sbit, self.state, 0)
            self.unhook(squbit)
            return 1

        if m.move_type == enums.MoveType.MERGE_JUMP:
            sbit2 = square_to_bit(m.source2)
            squbit2 = bit_to_qubit(sbit2)
            self.add_entangled(squbit, squbit2, tqubit)
            self.circuit.append(qm.merge_move(squbit, squbit2, tqubit))
            # TODO: unhook source qubit?
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
                    not rook_tqubit in self.entangled_squares):
                return 0
            if (nth_bit_of(tbit, self.state) and
                    not tqubit in self.entangled_squares):
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
                castle_allowed = self.post_select_on(castle_ancilla)
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
            is_there = self.post_select_on(measure_qubit)
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
                    not rook_tqubit in self.entangled_squares):
                return 0
            if (nth_bit_of(tbit, self.state) and
                    not tqubit in self.entangled_squares):
                return 0
            if (b_bit is not None and nth_bit_of(b_bit, self.state) and
                    not b_qubit in self.entangled_squares):
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
                castle_allowed = self.post_select_on(castle_ancilla)
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
            is_there = self.post_select_on(measure_qubit)
            if is_there:
                return 0
            if b_qubit not in self.entangled_squares:
                self.set_castle(sbit, rook_sbit, tbit, rook_tbit)
            else:
                self.queenside_castle(squbit, rook_squbit, tqubit, rook_tqubit,
                                      b_qubit)
            return 1

        raise ValueError(f'Move type {m.move_type} not supported')
