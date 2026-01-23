# Copyright 2025 Google
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

"""The magic square game is a quantum parity challenge where two players, Alice and Bob, must fill
rows and columns of a 3x3 grid with values of +1 or -1. Alice is assigned a random
row and Bob a random column, and they win if their shared cell matches while Alice’s row product
is +1 and Bob’s column product is -1. While no classical strategy can guarantee a win every
time, players using shared entangled quantum states can win with 100% certainty by exploiting
non-local correlations. The game serves as a "proof" of quantum contextuality, demonstrating that
the outcome of a measurement depends on the other measurements performed alongside it (its context).
In a classical world, each cell in the square would have a fixed, pre-existing value regardless
of whether it's being measured as part of a row or a column. However, the Mermin-Peres game
proves that these values cannot exist independently of their measurement context,
as the algebraic requirements for the rows and columns are mathematically impossible to satisfy
simultaneously with fixed values.
"""

import dataclasses
from typing import Any, Literal

import cirq
import numpy as np

type GameType = Literal[
    "guess_3rd",
    "measure_3rd_classical_multiplication",
    "measure_3rd_quantum_multiplication",
]


@dataclasses.dataclass
class ContextualityResult:
    """Result of the contextuality warm-up experiment.

    Attributes:
        alice_measurements: Alice's measurements. Shape is (mermin_row_alice, mermin_col_bob,
            repetition, mermin_col).
        bob_measurements: Bob's measurements. Shape is (mermin_row_alice, mermin_col_bob,
            repetition, mermin_row).
    """

    alice_measurements: np.ndarray
    bob_measurements: np.ndarray

    def _select_choices_from_second_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.alice_measurements, self.bob_measurements

    def _generate_choices_from_rules_guess_3rd(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate choices from Alice and Bob's measurements by inferring the third number from the
        first two.

        Returns:
            Alice and Bob's choices in the game.
        """
        repetitions = self.alice_measurements.shape[2]
        alice_choices = np.zeros((3, 3, repetitions, 3), dtype=bool)
        bob_choices = np.zeros((3, 3, repetitions, 3), dtype=bool)

        alice_choices[0, :, :, 0] = self.alice_measurements[0, :, :, 1]
        alice_choices[0, :, :, 1] = self.alice_measurements[0, :, :, 0]
        alice_choices[1, :, :, :2] = self.alice_measurements[1, :, :, :2]
        alice_choices[2, :, :, :2] = 1 - self.alice_measurements[2, :, :, :2]
        bob_choices[:, 0, :, 0] = self.bob_measurements[:, 0, :, 1]
        bob_choices[:, 0, :, 1] = self.bob_measurements[:, 0, :, 0]
        bob_choices[:, 1:, :, :2] = self.bob_measurements[:, 1:, :, :2]

        alice_choices[:, :, :, 2] = np.sum(alice_choices, axis=3) % 2
        bob_choices[:, :, :, 2] = 1 - (np.sum(bob_choices, axis=3) % 2)
        return alice_choices, bob_choices

    def _generate_choices_from_rules_measure_3rd_classical_multiplication(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate choices from Alice and Bob's measurements by measuring
        two one-body observables and making a classical multiplication to get the result

        Returns:
            Alice and Bob's choices in the game.
        """
        repetitions = self.alice_measurements.shape[2]
        alice_choices = np.zeros((3, 3, repetitions, 3), dtype=bool)
        bob_choices = np.zeros((3, 3, repetitions, 3), dtype=bool)

        alice_choices[0, :, :, 0] = self.alice_measurements[0, :, :, 1]
        alice_choices[0, :, :, 1] = self.alice_measurements[0, :, :, 0]
        alice_choices[1, :, :, :2] = self.alice_measurements[1, :, :, :2]
        alice_choices[2, :, :, :2] = 1 - self.alice_measurements[2, :, :, :2]
        bob_choices[:, 0, :, 0] = self.bob_measurements[:, 0, :, 1]
        bob_choices[:, 0, :, 1] = self.bob_measurements[:, 0, :, 0]
        bob_choices[:, 1:, :, :2] = self.bob_measurements[:, 1:, :, :2]

        alice_choices[:, :, :, 2] = 1 - self.alice_measurements[:, :, :, 2]

        bob_choices[:, :, :, 2] = 1 - self.bob_measurements[:, :, :, 2]
        bob_choices[0, 0, :, 2] = self.bob_measurements[0, 0, :, 2]
        bob_choices[0, 1, :, 2] = self.bob_measurements[0, 1, :, 2]
        bob_choices[1, 0, :, 2] = self.bob_measurements[1, 0, :, 2]
        bob_choices[1, 1, :, 2] = self.bob_measurements[1, 1, :, 2]
        bob_choices[2, 0, :, 2] = self.bob_measurements[2, 0, :, 2]
        bob_choices[2, 1, :, 2] = self.bob_measurements[2, 1, :, 2]

        return alice_choices, bob_choices

    def _generate_choices_from_rules_measure_3rd_quantum_multiplication(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate choices from Alice and Bob's measurements by measuring
        one two body operator for the third observable.

        Returns:
            Alice and Bob's choices in the game.
        """
        repetitions = self.alice_measurements.shape[2]
        alice_choices = np.zeros((3, 3, repetitions, 3), dtype=bool)
        bob_choices = np.zeros((3, 3, repetitions, 3), dtype=bool)

        alice_choices[0, :, :, 0] = self.alice_measurements[0, :, :, 1]
        alice_choices[0, :, :, 1] = self.alice_measurements[0, :, :, 0]
        alice_choices[1, :, :, :2] = self.alice_measurements[1, :, :, :2]
        alice_choices[2, :, :, :2] = 1 - self.alice_measurements[2, :, :, :2]
        bob_choices[:, 0, :, 0] = self.bob_measurements[:, 0, :, 1]
        bob_choices[:, 0, :, 1] = self.bob_measurements[:, 0, :, 0]
        bob_choices[:, 1:, :, :2] = self.bob_measurements[:, 1:, :, :2]

        alice_choices[:, :, :, 2] = self.alice_measurements[:, :, :, 2]
        bob_choices[:, 0, :, 2] = 1 - self.bob_measurements[:, 0, :, 2]
        bob_choices[:, 1, :, 2] = 1 - self.bob_measurements[:, 1, :, 2]
        bob_choices[:, 2, :, 2] = self.bob_measurements[:, 2, :, 2]

        return alice_choices, bob_choices

    def generate_choices(
        self,
        game: GameType,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate choices from Alice and Bob's measurements.

        Args:
            game:
                guess_3rd means making two measurements and infering the third bit
                "measure_3rd_classical_multiplication": make 4 measurements for Alice and 4 for Bob.
                get three bits out of them by mutiplying two of them toghether.
                This corresponds to measure A and B to compute A*B.
                "measure_3rd_quantum_multiplication": make 3 measurements for Alice and 3 for Bob.
                Use a two qubit interaction to directly measure A*B.

        Returns:
            Alice and Bob's choices in the game.

        Raises:
            NotImplementedError: If Alice and Bob measure unequal numbers of Paulis.
        """
        # return self._select_choices_from_second_data()
        if game == "guess_3rd":
            return self._generate_choices_from_rules_guess_3rd()
        if game == "measure_3rd_classical_multiplication":
            return self._generate_choices_from_rules_measure_3rd_classical_multiplication()
        if game == "measure_3rd_quantum_multiplication":
            return self._generate_choices_from_rules_measure_3rd_quantum_multiplication()

    def get_win_matrix(self, game: GameType) -> np.ndarray:
        """Find the fraction of the time that Alice and Bob "agree" (in the intersection) given that
        they "multiply correctly" (alice to -1 and bob to +1).

        Args:
            game:
                guess_3rd means making two measurements and infering the third bit
                "measure_3rd_classical_multiplication": make 4 measurements for Alice and 4 for Bob.
                get three bits out of them by mutiplying two of them toghether.
                This corresponds to measure A and B to compute A*B.
                "measure_3rd_quantum_multiplication": make 3 measurements for Alice and 3 for Bob.
                Use a two qubit interaction to directly measure A*B.

        Returns:
            The fraction of the time that they "agree" given that they "multiply correctly".
        """
        alice_choices, bob_choices = self.generate_choices(game)
        win_matrix = np.zeros((3, 3))
        repetitions = alice_choices.shape[2]
        print(f"{repetitions=}")
        for row in range(3):
            for col in range(3):
                # if multiplication rules are not respected, there is no match
                number_of_matches = 0
                for rep in range(repetitions):
                    alice_triad = alice_choices[row, col, rep, :]
                    bob_triad = bob_choices[row, col, rep, :]
                    if np.sum(alice_triad) % 2 == 0 and np.sum(bob_triad) % 2 == 1:
                        number_of_matches += 1
                        if alice_triad[col] == bob_triad[row]:
                            win_matrix[row, col] += 1
                print("Times they both multiply correctly to +-1 =", number_of_matches)
                win_matrix[row, col] = (
                    win_matrix[row, col] / number_of_matches
                )
        return win_matrix

    def get_multiply_matrix(self, game: GameType) -> np.ndarray:
        """Find the fraction of the time that Alice and Bob
        "multiply correctly" (alice to -1 and bob to +1).

        Args:
            game:
                guess_3rd means making two measurements and infering the third bit
                "measure_3rd_classical_multiplication": make 4 measurements for Alice and 4 for Bob.
                get three bits out of them by mutiplying two of them toghether.
                This corresponds to measure A and B to compute A*B.
                "measure_3rd_quantum_multiplication": make 3 measurements for Alice and 3 for Bob.
                Use a two qubit interaction to directly measure A*B.

        Returns:
            The fraction of the time that they "multiply correctly".
        """
        alice_choices, bob_choices = self.generate_choices(game)
        win_matrix = np.zeros((3, 3))
        repetitions = alice_choices.shape[2]
        print(f"{repetitions=}")
        for row in range(3):
            for col in range(3):
                for rep in range(repetitions):
                    alice_triad = alice_choices[row, col, rep, :]
                    bob_triad = bob_choices[row, col, rep, :]
                    if np.sum(alice_triad) % 2 == 0 and np.sum(bob_triad) % 2 == 1:
                        win_matrix[row, col] += 1

                win_matrix[row, col] = win_matrix[row, col] / repetitions
        return win_matrix

    def get_agree_matrix(self, game: GameType) -> np.ndarray:
        """Fraction of the time that Alice and Bob Alice and Bob "agree" (in the intersection).

        Args:
            game:
                "guess_3rd": making two measurements and infering the third bit
                "measure_3rd_classical_multiplication": make 4 measurements for Alice and 4 for Bob.
                get three bits out of them by mutiplying two of them toghether.
                This corresponds to measure A and B to compute A*B.
                "measure_3rd_quantum_multiplication": make 3 measurements for Alice and 3 for Bob.
                Use a two qubit interaction to directly measure A*B.

        Returns:
            The fraction of the time that they "agree".
        """
        alice_choices, bob_choices = self.generate_choices(game)
        win_matrix = np.zeros((3, 3))
        repetitions = alice_choices.shape[2]
        print(f"{repetitions=}")
        for row in range(3):
            for col in range(3):
                for rep in range(repetitions):
                    alice_triad = alice_choices[row, col, rep, :]
                    bob_triad = bob_choices[row, col, rep, :]
                    if alice_triad[col] == bob_triad[row]:
                        win_matrix[row, col] += 1
                win_matrix[row, col] = win_matrix[row, col] / repetitions
        return win_matrix

    def get_agree_and_multiply_matrix(self, game: GameType) -> np.ndarray:
        """Find the fraction of the time that Alice and Bob
        Alice and Bob "agree" (in the intersection)
        and
        they "multiply correctly" (alice to -1 and bob to +1).

        Args:
            game:
                "guess_3rd": making two measurements and infering the third bit
                "measure_3rd_classical_multiplication": make 4 measurements for Alice and 4 for Bob.
                get three bits out of them by mutiplying two of them toghether.
                This corresponds to measure A and B to compute A*B.
                "measure_3rd_quantum_multiplication": make 3 measurements for Alice and 3 for Bob.
                Use a two qubit interaction to directly measure A*B.

        Returns:
            The fraction of the time that they "multiply correctly" AND "agree".
        """
        alice_choices, bob_choices = self.generate_choices(game)
        win_matrix = np.zeros((3, 3))
        repetitions = alice_choices.shape[2]
        print(f"{repetitions=}")
        for row in range(3):
            for col in range(3):
                for rep in range(repetitions):
                    alice_triad = alice_choices[row, col, rep, :]
                    bob_triad = bob_choices[row, col, rep, :]
                    if (
                        alice_triad[col] == bob_triad[row]
                        and np.sum(alice_triad) % 2 == 0
                        and np.sum(bob_triad) % 2 == 1
                    ):
                        win_matrix[row, col] += 1
                win_matrix[row, col] = win_matrix[row, col] / repetitions
        return win_matrix


def run_contextuality_experiment(
    sampler: cirq.Sampler,
    alice_qubits: list[cirq.GridQubit],
    bob_qubits: list[cirq.GridQubit],
    game: GameType,
    sub_case: Literal["square_1", "square_2", "only_two_qubits"],
    repetitions: int = 10_000,
    add_dd: bool = True,
    dd_scheme: tuple[cirq.Gate, ...] = (cirq.X, cirq.Y, cirq.X, cirq.Y),
) -> ContextualityResult:
    """Run the contextuality warm-up experiment.

    Args:
        sampler: The hardware sampler or simulator.
        alice_qubits: Alice's qubits, order is (measure, measure, data, data, measure) or (measure,
            data, data, measure).
        bob_qubits: Bob's qubits, (order is measure, measure, data, data, measure) or (measure,
            data, data, measure).
        repetitions: The number of repetitions for each row and column of the Mermin-Peres square.
        add_dd: Whether to add dynamical decoupling.
        dd_scheme: The dynamical decoupling sequence to use if doing DD.
        game:
                "guess_3rd": making two measurements and infering the third bit
                "measure_3rd_classical_multiplication": make 4 measurements for Alice and 4 for Bob.
                get three bits out of them by mutiplying two of them toghether.
                This corresponds to measure A and B to compute A*B.
                "measure_3rd_quantum_multiplication": make 3 measurements for Alice and 3 for Bob.
                Use a two qubit interaction to directly measure A*B.
        sub_case:
            square1 is the wikipedia square: https://en.wikipedia.org/wiki/Quantum_pseudo-telepathy
            square2 is not implemented, all blocks have 2body observables
            only_two_qubits means with 4q total (2 per player). Can only do guess_3rd

    Returns:
        A ContextualityResult object containing the experiment results.
    """

    all_circuits = []
    for mermin_row in range(3):
        for mermin_col in range(3):
            all_circuits.append(
                construct_contextuality_circuit(
                    alice_qubits,
                    bob_qubits,
                    mermin_row,
                    mermin_col,
                    add_dd,
                    game,
                    sub_case,
                    dd_scheme,
                )
            )
    results = sampler.run_batch(all_circuits, repetitions=repetitions)

    alice_measurements = np.zeros((3, 3, repetitions, 3), dtype=bool)
    bob_measurements = np.zeros((3, 3, repetitions, 3), dtype=bool)
    idx = 0

    for row in range(3):
        for col in range(3):
            if sub_case == "only_two_qubits":
                alice_measurements[row, col, :, :2] = results[idx][0].measurements[
                    "alice_datas"
                ]

                bob_measurements[row, col, :, :2] = results[idx][0].measurements["bob_datas"]
            else:
                alice_measurements[row, col, :, :2] = results[idx][0].measurements["alice"]
                bob_measurements[row, col, :, :2] = results[idx][0].measurements["bob"]

            if game == "measure_3rd_classical_multiplication":
                alice_measurements[row, col, :, 2] = multiply_bool(
                    results[idx][0].measurements["alice_datas"][:, 1],
                    results[idx][0].measurements["alice_datas"][:, 0],
                )
                bob_measurements[row, col, :, 2] = multiply_bool(
                    results[idx][0].measurements["bob_datas"][:, 1],
                    results[idx][0].measurements["bob_datas"][:, 0],
                )
            if game == "measure_3rd_quantum_multiplication":
                alice_measurements[row, col, :, 2] = results[idx][0].measurements[
                    "alice_datas"
                ][:, 0]
                bob_measurements[row, col, :, 2] = results[idx][0].measurements["bob_datas"][
                    :, 0
                ]

            idx += 1

    return ContextualityResult(alice_measurements, bob_measurements)


def multiply_bool(bool_0: list[bool], bool_1: list[bool]) -> list[bool]:
    """Perform boolean multiplication. Useful for "measure_3rd_classical_multiplication".""""
    return [el0 == el1 for el0, el1 in zip(bool_0, bool_1)]


def bell_pair_prep_circuit(q0: cirq.GridQubit, q1: cirq.GridQubit) -> cirq.Circuit:
    """Prepare a Bell state between qubits q0 and q1.

    Args:
        q0: One qubit.
        q1: The other qubit.

    Returns:
        A circuit creating a Bell state.
    """
    return cirq.Circuit(cirq.H.on_each(q0, q1), cirq.CZ(q0, q1), cirq.H(q1))


def state_prep_circuit(
    alice_data_qubits: tuple[cirq.GridQubit, cirq.GridQubit],
    bob_data_qubits: tuple[cirq.GridQubit, cirq.GridQubit],
) -> cirq.Circuit:
    """Construct a circuit to produce the initial Bell pairs.

    Args:
        alice_data_qubits: Alice's data qubits.
        bob_data_qubits: Bob's data qubits.

    Returns:
        A circuit preparing the Bell states.
    """
    pairs = ((alice_data_qubits[0], bob_data_qubits[0]), (alice_data_qubits[1], bob_data_qubits[1]))
    return bell_pair_prep_circuit(*pairs[0]).zip(bell_pair_prep_circuit(*pairs[1]))


def construct_measure_circuit(
    alice_qubits: list[cirq.GridQubit],
    bob_qubits: list[cirq.GridQubit],
    mermin_row: int,
    mermin_col: int,
    game: GameType,
    sub_case: Literal["square_1", "square_2", "only_two_qubits"],
) -> cirq.Circuit:
    """Construct a circuit to implement the measurement.

    We use the conventions described in https://en.wikipedia.org/wiki/Quantum_pseudo-telepathy.
    In particular, the Mermin-Peres table is

     I ⊗ Z | Z ⊗ I | Z ⊗ Z
     X ⊗ I | I ⊗ X | X ⊗ X
    -X ⊗ Z |-Z ⊗ X | Y ⊗ Y

    Args:
        alice_qubits: The line of qubits to use for Alice (measure, data, data, measure).
        bob_qubits: The line of qubits to use for Bob (measure, data, data, measure).
        mermin_row: The row of the Mermin-Peres square to measure.
        mermin_col: The column of the Mermin-Peres square to measure.
        game:
            "guess_3rd": making two measurements and infering the third bit
            "measure_3rd_classical_multiplication": make 4 measurements for Alice and 4 for Bob.
            get three bits out of them by mutiplying two of them toghether.
            This corresponds to measure A and B to compute A*B.
            "measure_3rd_quantum_multiplication": make 3 measurements for Alice and 3 for Bob.
            Use a two qubit interaction to directly measure A*B.
        sub_case:
            square1 is the wikipedia square: https://en.wikipedia.org/wiki/Quantum_pseudo-telepathy
            square2 is not implemented, all blocks have 2body observables
            only_two_qubits means with 4q total (2 per player). Can only do guess_3rd

    Returns:
        A circuit implementing the measurement.
    """

    q = alice_qubits[1:3]  # data qubits
    m = (alice_qubits[0], alice_qubits[3])  # measure qubits
    if mermin_row == 0:
        if sub_case == "square_1":
            alice_circuit = cirq.Circuit(
                # map datas into two measures to measure I ⊗ Z  and Z ⊗ I on them
                cirq.H.on_each(*m),
                cirq.CZ.on_each(*zip(m, q)),
                cirq.H.on_each(*m),
                # map the two datas onto the second data
                cirq.Moment(cirq.M(*m, key="alice")),
            )

            if game == "guess_3rd":
                pass
            elif game == "measure_3rd_classical_multiplication":
                alice_circuit.append(cirq.M(*q, key="alice_datas"))

            elif game == "measure_3rd_quantum_multiplication":
                alice_circuit.append([
                    cirq.H.on(q[1]),
                    cirq.CZ.on(*q),
                    cirq.H.on(q[1]),
                    cirq.M(q[1], key="alice_datas"),
                ])

        if sub_case == "only_two_qubits":
            alice_circuit = cirq.Circuit(cirq.Moment(cirq.M(*q, key="alice_datas")))

            if game == "guess_3rd":
                pass
            else:
                raise ValueError("You can only game = guess_3rd if you sub_case = only_two_qubits")

    elif mermin_row == 1:
        if sub_case == "square_1":
            alice_circuit = cirq.Circuit(
                cirq.H.on_each(*q, *m),
                cirq.CZ.on_each(*zip(q, m)),
                cirq.H.on_each(*m),
                cirq.Moment(cirq.M(*m, key="alice")),
            )

            if game == "guess_3rd":
                pass

            elif game == "measure_3rd_classical_multiplication":
                alice_circuit.append(cirq.M(*q, key="alice_datas"))

            elif game == "measure_3rd_quantum_multiplication":
                alice_circuit.append([
                    cirq.H.on(q[1]),
                    cirq.CZ.on(*q),
                    cirq.H.on(q[1]),
                    cirq.M(q[1], key="alice_datas"),
                ])

        if sub_case == "only_two_qubits":
            alice_circuit = cirq.Circuit(
                cirq.Moment(cirq.H.on_each(*q)), cirq.Moment(cirq.M(*q, key="alice_datas"))
            )

            if game == "guess_3rd":
                pass
            else:
                raise ValueError("You can only game = guess_3rd if you sub_case = only_two_qubits")

    elif mermin_row == 2:
        if sub_case == "square_1":
            alice_circuit = cirq.Circuit(
                cirq.CZ(*q),
                cirq.Moment(cirq.H.on_each(*q, *m)),
                cirq.CZ.on_each(*zip(m, q)),
                cirq.H.on_each(*m),
                cirq.Moment(cirq.H.on_each(*q)),
                cirq.CZ(*q),
                cirq.Moment(cirq.M(*m, key="alice")),
            )

            if game == "guess_3rd":
                pass
            elif game == "measure_3rd_classical_multiplication":
                alice_circuit.append([
                    cirq.Rx(rads=np.pi / 2).on_each(*q),
                    cirq.M(*q, key="alice_datas"),
                ])


            elif game == "measure_3rd_quantum_multiplication":
                alice_circuit.append([
                    cirq.Rx(rads=np.pi / 2).on_each(*q),
                    cirq.H.on(q[1]),
                    cirq.CZ.on(*q),
                    cirq.H.on(q[1]),
                    cirq.M(q[1], key="alice_datas"),
                ])


        if sub_case == "only_two_qubits":
            alice_circuit = cirq.Circuit(
                cirq.ry(np.pi / 2)(q[0]),
                cirq.H.on(q[0]),
                cirq.CZ.on(*q),
                cirq.Moment(cirq.H.on_each(*q)),
                cirq.X.on(q[0]),
                cirq.Moment(cirq.M(*q, key="alice_datas")),
            )

            if game == "guess_3rd":
                pass
            else:
                raise ValueError("You can only game = guess_3rd if you sub_case = only_two_qubits")

    q = bob_qubits[1:3]  # data qubits
    m = (bob_qubits[0], bob_qubits[3])  # measure qubits
    if mermin_col == 0:
        if sub_case == "square_1":
            bob_circuit = cirq.Circuit(
                # map datas onto measures to measure I ⊗ Z and X ⊗ I on them
                cirq.H.on_each(*m, q[0]),
                cirq.CZ.on_each(*zip(m, q)),
                cirq.H.on_each(*m),
                cirq.Moment(cirq.M(*m, key="bob")),
            )

            if game == "guess_3rd":
                pass
            elif game == "measure_3rd_classical_multiplication":
                bob_circuit.append(cirq.M(*q, key="bob_datas"))


            elif game == "measure_3rd_quantum_multiplication":
                bob_circuit.append([
                    cirq.H.on(q[1]),
                    cirq.CZ.on(*q),
                    cirq.H.on(q[1]),
                    cirq.M(q[1], key="bob_datas"),
                ])

        if sub_case == "only_two_qubits":
            bob_circuit = cirq.Circuit(cirq.H.on(q[0]), cirq.Moment(cirq.M(*q, key="bob_datas")))

            if game == "guess_3rd":
                pass
            else:
                raise ValueError("You can only game = guess_3rd if you sub_case = only_two_qubits")

    elif mermin_col == 1:
        if sub_case == "square_1":
            bob_circuit = cirq.Circuit(
                # map datas onto measures to measure Z ⊗ I and I ⊗ X on them
                cirq.H.on_each(*m, q[1]),
                cirq.CZ.on_each(*zip(m, q)),
                cirq.H.on_each(*m),
                cirq.Moment(cirq.M(*m, key="bob")),
            )

            if game == "guess_3rd":
                pass
            elif game == "measure_3rd_classical_multiplication":
                bob_circuit.append(cirq.M(*q, key="bob_datas"))

            elif game == "measure_3rd_quantum_multiplication":
                bob_circuit.append([
                    cirq.H.on(q[1]),
                    cirq.CZ.on(*q),
                    cirq.H.on(q[1]),
                    cirq.M(q[1], key="bob_datas"),
                ])


        if sub_case == "only_two_qubits":
            bob_circuit = cirq.Circuit(cirq.H.on(q[1]), cirq.Moment(cirq.M(*q, key="bob_datas")))

            if game == "guess_3rd":
                pass
            else:
                raise ValueError("You can only game = guess_3rd if you sub_case = only_two_qubits")

    elif mermin_col == 2:
        if sub_case == "square_1":
            bob_circuit = cirq.Circuit(
                cirq.H(q[0]),
                cirq.CZ(*q),
                cirq.Moment(cirq.H.on_each(*q, *m)),
                cirq.CZ.on_each(*zip(m, q)),
                cirq.H.on_each(*m),
                # re-add Eliott's dropped to circuit
                cirq.Moment(cirq.H.on_each(*q)),
                cirq.CZ(*q),
                cirq.H(q[0]),
                cirq.Moment(cirq.M(*m, key="bob")),
            )

            if game == "guess_3rd":
                pass
            elif game == "measure_3rd_classical_multiplication":
                bob_circuit.append([
                    cirq.Rx(rads=np.pi / 2).on_each(*q),
                    cirq.M(*q, key="bob_datas"),
                ])

            elif game == "measure_3rd_quantum_multiplication":
                bob_circuit.append([
                    cirq.Rx(rads=np.pi / 2).on_each(*q),
                    cirq.H.on(q[1]),
                    cirq.CZ.on(*q),
                    cirq.H.on(q[1]),
                    cirq.M(q[1], key="bob_datas"),
                ])



        if sub_case == "only_two_qubits":
            bob_circuit = cirq.Circuit(
                cirq.H.on(q[0]),
                cirq.CZ.on(*q),
                cirq.H.on_each(*q),
                cirq.Moment(cirq.M(*q, key="bob_datas")),
            )



            if game == "guess_3rd":
                pass
            else:
                raise ValueError("You can only sub_case = only_two_qubits if game = guess_3rd")
    return cirq.align_right(alice_circuit + bob_circuit)

def construct_contextuality_circuit(
    alice_qubits: list[cirq.GridQubit],
    bob_qubits: list[cirq.GridQubit],
    mermin_row: int,
    mermin_col: int,
    add_dd: bool,
    game: GameType,
    sub_case: Literal["square_1", "square_2", "only_two_qubits"],
    dd_scheme: tuple[cirq.Gate, ...] = (cirq.X, cirq.Y, cirq.X, cirq.Y),
) -> cirq.Circuit:
    """Construct a circuit to implement the contextuality experiment on hardware.

    We use the conventions described in https://en.wikipedia.org/wiki/Quantum_pseudo-telepathy.
    In particular, the Mermin-Peres table is

     I ⊗ Z | Z ⊗ I | Z ⊗ Z
     X ⊗ I | I ⊗ X | X ⊗ X
    -X ⊗ Z |-Z ⊗ X | Y ⊗ Y

    Args:
        alice_qubits: The line of qubits to use for Alice (measure, data, data, measure).
        bob_qubits: The line of qubits to use for Bob (measure, data, data, measure).
        mermin_row: The row of the Mermin-Peres square to measure.
        mermin_col: The column of the Mermin-Peres square to measure.
        add_dd: Whether to add dynamical decoupling.
        dd_scheme: The dynamical decoupling sequence to use if doing DD.
        game:
            "guess_3rd": making two measurements and infering the third bit
            "measure_3rd_classical_multiplication": make 4 measurements for Alice and 4 for Bob.
            get three bits out of them by mutiplying two of them toghether.
            This corresponds to measure A and B to compute A*B.
            "measure_3rd_quantum_multiplication": make 3 measurements for Alice and 3 for Bob.
            Use a two qubit interaction to directly measure A*B.

    Returns:
        A circuit implementing the game.
    """
    alice_data_qubits = (alice_qubits[1], alice_qubits[2])
    bob_data_qubits = (bob_qubits[1], bob_qubits[2])
    prep_circuit = state_prep_circuit(alice_data_qubits, bob_data_qubits)  # test cirq.Circuit() #
    measure_circuit = construct_measure_circuit(
        alice_qubits, bob_qubits, mermin_row, mermin_col, game, sub_case
    )
    circuit = prep_circuit + measure_circuit
    if add_dd:
        circuit = cirq.add_dynamical_decoupling(
            circuit, single_qubit_gate_moments_only=True, schema=dd_scheme
        )
    return circuit
