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

from __future__ import annotations
import warnings
import collections
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Literal

import attrs
import cirq
import matplotlib.pyplot as plt
import numpy as np

Axis = Literal["row", "col"]
QndStrategy = Literal["map_all_paulis_to_ancilla", "map_2q_paulis_to_ancilla"]
CircuitStyle = Literal["flat", "jt", "jt_pauli", "jt_pauli_pivot"]


_CIRCUIT_DD_SUPPORTED = hasattr(cirq, "add_dynamical_decoupling")

if _CIRCUIT_DD_SUPPORTED:
    _ADD_DD_FUNC = cirq.add_dynamical_decoupling
else:
    _ADD_DD_FUNC = None


@attrs.frozen
class MagicSquare:
    """Represents a MagicSquare.

    A MagicSquare is a 3x3 grid of two-qubit Pauli operators used to demonstrate quantum
    contextuality, specifically the Kochen-Specker theorem.

    The operators in each row and each column mutually commute.
    The product of operators in each row is +I.
    The product of operators in each column is -I.

    Attributes:
        paulis: A 3x3 sequence of strings representing the Pauli operators in the square.
    """

    paulis: Sequence[Sequence[str]]

    def build_circuit_for_context(
        self,
        context: tuple[Axis, int],
        trios: Sequence[tuple[cirq.Qid, cirq.Qid, cirq.Qid]],
        qnd_strategy: QndStrategy = "map_all_paulis_to_ancilla",
        virtual_z_half_turns: float = 0.0,
        add_ancilla_to_measure_key: bool = False,
    ) -> cirq.FrozenCircuit:
        """Circuit to measure the idx'th col or row of this magic square.

        Args:
            context: Tuple of (Axis, idx). Axis corresponds to either a row or a column, while idx
                determines which row or col to measure from the Mermin-Peres square.
            trios: Sequence of three-element tuples of the form (q0, q1, ancilla).
            qnd_strategy: Whether to use ancillae for measuring single-qubit paulis.
            virtual_z_half_turns: This applied only for Paulis mapped to an ancilla. If not zero,
                amount of half turns around the Z axis to apply on the ancilla during the mapping of
                the paulis from the data qubits (q0, q1). This virtualZ is applied before the last
                Hadamard during the mapping of the Pauli to the ancilla.
            add_ancilla_to_measure_key: Whether to include the ancilla qubit name in the measure
                key or not.
        """
        axis, idx = context
        if axis == "row":
            paulis = [self.paulis[idx][c] for c in range(3)]
        else:
            paulis = [self.paulis[r][idx] for r in range(3)]
        return cirq.FrozenCircuit(
            cirq.FrozenCircuit.zip(
                *[
                    generate_qnd_pauli_circuit(
                        q0,
                        q1,
                        anc,
                        pauli,
                        measure_key=(
                            f"{pauli}_{axis}_{anc}"
                            if add_ancilla_to_measure_key
                            else f"{pauli}_{axis}"
                        ),
                        qnd_strategy=qnd_strategy,
                        virtual_z_half_turns=virtual_z_half_turns,
                    )
                    for q0, q1, anc in trios
                ]
            )
            for pauli in paulis
        )

    def kochen_specker_expt(
        self,
        q0: cirq.Qid,
        q1: cirq.Qid,
        anc: cirq.Qid,
        num_contexts: int = 36,
        seed: int = 0,
        qnd_strategy: QndStrategy = "map_all_paulis_to_ancilla",
        virtual_z_half_turns: float = 0.0,
    ) -> MagicSquareExperiment:
        """Builds a Kochen-Specker experiment, which measures state-independent contextuality.

        See https://arxiv.org/pdf/2512.02284 for more details.

        Args:
            q0: First data qubit.
            q1: Second data qubit.
            anc: Ancilla qubit.
            num_contexts: Number of rows+cols measured out of the 6
                possible row/col.
            seed: Seed for random number generator. Specifying this will give
                reproducible results for the random axes and indices.
            qnd_strategy: Whether to map all paulis to an ancilla, or only 2q paulis.
            virtual_z_half_turns: This applied only for Paulis mapped to an ancilla. If not zero,
                amount of half turns around the Z axis to apply on the ancilla during the mapping of
                the paulis from the data qubits (q0, q1). This virtualZ is applied before the last
                Hadamard during the mapping of the Pauli to the ancilla.

        Returns:
            A `MagicSquareExperiment` object containing the constructed circuit and context info.
        """

        axes: list[Axis] = ["col", "row"]
        indices = [0, 1, 2]

        random.seed(seed)
        random_axes = random.choices(axes, k=num_contexts)
        random_idx = random.choices(indices, k=num_contexts)

        context_to_circuit: dict[tuple[str, int], cirq.FrozenCircuit] = {
            (axis, idx): self.build_circuit_for_context(
                trios=[(q0, q1, anc)],
                context=(axis, idx),
                qnd_strategy=qnd_strategy,
                virtual_z_half_turns=virtual_z_half_turns,
            )
            for axis in axes
            for idx in indices
        }

        def pauli_key(row: int, col: int, axis: Axis) -> PauliKey:
            return PauliKey(f"{self.paulis[row][col]}_{axis}")

        contexts: dict[tuple[Axis, int], Sequence[PauliKey]] = {
            **{
                ("row", r): [pauli_key(r, c, "row") for c in range(3)] for r in range(3)
            },
            **{
                ("col", c): [pauli_key(r, c, "col") for r in range(3)] for c in range(3)
            },
        }

        ks_circuits = [
            context_to_circuit[axis, idx] for axis, idx in zip(random_axes, random_idx)
        ]
        circuit = transform_circuit_for_jump_table(
            ks_circuits=ks_circuits,
        )
        return MagicSquareExperiment(self, circuit, contexts)


EASY_SQUARE = MagicSquare(
    [["1Z", "Z1", "ZZ"], ["X1", "1X", "XX"], ["-XZ", "-ZX", "YY"]]
)
HARD_SQUARE = MagicSquare([["ZX", "YY", "XZ"], ["XY", "ZZ", "YX"], ["YZ", "XX", "ZY"]])


def transform_circuit_for_jump_table(
    ks_circuits: list[cirq.FrozenCircuit],
) -> cirq.FrozenCircuit:
    """Transform the circuit.

    Args:
        ks_circuits: Sequence of circuits to measure a sequence of contexts.
    """
    return cirq.FrozenCircuit(ks_circuits)


def _copy_circuit(circuit: cirq.FrozenCircuit) -> cirq.FrozenCircuit:
    return cirq.FrozenCircuit.from_moments(*circuit).with_tags(*circuit.tags)


@attrs.frozen
class MagicSquareExperiment:
    """An experiment to test for contextuality using a magic square.

    Attributes:
        square: The `MagicSquare` instance used for the experiment.
        circuit: The `cirq.FrozenCircuit` that will be run. This circuit
            is composed of measurements for a sequence of randomly chosen
            contexts (rows or columns of the magic square).
        contexts: A dictionary mapping each possible context (row or column) to
            the sequence of `PauliKey`s that identify the measurements for that
            context.
    """

    square: MagicSquare
    circuit: cirq.FrozenCircuit
    contexts: dict[tuple[Axis, int], Sequence[PauliKey]]

    def run(self, sampler: cirq.Sampler, repetitions: int = 1000) -> MagicSquareResult:
        """Runs the experiment circuit on a sampler.

        Args:
            sampler: The `cirq.Sampler` to run the experiment on.
            repetitions: The number of times to repeat the circuit.

        Returns:
            A `MagicSquareResult` object containing the results of the experiment.
        """
        result = sampler.run(self.circuit, repetitions=repetitions)
        return MagicSquareResult(self, result)


@attrs.frozen
class PauliKey:
    """A key to identify a specific Pauli measurement in a circuit.

    Attributes:
        key: The measurement key string for the Pauli measurement.
        slice: A slice or list of indices to select specific measurement instances
            if the key appears multiple times in a circuit for a given context.
            Defaults to `slice(None)` to select all instances.
    """

    key: str
    slice: list[int] | slice = slice(None)


@attrs.frozen(slots=False)
class MagicSquareResult:
    """The results of a `MagicSquareExperiment`.

    This class holds the `cirq.Result` from running the experiment and provides
    methods to analyze the results, such as estimating the Kochen-Specker value (chi)
    and other statistical properties.

    Attributes:
        expt: The `MagicSquareExperiment` that was run.
        result: The `cirq.Result` object from the sampler.
    """

    expt: MagicSquareExperiment
    result: cirq.Result

    def estimate_chi(self) -> tuple[float, float]:
        """Estimate chi out of the 6 parity averages.

        This method takes the cirq.Result output by the Kochen-Specker
        experiment and returns an estimate of chi. We assume that the length of the
        circuit ran was long enough to contain at least one instance of each of the
        six 3-observable parities, e.g. (1X X1 XX).
        """

        chi_mean, chi_error = estimate_chi(self.squeezed_records(), self.expt.contexts)

        return chi_mean, chi_error

    def squeezed_records(self) -> Mapping[str, np.ndarray]:
        """Squeeze the ResultDict.records.values() ndarrays to 2D.

        Since there is only one qubit per measurement key in our experiments, the third index is
        a dummy index. This helps us process data more easily.
        """
        squeezed_records = {
            pauli_key: np.squeeze(records_values)
            for pauli_key, records_values in self.result.records.items()
        }
        return squeezed_records

    def estimate_parities(self) -> dict[tuple[Axis, int], tuple[float, float]]:
        """Estimate average parity and error of each of the 6 contexts.

        Useful to get more nuanced information on each row/col. We first evaluate the parity for
        each time every context has appeared in the circuit. We then average the parities for
        each context and estimate the error of this average. For references on the error estimation,
        check the error of a binomial distribution.

        Returns:
            context_averages: A dictionary mapping a context (axis, idx) to a tuple
            (mean_parity, parity_error)
        """
        context_averages = estimate_parities(
            self.squeezed_records(), self.expt.contexts
        )
        return context_averages

    def estimate_coincidences(
        self,
    ) -> tuple[
        dict[str, list[float]], dict[str, list[float]], dict[str, list[tuple[str, str]]]
    ]:
        """Each PauliString, get coincidence probability for compatible/incompatible measurements.

        A compatible pair of measurements of a given PauliString P_i (i = 0, 1, ..., 8), say ZZ,
        is a pair of outcomes of ZZ where all other PauliStrings P_j (j != i) measured in between
        these two outcomes are compatible with ZZ. Essentially, this happens when each and every
        P_j commutes with the P_i in question. An incompatible pair of measurements of a
        PauliString P_i happens when in between a pair of outcomes of P_i, there is at least one
        P_k that does not commute with P_i, scrambling its measurement outcome.

        For example, say the order of the contexts measured (i.e. measurement dna) of the given
        circuit is, in the EASY_SQUARE:

        dna:
         ('1Z_row', 'Z1_row', 'ZZ_row') - First context
         ('ZZ_col', 'XX_col', 'YY_col') - Second context
         ('X1_row', '1X_row', 'XX_row') - Third context
         ('ZZ_col', 'XX_col', 'YY_col') - Fourth context
         ('ZZ_col', 'XX_col', 'YY_col') - Fifth context
         ('X1_row', '1X_row', 'XX_row') - Sixth context
         ('ZZ_col', 'XX_col', 'YY_col') - Seventh context

         ...

        When considering the PauliString ZZ, a compatible pair of ZZ measurements happens between
        (first, second) contexts and between (fourth, fifth) contexts, since ZZ commutes with
        every element of each tuple of contexts. Nonetheless, pair of ZZ measurements that
        happens between (second, fourth) contexts and (fifth, seventh) is an incompatible pair of
        measurements of ZZ. For example, since the third context does not contain a ZZ
        measurement and no element in that context commutes with ZZ, measuring that third
        context will scramble the measurement outcome of ZZ once we measure this PauliString in the
        fourth context. Same happens when measuring the sixth context.

        In theory, for compatible measurements of a given PauliString, we expect the measurement
        outcomes to have a probability of coincidence P=1, while for incompatible measurements
        we expect the probability of coincidence P=0.5 (since there are measurements in between
        that do not commute with such Pauli, so its outcome gets scrambled).

        Args:
            circuit: a cirq.FrozenCircuit. Need to pass a circuit without using the jump table
                (flatten=True).

        Returns:
            - pauli_to_compatible_coincidences: dict
            - pauli_to_incompatible_coincidences: dict
            - pauli_to_compatible_contexts: dict

            Explanations:
            - pauli_to_compatible_coincidences and pauli_to_incompatible_coincidences:
                Two dictionaries. They map each PauliString in the MagicSquare to a list
                of 0's and 1's for compatible and incompatible pairs of measurements, respectively.
                    1: indicates coincidence between the pair of measurements.
                    0: indicates no coincidence between the pair of measurements.

                Given the dna of PauliStrings presented above, ZZ was measured 5 times in the
                circuit. The pairs contexts that contain ZZ are (1st, 2nd), (2nd, 4th), (4th, 5th),
                and (5th, 7th). Say that the coincidences of each pair gives the array

                coincidences_array = [1, 0, 1, 1].

                We know that, in this circuit dna, the odd elements correspond to compatible
                contexts [(1st, 2nd), (4th, 5th)], while the even elements correspond to
                incompatible contexts [(2nd, 4th), (5th, 7th)]. Then the values for the ZZ key in
                each dictionary would be:

                pauli_to_compatible_coincidences[ZZ] = [1, 1]
                pauli_to_incompatible_coincidences[ZZ] = [0, 1]

            - pauli_to_compatible_contexts:
                This dictionary is useful for debugging purposes.
                Each PauliString can show up in a circuit as part of a context that
                corresponds to either a row or a column. For the compatible pairs of
                measurements in a Kochen Specker circuit (and only the compatible pairs),
                this dictionary is a mapping between each PauliString in the MagicSquare to a
                list of tuples containing whether each tuples of previous context, where each
                tuple contains the axis context corresponds to a row or column context. For the
                example we are analysing, this would give:

                pauli_to_compatible_contexts[ZZ] = [('row', 'col'), ('col', 'col')]
        """
        circuit = self.expt.circuit
        repetitions = self.result.repetitions
        dna = get_paulis_dna_for_circuit(circuit)
        paulis_set = set(dna)  # length = 18
        paulis = [pauli for row in self.expt.square.paulis for pauli in row]

        # Map each pauli_context, e.g. 1Z_row, to a list of N iterators. N = repetitions and
        # each nested list has length equal to the number of times such 1Z_row shows up in the
        # circuit.
        pauli_context_outcomes = {
            pauli_context: [
                iter(self.squeezed_records()[pauli_context][repetition].flatten())
                for repetition in range(repetitions)
            ]
            for pauli_context in paulis_set
        }

        # Build a dictionary pauli_outcomes that maps pauliStrings (e.g. ZZ) to a 2D list of
        # lists of shape(repetitions, num_measurements). The elements of the nested lists have the
        # outcomes (0's and 1's) of such pauli in the order they were measured and recorded in
        # the circuit.

        # Create this map once before the pauli_outcomes loop
        pauli_to_dna_indices = collections.defaultdict(list)
        for i, op in enumerate(dna):
            pauli = op[:-4]  # e.g., 'ZZ_row' -> 'ZZ'
            pauli_to_dna_indices[pauli].append(i)

        pauli_outcomes: dict[str, list[list[int]]] = {
            pauli: [
                [
                    next(pauli_context_outcomes[dna[i]][rep])
                    for i in pauli_to_dna_indices[pauli]
                ]
                for rep in range(repetitions)
            ]
            for pauli in paulis
        }

        pauli_average_coincidences = {}
        for pauli, outcome_lists in pauli_outcomes.items():
            # Convert to a 2D NumPy array of shape (repetitions, num_measurements)
            outcomes_array = np.array(outcome_lists, dtype=np.int8)

            # Vectorized comparison: compare each column with the next one
            # This creates a boolean array of shape (repetitions, num_measurements - 1)
            coincidences_array = outcomes_array[:, :-1] == outcomes_array[:, 1:]

            # Calculate the mean directly along the repetitions axis (axis=0)
            pauli_average_coincidences[pauli] = np.mean(coincidences_array, axis=0)

        # pauli_compatible_mask maps paulis to a list of True's and False's.
        # True in the indices when such pauli was also measured in the prev dna triplet (compatible)
        # False otherwise (incompatible).

        # We expect the True's indices to correlate with the indices in pauli_coincidences with
        # probability == 1
        # We expect the False's indices to correlate with the indices in pauli_coincidences with
        # probability == 0.5
        pauli_compatible_mask: dict[str, list[bool]] = {}

        # Maps paulis to a list of tuples of contexts, e.g. ("row", "row") or ("col",
        # "row). There are 4 different combinations of tuples of contexts. Will be useful to debug
        # coincidence errors within compatible measurements.
        pauli_to_compatible_contexts: dict[str, list[tuple[str, str]]] = {}

        prev_dna_triplet: list[str] = ["", "", ""]
        prev_context_axis: str = ""
        contexts_are_compatible = False
        for ii in range(int(len(dna) / 3)):
            current_dna_triplet = dna[(ii * 3) : (ii * 3 + 3)]
            current_context_axis = dna[(ii * 3)][-3:]  # e.g. "row" or "col"
            for dna_elem in current_dna_triplet:
                pauli = dna_elem[:-4]
                # when it first appears, just create the lists
                if pauli not in pauli_compatible_mask.keys():
                    pauli_compatible_mask[pauli] = []
                    pauli_to_compatible_contexts[pauli] = []
                else:  # if list already exist, compare with previous_triplet
                    contexts_are_compatible = any(
                        pauli in prev_pauli for prev_pauli in prev_dna_triplet
                    )
                    pauli_compatible_mask[pauli].append(contexts_are_compatible)
                if contexts_are_compatible:
                    pauli_to_compatible_contexts[pauli].append(
                        (
                            prev_context_axis,
                            current_context_axis,
                        )
                    )
            prev_dna_triplet = dna[(ii * 3) : (ii * 3 + 3)]
            prev_context_axis = dna[(ii * 3)][-3:]

        # Build compatible coincidences and incompatible coincidences dictionaries
        pauli_compatible = {
            pauli: pauli_average_coincidences[pauli][pauli_compatible_mask[pauli]]
            for pauli in paulis
        }

        pauli_incompatible = {
            pauli: pauli_average_coincidences[pauli][
                [not elem for elem in pauli_compatible_mask[pauli]]
            ]
            for pauli in paulis
        }

        return pauli_compatible, pauli_incompatible, pauli_to_compatible_contexts

    def get_pauli_coincidences_per_context(
        self,
    ) -> dict[str, dict[tuple[str, str], list[float]]]:
        """Nested mapping, Paulis to compatible measurement context pairs to coincidence probs.

        See the docstrings of the third dictionary returned by self.estimate_coincidences for
        more info on the meaning of "compatible measurement context pairs". An example of an
        entry of the dictionary that self.get_pauli_coincidences_per_context returns is:

        {ZZ: {(row, row): 0.98, (row, col): 0.97, (col, row): 0.93, (col, col): 0.95}.

        Useful for debugging purposes.
        """
        pauli_compatible_per_context: dict[str, dict[tuple[str, str], list[float]]] = {}
        pauli_compatible, _, pauli_to_compatible_contexts = self.estimate_coincidences()

        for pauli in pauli_compatible:
            for probability, context_tuple in zip(
                pauli_compatible[pauli], pauli_to_compatible_contexts[pauli]
            ):
                pauli_compatible_per_context.setdefault(pauli, {}).setdefault(
                    context_tuple, []
                ).append(probability)

        return pauli_compatible_per_context

    def plot_coincidences_timeline(self, include_legend: bool = True) -> None:
        """Plot coincidences coincidences over time.

        The coincidences over time show the evolution of the coincidence probability over time,
        with time ticks given by the position of each Pauli along the circuit.
        """
        pauli_compatible, pauli_incompatible, _ = self.estimate_coincidences()
        paulis = [pauli for row in self.expt.square.paulis for pauli in row]

        # Plot coincidence probabilities over time
        f, axs = plt.subplots(1, 2)

        for pauli in paulis:
            axs[0].plot(pauli_compatible[pauli], label=pauli, marker="o")
            axs[1].plot(pauli_incompatible[pauli], label=pauli, marker="o")

        for ax in axs:
            ax.set_ylim(0, 1.1)
            if include_legend:
                ax.legend()
            ax.set_xlabel("Pauli iteration within circuit")
            ax.set_ylabel("Probability of coincidence")

        axs[0].set_title("Compatible contexts")
        axs[1].set_title("Incompatible contexts")
        f.tight_layout()

    def plot_coincidences_histograms_per_pauli(self) -> None:
        """Plots histograms for each Pauli measurement in pauli_compatible dict."""
        pauli_compatible, _, _ = self.estimate_coincidences()

        keys = list(pauli_compatible.keys())
        num_keys = len(keys)

        # Calculate number of rows and columns for a roughly square grid
        num_cols = 3
        num_rows = int(np.ceil(num_keys / num_cols))

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4)
        )
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

        for i, key in enumerate(keys):
            ax = axes[i]
            data = pauli_compatible[key]

            ax.hist(data, bins=20, edgecolor="black", alpha=0.7)
            ax.set_title(f"Pauli: {key}")
            ax.set_xlabel("P_agree")
            ax.set_ylabel("Counts")
            ax.grid(axis="y", alpha=0.75)

        # Hide any unused subplots
        for j in range(num_keys, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

    def plot_coincidences_histogram_combined_paulis(
        self, ax: plt.Axes | None = None
    ) -> None:
        """Plots a single histogram combining all Pauli measurements from pauli_compatible."""
        pauli_compatible, _, _ = self.estimate_coincidences()
        all_data = np.concatenate(list(pauli_compatible.values()))
        if ax is None:
            f, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_data, bins=20, edgecolor="black", alpha=0.7)
        ax.set_title("Combined Pauli Coincidence Rates")
        ax.set_xlabel("P_agree")
        ax.set_ylabel("Counts")
        ax.grid(axis="y", alpha=0.75)


def get_paulis_dna_for_circuit(circuit: cirq.FrozenCircuit) -> list[str]:
    """Extract the string of all Paulis measured in the circuit in the order they appear.

    Returns:
        paulis_dna: list of strings (Paulis).
            For example:
            dna = [
                    "1Z_row", "Z1_row", "ZZ_row",
                    "X1_row", "1X_row", "XX_row",
                    ...
                    "ZZ_col", "XX_col", "YY_col"
                ]
    """

    def meas_keys(circuit: cirq.FrozenCircuit) -> Iterator[str]:
        """Yields all non-circuit operations from a possibly nested circuit.

        Recurses into subcircuits to yield their non-circuit operations.
        """
        for moment in circuit.moments:
            if len(moment) > 1 and any(
                isinstance(op.untagged, cirq.CircuitOperation) for op in moment
            ):
                raise ValueError(
                    f"CircuitOperation must be the only operation in a moment: {moment}"
                )
            for op in moment:
                if isinstance(circuit_op := op.untagged, cirq.CircuitOperation):
                    if (reps := circuit_op.repetitions) == 1:
                        yield from meas_keys(circuit_op.circuit)
                    else:
                        yield from list(meas_keys(circuit_op.circuit)) * reps
                elif cirq.is_measurement(op):
                    yield cirq.measurement_key_name(op)

    return list(meas_keys(circuit))


def get_rot_gate_from_pauli(pauli: cirq.Gate | str) -> tuple[cirq.Gate, cirq.Gate]:
    """Get the change of basis rotation and its inverse to measure a given Pauli. If the Pauli
    doesn't require a basis rotation we add a WaitLike gate with the duration of cirq.rx to
    guarantee that each context has the same length.

    Args:
        pauli: Which Pauli we want to measure

    Returns:
        The pair of gates to perform the rotation and its inverse.

    Raises:
        ValueError: If Pauli is not X, Y, or Z.
    """
    if pauli == "X" or pauli == cirq.X:
        return cirq.ry(-np.pi / 2), cirq.ry(np.pi / 2)
    elif pauli == "-X":
        return cirq.ry(np.pi / 2), cirq.ry(-np.pi / 2)
    elif pauli == "Y" or pauli == cirq.Y:
        return cirq.rx(np.pi / 2), cirq.rx(-np.pi / 2)
    elif pauli == "-Y":
        return cirq.rx(-np.pi / 2), cirq.rx(np.pi / 2)
    elif pauli == "Z" or pauli == cirq.Z:
        return cirq.I, cirq.I
    elif pauli == "-Z":
        return cirq.X, cirq.X
    elif pauli == "1":
        return cirq.I, cirq.I
    else:
        raise ValueError(f"pauli must be X, Y, or Z: {pauli}")


def _generate_qnd_pauli_circuit(
    q0: cirq.Qid,
    q1: cirq.Qid,
    anc: cirq.Qid,
    paulis: tuple[str, str],
    measure_key: str | None,
    virtual_z_half_turns: float = 0.0,
) -> cirq.FrozenCircuit:
    """Make a circuit to perform QND measurement of the specified Pauli.

    If len(pauli)==1, we do not map the qubit to an ancilla, and just perform a QND measurement on
    the qubit itself.

    Args:
        q0: Data qubit 0
        q1: Data qubit 1
        anc: Ancilla qubit
        paulis: The Paulis to measure on each qubit.
        measure_key: A key to associate to the measurement.
        virtual_z_half_turns: This applied only for Paulis mapped to an ancilla. If not zero,
            amount of half turns around the Z axis to apply on the ancilla during the mapping of
            the paulis from the data qubits (q0, q1). This virtualZ is applied twice, once before
            each Hadamard during the mapping of the Pauli to the ancilla.

    Returns:
        A circuit to measure the indicated Pauli operator
    """
    if "1" in paulis:
        if paulis[0] == "1":
            pauli, qubit, other = paulis[1], q1, q0
        else:
            pauli, qubit, other = paulis[0], q0, q1
        pre, post = get_rot_gate_from_pauli(pauli)
        circuit = cirq.FrozenCircuit.from_moments(
            [pre(qubit), cirq.I(other)],
            cirq.M(qubit, key=measure_key),
            post(qubit),
        )
        if _CIRCUIT_DD_SUPPORTED:
            circuit = _ADD_DD_FUNC(circuit)
        else:
            warnings.warn(
                "Skipping Dynamical Decoupling: This feature requires Cirq 1.4 or later. "
                "Please upgrade your Cirq version to enable noise mitigation.",
                UserWarning,
            )
        return circuit
    return _generate_qnd_pauli_circuit_with_ancillas(
        q0, q1, anc, paulis, measure_key, virtual_z_half_turns=virtual_z_half_turns
    )


def _generate_qnd_pauli_circuit_with_ancillas(
    q0: cirq.Qid,
    q1: cirq.Qid,
    anc: cirq.Qid,
    paulis: tuple[str, str],
    measure_key: str | None,
    virtual_z_half_turns: float = 0.0,
) -> cirq.FrozenCircuit:
    """Make a circuit to perform a QND measurement of the specified Pauli.

    This method is different from generate_qnd_pauli_circuit since we always map the observables to
    an ancilla, regardless if the observable is a 1-body or a 2-body Pauli.

    Args:
        q0: Data qubit 0
        q1: Data qubit 1
        anc: Ancilla qubit
        paulis: The Paulis to measure on each qubit.
        measure_key: A key to associate to the measurement.
        virtual_z_half_turns: This applied only for Paulis mapped to an ancilla. If not zero,
            amount of half turns around the Z axis to apply on the ancilla during the mapping of
            the paulis from the data qubits (q0, q1). This virtualZ is applied twice, once before
            each Hadamard during the mapping of the Pauli to the ancilla.
    """
    # Get pre- and post- rotations for the pauli to measure.
    pre0, post0 = (rot(q0) for rot in get_rot_gate_from_pauli(paulis[0]))
    pre1, post1 = (rot(q1) for rot in get_rot_gate_from_pauli(paulis[1]))
    return cirq.FrozenCircuit.from_moments(
        [pre0, pre1, cirq.H(anc)],
        *(cirq.CZ(q, anc) for q, pauli in zip((q0, q1), paulis) if pauli != "1"),
        cirq.Z(anc) ** virtual_z_half_turns,
        [post0, post1, cirq.H(anc)],
        *measure_reset_circuit(q0, q1, anc, measure_key),
    )


def measure_reset_circuit(
    q0: cirq.Qid, q1: cirq.Qid, anc: cirq.Qid, measure_key: str | None
) -> cirq.FrozenCircuit:
    m = cirq.M(anc, key=measure_key)
    circuit = cirq.FrozenCircuit.from_moments(
        [
            m,
            cirq.I(q0),
            cirq.I(q1),
        ],
        cirq.R(anc),
    )
    if _CIRCUIT_DD_SUPPORTED:
        circuit = _ADD_DD_FUNC(circuit)
    else:
        warnings.warn(
            "Skipping Dynamical Decoupling: This feature requires Cirq 1.4 or later. "
            "Please upgrade your Cirq version to enable noise mitigation.",
            UserWarning,
        )
    return circuit


def generate_qnd_pauli_circuit(
    q0: cirq.Qid,
    q1: cirq.Qid,
    anc: cirq.Qid,
    pauli: str,
    measure_key: str | None,
    qnd_strategy: QndStrategy = "map_all_paulis_to_ancilla",
    virtual_z_half_turns: float = 0.0,
) -> cirq.FrozenCircuit:
    """Generate a circuit to measure the one-qubit or two-qubit pauli (PauliString).

    Args:
        q0: First qubit.
        q1: Second qubit.
        anc: Measure/ancilla qubit.
        pauli: PauliString, can be one-qubit or two-qubit Pauli.
        measure_key: Key for the cirq.M gate.
        qnd_strategy: Whether to map all PauliStrings to the ancilla or only map two-qubit Paulis.
        virtual_z_half_turns: This applied only for Paulis mapped to an ancilla. If not zero,
            amount of half turns around the Z axis to apply on the ancilla during the mapping of
            the paulis from the data qubits (q0, q1). This virtualZ is applied twice, once before
            each Hadamard during the mapping of the Pauli to the ancilla.
    """
    # Split into signed single-qubit paulis. Include the sign on an X or Y term if possible.
    unsigned = pauli.removeprefix("-")
    p0, p1 = unsigned[0], unsigned[1]
    if pauli.startswith("-"):
        if p0 in ("X", "Y"):
            p0 = f"-{p0}"
        elif p1 in ("X", "Y"):
            p1 = f"-{p1}"
        elif p0 == "Z":
            p0 = "-Z"
        elif p1 == "Z":
            p1 = "-Z"
        else:
            raise ValueError(f"invalid pauli string: {pauli}")
    if qnd_strategy == "map_2q_paulis_to_ancilla":
        return _generate_qnd_pauli_circuit(
            q0,
            q1,
            anc,
            (p0, p1),
            measure_key,
            virtual_z_half_turns=virtual_z_half_turns,
        )
    elif qnd_strategy == "map_all_paulis_to_ancilla":
        return _generate_qnd_pauli_circuit_with_ancillas(
            q0,
            q1,
            anc,
            (p0, p1),
            measure_key,
            virtual_z_half_turns=virtual_z_half_turns,
        )
    else:
        raise ValueError(
            'qnd_strategy can only be "map_all_paulis_to_ancilla" or "map_2q_paulis_to_ancilla"'
            f" got {qnd_strategy=}."
        )


def estimate_parities(
    records: Mapping[str, np.ndarray],
    contexts: Mapping[tuple[Axis, int], Sequence[PauliKey]],
) -> dict[tuple[Axis, int], tuple[float, float]]:
    """Estimate average parity and error of each of the 6 contexts.

    Useful to get more nuanced information on each row/col. We first evaluate the parity for
    each time every context has appeared in the circuit. We then average the parities for
    each context and estimate the error of this average. For references on the error estimation,
    check the error of a binomial distribution.

    Args:
        records: The value for each key is a 2-D array of booleans, with the first index running
            over circuit repetitions, the second index running over instances of the measurement
            key in the circuit.
        contexts: Mapping contexts to their corresponding pauli keys.

    Returns:
        context_averages: A dictionary mapping a context (axis, idx) to a tuple
        (mean_parity, parity_error)
    """
    context_parities = {}
    for context, pauli_keys in contexts.items():
        # Sum all outcome arrays for this context first. Summing the arrays first and doing
        # one exponentiation is much faster than doing multiple exponentiations and then a
        # product.
        outcome_sum = np.sum(
            [
                records[pauli_key.key][:, pauli_key.slice].astype(int)
                for pauli_key in pauli_keys
            ],
            axis=0,
        )
        # Now calculate parity in one vectorized operation
        context_parities[context] = (-1) ** outcome_sum

    context_averages = {}
    for context, parities in context_parities.items():
        mean_parity = np.mean(parities)

        # Arithmetically identical to np.mean([(parity + 1) / 2 and much faster
        p_mean = abs((mean_parity + 1) / 2)

        parity_error = 2 * np.sqrt(p_mean * (1 - p_mean) / parities.size)

        context_averages[context] = (float(mean_parity), float(parity_error))

    return context_averages


def estimate_chi(
    records: Mapping[str, np.ndarray],
    contexts: Mapping[tuple[Axis, int], Sequence[PauliKey]],
) -> tuple[float, float]:
    """Estimate chi out of the 6 parity averages.

    This method takes the output of the Kochen-Specker experiment and returns an estimate of chi.
    We assume that the length of the circuit ran was long enough to contain at least one instance of
    each of the six 3-observable parities, e.g. (1X X1 XX).
    """
    context_averages = estimate_parities(records, contexts)

    chi_mean = sum(
        avg * (1 if axis == "row" else -1)
        for (axis, _idx), (avg, _unc) in context_averages.items()
    )
    chi_error = float(np.sqrt(sum(unc**2 for _avg, unc in context_averages.values())))

    return chi_mean, chi_error


# Utils
def get_parities_from_single_context_experiments(
    context_to_kochen_specker_result: dict[tuple[Axis, int], MagicSquareResult],
) -> dict[tuple[Axis, int], tuple[float, float]]:
    """Get the parities (with uncertainties) for each context.

    Assumes that we ran 6 single-context Kochen Specker experiments. Returns a mapping from
    context to their parities (with uncertainties).

    Args:
        context_to_kochen_specker_result: maps context to a MagicSquareResult instance. Object is
            usually returned by "run_kochen_specker_per_context".
    """
    axes: list[Axis] = ["col", "row"]
    indices = [0, 1, 2]
    all_contexts: list[tuple[Axis, int]] = [
        (axis, idx) for axis in axes for idx in indices
    ]
    context_to_parities = {}
    context_to_parities.update(
        {
            context: context_to_kochen_specker_result[context].estimate_parities()[
                context
            ]
            for context in all_contexts
        }
    )
    return context_to_parities


def get_chi_from_single_context_experiment(
    context_to_parities: dict[tuple[Axis, int], tuple[float, float]],
) -> tuple[float, float]:
    """Estimate chi (with uncertainty) from a context_to_parities mapping."""
    chi_mean = sum(
        avg * (1 if axis == "row" else -1)
        for (axis, _idx), (avg, _unc) in context_to_parities.items()
    )
    chi_error = np.sqrt(sum(unc**2 for _avg, unc in context_to_parities.values()))

    return chi_mean, chi_error
