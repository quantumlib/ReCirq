import os
import pickle
from copy import deepcopy
from functools import partial
from typing import cast, Sequence

import cirq
import cirq.transformers.dynamical_decoupling as dd
import numpy as np
from tqdm import tqdm

import concurrent.futures
from . import dfl_1d as dfl
from . import dfl_2d_second_order_trotter as dfl_2d


def _apply_gauge_compiling(seed: int, circuit: cirq.Circuit) -> cirq.Circuit:
    return cirq.merge_single_qubit_moments_to_phxz(
        cirq.transformers.gauge_compiling.CPhaseGaugeTransformerMM()(
            circuit, rng_or_seed=seed
        )
    )


def _distance(q1: cirq.GridQubit, q2: cirq.GridQubit) -> int:
    """Return the Manhattan distance between two qubits.

    Args:
        q1: The first qubit.
        q2: The second qubit.

    Returns:
        The distance between the qubits.
    """
    return abs(q1.row - q2.row) + abs(q1.col - q2.col)


class DFLExperiment:
    """A class for performing the 1D DFL experiment (Fig 1 of the paper).

    Attrs:
        qubits: The qubits to use for the experiment.
        sampler: The cirq sampler to use.
        j: The coefficient on the hopping term.
        h: The coefficient on the gauge X term.
        mu: The coefficient on the matter sigma_x term.
        tau: The Trotter step size.
        pbc: Whether to use periodic boundary conditions.
        rng: The pseudorandom number generator to use for readout benchmarking.
        cycles: The number of cycles for which to run the experiment.
        num_gc_instances: The number of gauge compiling instances to use.
        include_zero_trotter: Whether to include circuits with the Trotter step set to 0.
    """

    def __init__(
        self,
        qubits: list[cirq.GridQubit],
        sampler: cirq.Sampler,
        save_directory: str,
        j: float = 1.0,
        h: float = 1.3,
        mu: float = 1.5,
        tau: float = 0.25,
        pbc: bool = True,
        rng: np.random.Generator = np.random.default_rng(),
        cycles: np.ndarray = np.arange(31),
        num_gc_instances: int = 40,
        use_cphase: bool = False,
        include_zero_trotter: bool = True,
    ):
        self.qubits = qubits
        self.sampler = sampler
        self.j = j
        self.h = h
        self.mu = mu
        self.tau = tau
        assert len(qubits) % 2 == 0
        self.pbc = pbc
        self.rng = rng
        self.cycles = cycles
        self.all_circuits: list[cirq.Circuit] = []
        self.e0 = np.array([])
        self.e1 = np.array([])
        self.num_gc_instances = num_gc_instances
        self.executor = concurrent.futures.ProcessPoolExecutor()
        self.use_cphase = use_cphase
        self.readout_ideal_bitstrs = np.array([])
        self.save_directory = save_directory
        self.include_zero_trotter = include_zero_trotter
        self.matter_sites = np.arange(0, len(qubits), 2)
        self.gauge_sites = np.arange(1, len(qubits), 2)

    def save_to_file(self, filename: str):
        d_to_save = {
            "measurements": self.measurements,
            "readout_ideal_bitstrs": self.readout_ideal_bitstrs,
            "j": self.j,
            "h": self.h,
            "mu": self.mu,
            "tau": self.tau,
            "pbc": self.pbc,
            "cycles": self.cycles,
            "num_gc_instances": self.num_gc_instances,
            "use_cphase": self.use_cphase,
            "e0": self.e0,
            "e1": self.e1,
            "qubits": self.qubits,
            "save_directory": self.save_directory,
            "include_zero_trotter": self.include_zero_trotter,
        }

        pickle.dump(d_to_save, open(filename, "wb"))

    def load_from_file(self, filename: str):
        d = pickle.load(open(filename, "rb"))
        self.measurements = d["measurements"]
        self.readout_ideal_bitstrs = d["readout_ideal_bitstrs"]
        self.j = d["j"]
        self.h = d["h"]
        self.mu = d["mu"]
        self.tau = d["tau"]
        self.pbc = d["pbc"]
        self.cycles = d["cycles"]
        self.num_gc_instances = d["num_gc_instances"]
        self.use_cphase = d["use_cphase"]
        self.e0 = d["e0"]
        self.e1 = d["e1"]
        self.qubits = d["qubits"]
        self.save_directory = d.get("save_directory", "temp")
        self.include_zero_trotter = d.get("include_zero_trotter", True)

    def generate_circuit_instances(
        self,
        initial_state: str,
        ncycles: int,
        basis: str,
        gauge_compile: bool = True,
        dynamical_decouple: bool = True,
        dd_sequence: tuple[cirq.Gate, ...] = (cirq.X, cirq.Y, cirq.X, cirq.Y),
        zero_trotter: bool = False,
    ) -> list[cirq.Circuit]:
        if initial_state == "gauge_invariant":
            initial_state = "single_sector"

        assert initial_state in ["single_sector", "superposition"]

        circuit = dfl.get_1d_dfl_experiment_circuits(
            self.qubits,
            initial_state,
            [ncycles],
            [self.qubits[1]],  # put the excitation on the first gauge site
            0.0 if zero_trotter else self.tau,
            self.h,
            self.mu,
            1,
            "cphase" if self.use_cphase else "cz",
            "dual",
        )[["z", "x"].index(basis)]

        if gauge_compile:
            circuits = list(
                self.executor.map(
                    partial(_apply_gauge_compiling, circuit=circuit),
                    range(self.num_gc_instances),
                )
            )
        else:
            circuits = [circuit]

        if dynamical_decouple:
            circuits_dd = self.executor.map(
                partial(dd.add_dynamical_decoupling, schema=dd_sequence), circuits
            )
            circuits = list(circuits_dd)

        return circuits

    def create_readout_benchmark_circuits(
        self, num_random_bitstrings: int = 30
    ) -> tuple[np.ndarray, list[cirq.Circuit]]:
        n = len(self.qubits)
        x_or_I = lambda bit: cirq.X if bit == 1 else cirq.I
        bitstrs = self.rng.integers(0, 2, size=(num_random_bitstrings, n))
        random_circuits = []
        random_circuits = [
            cirq.Circuit(
                [
                    x_or_I(bit)(qubit)
                    for bit, qubit in zip(bitstr, self.qubits[: len(bitstr)])
                ]
                + [cirq.M(*self.qubits[: len(bitstr)], key="m")]
            )
            for bitstr in bitstrs
        ]
        return bitstrs, random_circuits

    def identify_ideal_readout_bitstrings_from_circuits(self):
        # first identify the number of readout bitstrings:
        for num_bitstrs, circuit in enumerate(self.all_circuits[::-1]):
            if len(circuit) > 2:
                break

        bit_fn = lambda gate: 1 if gate == cirq.X else 0
        self.readout_ideal_bitstrs = np.array(
            [
                [bit_fn(circuit[0][qubit].gate) for qubit in self.qubits]
                for circuit in self.all_circuits[-num_bitstrs:]
            ]
        )

    def generate_all_circuits(
        self,
        gauge_compile: bool = True,
        dynamical_decouple: bool = True,
        dd_sequence: tuple[cirq.Gate, ...] = (cirq.X, cirq.Y, cirq.X, cirq.Y),
        num_random_bitstrings: int = 30,
    ):
        if not gauge_compile:
            self.num_gc_instances = 1
        num_gc_instances = self.num_gc_instances

        zero_trotter_options = [False, True] if self.include_zero_trotter else [False]

        all_circuits = []
        with tqdm(
            total=2
            * 2
            * len(zero_trotter_options)
            * len(self.cycles)
            * num_gc_instances
        ) as pbar:
            for initial_state in ["gauge_invariant", "superposition"]:
                for basis in ["z", "x"]:
                    for zero_trotter in zero_trotter_options:
                        for ncycles in self.cycles:
                            all_circuits += self.generate_circuit_instances(
                                initial_state,
                                ncycles,
                                basis,
                                gauge_compile,
                                dynamical_decouple,
                                dd_sequence,
                                zero_trotter,
                            )
                            pbar.update(num_gc_instances)

        bitstrs, readout_circuits = self.create_readout_benchmark_circuits(
            num_random_bitstrings
        )
        all_circuits += readout_circuits

        expected_number = (
            len(self.cycles) * 2 * 2 * len(zero_trotter_options) * num_gc_instances
            + num_random_bitstrings
        )
        assert len(all_circuits) == expected_number

        self.all_circuits = all_circuits
        self.readout_ideal_bitstrs = bitstrs

    def load_circuits_from_file(
        self, filename: str, old_qubits_list: list[cirq.Qid] | None = None
    ):
        """Load the circuits from a file.

        Args:
            filename: The filename from which to load the circuits.
            old_qubits_list: The previous ordered list of qubits if mapping to different qubits.
        """
        circuits = pickle.load(open(filename, "rb"))
        if old_qubits_list is not None and old_qubits_list != self.qubits:
            self.all_circuits = [
                circuit.transform_qubits(dict(zip(old_qubits_list, self.qubits)))
                for circuit in tqdm(circuits, total=len(circuits))
            ]
        else:
            self.all_circuits = circuits
        self.identify_ideal_readout_bitstrings_from_circuits()

    def shuffle_circuits_and_save(
        self, batch_size: int = 500, delete_circuit_list: bool = True
    ):
        """Shuffle all of the circuits and save them to files.

        Args:
            batch_size: The maximum number of circuits per file.
            delete_circuit_list: Whether to delete the circuit list from this object after saving, saves memory.
        """
        if not os.path.isdir(self.save_directory + "/shuffled_circuits"):
            os.mkdir(self.save_directory + "/shuffled_circuits")

        # shuffle the circuits:
        circuits = self.all_circuits
        indices = np.arange(len(circuits))
        self.rng.shuffle(indices)
        self.run_order = indices
        inv_map = np.array([list(indices).index(_) for _ in range(len(circuits))])
        circuits_shuffled = [circuits[_] for _ in indices]
        # save the shuffled circuits
        for start_idx in tqdm(range(0, len(circuits), batch_size)):
            circuits_i = circuits_shuffled[start_idx : (start_idx + batch_size)]
            pickle.dump(
                circuits_i,
                open(
                    self.save_directory + f"/shuffled_circuits/{start_idx}.pickle", "wb"
                ),
            )

        params = {
            "indices": indices,
            "inv_map": inv_map,
            "readout_ideal_bitstrs": self.readout_ideal_bitstrs,
            "batch_size": batch_size,
        }
        pickle.dump(
            params, open(self.save_directory + "/shuffled_circuits/params.pickle", "wb")
        )

        if delete_circuit_list:
            del self.all_circuits

    def run_experiment(
        self,
        repetitions_post_selection: int | list[int] = 2000,
        repetitions_non_post_selection: int | list[int] = 2000,
        batch_size: int = 500,
        readout_repetitions: int = 1000,
        gauge_compile: bool = True,
        dynamical_decouple: bool = True,
        dd_sequence: tuple[cirq.Gate, ...] = (cirq.X, cirq.Y, cirq.X, cirq.Y),
        num_random_bitstrings: int = 30,
    ):
        """Run the experiment. First, shuffles and saves the circuits and then runs from the saved
        circuits.

        Calls generate_all_circuits, then shuffle_circuits_and_save, then run_experiment_from_saved_circuits, then load_shuffled_measurements.

        Note: These default values are used for the 1D experiment. For the 2D experiment, we use
        ```
        repetitions_post_selection = [
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1136,
            1317,
            1604,
            2267,
            4139,
            5655,
            6579,
            7688,
            11550,
            17332,
            25493,
            37463,
            46275,
            63527,
            69691,
            111862,
            137194,
            227824,
            348209,
            346286,
            406434,
            500000,
            682846,
        ],
        repetitions_non_post_selection = 1000
        ```

        Args:
            repetitions_post_selection: How many repetitions to use for the gauge invariant initial state at each cycle number.
            repetitions_non_post_selection: How many repetitions to use for the superposition initial state.
            batch_size: The maximum number of circuits per file and per run_batch call.
            gauge_compile: Whether to add gauge compiling.
            dynamical_decouple: Whether to add dynamical decoupling.
            dd_sequence: The dynamical decoupling sequence to use.
            num_random_bitstrings: The number of bitstrings to use for readout mitigation.
            readout_repetitions: The number of repetitions to use for readout benchmarking.
        """
        self.generate_all_circuits(
            gauge_compile=gauge_compile,
            dynamical_decouple=dynamical_decouple,
            dd_sequence=dd_sequence,
            num_random_bitstrings=num_random_bitstrings,
        )
        self.shuffle_circuits_and_save(batch_size=batch_size, delete_circuit_list=True)
        self.run_experiment_from_saved_circuits(
            repetitions_post_selection=repetitions_post_selection,
            repetitions_non_post_selection=repetitions_non_post_selection,
            initial_start_index=0,
            readout_repetitions=readout_repetitions,
        )
        self.load_shuffled_measurements()
        self.extract_readout_error_rates()

    def run_experiment_from_saved_circuits(
        self,
        repetitions_post_selection: int | list[int] = 2000,
        repetitions_non_post_selection: int | list[int] = 2000,
        initial_start_index: int = 0,
        old_qubits: list[cirq.Qid] | None = None,
        new_qubits: list[cirq.Qid] | None = None,
        readout_repetitions: int = 1000,
    ):
        """Run the experiment from circuits that are saved ahead of time. Saves the results to
        files.

        To use this, the circuits should have been saved with shuffle_circuits_and_save.

        The location of the saved files is `self.save_directory + "/shuffled_results"`.

        Note: These default values are used for the 1D experiment. For the 2D experiment, we use
        ```
        repetitions_post_selection = [
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1136,
            1317,
            1604,
            2267,
            4139,
            5655,
            6579,
            7688,
            11550,
            17332,
            25493,
            37463,
            46275,
            63527,
            69691,
            111862,
            137194,
            227824,
            348209,
            346286,
            406434,
            500000,
            682846,
        ],
        repetitions_non_post_selection = 1000
        ```

        Args:
            repetitions_post_selection: How many repetitions to use for the gauge invariant initial state at each cycle number.
            repetitions_non_post_selection: How many repetitions to use for the superposition initial state.
            initial_start_index: The circuit number to start at (inteneded for resuming datataking if it crashed before).
            old_qubits: The order of qubits for which the circuits were originally generated.
            new_qubits: The order of qubits to use now.
            readout_repetitions: The number of repetitions to use for readout benchmarking.
        """

        # convert to lists:
        repetitions_post_selection_list = list(
            np.zeros(len(self.cycles), dtype=int) + repetitions_post_selection
        )
        repetitions_non_post_selection_list = list(
            np.zeros(len(self.cycles), dtype=int) + repetitions_non_post_selection
        )

        # load the parameters for the shuffled circuits:
        params = pickle.load(
            open(self.save_directory + "/shuffled_circuits/params.pickle", "rb")
        )
        indices = params["indices"]
        inv_map = params["inv_map"]
        self.readout_ideal_bitstrs = params["readout_ideal_bitstrs"]
        batch_size = params["batch_size"]
        self.run_order = indices
        num_random_bitstrings = len(self.readout_ideal_bitstrs)
        zero_trotter_options = [0, 1] if self.include_zero_trotter else [0]
        tot_num_circuits = (
            len(self.cycles) * 2 * 2 * len(zero_trotter_options) * self.num_gc_instances
            + num_random_bitstrings
        )

        repetitions = [readout_repetitions] * tot_num_circuits

        # fill in other repetition numbers
        for init_state_number, reps_list in [
            (0, repetitions_post_selection_list),
            (1, repetitions_non_post_selection_list),
        ]:
            for basis_number in [0, 1]:
                for zero_trotter_number in zero_trotter_options:
                    for cycle_number in self.cycles:
                        start_index = (
                            init_state_number
                            * 2
                            * len(zero_trotter_options)
                            * len(self.cycles)
                            * self.num_gc_instances
                            + basis_number
                            * len(zero_trotter_options)
                            * len(self.cycles)
                            * self.num_gc_instances
                            + zero_trotter_number
                            * len(self.cycles)
                            * self.num_gc_instances
                            + cycle_number * self.num_gc_instances
                        )
                        end_index = start_index + self.num_gc_instances
                        repetitions[start_index:end_index] = [
                            reps_list[cycle_number]
                        ] * self.num_gc_instances

        # now apply the shuffle
        repetitions = [repetitions[index] for index in indices]

        # run the experiment and save shuffled measurements to files (avoids memory issues)
        for start_idx in tqdm(range(initial_start_index, tot_num_circuits, batch_size)):
            circuits_i = pickle.load(
                open(
                    self.save_directory + f"/shuffled_circuits/{start_idx}.pickle", "rb"
                )
            )
            if old_qubits is not None and new_qubits is not None:
                print("Transforming qubits")
                circuits_i = [
                    circuit.transform_qubits(dict(zip(old_qubits, new_qubits)))
                    for circuit in circuits_i
                ]
            print(
                f"Shots this batch: {sum(repetitions[start_idx : (start_idx + batch_size)])}"
            )
            result = self.sampler.run_batch(
                circuits_i,
                repetitions=repetitions[start_idx : (start_idx + batch_size)],
            )
            results_i = [res[0].measurements["m"].astype(bool) for res in result]
            if not os.path.isdir(self.save_directory + f"/shuffled_results"):
                os.mkdir(self.save_directory + f"/shuffled_results")
            pickle.dump(
                results_i,
                open(
                    self.save_directory + f"/shuffled_results/{start_idx}.pickle", "wb"
                ),
            )

    def load_shuffled_measurements(self):
        """Loads the measurement results from files.

        First, you should have run `run_experiment_from_saved_circuits` or `run_experiment`.
        """
        params = pickle.load(
            open(self.save_directory + "/shuffled_circuits/params.pickle", "rb")
        )
        indices = params["indices"]
        inv_map = params["inv_map"]
        self.readout_ideal_bitstrs = params["readout_ideal_bitstrs"]
        batch_size = params["batch_size"]
        num_random_bitstrings = len(self.readout_ideal_bitstrs)
        zero_trotter_options = [False, True] if self.include_zero_trotter else [False]
        tot_num_circuits = (
            len(self.cycles) * 2 * 2 * len(zero_trotter_options) * self.num_gc_instances
            + num_random_bitstrings
        )
        measurements_shuffled = []
        for start_idx in tqdm(range(0, tot_num_circuits, batch_size)):
            measurements_shuffled += pickle.load(
                open(
                    self.save_directory + f"/shuffled_results/{start_idx}.pickle", "rb"
                )
            )

        self.measurements = [measurements_shuffled[i] for i in inv_map]

    def extract_readout_error_rates(self) -> None:
        ideal_bitstrs = self.readout_ideal_bitstrs
        num_bitstrs = len(ideal_bitstrs)
        readout_measurements = np.array(self.measurements[-num_bitstrs:])
        repetitions = len(readout_measurements[0])
        num_prep_0 = np.sum(1 - ideal_bitstrs, axis=0)
        num_prep_1 = np.sum(ideal_bitstrs, axis=0)
        self.e0 = np.einsum("ik,ijk->k", (1 - ideal_bitstrs), readout_measurements) / (
            num_prep_0 * repetitions
        )
        self.e1 = np.einsum("ik,ijk->k", ideal_bitstrs, (1 - readout_measurements)) / (
            num_prep_1 * repetitions
        )

        return None

    def extract_measurements(
        self,
        basis_number: int,
        initial_state: str,
        cycle_number: int,
        zero_trotter: bool = False,
        post_select: bool = False,
    ) -> np.ndarray:
        """Extract the portion of the measurement results corresponding to the requested data.

        Args:
            basis_number: 0 for z-basis and 1 for x-basis.
            initial_state: Either "superposition" or "gauge_invariant"
            cycle_number: The number of Trotter steps (can be 0).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).

        Returns:
            The measurement results.

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if cycle_number < 0:
            raise ValueError("Cycle number should be nonnegative")
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )
        if initial_state == "single_sector":
            initial_state = "gauge_invariant"
        init_state_number = ["gauge_invariant", "superposition"].index(initial_state)
        zero_trotter_options = [False, True] if self.include_zero_trotter else [False]
        zero_trotter_number = zero_trotter_options.index(zero_trotter)

        start_index = (
            init_state_number
            * 2
            * len(zero_trotter_options)
            * len(self.cycles)
            * self.num_gc_instances
            + basis_number
            * len(zero_trotter_options)
            * len(self.cycles)
            * self.num_gc_instances
            + zero_trotter_number * len(self.cycles) * self.num_gc_instances
            + cycle_number * self.num_gc_instances
        )

        measurements = np.array(
            self.measurements[start_index : (start_index + self.num_gc_instances)]
        )
        repetitions = len(measurements[0])
        measurements = measurements.reshape(
            repetitions * self.num_gc_instances, len(self.qubits)
        )
        if post_select:
            mask = np.all(measurements[:, self.matter_sites] == False, axis=1)
            measurements = measurements[mask, :]
        return measurements

    def extract_zzz_single_cycle(
        self,
        initial_state: str,
        cycle_number: int,
        zero_trotter: bool = False,
        readout_mitigate: bool = True,
        post_select: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ) -> np.ndarray:
        """Get the expectation value of the interaction term (ZZZ in the LGT basis but implemented
        here as Z_gauge in the dual basis).

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            cycle_number: The number of Trotter steps (can be 0).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if cycle_number < 0:
            raise ValueError("Cycle number should be nonnegative")
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        if readout_mitigate and len(self.e0) == 0 or len(self.e1) == 0:
            self.extract_readout_error_rates()

        measurements = self.extract_measurements(
            0,
            initial_state,
            cycle_number,
            zero_trotter,
            post_select and tolerated_distance_to_error == np.inf,
        )
        if post_select and tolerated_distance_to_error < np.inf:
            raise NotImplementedError
        else:
            p1 = np.mean(measurements[:, self.gauge_sites], axis=0)
            dp1 = np.sqrt(p1 * (1 - p1) / len(measurements))
        if not readout_mitigate:
            x_dx = np.array([1 - 2 * p1, 2 * dp1])  # value, uncertainty
        else:
            x_raw = 1 - 2 * p1
            e0 = self.e0[self.gauge_sites]
            e1 = self.e1[self.gauge_sites]
            x_mitigated = (x_raw + e0 - e1) / (
                1.0 - e0 - e1
            )  # see Eq. F3 of https://arxiv.org/pdf/2106.01264
            x_dx = np.array([x_mitigated, 2 * dp1 / (1.0 - e0 - e1)])
        return x_dx

    def extract_interaction(
        self,
        initial_state: str,
        zero_trotter: bool = False,
        readout_mitigate: bool = True,
        post_select: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ):
        """Get the expectation value of the interaction term (ZZZ in the LGT basis but implemented
        here as Z_gauge in the dual basis) at all cycles.

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [cycle_number, value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        return np.array(
            [
                self.extract_zzz_single_cycle(
                    initial_state,
                    cycle,
                    zero_trotter,
                    readout_mitigate,
                    post_select,
                    tolerated_distance_to_error,
                )
                for cycle in tqdm(self.cycles, total=len(self.cycles))
            ]
        )

    def extract_gauge_x_single_cycle(
        self,
        initial_state: str,
        cycle_number: int,
        zero_trotter: bool = False,
        readout_mitigate: bool = True,
        post_select: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ) -> np.ndarray:
        """Get the expectation value of the h term, which is X_gauge in both the LGT and dual bases.

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            cycle_number: The number of Trotter steps (can be 0).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if cycle_number < 0:
            raise ValueError("Cycle number should be nonnegative")
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        if readout_mitigate and len(self.e0) == 0 or len(self.e1) == 0:
            self.extract_readout_error_rates()

        measurements = self.extract_measurements(
            1,
            initial_state,
            cycle_number,
            zero_trotter,
            post_select and tolerated_distance_to_error == np.inf,
        )
        if post_select and tolerated_distance_to_error < np.inf:
            raise NotImplementedError
        else:
            p1 = np.mean(measurements[:, self.gauge_sites], axis=0)
            dp1 = np.sqrt(p1 * (1 - p1) / len(measurements))
        if not readout_mitigate:
            x_dx = np.array([1 - 2 * p1, 2 * dp1])  # value, uncertainty
        else:
            x_raw = 1 - 2 * p1
            e0 = self.e0[self.gauge_sites]
            e1 = self.e1[self.gauge_sites]
            x_mitigated = (x_raw + e0 - e1) / (
                1.0 - e0 - e1
            )  # see Eq. F3 of https://arxiv.org/pdf/2106.01264
            x_dx = np.array([x_mitigated, 2 * dp1 / (1.0 - e0 - e1)])
        return x_dx

    def extract_gauge_x(
        self,
        initial_state: str,
        readout_mitigate: bool = True,
        post_select: bool = False,
        zero_trotter: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ) -> np.ndarray:
        """Get the expectation value of the h term, which is X_gauge in both the LGT and dual bases
        at all cycle numbers.

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [cycle_number, value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        return np.array(
            [
                self.extract_gauge_x_single_cycle(
                    initial_state,
                    cycle,
                    readout_mitigate=readout_mitigate,
                    post_select=post_select,
                    zero_trotter=zero_trotter,
                    tolerated_distance_to_error=tolerated_distance_to_error,
                )
                for cycle in self.cycles
            ]
        )

    def extract_matter_x_single_cycle(
        self,
        initial_state: str,
        cycle_number: int,
        zero_trotter: bool = False,
        readout_mitigate: bool = True,
        post_select: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ) -> np.ndarray:
        """Get the expectation value of the mu term.

        This is X_matter in the LGT basis and a product of Xs on a matter site and the neighboring gauge sites in the dual basis.

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            cycle_number: The number of Trotter steps (can be 0).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if cycle_number < 0:
            raise ValueError("Cycle number should be nonnegative")
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        if readout_mitigate and len(self.e0) == 0 or len(self.e1) == 0:
            self.extract_readout_error_rates()

        measurements = self.extract_measurements(
            1,
            initial_state,
            cycle_number,
            zero_trotter,
            post_select and tolerated_distance_to_error == np.inf,
        )
        x = []
        dx = []
        e0 = deepcopy(self.e0)
        e1 = deepcopy(self.e1)
        if post_select and readout_mitigate:
            for idx in self.matter_sites:
                e0[idx] = 0.0
                e1[idx] = 0.0
        for q_idx in self.matter_sites:
            qubits_i = [
                self.qubits[q_idx - 1],
                self.qubits[q_idx],
                self.qubits[(q_idx + 1) % len(self.qubits)],
            ]
            indices = np.array(
                [self.qubits.index(cast(cirq.GridQubit, qi)) for qi in qubits_i]
            )
            if post_select and tolerated_distance_to_error < np.inf:
                raise NotImplementedError
            else:
                measurements_i = measurements[:, indices]
            if readout_mitigate:
                single_qubit_cmats = []
                qubits_to_mitigate = []

                for i in indices:
                    e0_sq = e0[i]
                    e1_sq = e1[i]
                    single_qubit_cmats.append(
                        np.array([[1 - e0_sq, e1_sq], [e0_sq, 1 - e1_sq]])
                    )
                    qubits_to_mitigate.append(self.qubits[i])

                tcm = cirq.experiments.TensoredConfusionMatrices(
                    single_qubit_cmats,
                    [[q] for q in qubits_to_mitigate],
                    repetitions=measurements_i.shape[0],
                    timestamp=0.0,
                )

                x_i, dx_i = tcm.readout_mitigation_pauli_uncorrelated(
                    qubits_to_mitigate, measurements_i
                )
            else:
                repetitions = measurements_i.shape[0]
                p1 = np.mean(np.sum(measurements_i, axis=1) % 2)
                dp1 = np.sqrt(p1 * (1 - p1) / (repetitions * self.num_gc_instances))
                x_i = 1 - 2 * p1
                dx_i = 2 * dp1
            x.append(x_i)
            dx.append(dx_i)
        return np.array([x, dx])

    def extract_matter_x(
        self,
        initial_state: str,
        readout_mitigate: bool = True,
        post_select: bool = False,
        zero_trotter: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ) -> np.ndarray:
        """Get the expectation value of the mu term at all cycles.

        This is X_matter in the LGT basis and a product of Xs on a matter site and the neighboring gauge sites in the dual basis.

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [cycle_number, value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        return np.array(
            [
                self.extract_matter_x_single_cycle(
                    initial_state,
                    cycle,
                    readout_mitigate=readout_mitigate,
                    post_select=post_select,
                    zero_trotter=zero_trotter,
                    tolerated_distance_to_error=tolerated_distance_to_error,
                )
                for cycle in self.cycles
            ]
        )


class DFLExperiment2D(DFLExperiment):
    """A class for performing the 2D DFL experiment (Fig 3 of the paper).

    Attrs:
        sampler: The cirq sampler to use.
        qubits: The grid of qubits.
        origin_qubit: One of the matter qubits.
        h: The coefficient on the gauge X term.
        mu: The coefficient on the matter sigma_x term.
        tau: The Trotter step size.
        rng: The pseudorandom number generator to use for readout benchmarking.
        cycles: The number of cycles for which to run the experiment.
        num_gc_instances: The number of gauge compiling instances to use.
        excited_qubits: Which qubits to excite.
        use_cphase: Whether to use cphase gates.
        include_zero_trotter: Whether to include circuits with the Trotter step set to 0.
    """

    def __init__(
        self,
        sampler: cirq.Sampler,
        qubits: list[cirq.GridQubit],
        origin_qubit: cirq.GridQubit,
        save_directory: str,
        h: float = 1.5,
        mu: float = 3.5,
        tau: float = 0.35,
        rng: np.random.Generator = np.random.default_rng(),
        cycles: np.ndarray = np.arange(31),
        num_gc_instances: int = 40,
        excited_qubits: list[cirq.GridQubit] = [cirq.GridQubit(0, 7)],
        use_cphase: bool = True,
        include_zero_trotter: bool = True,
    ):
        self.sampler = sampler
        self.h = h
        self.mu = mu
        self.tau = tau
        self.rng = rng
        self.cycles = cycles
        self.all_circuits: list[cirq.Circuit] = []
        self.e0 = np.array([])
        self.e1 = np.array([])
        self.num_gc_instances = num_gc_instances
        self.executor = concurrent.futures.ProcessPoolExecutor()
        self.readout_ideal_bitstrs = np.array([])
        self.save_directory = save_directory
        self.lgtdfl = dfl_2d.LGTDFL(qubits, origin_qubit, tau, h, mu)
        self.lgtdfl_zero_trotter = dfl_2d.LGTDFL(qubits, origin_qubit, 0.0, h, mu)
        self.qubits = self.lgtdfl.all_qubits
        self.excited_qubits = excited_qubits
        self.use_cphase = use_cphase
        self.origin_qubit = origin_qubit
        self.include_zero_trotter = include_zero_trotter
        self.matter_sites = np.array(self.lgtdfl._matter_indices())
        self.gauge_sites = np.array(self.lgtdfl._gauge_indices())

    def save_to_file(self, filename: str):
        """Save a dictionary containing parameters and measurements.

        Args:
            filename: The filename to save to.
        """

        d_to_save = {
            "measurements": self.measurements,
            "readout_ideal_bitstrs": self.readout_ideal_bitstrs,
            "h": self.h,
            "mu": self.mu,
            "tau": self.tau,
            "cycles": self.cycles,
            "num_gc_instances": self.num_gc_instances,
            "e0": self.e0,
            "e1": self.e1,
            "qubits": self.qubits,
            "save_directory": self.save_directory,
            "excited_qubits": self.excited_qubits,
            "use_cphase": self.use_cphase,
            "origin_qubit": self.origin_qubit,
            "include_zero_trotter": self.include_zero_trotter,
        }

        pickle.dump(d_to_save, open(filename, "wb"))

    def load_from_file(self, filename: str):
        """Load parameters and measurements from a dictionary.

        Args:
            filename: The filename to load from.
        """

        d = pickle.load(open(filename, "rb"))
        self.measurements = d["measurements"]
        self.readout_ideal_bitstrs = d["readout_ideal_bitstrs"]
        self.h = d["h"]
        self.mu = d["mu"]
        self.tau = d["tau"]
        self.cycles = d["cycles"]
        self.num_gc_instances = d["num_gc_instances"]
        self.e0 = d["e0"]
        self.e1 = d["e1"]
        self.qubits = d["qubits"]
        self.save_directory = d.get("save_directory", "temp")
        self.excited_qubits = d["excited_qubits"]
        self.use_cphase = d["use_cphase"]
        self.origin_qubit = d["origin_qubit"]
        self.include_zero_trotter = d.get("include_zero_trotter", True)

    def generate_circuit_instances(
        self,
        initial_state: str,
        ncycles: int,
        basis: str,
        gauge_compile: bool = True,
        dynamical_decouple: bool = True,
        dd_sequence: tuple[cirq.Gate, ...] = (cirq.X, cirq.Y, cirq.X, cirq.Y),
        zero_trotter: bool = False,
        qubits_to_fix_disorder: list[cirq.Qid] = [],
        disorder_pattern: list[bool] = [],
    ) -> list[cirq.Circuit]:
        """Generate the circuit instances for a given initial state, number of cycles, and
        measurement basis.

        Args:
            initial_state: Which initial state, either "gauge_invariant" or "superposition".
            ncycles: The number of Trotter steps (can be 0).
            basis: The basis in which to measure. Either "x" or "z".
            gauge_compile: Whether to apply gauge compiling.
            dynamical_decouple: Whether to apply dynamical decoupling.
            dd_sequence: The dynamical decoupling sequence to use.
            zero_trotter: Whether to set the trotter step size to 0 (used to calibrate error mitigation).
            qubits_to_fix_disorder: Qubits on which to fix a disorder pattern for the superposition initial state.
            disorder_pattern: The disorder pattern to fix.

        Returns:
            A list of the circuit instances.

        Raises:
            ValueError: If initial_state, ncycles, or basis is not valid.
        """

        for q in qubits_to_fix_disorder:
            assert (
                q in self.lgtdfl.matter_qubits
            ), "qubits_to_fix_disorder must all be matter qubits"

        if initial_state not in ["gauge_invariant", "single_sector", "superposition"]:
            raise ValueError(
                "initial_state should be 'gauge_invariant' or 'superposition' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if ncycles < 0:
            raise ValueError("ncycles must be nonnegative")
        if basis not in ["z", "x"]:
            raise ValueError("basis must be 'z' or 'x'")

        basis_index = ["z", "x"].index(basis)
        if initial_state == "gauge_invariant":
            initial_state = "single_sector"
        if zero_trotter:
            lgtdfl = self.lgtdfl_zero_trotter
        else:
            lgtdfl = self.lgtdfl
        circuit = lgtdfl.get_2d_dfl_experiment_circuits(
            initial_state,
            [ncycles],
            excited_qubits=self.excited_qubits,
            n_instances=1,
            two_qubit_gate=(
                "cphase_simultaneous" if self.use_cphase else "cz_simultaneous"
            ),
        )[basis_index]
        new_moment_0 = cirq.Moment(
            list(circuit[0].operations)
            + [
                cirq.Ry(rads=(1 - 2 * disorder) * np.pi / 2)(q)
                for q, disorder in zip(qubits_to_fix_disorder, disorder_pattern)
            ]
        )
        circuit = cirq.Circuit(new_moment_0) + circuit[1:]

        if gauge_compile:
            circuits = list(
                self.executor.map(
                    partial(_apply_gauge_compiling, circuit=circuit),
                    range(self.num_gc_instances),
                )
            )
        else:
            circuits = [circuit]

        if dynamical_decouple:
            circuits_dd = list(
                self.executor.map(
                    partial(cirq.add_dynamical_decoupling, schema=dd_sequence), circuits
                )
            )
            circuits = circuits_dd

        return circuits

    def post_select_measurements_on_distance(
        self,
        measurements: np.ndarray,
        tolerated_distance_to_error: int | float,
        operator_qubits: list[cirq.GridQubit],
    ) -> np.ndarray:
        """Return the subset of measurements where the errors occur at least a given distance from
        the operator qubits.

        Args:
            measurements: The non-post-selected measurements.
            tolerated_distance_to_error: Allow errors greater than this distance from operator_qubits.
            operator_qubits: The qubits on which we are measuring an operator.

        Returns:
            The post-selected measurements.
        """
        matter_qubits = self.lgtdfl.matter_qubits
        matter_indices = np.array(self.lgtdfl._matter_indices())
        distances = np.array(
            [
                min(_distance(q_operator, qubit) for q_operator in operator_qubits)
                for qubit in matter_qubits
            ]
        )
        errors = measurements[:, matter_indices]
        distance_to_error = np.min(np.where(errors, distances, np.inf), axis=1)
        return measurements[distance_to_error >= tolerated_distance_to_error]

    def extract_zzz_single_cycle(
        self,
        initial_state: str,
        cycle_number: int,
        zero_trotter: bool = False,
        readout_mitigate: bool = True,
        post_select: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ) -> np.ndarray:
        """Get the expectation value of the interaction term (ZZZ in the LGT basis but implemented
        here as Z_gauge in the dual basis).

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            cycle_number: The number of Trotter steps (can be 0).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if cycle_number < 0:
            raise ValueError("Cycle number should be nonnegative")
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        if readout_mitigate and len(self.e0) == 0 or len(self.e1) == 0:
            self.extract_readout_error_rates()

        measurements = self.extract_measurements(
            0,
            initial_state,
            cycle_number,
            zero_trotter,
            post_select and tolerated_distance_to_error == np.inf,
        )
        if post_select and tolerated_distance_to_error < np.inf:
            p1 = np.zeros(len(self.lgtdfl.gauge_qubits))
            dp1 = np.zeros(len(self.lgtdfl.gauge_qubits))
            for idx, (op_qubit, op_index) in enumerate(
                zip(self.lgtdfl.gauge_qubits, self.lgtdfl._gauge_indices())
            ):
                measurements_q = self.post_select_measurements_on_distance(
                    measurements, tolerated_distance_to_error, [op_qubit]
                )[:, op_index]
                print(
                    f"Interaction term, cycle {cycle_number}, qubit {op_qubit}, {len(measurements_q)} counts of {len(measurements)}"
                )
                if os.path.isfile("dfl_surviving_counts.pickle"):
                    log_dict = pickle.load(open("dfl_surviving_counts.pickle", "rb"))
                else:
                    log_dict = {
                        "gauge_x": [{} for _ in range(31)],
                        "matter_x": [{} for _ in range(31)],
                        "interaction": [{} for _ in range(31)],
                        "gauge_x_zero_trotter": [{} for _ in range(31)],
                        "matter_x_zero_trotter": [{} for _ in range(31)],
                        "interaction_zero_trotter": [{} for _ in range(31)],
                        "total_measurements": [0 for _ in range(31)],
                    }
                key = "interaction"
                if zero_trotter:
                    key += "_zero_trotter"
                log_dict[key][cycle_number][op_qubit] = len(measurements_q)
                log_dict["total_measurements"][cycle_number] = len(measurements)
                pickle.dump(log_dict, open("dfl_surviving_counts.pickle", "wb"))

                if len(measurements_q) == 0:
                    p1[idx] = np.nan
                    dp1[idx] = np.nan
                else:
                    p1[idx] = np.mean(measurements_q)
                    dp1[idx] = np.sqrt(p1[idx] * (1 - p1[idx]) / len(measurements_q))
        else:
            p1 = np.mean(measurements[:, self.lgtdfl._gauge_indices()], axis=0)
            dp1 = np.sqrt(p1 * (1 - p1) / len(measurements))
        if not readout_mitigate:
            x_dx = np.array([1 - 2 * p1, 2 * dp1])  # value, uncertainty
        else:
            x_raw = 1 - 2 * p1
            e0 = self.e0[self.lgtdfl._gauge_indices()]
            e1 = self.e1[self.lgtdfl._gauge_indices()]
            x_mitigated = (x_raw + e0 - e1) / (
                1.0 - e0 - e1
            )  # see Eq. F3 of https://arxiv.org/pdf/2106.01264
            x_dx = np.array([x_mitigated, 2 * dp1 / (1.0 - e0 - e1)])
        return x_dx

    def extract_gauge_x_single_cycle(
        self,
        initial_state: str,
        cycle_number: int,
        zero_trotter: bool = False,
        readout_mitigate: bool = True,
        post_select: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ) -> np.ndarray:
        """Get the expectation value of the h term, which is X_gauge in both the LGT and dual bases.

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            cycle_number: The number of Trotter steps (can be 0).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if cycle_number < 0:
            raise ValueError("Cycle number should be nonnegative")
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        if readout_mitigate and len(self.e0) == 0 or len(self.e1) == 0:
            self.extract_readout_error_rates()

        measurements = self.extract_measurements(
            1,
            initial_state,
            cycle_number,
            zero_trotter,
            post_select and tolerated_distance_to_error == np.inf,
        )
        if post_select and tolerated_distance_to_error < np.inf:
            p1 = np.zeros(len(self.lgtdfl.gauge_qubits))
            dp1 = np.zeros(len(self.lgtdfl.gauge_qubits))
            for idx, (op_qubit, op_index) in enumerate(
                zip(self.lgtdfl.gauge_qubits, self.lgtdfl._gauge_indices())
            ):
                measurements_q = self.post_select_measurements_on_distance(
                    measurements, tolerated_distance_to_error, [op_qubit]
                )[:, op_index]
                print(
                    f"Gauge X term, cycle {cycle_number}, qubit {op_qubit}, {len(measurements_q)} counts of {len(measurements)}"
                )

                if os.path.isfile("dfl_surviving_counts.pickle"):
                    log_dict = pickle.load(open("dfl_surviving_counts.pickle", "rb"))
                else:
                    log_dict = {
                        "gauge_x": [{} for _ in range(31)],
                        "matter_x": [{} for _ in range(31)],
                        "interaction": [{} for _ in range(31)],
                        "gauge_x_zero_trotter": [{} for _ in range(31)],
                        "matter_x_zero_trotter": [{} for _ in range(31)],
                        "interaction_zero_trotter": [{} for _ in range(31)],
                        "total_measurements": [0 for _ in range(31)],
                    }
                key = "gauge_x"
                if zero_trotter:
                    key += "_zero_trotter"
                log_dict[key][cycle_number][op_qubit] = len(measurements_q)
                log_dict["total_measurements"][cycle_number] = len(measurements)
                pickle.dump(log_dict, open("dfl_surviving_counts.pickle", "wb"))

                if len(measurements_q) == 0:
                    p1[idx] = np.nan
                    dp1[idx] = np.nan
                else:
                    p1[idx] = np.mean(measurements_q)
                    dp1[idx] = np.sqrt(p1[idx] * (1 - p1[idx]) / len(measurements_q))
        else:
            p1 = np.mean(measurements[:, self.lgtdfl._gauge_indices()], axis=0)
            dp1 = np.sqrt(p1 * (1 - p1) / len(measurements))
        if not readout_mitigate:
            x_dx = np.array([1 - 2 * p1, 2 * dp1])  # value, uncertainty
        else:
            x_raw = 1 - 2 * p1
            e0 = self.e0[self.lgtdfl._gauge_indices()]
            e1 = self.e1[self.lgtdfl._gauge_indices()]
            x_mitigated = (x_raw + e0 - e1) / (
                1.0 - e0 - e1
            )  # see Eq. F3 of https://arxiv.org/pdf/2106.01264
            x_dx = np.array([x_mitigated, 2 * dp1 / (1.0 - e0 - e1)])
        return x_dx

    def extract_matter_x_single_cycle(
        self,
        initial_state: str,
        cycle_number: int,
        zero_trotter: bool = False,
        readout_mitigate: bool = True,
        post_select: bool = False,
        tolerated_distance_to_error: int | float = np.inf,
    ) -> np.ndarray:
        """Get the expectation value of the mu term.

        This is X_matter in the LGT basis and a product of Xs on a matter site and the neighboring gauge sites in the dual basis.

        Args:
            initial_state: Either "superposition" or "gauge_invariant"
            cycle_number: The number of Trotter steps (can be 0).
            zero_trotter: Whether to set the time step to 0 (for calibrating error mitigation).
            readout_mitigate: Whether to use readout error mitigation.
            post_select: Whether to post select on the gauge charges (intended for the gauge_invariant initial state only).
            tolerated_distance_to_error: Allow errors greater than this distance.

        Returns:
            The expectation value and statistical uncertainty of the interaction term. Shape is [value/uncertainty, site_number].

        Raises:
            ValueError: If the input arguments are not allowed.
        """
        if initial_state not in ["single_sector", "gauge_invariant", "superposition"]:
            raise ValueError(
                "initial_state should be 'superposition' or 'gauge_invariant' (or 'single_sector' which is the same as 'gauge_invariant')"
            )
        if cycle_number < 0:
            raise ValueError("Cycle number should be nonnegative")
        if initial_state == "superposition" and post_select:
            raise ValueError(
                "Post selection is intended for the gauge invariant initial state."
            )

        if readout_mitigate and len(self.e0) == 0 or len(self.e1) == 0:
            self.extract_readout_error_rates()

        measurements = self.extract_measurements(
            1,
            initial_state,
            cycle_number,
            zero_trotter,
            post_select and tolerated_distance_to_error == np.inf,
        )
        x = []
        dx = []
        e0 = deepcopy(self.e0)
        e1 = deepcopy(self.e1)
        if post_select and readout_mitigate:
            for idx in self.lgtdfl._matter_indices():
                e0[idx] = 0.0
                e1[idx] = 0.0
        for q in self.lgtdfl.matter_qubits:
            qubits_i = [
                q_n for q_n in q.neighbors() if q_n in self.lgtdfl.all_qubits
            ] + [q]
            indices = np.array(
                [
                    self.lgtdfl.all_qubits.index(cast(cirq.GridQubit, qi))
                    for qi in qubits_i
                ]
            )
            if post_select and tolerated_distance_to_error < np.inf:
                measurements_i = self.post_select_measurements_on_distance(
                    measurements,
                    tolerated_distance_to_error,
                    cast(list[cirq.GridQubit], qubits_i),
                )[:, indices]
                print(
                    f"Matter X term, cycle {cycle_number}, qubit {q}, {len(measurements_i)} counts of {len(measurements)}"
                )

                if os.path.isfile("dfl_surviving_counts.pickle"):
                    log_dict = pickle.load(open("dfl_surviving_counts.pickle", "rb"))
                else:
                    log_dict = {
                        "gauge_x": [{} for _ in range(31)],
                        "matter_x": [{} for _ in range(31)],
                        "interaction": [{} for _ in range(31)],
                        "gauge_x_zero_trotter": [{} for _ in range(31)],
                        "matter_x_zero_trotter": [{} for _ in range(31)],
                        "interaction_zero_trotter": [{} for _ in range(31)],
                        "total_measurements": [0 for _ in range(31)],
                    }
                key = "matter_x"
                if zero_trotter:
                    key += "_zero_trotter"
                log_dict[key][cycle_number][q] = len(measurements_i)
                log_dict["total_measurements"][cycle_number] = len(measurements)
                pickle.dump(log_dict, open("dfl_surviving_counts.pickle", "wb"))

            else:
                measurements_i = measurements[:, indices]
            if readout_mitigate:
                single_qubit_cmats = []
                qubits_to_mitigate = []

                for i in indices:
                    e0_sq = e0[i]
                    e1_sq = e1[i]
                    single_qubit_cmats.append(
                        np.array([[1 - e0_sq, e1_sq], [e0_sq, 1 - e1_sq]])
                    )
                    qubits_to_mitigate.append(self.qubits[i])

                tcm = cirq.experiments.TensoredConfusionMatrices(
                    single_qubit_cmats,
                    [[q] for q in qubits_to_mitigate],
                    repetitions=measurements_i.shape[0],
                    timestamp=0.0,
                )

                x_i, dx_i = tcm.readout_mitigation_pauli_uncorrelated(
                    qubits_to_mitigate, measurements_i
                )
            else:
                repetitions = measurements_i.shape[0]
                p1 = np.mean(np.sum(measurements_i, axis=1) % 2)
                dp1 = np.sqrt(p1 * (1 - p1) / (repetitions * self.num_gc_instances))
                x_i = 1 - 2 * p1
                dx_i = 2 * dp1
            x.append(x_i)
            dx.append(dx_i)
        return np.array([x, dx])
