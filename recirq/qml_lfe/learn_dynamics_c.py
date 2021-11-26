# Copyright 2021 Google
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

"""Draw simplified shadows data (Y-basis) from 1D random scrambling or tsym
style circuits.

learn_dynamics.py --n=6 --depth=5 --n_data=20 --batch_size=5 --n_shots=1000

Will create 40 total circuits, 20 scrambling and 20 tsym circuits all on
6 qubits with depth 5. Once the circuits are generated, batch_size circuits
will be sent for simulation/execution, drawing n_shots samples from each
one using `run_batch` in Cirq. By default the bitstring data will be saved
in the data folder. One can also set `use_engine` to True in order to
run this against a processor on quantum engine.
"""

from typing import List

import os
import cirq
import cirq_google
import numpy as np
import circuit_blocks

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


flags.DEFINE_integer("n", None, "System size (total qubits == 2 * n).")
flags.DEFINE_integer("depth", None, "Circuit depth (block-wise).")
flags.DEFINE_integer(
    "n_data",
    20,
    "Number of circuits generated for each class (total circuits == 2 * n_data).",
)

flags.DEFINE_integer(
    "batch_size",
    5,
    "Number of circuits to send over the wire per batch (value does not affect results).",
)

flags.DEFINE_integer(
    "n_shots", 2000, "Number of measurements to draw from each individual circuit."
)

flags.DEFINE_string(
    "save_dir", "./data", "Path to save experiment data (must already exist)."
)

flags.DEFINE_bool("use_engine", False, "Whether or not to use quantum engine.")


def flatten_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """Pack operations in circuit to the left as far as possible."""
    return cirq.Circuit([op for mom in circuit for op in mom])


def build_circuit(
    qubit_pairs: List[List[cirq.Qid]], use_tsym: bool, depth: int
) -> cirq.Circuit:
    """Generate 1D tsym or scrambling circuit with given depth on qubit_pairs."""
    inter_gen = circuit_blocks.scrambling_block
    if use_tsym:
        # Use tsym gates.
        inter_gen = circuit_blocks.tsym_block

    # Random source for circuits.
    random_source = np.random.uniform(0, 4, size=(depth * len(qubit_pairs), 2))

    ret_circuit = circuit_blocks.block_1d_circuit(
        [qubits[0] for qubits in qubit_pairs], depth, inter_gen, random_source
    )
    ret_circuit += [cirq.S(qubits[0]) for qubits in qubit_pairs]
    ret_circuit += [cirq.H(qubits[0]) for qubits in qubit_pairs]

    # Merge single qubit gates together and add measurements.
    cirq.merge_single_qubit_gates_into_phxz(ret_circuit)
    cirq.DropEmptyMoments().optimize_circuit(circuit=ret_circuit)
    ret_circuit = flatten_circuit(ret_circuit)

    for i, qubit in enumerate([qubits[0] for qubits in qubit_pairs]):
        ret_circuit += cirq.measure(qubit, key=f"q{i}")

    cirq.SynchronizeTerminalMeasurements().optimize_circuit(circuit=ret_circuit)
    logging.debug(
        f"Generated a new circuit w/ tsym={use_tsym} and depth {len(ret_circuit)}"
    )
    return ret_circuit


def _engine_sim_workaround(batch: List[cirq.Circuit]) -> List[cirq.Result]:
    """Use either simulator or engine."""
    if FLAGS.use_engine:
        project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
        engine = cirq_google.Engine(project_id=project_id)
        job = engine.run_batch(
            programs=batch,
            gate_set=cirq_google.SYC_GATESET,
            processor_ids=["weber"],
            repetitions=FLAGS.n_shots,
        )
        return job.results()

    sim = cirq.Simulator()
    return [sim.run(circuit, repetitions=FLAGS.n_shots) for circuit in batch]


def main(_):
    logging.info("Beginning conventional circuit generation.")
    # Choose system pairs so that they are consecutive neighbors.
    system_pairs = [
        [cirq.GridQubit(1, 5), cirq.GridQubit(0, 5)],
        [cirq.GridQubit(1, 6), cirq.GridQubit(0, 6)],
        [cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)],
        [cirq.GridQubit(2, 7), cirq.GridQubit(1, 7)],
        [cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)],
        [cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)],
        [cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)],
        [cirq.GridQubit(5, 6), cirq.GridQubit(4, 6)],
        # Avoid dead qubit 5, 4.
        # [cirq.GridQubit(5, 5), cirq.GridQubit(5, 4)],
        [cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)],
        [cirq.GridQubit(6, 5), cirq.GridQubit(5, 5)],
        [cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)],
        [cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)],
        [cirq.GridQubit(7, 3), cirq.GridQubit(7, 2)],
        [cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)],
        [cirq.GridQubit(6, 2), cirq.GridQubit(6, 1)],
        [cirq.GridQubit(5, 2), cirq.GridQubit(5, 1)],
        [cirq.GridQubit(4, 2), cirq.GridQubit(4, 1)],
        [cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)],
        [cirq.GridQubit(3, 3), cirq.GridQubit(3, 2)],
        [cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)],
        [cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)],
        [cirq.GridQubit(2, 5), cirq.GridQubit(2, 4)],
    ]
    system_pairs = system_pairs[: FLAGS.n]

    to_run_scramb = [
        build_circuit(system_pairs, False, FLAGS.depth) for _ in range(FLAGS.n_data)
    ]
    to_run_tsym = [
        build_circuit(system_pairs, True, FLAGS.depth) for _ in range(FLAGS.n_data)
    ]

    logging.info(
        f"Circuit generation complete. Generated {len(to_run_tsym) + len(to_run_scramb)} total circuits"
    )
    for k in range(0, FLAGS.n_data, FLAGS.batch_size):
        logging.info(f"Running batch: [{k}-{k + FLAGS.batch_size}) / {FLAGS.n_data}")
        for is_tsym in [0, 1]:

            # Alternate between scrambling and tsym to
            # mitigate drift related noise artifacts.
            batch = to_run_scramb[k : k + FLAGS.batch_size]
            if is_tsym:
                batch = to_run_tsym[k : k + FLAGS.batch_size]

            # Upload and run the circuit batch.
            results = _engine_sim_workaround(batch)

            for j, single_circuit_samples in enumerate(results):
                name0 = (
                    f"1D-scramble-C-size-{FLAGS.n}"
                    f"-depth-{FLAGS.depth}"
                    f"-type-{is_tsym}"
                    f"-batch-{k}-number-{j}"
                )
                qubit_order = [f"q{i}" for i in range(FLAGS.n)]
                out0 = single_circuit_samples.data[qubit_order].to_numpy()
                np.save(os.path.join(FLAGS.save_dir, name0), out0)
                logging.debug("Saved: " + name0)


if __name__ == "__main__":
    app.run(main)
