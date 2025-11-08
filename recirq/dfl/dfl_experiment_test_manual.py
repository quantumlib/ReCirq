import os

import cirq
import numpy as np

from . import dfl_experiment as dfl

if not os.path.isdir("test_circuits"):
    os.mkdir("test_circuits")


def test_dfl_experiment_3x3():
    """Performs a comprehensive integration test of the 2D DFL experiment setup.

    Note: This test is designed to be run manually as it is time-consuming.
    """
    qubits = cirq.GridQubit.rect(3, 3, 0, 0)
    sampler = cirq.Simulator()
    for num_gc_instances in [1, 2]:
        for use_cphase in [False, True]:
            dfl_expt = dfl.DFLExperiment2D(
                sampler,
                qubits,
                origin_qubit=cirq.q(0, 0),
                save_directory="test_circuits",
                num_gc_instances=num_gc_instances,
                excited_qubits=[cirq.q(0, 1)],
                cycles=np.array([0, 1]),
                use_cphase=use_cphase,
            )
            for gauge_compile in [False, True]:
                for dynamical_decouple in [False, True]:
                    dfl_expt.run_experiment(
                        gauge_compile=gauge_compile,
                        dynamical_decouple=dynamical_decouple,
                        repetitions_post_selection=np.ones(
                            len(dfl_expt.cycles), dtype=int
                        )
                                                   * 10_000,
                    )

                    # check readout error rates
                    assert np.all(dfl_expt.e0 == np.zeros(len(dfl_expt.qubits)))
                    assert np.all(dfl_expt.e1 == np.zeros(len(dfl_expt.qubits)))

                    # check gauge_x
                    signs = np.ones(len(dfl_expt.lgtdfl._gauge_indices()))
                    signs[0] = -1
                    gauge_expected = dfl_expt.h / np.sqrt(dfl_expt.h ** 2 + 1) * signs

                    for readout_mitigate in [False, True]:
                        for initial_state in ["gauge_invariant", "superposition"]:
                            for zero_trotter in [False, True]:
                                for post_select in [False, True]:
                                    if post_select and initial_state == "superposition":
                                        continue
                                    else:
                                        gauge_x = dfl_expt.extract_gauge_x(
                                            initial_state,
                                            readout_mitigate=readout_mitigate,
                                            post_select=post_select,
                                            zero_trotter=zero_trotter,
                                        )
                                        assert np.all(
                                            np.isclose(
                                                gauge_x[0, 0],
                                                gauge_expected,
                                                atol=5 * gauge_x[0, 1],
                                            )
                                        )
                                        if zero_trotter:
                                            assert np.all(
                                                np.isclose(
                                                    gauge_x[0, 0],
                                                    gauge_x[1, 0],
                                                    atol=5
                                                         * np.sqrt(
                                                        gauge_x[0, 1] ** 2
                                                        + gauge_x[1, 1] ** 2
                                                    ),
                                                )
                                            )

                    # check matter_x
                    matter_expected = []
                    for q_matter in dfl_expt.lgtdfl.matter_qubits:
                        neighbor_indices = np.array(
                            [
                                list(dfl_expt.lgtdfl.gauge_qubits).index(q)
                                for q in q_matter.neighbors()
                                if q in dfl_expt.qubits
                            ]
                        )
                        matter_expected.append(
                            np.prod(gauge_expected[neighbor_indices])
                        )

                    for readout_mitigate in [False, True]:
                        for post_select in [False, True]:
                            for zero_trotter in [False, True]:
                                matter_x = dfl_expt.extract_matter_x(
                                    "gauge_invariant",
                                    readout_mitigate=readout_mitigate,
                                    post_select=post_select,
                                    zero_trotter=zero_trotter,
                                )
                                assert np.all(
                                    np.isclose(
                                        matter_x[0, 0],
                                        matter_expected,
                                        atol=5 * matter_x[0, 1],
                                    )
                                )
                                if zero_trotter:
                                    assert np.all(
                                        np.isclose(
                                            matter_x[0, 0],
                                            matter_x[1, 0],
                                            atol=5
                                                 * np.sqrt(
                                                matter_x[0, 1] ** 2
                                                + matter_x[1, 1] ** 2
                                            ),
                                        )
                                    )

                    for readout_mitigate in [False, True]:
                        for zero_trotter in [False, True]:
                            matter_x = dfl_expt.extract_matter_x(
                                "superposition",
                                readout_mitigate=readout_mitigate,
                                post_select=False,
                                zero_trotter=zero_trotter,
                            )
                            assert np.all(
                                np.isclose(matter_x[0, 0], 0.0, atol=5 * matter_x[0, 1])
                            )
                            if zero_trotter:
                                assert np.all(
                                    np.isclose(
                                        matter_x[0, 0],
                                        matter_x[1, 0],
                                        atol=5
                                             * np.sqrt(
                                            matter_x[0, 1] ** 2 + matter_x[1, 1] ** 2
                                        ),
                                    )
                                )

                    # check interaction
                    interaction_expected = 1 / np.sqrt(dfl_expt.h ** 2 + 1) * signs
                    for readout_mitigate in [False, True]:
                        for initial_state in ["gauge_invariant", "superposition"]:
                            for zero_trotter in [False, True]:
                                for post_select in [False, True]:
                                    if post_select and initial_state == "superposition":
                                        continue
                                    else:
                                        interaction = dfl_expt.extract_interaction(
                                            initial_state,
                                            readout_mitigate=readout_mitigate,
                                            post_select=post_select,
                                            zero_trotter=zero_trotter,
                                        )
                                        assert np.all(
                                            np.isclose(
                                                interaction[0, 0],
                                                interaction_expected,
                                                atol=5 * interaction[0, 1],
                                            )
                                        )
                                        if zero_trotter:
                                            assert np.all(
                                                np.isclose(
                                                    interaction[0, 0],
                                                    interaction[1, 0],
                                                    atol=5
                                                         * np.sqrt(
                                                        interaction[0, 1] ** 2
                                                        + interaction[1, 1] ** 2
                                                    ),
                                                )
                                            )
