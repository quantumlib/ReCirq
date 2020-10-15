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

from typing import List, Optional

import numpy as np

from recirq.fermi_hubbard.execution import (
    ExperimentResult,
    ExperimentRun,
    FermiHubbardExperiment
)
from recirq.fermi_hubbard.layouts import (
    LineLayout
)
from recirq.fermi_hubbard.parameters import (
    FermiHubbardParameters,
    Hamiltonian,
    IndependentChainsInitialState,
    UniformSingleParticle
)
from recirq.fermi_hubbard.post_processing import InstanceBundle


def test_up_down_density() -> None:
    experiments = [
        _create_experiment(
            sites_count=3,
            trotter_steps=[1, 2],
            results_post_selected=[
                ExperimentResult(
                    up_density=(0.0, 1.0, 2.0),
                    down_density=(3.0, 4.0, 5.0),
                    measurements_count=1
                ),
                ExperimentResult(
                    up_density=(6.0, 7.0, 8.0),
                    down_density=(9.0, 10.0, 11.0),
                    measurements_count=1
                )
            ]
        ),
        _create_experiment(
            sites_count=3,
            trotter_steps=[1, 2],
            results_post_selected=[
                ExperimentResult(
                    up_density=(12.0, 13.0, 14.0),
                    down_density=(17.0, 16.0, 15.0),
                    measurements_count=1
                ),
                ExperimentResult(
                    up_density=(18.0, 19.0, 20.0),
                    down_density=(23.0, 22.0, 21.0),
                    measurements_count=1
                )
            ]
        )
    ]

    bundle = InstanceBundle(experiments)
    up_down_density = bundle.up_down_density(rescaled=False)

    np.testing.assert_allclose(
        up_down_density.values,
        [  # Chains
            [  # Trotter steps
                [  # Experiments
                    [0, 1, 2],  # Sites
                    [12, 13, 14]
                ],
                [
                    [6, 7, 8],
                    [18, 19, 20]
                ]
            ],
            [
                [
                    [3, 4, 5],
                    [17, 16, 15]
                ],
                [
                    [9, 10, 11],
                    [23, 22, 21]
                ]
            ]
        ]
    )

    np.testing.assert_allclose(
        up_down_density.average,
        [  # Chains
            [  # Trotter steps
                [6, 7, 8],  # Sites
                [12, 13, 14]
            ],
            [
                [10, 10, 10],
                [16, 16, 16]
            ]
        ]
    )

    std_dev_5 = np.sqrt(5 ** 2 + 5 ** 2)
    std_dev_6 = np.sqrt(6 ** 2 + 6 ** 2)
    std_dev_7 = np.sqrt(7 ** 2 + 7 ** 2)

    np.testing.assert_allclose(
        up_down_density.std_dev,
        [  # Chains
            [  # Trotter steps
                [std_dev_6, std_dev_6, std_dev_6],  # Sites
                [std_dev_6, std_dev_6, std_dev_6]
            ],
            [
                [std_dev_7, std_dev_6, std_dev_5],
                [std_dev_7, std_dev_6, std_dev_5]
            ]
        ]
    )

    std_error_5 = std_dev_5 / np.sqrt(2)
    std_error_6 = std_dev_6 / np.sqrt(2)
    std_error_7 = std_dev_7 / np.sqrt(2)

    np.testing.assert_allclose(
        up_down_density.std_error,
        [  # Chains
            [  # Trotter steps
                [std_error_6, std_error_6, std_error_6],  # Sites
                [std_error_6, std_error_6, std_error_6]
            ],
            [
                [std_error_7, std_error_6, std_error_5],
                [std_error_7, std_error_6, std_error_5]
            ]
        ]
    )


def test_up_down_position_average() -> None:
    experiments = [
        _create_experiment(
            sites_count=3,
            trotter_steps=[1, 2],
            results_post_selected=[
                ExperimentResult(
                    up_density=(0.0, 1.0, 2.0),
                    down_density=(3.0, 4.0, 5.0),
                    measurements_count=1
                ),
                ExperimentResult(
                    up_density=(6.0, 7.0, 8.0),
                    down_density=(9.0, 10.0, 11.0),
                    measurements_count=1
                )
            ]
        ),
        _create_experiment(
            sites_count=3,
            trotter_steps=[1, 2],
            results_post_selected=[
                ExperimentResult(
                    up_density=(12.0, 13.0, 14.0),
                    down_density=(17.0, 16.0, 15.0),
                    measurements_count=1
                ),
                ExperimentResult(
                    up_density=(18.0, 19.0, 20.0),
                    down_density=(23.0, 22.0, 21.0),
                    measurements_count=1
                )
            ]
        )
    ]

    bundle = InstanceBundle(experiments)
    up_down_position_average = bundle.up_down_position_average(rescaled=False)

    np.testing.assert_allclose(
        up_down_position_average.values,
        [  # Chains
            [  # Trotter steps
                [8, 80],  # Experiments
                [44, 116]
            ],
            [
                [26, 94],
                [62, 130]
            ]
        ]
    )

    np.testing.assert_allclose(
        up_down_position_average.average,
        [  # Chains
            [44, 80],  # Trotter steps
            [60, 96]
        ]
    )

    std_dev_34 = np.sqrt(34 ** 2 + 34 ** 2)
    std_dev_36 = np.sqrt(36 ** 2 + 36 ** 2)

    np.testing.assert_allclose(
        up_down_position_average.std_dev,
        [  # Chains
            [std_dev_36, std_dev_36],  # Trotter steps
            [std_dev_34, std_dev_34]
        ]
    )

    std_error_34 = std_dev_34 / np.sqrt(2)
    std_error_36 = std_dev_36 / np.sqrt(2)

    np.testing.assert_allclose(
        up_down_position_average.std_error,
        [  # Chains
            [std_error_36, std_error_36],  # Trotter steps
            [std_error_34, std_error_34]
        ]
    )


def test_charge_spin_density() -> None:
    experiments = [
        _create_experiment(
            sites_count=3,
            trotter_steps=[1, 2],
            results_post_selected=[
                ExperimentResult(
                    up_density=(0.0, 1.0, 2.0),
                    down_density=(3.0, 4.0, 5.0),
                    measurements_count=1
                ),
                ExperimentResult(
                    up_density=(6.0, 7.0, 8.0),
                    down_density=(9.0, 10.0, 11.0),
                    measurements_count=1
                )
            ]
        ),
        _create_experiment(
            sites_count=3,
            trotter_steps=[1, 2],
            results_post_selected=[
                ExperimentResult(
                    up_density=(12.0, 13.0, 14.0),
                    down_density=(17.0, 16.0, 15.0),
                    measurements_count=1
                ),
                ExperimentResult(
                    up_density=(18.0, 19.0, 20.0),
                    down_density=(23.0, 22.0, 21.0),
                    measurements_count=1
                )
            ]
        )
    ]

    bundle = InstanceBundle(experiments)
    charge_spin = bundle.charge_spin_density(rescaled=False)

    np.testing.assert_allclose(
        charge_spin.values,
        [  # Chains
            [  # Trotter steps
                [  # Experiments
                    [3, 5, 7],  # Sites
                    [29, 29, 29]
                ],
                [
                    [15, 17, 19],
                    [41, 41, 41]
                ]
            ],
            [
                [
                    [-3, -3, -3],
                    [-5, -3, -1]
                ],
                [
                    [-3, -3, -3],
                    [-5, -3, -1]
                ]
            ]
        ]
    )

    np.testing.assert_allclose(
        charge_spin.average,
        [  # Chains
            [  # Trotter steps
                [16, 17, 18],  # Sites
                [28, 29, 30]
            ],
            [
                [-4, -3, -2],
                [-4, -3, -2]
            ]
        ]
    )

    std_dev_0 = np.sqrt(0 ** 2 + 0 ** 2)
    std_dev_1 = np.sqrt(1 ** 2 + 1 ** 2)
    std_dev_11 = np.sqrt(11 ** 2 + 11 ** 2)
    std_dev_12 = np.sqrt(12 ** 2 + 12 ** 2)
    std_dev_13 = np.sqrt(13 ** 2 + 13 ** 2)


    np.testing.assert_allclose(
        charge_spin.std_dev,
        [  # Chains
            [  # Trotter steps
                [std_dev_13, std_dev_12, std_dev_11],  # Sites
                [std_dev_13, std_dev_12, std_dev_11]
            ],
            [
                [std_dev_1, std_dev_0, std_dev_1],
                [std_dev_1, std_dev_0, std_dev_1]
            ]
        ]
    )

    std_error_0 = std_dev_0 / np.sqrt(2)
    std_error_1 = std_dev_1 / np.sqrt(2)
    std_error_11 = std_dev_11 / np.sqrt(2)
    std_error_12 = std_dev_12 / np.sqrt(2)
    std_error_13 = std_dev_13 / np.sqrt(2)

    np.testing.assert_allclose(
        charge_spin.std_error,
        [  # Chains
            [  # Trotter steps
                [std_error_13, std_error_12, std_error_11],  # Sites
                [std_error_13, std_error_12, std_error_11]
            ],
            [
                [std_error_1, std_error_0, std_error_1],
                [std_error_1, std_error_0, std_error_1]
            ]
        ]
    )


def test_charge_spin_spreading() -> None:
    experiments = [
        _create_experiment(
            sites_count=3,
            trotter_steps=[1, 2],
            results_post_selected=[
                ExperimentResult(
                    up_density=(0.0, 1.0, 2.0),
                    down_density=(3.0, 4.0, 5.0),
                    measurements_count=1
                ),
                ExperimentResult(
                    up_density=(6.0, 7.0, 8.0),
                    down_density=(9.0, 10.0, 11.0),
                    measurements_count=1
                )
            ]
        ),
        _create_experiment(
            sites_count=3,
            trotter_steps=[1, 2],
            results_post_selected=[
                ExperimentResult(
                    up_density=(12.0, 13.0, 14.0),
                    down_density=(17.0, 16.0, 15.0),
                    measurements_count=1
                ),
                ExperimentResult(
                    up_density=(18.0, 19.0, 20.0),
                    down_density=(23.0, 22.0, 21.0),
                    measurements_count=1
                )
            ]
        )
    ]

    bundle = InstanceBundle(experiments)
    charge_spin_spreading = bundle.charge_spin_spreading(rescaled=False)

    np.testing.assert_allclose(
        charge_spin_spreading.values,
        [  # Chains
            [  # Trotter steps
                [10, 58],  # Experiments
                [34, 82]
            ],
            [
                [-6, -6],
                [-6, -6]
            ]
        ]
    )

    np.testing.assert_allclose(
        charge_spin_spreading.average,
        [  # Chains
            [34, 58],  # Trotter steps
            [-6, -6]
        ]
    )

    std_dev_0 = np.sqrt(0 ** 2 + 0 ** 2)
    std_dev_24 = np.sqrt(24 ** 2 + 24 ** 2)

    np.testing.assert_allclose(
        charge_spin_spreading.std_dev,
        [  # Chains
            [std_dev_24, std_dev_24],  # Trotter steps
            [std_dev_0, std_dev_0]
        ]
    )

    std_error_0 = std_dev_0 / np.sqrt(2)
    std_error_24 = std_dev_24 / np.sqrt(2)

    np.testing.assert_allclose(
        charge_spin_spreading.std_error,
        [  # Chains
            [std_error_24, std_error_24],  # Trotter steps
            [std_error_0, std_error_0]
        ]
    )


def test_post_selection() -> None:
    experiments = [
        _create_experiment(
            sites_count=1,
            trotter_steps=[1, 2],
            results=[
                ExperimentResult(
                    up_density=(0.0,),
                    down_density=(0.0,),
                    measurements_count=10
                ),
                ExperimentResult(
                    up_density=(0.0,),
                    down_density=(0.0,),
                    measurements_count=10
                )
            ],
            results_post_selected=[
                ExperimentResult(
                    up_density=(0.0,),
                    down_density=(0.0,),
                    measurements_count=3
                ),
                ExperimentResult(
                    up_density=(0.0,),
                    down_density=(0.0,),
                    measurements_count=4
                )
            ]
        ),
        _create_experiment(
            sites_count=1,
            trotter_steps=[1, 2],
            results=[
                ExperimentResult(
                    up_density=(0.0,),
                    down_density=(0.0,),
                    measurements_count=10
                ),
                ExperimentResult(
                    up_density=(0.0,),
                    down_density=(0.0,),
                    measurements_count=10
                )
            ],
            results_post_selected=[
                ExperimentResult(
                    up_density=(0.0,),
                    down_density=(0.0,),
                    measurements_count=5
                ),
                ExperimentResult(
                    up_density=(0.0,),
                    down_density=(0.0,),
                    measurements_count=6
                )
            ]
        )
    ]

    bundle = InstanceBundle(experiments)
    post_selection = bundle.post_selection()

    np.testing.assert_allclose(
        post_selection.values,
        [  # Chains
            [  # Trotter steps
                [3 / 10, 5 / 10],  # Experiments
                [4 / 10, 6 / 10]
            ]
        ]
    )

    np.testing.assert_allclose(
        post_selection.average,
        [  # Chains
            [4 / 10, 5 / 10]  # Trotter steps
        ]
    )

    std_dev_1_10 = np.sqrt((1 / 10) ** 2 + (1 / 10) ** 2)

    np.testing.assert_allclose(
        post_selection.std_dev,
        [  # Chains
            [std_dev_1_10, std_dev_1_10]  # Trotter steps
        ]
    )

    std_error_1_10 = std_dev_1_10 / np.sqrt(2)

    np.testing.assert_allclose(
        post_selection.std_error,
        [  # Chains
            [std_error_1_10, std_error_1_10]  # Trotter steps
        ]
    )


def _create_experiment(
        sites_count: int,
        trotter_steps: List[int],
        results: Optional[List[ExperimentResult]] = None,
        results_post_selected: Optional[List[ExperimentResult]] = None
) -> FermiHubbardExperiment:

    def create_null_results() -> List[ExperimentResult]:
        return [
            ExperimentResult(up_density=(0.0,) * sites_count,
                             down_density=(0.0,) * sites_count,
                             measurements_count=0)
            for _ in range(len(trotter_steps))
        ]

    if results is None:
        results = create_null_results()

    if results_post_selected is None:
        results_post_selected = create_null_results()

    runs = []
    for step, result, results_post_selected in zip(
            trotter_steps, results, results_post_selected):
        runs.append(ExperimentRun(
            trotter_steps=step,
            end_timestamp_sec=0.0,
            result=result,
            result_post_selected=results_post_selected)
        )

    return FermiHubbardExperiment(
        parameters=FermiHubbardParameters(
            hamiltonian=Hamiltonian(
                sites_count=sites_count,
                j=1.0,
                u=2.0
            ),
            initial_state=IndependentChainsInitialState(
                up=UniformSingleParticle(),
                down=UniformSingleParticle()
            ),
            layout=LineLayout(size=sites_count),
            dt=0.3
        ),
        runs=runs
    )
