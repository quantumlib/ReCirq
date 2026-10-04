# Copyright 2026 Google
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

import recirq.qec_memory.analysis as qec_analysis
import numpy as np
import matplotlib.pyplot as plt
import copy


def test_fit_logical_error_per_cycle():
    cycles = np.arange(1, 101)
    amp = 0.8
    epsilon = 0.01
    fidelity = amp * (1 - 2 * epsilon) ** cycles
    lep = (1 - fidelity) / 2
    d_lep = 0.001
    popt, cov = qec_analysis.fit_logical_error_per_cycle(cycles, lep, d_lep)
    assert np.isclose(popt[0], amp)
    assert np.isclose(popt[1], epsilon)


def test_lambda_experiment_results():
    cycles = np.arange(1, 201)
    for num_sweep_bit_choices in [1, 4]:
        lambda_results = qec_analysis.LambdaExperimentResults(
            cycles_list=copy.deepcopy(cycles),
            repetitions=1000,
            num_sweep_bit_choices=num_sweep_bit_choices,
        )
        epsilon_7 = 1e-3
        epsilon_5 = 2 * epsilon_7
        epsilon_3 = 2 * epsilon_5
        amp = 0.8
        cycles_all = np.repeat(cycles, num_sweep_bit_choices)
        rng = np.random.default_rng(0)
        for distance, epsilon in [(3, epsilon_3), (5, epsilon_5), (7, epsilon_7)]:
            for shift in [(0, 0), (0, 4), (4, 0)]:
                for basis in ["H", "V"]:
                    rng.shuffle(cycles_all)
                    fidelity = amp * (1 - 2 * epsilon) ** cycles_all
                    lep = (1 - fidelity) / 2
                    lambda_results.add_result(
                        qec_analysis.SurfaceCodeParams(
                            distance=distance, shift=shift, observable=basis
                        ),
                        lep,
                        copy.deepcopy(cycles_all),
                    )
        lambda_results.fit_exponential()
        assert np.isclose(lambda_results.fitted_ler[3], epsilon_3)
        assert np.isclose(lambda_results.fitted_ler[5], epsilon_5)
        assert np.isclose(lambda_results.fitted_ler[7], epsilon_7)
        for d in [3, 5, 7]:
            assert lambda_results.d_fitted_ler[d] < 1e-5
            assert np.isclose(lambda_results.fitted_scale[d], amp)
            assert lambda_results.d_fitted_scale[d] < 1e-5
        fig, ax = plt.subplots(dpi=150, facecolor="white")
        ax = lambda_results.plot(ax=ax)
        num_lines = len(ax.get_lines())
        assert num_lines == 30
        fig.canvas.draw()  # Force mathtext parsing of labels and annotations.
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend_labels == [
            "$\\varepsilon_{3}=0.400\\% \\pm 0.000\\%$",
            "$\\varepsilon_{5}=0.200\\% \\pm 0.000\\%$",
            "$\\varepsilon_{7}=0.100\\% \\pm 0.000\\%$",
            "$d = 3$",
            "$d = 5$",
            "$d = 7$",
        ]
        plt.close(fig)
