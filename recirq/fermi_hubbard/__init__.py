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

from recirq.fermi_hubbard.circuits import (
    align_givens_circuit,
    create_initial_circuit,
    create_line_circuits,
    create_line_trotter_circuit,
    create_line_trotter_step_circuit,
    create_measurement_circuit,
    create_one_particle_circuit,
    create_zigzag_circuits,
    create_zigzag_trotter_circuit,
    create_zigzag_trotter_step_circuit,
    run_in_parallel
)
from recirq.fermi_hubbard.converting_sampler import (
    ConvertingSampler
)
from recirq.fermi_hubbard.data_plotting import (
    default_top_label,
    plot_quantity,
    quantity_data_frame
)
from recirq.fermi_hubbard.decomposition import (
    CPhaseEchoGate,
    ConvertToNonUniformSqrtIswapGates,
    DecomposeCallable,
    Decomposition,
    GateDecompositionError,
    ParticleConservingParameters,
    decompose_preserving_moments
)
from recirq.fermi_hubbard.execution import (
    ExperimentResult,
    ExperimentRun,
    FermiHubbardExperiment,
    create_circuits,
    extract_result,
    load_experiment,
    run_experiment,
    save_experiment
)
from recirq.fermi_hubbard.layouts import (
    LineLayout,
    ZigZagLayout
)
from recirq.fermi_hubbard.parameters import (
    FermiHubbardParameters,
    FixedSingleParticle,
    FixedTrappingPotential,
    GaussianTrappingPotential,
    Hamiltonian,
    IndependentChainsInitialState,
    InitialState,
    PhasedGaussianSingleParticle,
    UniformSingleParticle,
    UniformTrappingPotential
)
from recirq.fermi_hubbard.post_processing import (
    AggregatedQuantity,
    InstanceBundle,
    PerSiteQuantity,
    Rescaling,
    apply_rescalings_to_bundles,
    find_bundles_rescalings
)
