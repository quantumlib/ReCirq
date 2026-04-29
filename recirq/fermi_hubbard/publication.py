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
"""Data specific to experiment published in arXiv:2010.07965."""

from io import BytesIO
from copy import deepcopy
import os
import requests
from typing import Callable, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np

from recirq.fermi_hubbard.decomposition import (
    ConvertToNonUniformSqrtIswapGates,
    ParticleConservingParameters
)
from recirq.fermi_hubbard.layouts import (
    LineLayout,
    QubitsLayout,
    ZigZagLayout
)
from recirq.fermi_hubbard.parameters import (
    FermiHubbardParameters,
    GaussianTrappingPotential,
    Hamiltonian,
    IndependentChainsInitialState,
    PhasedGaussianSingleParticle,
    UniformTrappingPotential
)


def gaussian_1u1d_instance(layout: QubitsLayout,
                           u: float,
                           dt: float = 0.3) -> FermiHubbardParameters:
    """Predefined instance of two colliding Gaussian wavepackets.

    Args:
        layout: Layout with qubits mapping.
        u: Value of the interaction strength.
        dt: Trotter step length.

    Returns:
        Colliding Gaussian wavepackets problem parameters.
    """

    hamiltonian = Hamiltonian(
        sites_count=layout.size,
        j=1.0,
        u=u
    )

    initial_state = IndependentChainsInitialState(
        up=PhasedGaussianSingleParticle(
            k=1.2 * 7,
            sigma=1.2 / 7,
            position=1.5 / 7
        ),
        down=PhasedGaussianSingleParticle(
            k=-1.2 * 7,
            sigma=1.2 / 7,
            position=5.5 / 7)
    )

    return FermiHubbardParameters(
        hamiltonian=hamiltonian,
        initial_state=initial_state,
        layout=layout,
        dt=dt
    )


def trapping_instance(layout: QubitsLayout,
                      u: float,
                      dt: float = 0.3,
                      up_particles: int = 2,
                      down_particles: int = 2) -> FermiHubbardParameters:
    """Predefined initial state with up chain initialized by trapping potential.

    Args:
        layout: Layout with qubits mapping.
        u: Value of the interaction strength.
        dt: Trotter step length.
        up_particles: Number of up particles.
        down_particles: Number of down particles.

    Returns:
        Up particles trapped in Gaussian potential problem parameters.
    """

    hamiltonian = Hamiltonian(
        sites_count=layout.size,
        j=1.0,
        u=u
    )

    initial_state = IndependentChainsInitialState(
        up=GaussianTrappingPotential(
            particles=up_particles,
            center=0.5,
            sigma=1 / 7,
            scale=-4
        ),
        down=UniformTrappingPotential(particles=down_particles)
    )

    return FermiHubbardParameters(
        hamiltonian=hamiltonian,
        initial_state=initial_state,
        layout=layout,
        dt=dt
    )


def parasitic_cphase_compensation(
        cphase_angle: float
) -> Callable[[FermiHubbardParameters], FermiHubbardParameters]:
    """Transformation of problem parameters which account for parasitic cphase.

    This transformation compensates for parasitic cphase effects by adding the
    nearest-neighbor interaction terms V to the problem Hamiltonian, which are
    dependent on the qubits layout used.

    The result of this function can be passed as a value to the
    numerics_transform argument of the InstanceBundle class.

    Args:
        cphase_angle: Average parasitic cphase angle value over all the
            two-qubit interactions.

    Returns:
        Fermi-Hubbard problem parameters transformation function which adds the
        parasitic cphase compensation by adding nearest-neighbor interaction
        terms V to the Hamiltonian.
    """

    def compensate(parameters: FermiHubbardParameters
                   ) -> FermiHubbardParameters:

        cphase = cphase_angle / parameters.dt
        if isinstance(parameters.layout, ZigZagLayout):
            v = np.zeros(parameters.sites_count - 1)
            v[0::2] = 2.0 * cphase
            v[1::2] = 4.0 * cphase
        elif isinstance(parameters.layout, LineLayout):
            v = np.full(parameters.sites_count - 1, 2.0 * cphase)
        else:
            raise ValueError(f'Unsupported layout {parameters.layout}')

        v_parameters = deepcopy(parameters)
        v_parameters.hamiltonian.v = tuple(v)
        return v_parameters

    return compensate


def ideal_sqrt_iswap_converter() -> ConvertToNonUniformSqrtIswapGates:
    """Creates a converter which can decompose circuits to sqrt_iswap gate set.
    """
    return ConvertToNonUniformSqrtIswapGates(
        parameters={},
        parameters_when_missing=ParticleConservingParameters(
            theta=np.pi / 4,
            delta=0.0,
            chi=0.0,
            gamma=0.0,
            phi=0.0
        )
    )


def google_sqrt_iswap_converter() -> ConvertToNonUniformSqrtIswapGates:
    """Creates a converter which can decompose circuits to imperfect sqrt_iswap
    gate set.

    This converter assumes that each sqrt_iswap gate is really a
    cirq.FSim(π/4, π/24) gate.
    """
    return ConvertToNonUniformSqrtIswapGates(
        parameters={},
        parameters_when_missing=ParticleConservingParameters(
            theta=np.pi / 4,
            delta=0.0,
            chi=0.0,
            gamma=0.0,
            phi=np.pi / 24
        )
    )


def rainbow23_layouts(sites_count: int = 8) -> Tuple[ZigZagLayout]:
    """Creates a list of 16 that can be run on 23-qubit sub-grid of Rainbow
    processor.
    """
    return tuple(ZigZagLayout(size=sites_count,
                              origin=origin,
                              rotation=rotation,
                              flipped=flip,
                              exchange_chains=exchange,
                              reverse_chains=reverse)
                 for origin, rotation in (((4, 1), 0), ((8, 5), 180))
                 for flip in (False, True)
                 for exchange in (False, True)
                 for reverse in (False, True))


def fetch_publication_data(
        base_dir: str = "fermi_hubbard_data",
        exclude: Optional[List[str]] = None,
        auth_token: Optional[str] = None,
) -> None:
    """Downloads and extracts publication data from the Dryad repository at
    https://doi.org/10.5061/dryad.crjdfn32v, saving to disk.

    The following data are downloaded and saved:

    - gaussians_1u1d_nofloquet
    - gaussians_1u1d
    - trapping_2u2d
    - trapping_3u3d

    unless they already exist in the `base_dir` or are present in `exclude`.

    Args:
        base_dir: Base directory as a relative path to save data in.
            Set to "fermi_hubbard_data" if not specified.
        exclude: List of data to skip while downloading. See above for options.
        auth_token: OAuth2 token for Dryad API.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    api_url = 'https://datadryad.org/api/v2'

    # These IDs are specific to version 1 of DOI 10.5061/dryad.crjdfn32v
    # If the dataset was updated, these IDs might have changed!
    data = {
        "gaussians_1u1d_nofloquet": "451326",
        "gaussians_1u1d": "451327",
        "trapping_2u2d": "451328",
        "trapping_3u3d": "451329"
    }

    if exclude:
        data = {path: key for path, key in data.items() if path not in exclude}

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {auth_token}"
    }

    for file_name, file_id in data.items():
        print(f"Downloading {file_name}...")

        if os.path.exists(os.path.join(base_dir, file_name)):
            print("Data already exists. Skipping.\n")
            continue

        # Correct endpoint for file download
        url = f"{api_url}/files/{file_id}/download"

        # stream=True is better for large zip files
        response = requests.get(url, headers=headers, stream=True)

        # If you still get a 400, this will print the server's explanation
        if response.status_code != 200:
            print(f"Server returned {response.status_code}: {response.text}")
            continue

        with ZipFile(BytesIO(response.content)) as zfile:
            zfile.extractall(base_dir)
            print(f"Successfully downloaded and extracted {file_name}.\n")

