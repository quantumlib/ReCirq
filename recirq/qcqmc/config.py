# Copyright 2024 Google
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

import os
import pathlib

import attrs


@attrs.frozen
class OutputDirectories:
    """Default output directories for qcqmc data."""

    DEFAULT_HAMILTONIAN_DIRECTORY: str = "./data/hamiltonians/"
    DEFAULT_TRIAL_WAVEFUNCTION_DIRECTORY: str = "./data/trial_wfs/"
    DEFAULT_QMC_DIRECTORY: str = "./data/afqmc/"
    DEFAULT_BLUEPRINT_DIRECTORY: str = "./data/blueprints/"
    DEFAULT_EXPERIMENT_DIRECTORY: str = "./data/experiments/"
    DEFAULT_ANALYSIS_DIRECTORY: str = "./data/analyses/"

    def make_output_directories(self) -> None:
        """Make the output directories given in OUTDIRS"""
        for _, dirpath in attrs.asdict(self).items():
            try:
                os.makedirs(f"{dirpath}")
            except FileExistsError:
                pass


OUTDIRS = OutputDirectories()

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
DEVICE_LOCK_PATH = pathlib.Path(ROOT_DIR + "/data/data_taking.lock")

SINGLE_PRECISION_DEFAULT = True
DO_INVERSE_SIMULATION_QUBIT_NUMBER_CUTOFF = 6

# Controls a variety of print statement.
VERBOSE_EXECUTION = True
