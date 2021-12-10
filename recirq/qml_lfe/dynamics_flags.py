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

"""Executable flags used for learn_dynamics_*** modules."""

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("n", None, "System size.")
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
    "save_dir",
    "./recirq/qml_lfe/data",
    "Path to save experiment data (must already exist).",
)

flags.DEFINE_bool("use_engine", False, "Whether or not to use quantum engine.")
