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


# The version number is defined here.
# We mandate that at least Python 3.6 is required.
# This file is `exec`-ed by setup.py to maintain a single source of the
# version number.
# https://packaging.python.org/guides/single-sourcing-package-version/

import sys

if sys.version_info < (3, 6, 0):
    # coverage: ignore
    raise SystemError("ReCirq requires at least Python 3.6")

__version__ = "0.1.dev"
