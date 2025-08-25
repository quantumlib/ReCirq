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

from functools import lru_cache
from typing import Optional

from cirq.protocols.json_serialization import ObjectFactory, DEFAULT_RESOLVERS
from .sk_model import (
    SKModelQAOASpec,
)


@lru_cache()
def _resolve_json(cirq_type: str) -> Optional[ObjectFactory]:
    """Resolve the types of `recirq.qaoa.` json objects.

    This is a Cirq JSON resolver suitable for appending to
    `cirq.protocols.json_serialization.DEFAULT_RESOLVERS`.
    """
    if not cirq_type.startswith('recirq.qaoa.'):
        return None

    cirq_type = cirq_type[len('recirq.qaoa.'):]
    return {k.__name__: k for k in [
        SKModelQAOASpec,
    ]}.get(cirq_type, None)


DEFAULT_RESOLVERS.append(_resolve_json)
