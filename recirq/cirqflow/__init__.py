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
from functools import lru_cache
from typing import Optional

from cirq.protocols.json_serialization import DEFAULT_RESOLVERS, ObjectFactory

@lru_cache()
def resolve_cirq_google(cirq_type: str) -> Optional[ObjectFactory]:
    if not cirq_type.startswith('cirq.google.'):
        return None

    cirq_type = cirq_type[len('cirq.google.'):]
    import recirq.cirqflow.quantum_executable as qe
    import recirq.cirqflow.quantum_runtime as qrt
    import recirq.cirqflow.qubit_placement as qp
    return {
        'Bitstrings': qe.Bitstrings,
        'QuantumExecutable': qe.QuantumExecutable,
        'QuantumExecutableGroup': qe.QuantumExecutableGroup,
        'QCSBackend': qrt.QCSBackend,
        'SimulatorBackend': qrt.SimulatorBackend,
        'SharedRuntimeInfo': qrt.SharedRuntimeInfo,
        'QuantumRuntimeInfo': qrt.QuantumRuntimeInfo,
        'RawExecutableResult': qrt.RawExecutableResult,
        'ExecutionResult': qrt.ExecutionResult,
        'QuantumRuntimeConfiguration': qrt.QuantumRuntimeConfiguration,
        'NaiveQubitPlacer': qp.NaiveQubitPlacer,

    }.get(cirq_type, None)


DEFAULT_RESOLVERS.append(resolve_cirq_google)
