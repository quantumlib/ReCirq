# Copyright 2019 The Cirq Developers
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

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

import dataclasses
from matplotlib import pyplot as plt
import cirq

CrossEntropyPair = NamedTuple(
    "CrossEntropyPair", [("num_cycle", int), ("xeb_fidelity", float)]
)
SpecklePurityPair = NamedTuple(
    "SpecklePurityPair", [("num_cycle", int), ("purity", float)]
)


@dataclasses.dataclass(frozen=True)
class CrossEntropyResult:
    """Results from a cross-entropy benchmarking (XEB) experiment.

    May also include results from speckle purity benchmarking (SPB) performed
    concomitantly.

    Attributes:
        data: A sequence of NamedTuples, each of which contains two fields:
            num_cycle: the circuit depth as the number of cycles, where
            a cycle consists of a layer of single-qubit gates followed
            by a layer of two-qubit gates.
            xeb_fidelity: the XEB fidelity after the given cycle number.
        repetitions: The number of circuit repetitions used.
        purity_data: A sequence of NamedTuples, each of which contains two
            fields:
            num_cycle: the circuit depth as the number of cycles, where
            a cycle consists of a layer of single-qubit gates followed
            by a layer of two-qubit gates.
            purity: the purity after the given cycle number.
    """

    data: List[CrossEntropyPair]
    repetitions: int
    purity_data: Optional[List[SpecklePurityPair]] = None

    def plot(self, ax: Optional[plt.Axes] = None, **plot_kwargs: Any) -> plt.Axes:
        """Plots the average XEB fidelity vs the number of cycles.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        num_cycles = [d.num_cycle for d in self.data]
        fidelities = [d.xeb_fidelity for d in self.data]
        ax.set_ylim([0, 1.1])
        ax.plot(num_cycles, fidelities, "ro-", **plot_kwargs)
        ax.set_xlabel("Number of Cycles")
        ax.set_ylabel("XEB Fidelity")
        if show_plot:
            fig.show()
        return ax

    @classmethod
    def _from_json_dict_(cls, data, repetitions, **kwargs):
        purity_data = kwargs.get("purity_data", None)
        if purity_data is not None:
            purity_data = [SpecklePurityPair(d, f) for d, f in purity_data]
        return cls(
            data=[CrossEntropyPair(d, f) for d, f in data],
            repetitions=repetitions,
            purity_data=purity_data,
        )

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    def __repr__(self) -> str:
        args = (
            f"data={[tuple(p) for p in self.data]!r}, repetitions={self.repetitions!r}"
        )
        if self.purity_data is not None:
            args += f", purity_data={[tuple(p) for p in self.purity_data]!r}"
        return f"cirq.experiments.CrossEntropyResult({args})"


@dataclasses.dataclass
class CrossEntropyResultDict(Mapping[Tuple["cirq.Qid", ...], CrossEntropyResult]):
    """Per-qubit-tuple results from cross-entropy benchmarking.

    Attributes:
        results: Dictionary from qubit tuple to cross-entropy benchmarking
            result for that tuple.
    """

    results: Dict[Tuple["cirq.Qid", ...], CrossEntropyResult]

    def _json_dict_(self) -> Dict[str, Any]:
        return {"results": list(self.results.items())}

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    @classmethod
    def _from_json_dict_(
        cls, results: List[Tuple[List[cirq.Qid], CrossEntropyResult]], **kwargs
    ) -> "CrossEntropyResultDict":
        return cls(results={tuple(qubits): result for qubits, result in results})

    def __repr__(self) -> str:
        return f"cirq.experiments.CrossEntropyResultDict(results={self.results!r})"

    def __getitem__(self, key: Tuple[cirq.Qid, ...]) -> CrossEntropyResult:
        return self.results[key]

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)
