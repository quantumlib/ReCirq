from functools import lru_cache
from typing import Optional

from cirq.protocols.json_serialization import DEFAULT_RESOLVERS, ObjectFactory

from .analysis import OverlapAnalysisData, OverlapAnalysisParams
from .blueprint import BlueprintData, BlueprintParamsRobustShadow, BlueprintParamsTrialWf
from .experiment import ExperimentData, SimulatedExperimentParams
from .hamiltonian import HamiltonianData, LoadFromFileHamiltonianParams, PyscfHamiltonianParams
from .trial_wf import (
    FermionicMode,
    LayerSpec,
    PerfectPairingPlusTrialWavefunctionParams,
    TrialWavefunctionData,
)


@lru_cache()
def _resolve_json(cirq_type: str) -> Optional[ObjectFactory]:
    """Resolve the types of `recirq.qcqmc.` json objects.

    This is a Cirq JSON resolver suitable for appending to
    `cirq.protocols.json_serialization.DEFAULT_RESOLVERS`.
    """
    if not cirq_type.startswith('recirq.qcqmc.'):
        return None

    cirq_type = cirq_type[len('recirq.qcqmc.') :]
    return {
        k.__name__: k
        for k in [
            LoadFromFileHamiltonianParams,
            PyscfHamiltonianParams,
            HamiltonianData,
            FermionicMode,
            LayerSpec,
            PerfectPairingPlusTrialWavefunctionParams,
            TrialWavefunctionData,
            BlueprintParamsTrialWf,
            BlueprintParamsRobustShadow,
            BlueprintData,
            SimulatedExperimentParams,
            ExperimentData,
            OverlapAnalysisParams,
            OverlapAnalysisData,
        ]
    }.get(cirq_type, None)


DEFAULT_RESOLVERS.append(_resolve_json)
