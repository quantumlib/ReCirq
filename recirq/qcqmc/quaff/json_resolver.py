from typing import cast, Optional

from cirq.protocols.json_serialization import ObjectFactory


def json_resolver(quaff_type: str) -> Optional[ObjectFactory]:
    if quaff_type.startswith("qc_afqmc.quaff."):
        return cast(
            ObjectFactory, eval(f"__import__('qc_afqmc').{quaff_type.split('.', 1)[1]}", {}, {})
        )
    return None
