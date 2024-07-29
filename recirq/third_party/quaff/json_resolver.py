from typing import Optional, cast

from cirq.protocols.json_serialization import ObjectFactory


def json_resolver(quaff_type: str) -> Optional[ObjectFactory]:
    if quaff_type.startswith("quaff"):
        return cast(
            ObjectFactory,
            eval(f"__import__('recirq').third_party.{quaff_type}", {}, {}),
        )
    return None
