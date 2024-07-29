from typing import Optional, cast

from cirq.protocols.json_serialization import ObjectFactory

from recirq.third_party import quaff


def json_resolver(quaff_type: str) -> Optional[ObjectFactory]:
    if quaff_type.startswith("quaff."):
        return cast(
            ObjectFactory,
            getattr(quaff, quaff_type[6:])
        )
    return None
