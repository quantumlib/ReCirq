from typing import Optional, cast

from cirq.protocols.json_serialization import ObjectFactory


def json_resolver(quaff_type: str) -> Optional[ObjectFactory]:
    if quaff_type.startswith("recirq.third_party.quaff."):
        to_import = '.'.join(quaff_type.split('.')[1:])
        return cast(
            ObjectFactory,
            eval(f"__import__('recirq').{to_import}", {}, {}),
        )
    elif quaff_type.startswith("quaff"):
        to_import = f'third_party.{quaff_type}' 
        return cast(
            ObjectFactory,
            eval(f"__import__('recirq').{to_import}", {}, {}),
        )
    return None
