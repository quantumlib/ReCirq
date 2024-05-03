import cirq

DEFAULT_RESOLVERS = list(cirq.DEFAULT_RESOLVERS)


def _internal_register_resolver(resolver: cirq.JsonResolver):
    DEFAULT_RESOLVERS.append(resolver)


def read_json(*args, **kwargs):
    kwargs["resolvers"] = DEFAULT_RESOLVERS
    return cirq.read_json(*args, **kwargs)


def assert_json_roundtrip_works(obj):
    return cirq.testing.assert_json_roundtrip_works(obj, resolvers=DEFAULT_RESOLVERS)
