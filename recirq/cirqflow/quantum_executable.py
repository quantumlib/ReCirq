import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, Union, List, Optional, Tuple, Hashable, Dict, Sequence

import cirq.work
from cirq import NamedTopology
from cirq.protocols import dataclass_json_dict


class ExecutableSpec(metaclass=abc.ABCMeta):
    executable_family: str = NotImplemented


@dataclass(frozen=True)
class Histogrammer:
    pass


@dataclass(frozen=True)
class FrozenCollectionOfPauliSum:
    pass


class StructuredCircuit(cirq.Circuit):
    pass


class RunAndMeasureCircuit(cirq.Circuit):
    # TODO: what if something is structured *and* run-and-measure.
    pass


@dataclass(frozen=True)
class Bitstrings:
    n_repetitions: int
    measure_qubits: Optional[Tuple[cirq.Qid]] = None

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


TParamPair = Tuple[cirq.TParamKey, cirq.TParamVal]


@dataclass(frozen=True)
class QuantumExecutable:
    """An executable quantum program.

    This serves a similar purpose to `cirq.Circuit` with some key differences. First, a quantum
    executable contains all the relevant context for execution including parameters as well as
    the desired number of repetitions. Second, this object is immutable. Finally, there are
    optional fields enabling a higher level of abstraction for certain aspects of the executable.

    Attributes:
        circuit: A circuit describing the quantum operations to execute.
        measurement: A description of the type of measurement. Please see the documentation for
            each possible class type for more information. The lowest level of abstraction is
            to use MeasurementGate in your circuit and specify
            `measurement=Bitstrings(n_repetitions)`.
        params: An immutable version of cirq.ParamResolver represented as a tuple of key value
            pairs.
        info: Additional metadata about this executable that is not used by the quantum runtime.
            A tuple of key value pairs where the key is a string and the value is any immutable,
            hashable value.
        problem_topology: Description of the multiqubit gate topology present in the circuit.
            If not specified, the circuit must handle compatibility with device topology.
        initial_state: How to initialize the quantum system before running `circuit`. If not
            specified, the device will be initialized into the all-zeros state.
        uuid: A unique identifer for this executable. This will be automatically generated and
            should not be set by the user unless you are reconstructing a serialized executable.
    """
    circuit: cirq.FrozenCircuit
    measurement: Union[Bitstrings, FrozenCollectionOfPauliSum, Histogrammer]
    params: Tuple[TParamPair, ...] = None
    spec: ExecutableSpec = None
    problem_topology: NamedTopology = None
    initial_state: cirq.ProductState = None

    def __init__(self,
                 circuit: cirq.AbstractCircuit,
                 measurement: Union[Bitstrings, FrozenCollectionOfPauliSum, Histogrammer],
                 params: Union[Tuple[TParamPair, ...], cirq.ParamResolverOrSimilarType] = None,
                 spec: ExecutableSpec = None,
                 problem_topology: NamedTopology = None,
                 initial_state: cirq.ProductState = None,
                 ):
        """Initialize the quantum executable.

        The actual fields in this class are immutable, but we allow more liberal input types
        which will be frozen in this __init__ method.

        Args:
            circuit: The circuit. This will be frozen before set as an attribute
            measurement: A description of the type of measurement. Please see the documentation for
                each possible class type for more information.
            params: A cirq.ParamResolverOrSimilarType which will be frozen into a tuple of
                key value pairs.
            info: Additional metadata about this executable that is not used by the quantum runtime.
                If specified as a dictioanry, this will be frozen into a tuple of key value pairs.
            problem_topology: Description of the multiqubit gate topology present in the circuit.
            initial_state: How to initialize the quantum system before running `circuit`.
            uuid: A unique identifer for this executable. This will be automatically generated and
                should not be set by the user unless you are reconstructing a serialized executable.
        """

        # We care a lot about mutability in this class. No object is truly immutable in Python,
        # but we can get pretty close by following the example of dataclass(frozen=True), which
        # deletes this class's __setattr__ magic method. To set values ever, we use
        # object.__setattr__ in this __init__ function.
        #
        # We write our own __init__ function to be able to accept a wider range of input formats
        # that can be easily converted to our native, immutable format.
        object.__setattr__(self, 'circuit', circuit.freeze())

        if not isinstance(measurement, (Bitstrings, FrozenCollectionOfPauliSum, Histogrammer)):
            raise ValueError(f"measurement should be a Bitstrings, FrozenCollectionOfPauliSum, "
                             f"or Histogrammer instance, not {measurement}.")
        object.__setattr__(self, 'measurement', measurement)

        if isinstance(params, tuple) and \
                all(isinstance(param_kv, tuple) and len(param_kv) == 2 for param_kv in params):
            frozen_params = params
        elif isinstance(params, list) and \
                all(isinstance(param_kv, list) and len(param_kv) == 2 for param_kv in params):
            frozen_params = tuple((k, v) for k, v in params)
        else:
            param_resolver = cirq.ParamResolver(params)
            frozen_params = tuple(param_resolver.param_dict.items())
        object.__setattr__(self, 'params', frozen_params)

        if spec is None:
            raise ValueError()

        if dataclasses.is_dataclass(spec):
            # TODO: check for frozen?
            # TODO: update typing info or flatten into tuple
            frozen_spec = spec
        elif isinstance(spec, tuple) and \
                all(isinstance(info_kv, tuple) and len(info_kv) == 2 for info_kv in spec):
            frozen_spec = spec
        else:
            frozen_spec = tuple(spec.items())
        object.__setattr__(self, 'spec', frozen_spec)

        if problem_topology is not None and not isinstance(problem_topology, NamedTopology):
            raise ValueError(f"problem_topology should be a NamedTopology, "
                             f"not {problem_topology}.")
        object.__setattr__(self, 'problem_topology', problem_topology)

        if initial_state is not None and not isinstance(initial_state, cirq.ProductState):
            raise ValueError(f"initial_state should be a ProductState, not {initial_state}.")
        object.__setattr__(self, 'initial_state', initial_state)

        object.__setattr__(self, '_hash', hash(dataclasses.astuple(self)))

    # def __repr__(self):
    #     return f'QuantumExecutable(info={self.info_dict()})'

    def __str__(self):
        return f'QuantumExecutable(spec={self.spec})'

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


def _recurse_html_executablegroup(qexegroup: Union[QuantumExecutable, 'QuantumExecutableGroup'],
                                  depth=0):
    if isinstance(qexegroup, QuantumExecutableGroup):
        s = ' ' * depth + f'<li>{qexegroup}<ul>\n'

        for child in qexegroup.executables:
            s += _recurse_html_executablegroup(child, depth=depth + 1)
        s += ' ' * depth + '</ul></li>\n'
        return s

    s = ' ' * depth + '<li>' + str(qexegroup) + '</li>\n'
    return s


@dataclass(frozen=True)
class QuantumExecutableGroup:
    executables: Tuple[Union[QuantumExecutable, 'QuantumExecutableGroup'], ...]
    info: Tuple[Tuple[str, Hashable], ...] = None

    def __init__(self,
                 executables: Sequence[Union[QuantumExecutable, 'QuantumExecutableGroup']],
                 info: Union[Tuple[Tuple[str, Hashable], ...], Dict[str, Hashable]] = None,
                 ):

        if not isinstance(executables, tuple):
            executables = tuple(executables)
        object.__setattr__(self, 'executables', executables)

        if info is None:
            info = tuple()
        if isinstance(info, tuple) and \
                all(isinstance(info_kv, tuple) and len(info_kv) == 2 for info_kv in info):
            frozen_info = info
        else:
            frozen_info = tuple(info.items())
        object.__setattr__(self, 'info', frozen_info)

        object.__setattr__(self, '_hash', hash(dataclasses.astuple(self)))

    def __len__(self):
        return len(self.executables)

    def __iter__(self):
        yield from self.executables

    def info_dict(self):
        return dict(self.info)

    # def __repr__(self):
    #     return f'QuantumExecutable(info={self.info_dict()})'

    def __str__(self):
        return f'QuantumExecutable(info={self.info_dict()})'

    def __hash__(self):
        return self._hash

    def _repr_html_(self):
        return '<ul>\n' + _recurse_html_executablegroup(self) + '</ul>\n'

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


InfoTuple = Tuple[Tuple[str, Any], ...]


def _recurse_flatten_executablegroup(
        qexegroup: Union[QuantumExecutable, QuantumExecutableGroup],
        flat_list: List[Tuple[QuantumExecutable, InfoTuple, Tuple[QuantumExecutableGroup, ...]]],
        info: InfoTuple,
        parents: Tuple[QuantumExecutableGroup, ...]
):
    new_info = dict(info)  # todo: dict-like access to info
    pg_info = dict(qexegroup.info)  # todo: dict-like access to info
    for k in pg_info:
        if k in new_info:
            raise ValueError("Key already exists")
        new_info[k] = pg_info[k]
    new_info = tuple(new_info.items())

    if isinstance(qexegroup, QuantumExecutableGroup):
        for child in qexegroup.executables:
            _recurse_flatten_executablegroup(
                child, flat_list=flat_list, info=new_info, parents=(qexegroup,) + parents)
    else:
        flat_list.append((qexegroup, new_info, parents))


def flatten_executable_group(qexegroup: QuantumExecutableGroup):
    flat_list = []
    _recurse_flatten_executablegroup(qexegroup, flat_list, (), ())
    return flat_list
