import recirq.algo_benchmarks.loschmidt.loschmidt
from recirq.algo_benchmarks.loschmidt.loschmidt import *


def test_create_tilted_square_lattice_loschmidt_echo_circuit():
    topology = TiltedSquareLattice(width=3, height=2)
    macrocycle_depth = 2
    circuit = create_tilted_square_lattice_loschmidt_echo_circuit(
        topology=topology, macrocycle_depth=macrocycle_depth, rs=np.random.RandomState(52)
    )
    assert isinstance(circuit, cirq.FrozenCircuit)

    assert len(circuit.all_qubits()) == topology.n_nodes
    assert sorted(circuit.all_qubits()) == sorted(topology.nodes_as_gridqubits())

    edge_coloring_n = 4  # grid
    forward_backward = 2
    n_moment_per_microcycle = 2  # layer of single- and two- qubit gate
    measure_moment = 1
    extra_single_q_layer = 1
    assert len(circuit) == (edge_coloring_n * macrocycle_depth *
                            n_moment_per_microcycle + extra_single_q_layer) \
           * forward_backward + measure_moment


def test_tilted_square_lattice_loschmidt_spec(tmpdir):
    topology = TiltedSquareLattice(width=3, height=2)
    macrocycle_depth = 2
    spec1 = TiltedSquareLatticeLoschmidtSpec(
        topology=topology,
        macrocycle_depth=macrocycle_depth,
        instance_i=0,
        n_repetitions=10_000,
    )
    assert spec1.executable_family == 'recirq.otoc.loschmidt'

    fn = f'{tmpdir}/spec.json'
    cirq.to_json(spec1, fn)
    spec2 = cirq.read_json(fn)
    assert spec1 == spec2


def test_get_all_tilted_square_lattice_executables(monkeypatch):
    call_count = 0

    def mock_get_circuit(topology: TiltedSquareLattice, macrocycle_depth: int,
                         twoq_gate: cirq.Gate = cirq.FSimGate(np.pi / 4, 0.0),
                         rs: cirq.RANDOM_STATE_OR_SEED_LIKE = None):
        nonlocal call_count
        call_count += 1
        return cirq.Circuit()

    monkeypatch.setattr(recirq.algo_benchmarks.loschmidt.loschmidt,
                        "create_tilted_square_lattice_loschmidt_echo_circuit", mock_get_circuit)
    get_all_tilted_square_lattice_executables_flat()  # todo renamed flat -> ""
    n_instances = 10
    n_macrocycle_depths = 4  # 2,4,6,8
    n_side_lengths = 4  # width or height # of possibilities
    n_topos = n_side_lengths * (n_side_lengths + 1) / 2
    assert call_count == n_instances * n_macrocycle_depths * n_topos


def test_tilted_square_lattice_loschmidt_data(tmpdir):
    topology = TiltedSquareLattice(width=3, height=2)
    macrocycle_depth = 2
    spec = TiltedSquareLatticeLoschmidtSpec(
        topology=topology,
        macrocycle_depth=macrocycle_depth,
        instance_i=0,
        n_repetitions=10_000,
    )
    rt_info = QuantumRuntimeInfo(
        run_id='my-test-run-id',
        placement={cirq.LineQubit(4): cirq.GridQubit(1, 1)},
        timings={'placement': 0.123, 'sampler_run': 5.55}
    )
    data1 = TiltedSquareLatticeLoschmidtData.from_nested(
        spec=spec,
        rt_info=rt_info,
        success_probability=0.123,
    )

    fn = f'{tmpdir}/data.json'
    cirq.to_json(data1, fn)
    data2: TiltedSquareLatticeLoschmidtData = cirq.read_json(fn)
    assert data1 == data2
    assert hash(data1) == hash(data2)

    assert data2.width == spec.topology.width
    assert data2.height == spec.topology.height
    assert data2.n_qubits == spec.topology.n_nodes
    assert data2.success_probability == 0.123
