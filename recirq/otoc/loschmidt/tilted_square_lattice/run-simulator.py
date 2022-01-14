import cirq
from cirq_google.workflow import QuantumRuntimeConfiguration, execute
from recirq.cirqflow.quantum_backend import SimulatedEngineProcessorWithLocalDevice
from recirq.cirqflow.run_utils import get_unique_run_id
from recirq.otoc.loschmidt.tilted_square_lattice import TiltedSquareLatticeLoschmidtSpec

assert TiltedSquareLatticeLoschmidtSpec, 'register deserializer'

FN = 'loschmidt.tilted_square_lattice.small-v1.json.gz'


def main():
    exegroup = cirq.read_json_gzip(FN)
    rt_config = QuantumRuntimeConfiguration(
        processor=SimulatedEngineProcessorWithLocalDevice('rainbow', noise_strength=0.005),
        run_id=get_unique_run_id('simulated-{i}'),
        random_seed=52,
    )
    raw_results = execute(rt_config, exegroup)
    print("Finished run_id", raw_results.shared_runtime_info.run_id)


if __name__ == '__main__':
    main()
