"""
  Class to calculate latency and repetition rate.

  This class should replicate the acquisition of date for
  rep rate and latency for processor data sheets.
"""
import random
import cirq
import cirq_google as cg
import time
import sympy

from typing import List, Optional, Tuple


class RepRateCalculator:
    """Class to perform simple timing experiments using hardware.

    This class calculates the repetition rate for various
    types of circuits.  This is the approximate type per repetition
    (or shot) to execute this on the device.  This rate varies widely
    based on width, depth, and amount of sweeps/batching in the
    circuit.  This class will help test various rates on the device.

    This class is used to generate statistics on the processor datasheets
    for latency and repetition rates.
    """

    def __init__(self,
                 project_id: str,
                 processor_id: str,
                 gate_set: Optional[cg.SerializableGateSet] = None,
                 device: Optional[cirq.Device] = None,
                 sampler: Optional[cirq.Sampler] = None):
        """Initialize the tester object.

        Args:
            project_id: Cloud project id as a string
            processor_id: Processor to test on
            gate_set: gate set to use.  Defaults to square root of iswap
            device: device object to use. defaults to device provided by engine
            sampler: sampler object to use.  defaults to engine sampler
        """
        if not device or not sampler:
            engine = cg.Engine(project_id=project_id)
        gate_set = gate_set or cg.SQRT_ISWAP_GATESET
        self.device = device or engine.get_processor(processor_id).get_device(
            gate_sets=[gate_set])
        self.sampler = sampler or engine.sampler(processor_id=processor_id,
                                                 gate_set=gate_set)
        self.gate = cirq.ISWAP**0.5
        self.qubits = list(self.device.qubits)
        if gate_set == cg.SYC_GATESET:
            self.gate = cirq.google.SYC

        # log of print statements for testing
        self.print_log = ''

    def _flush_print_log(self) -> None:
        """Remove print log used fro debugging."""
        self.print_log = ''

    def _print(self, s: str) -> None:
        """Includes printing to testing log and flushing."""
        self.print_log += s
        self.print_log += '\n'
        print(s, flush=True)

    def _latency_circuit(self) -> cirq.Circuit:
        """Circuit for measuring round-trip latency.

        Use the most simple circuit possible, a measurement of a single
        random qubit.
        """
        return cirq.Circuit(cirq.measure(random.choice(self.qubits)))

    def _entangling_layer(self, qubits: List[cirq.Qid]) -> cirq.Moment:
        """Creates a layer of random two-qubit gates."""
        m = cirq.Moment()
        for q in qubits:
            if q in m.qubits:
                continue
            pairings = [
                q + adj
                for adj in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if q + adj in qubits and q + adj not in m.qubits
            ]
            if not pairings:
                continue
            q2 = random.choice(pairings)
            m = m.with_operation(self.gate(q, q2))
        return m

    def _sq_layer(self, qubits: List[cirq.Qid], parameterized: bool,
                  symbol_start: int) -> (cirq.Moment, int):
        """Creates a layer of single-qubit gates.

        If parameteried is true, this will add symbols to the qubits
        in order to test parameter resolution.
        """
        m = cirq.Moment()
        current_sym = symbol_start
        for q in qubits:
            if parameterized:
                symbol = f's_{current_sym}'
                current_sym += 1
                m = m.with_operation(cirq.X(q)**sympy.Symbol(symbol))
            else:
                m = m.with_operation(
                    cirq.PhasedXZGate(x_exponent=random.random(),
                                      z_exponent=random.random(),
                                      axis_phase_exponent=random.random())(q))
        return (m, current_sym)

    def _create_rep_rate_circuit(
            self, parameterized: bool, width: int, depth: int,
            num_sweeps: Optional[int]) -> Tuple[cirq.Circuit, cirq.Sweep]:
        """Creates a testing circuit based on parameters.

        The circuit will be of alternating single and two qubit layers
        using a number of qubits specified by width and a number of
        moments specified by depth.

        This function will also create a sweep if parameterized is true,
        by parameterized all single qubit layers with random values for
        angles.
        """
        qubits = self.qubits[:width]
        c = cirq.Circuit()
        symbol_count = 0
        for layer in range(depth):
            if layer % 2:
                moment, symbol_count = self._sq_layer(qubits, parameterized,
                                                      symbol_count)
                c.append(moment)
            else:
                c.append(self._entangling_layer(qubits))
        c.append(cirq.Moment(cirq.measure(*qubits)))

        if not parameterized:
            return (c, None)
        sweeps = cirq.Zip(*[
            cirq.Linspace('s_%d' % i, start=0, stop=1, length=num_sweeps)
            for i in range(symbol_count)
        ])
        return (c, sweeps)

    def get_samples(self, circuit: cirq.Circuit,
                    num_trials: int) -> List[float]:
        """Use a sampler to execute the circuit num_trial times.

        Returns:
           a list of durations, in seconds for the executions.
        """
        latencies = []
        for trial in range(num_trials):
            self._print(f'Executing Trial {trial}')
            start = time.time()
            self.sampler.run(circuit, repetitions=100)
            end = time.time()
            duration = end - start
            latencies.append(duration)
            self._print(f'Finished Trial {trial} with duration {duration}')
        return latencies

    def get_latency_samples(self, num_trials: int) -> List[float]:
        return self.get_samples(self._latency_circuit(), num_trials)

    def print_latency(self, num_trials: int) -> None:
        self._flush_print_log()
        latencies = self.get_latency_samples(num_trials)
        latencies = sorted(latencies)
        median = latencies[len(latencies) // 2]
        p05 = latencies[len(latencies) // 20]
        p95 = latencies[-len(latencies) // 20]
        p10 = latencies[len(latencies) // 20]
        p90 = latencies[-len(latencies) // 20]
        average = sum(latencies) / len(latencies)
        # Set an internal variable for testing
        self._print(str(latencies))
        self._print(f'Latency: Median={median}, p05={p05}, p10={p10}, ' +
                    f'p90={p90}, p95={p95}, avg={average}')

    def print_rep_rate(self,
                       parameterized: Optional[bool] = True,
                       width: Optional[int] = None,
                       depth: Optional[int] = 20,
                       num_sweeps: Optional[int] = 10,
                       repetitions: Optional[int] = 1000,
                       num_trials: Optional[int] = 10,
                       subtract_latency: Optional[float] = 0.0) -> None:
        """Executes a circuit to determine the repetition rate of the processor.

        Creates a circuit of alternating single and two qubit layers with the
        specified parameters.  Executes it the specified number of times
        to get an estimate for the repetition rate (how many repetitions of
        a circuit execution per second) for the given circuit.

        Displays the latencies for each circuit followed by the median,
        5,10,90, and 95th percentile values for latency and rep rate.
        Statistics are also logged to the member value print_log for
        testing and automatic processing.

        Args:
            parameterized: Whether to put in symbols for signle-qubit layers.
            width: number of qubits to use on the device.  qubits are used
                in the order of the device layout.
            depth: Number of moments for the circuit.  Moments will alternate
                between single-qubit and two-qubit gates.
            num_sweeps:  Number of different parameters to set for each single
                qubit gate.  Only relevant if parameterized is True.
            repetitions: Number of repetitions (samples) to run for each circuit.
            num_trials: The number of iterations to run each sweeped circuit.
                This will make num_trials independent calls to get a better
                idea of the average and variance of the rate.
            subtract_latency: A value, in seconds, to subtract from the end-to-end
                duration in order to determine repetition rate.  Useful if you
                would like to remove network latency or other round-trip effects
                for calculating run-times of circuits with high numbers of repetitions.
        """
        self._flush_print_log()
        if not width:
            width = len(self.qubits)
        circuit, sweep = self._create_rep_rate_circuit(parameterized, width,
                                                       depth, num_sweeps)
        latencies = []
        total_reps = repetitions
        if num_sweeps > 1:
            total_reps = num_sweeps * repetitions
        for trial in range(num_trials):
            self._print(f'Executing Trial {trial}')
            start = time.time()
            if sweep:
                self.sampler.run_sweep(circuit,
                                       params=sweep,
                                       repetitions=repetitions)
            else:
                self.sampler.run(circuit, repetitions=repetitions)
            end = time.time()
            duration = end - start
            self._print(f'Finished Trial {trial} with duration {duration}')
            latencies.append(duration)
        latencies = sorted(latencies)
        median = latencies[len(latencies) // 2]
        p05 = latencies[len(latencies) // 20]
        p95 = latencies[-len(latencies) // 20]
        p10 = latencies[len(latencies) // 10]
        p90 = latencies[-len(latencies) // 10]
        average = sum(latencies) / len(latencies)
        self._print(str(latencies))
        self._print(
            f'Latency:  Median={median}, p05={p05}, p10={p10}, p90={p90}, p95={p95}, avg={average}'
        )
        self._print(f'Total reps: {total_reps}')
        median = total_reps / (median - subtract_latency)
        p05 = total_reps / (p05 - subtract_latency)
        p95 = total_reps / (p95 - subtract_latency)
        p10 = total_reps / (p10 - subtract_latency)
        p90 = total_reps / (p90 - subtract_latency)
        average = total_reps / (average - subtract_latency)
        self._print(
            f'Rep Rate:  Median={median}, p05={p05}, p10={p10}, p90={p90}, p95={p95}, avg={average}'
        )
