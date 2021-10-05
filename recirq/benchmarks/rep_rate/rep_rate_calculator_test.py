import re
from typing import Dict

import cirq
import cirq_google as cg
import recirq.benchmarks.rep_rate.rep_rate_calculator as rep_rate


def _get_stats_from_print_line(s) -> Dict[str, str]:
    """Retrieves the statistics from the rep rate calculator output.

    Line should look like:
      'Latency, Median=xxx, p05=xxx, p10=xxx, p90=xxx, p95=xxx'

    Returns a dictionary of stat (e.g. 'Median') to value.
    """
    fields = re.split('[,:] *', s)
    stats_dict = {}
    for field in fields:
        key_value = re.split(' ?= ?', field)
        if len(key_value) >= 2:
            stats_dict[key_value[0]] = float(key_value[1])
    return stats_dict


def test_latency_with_simulator():
    tester = rep_rate.RepRateCalculator(device=cg.Foxtail,
                                        sampler=cirq.Simulator(),
                                        gate=cirq.ISWAP**0.5)
    latencies = tester.get_latency_samples(50)
    assert len(latencies) == 50

    # All of these samll circuits should finish in less
    # than 1 second
    assert all(0.0 < latency < 1.0 for latency in latencies)

    tester.print_latency(30)
    # Assuring that print statements are correct
    # Modify if print behavior changes
    for n in range(30):
        assert f'Executing Trial {n}' in tester.print_log
        assert f'Finished Trial {n}' in tester.print_log
    lines = tester.print_log.split('\n')
    latency_line = None
    for line in lines:
        if 'Latency' in line:
            latency_line = line
    assert latency_line

    stats = _get_stats_from_print_line(latency_line)
    assert stats['Median']
    assert stats['p05']
    assert stats['p10']
    assert stats['p90']
    assert stats['p95']
    assert stats['p05'] <= stats['p10'] <= stats['Median']
    assert stats['Median'] <= stats['p90'] <= stats['p95']


def test_rep_rate_with_simulator():
    tester = rep_rate.RepRateCalculator(device=cg.Foxtail,
                                        sampler=cirq.Simulator(),
                                        gate=cirq.ISWAP**0.5)
    tester.print_rep_rate(parameterized=True,
                          width=8,
                          depth=8,
                          num_sweeps=7,
                          repetitions=2000,
                          num_trials=10,
                          subtract_latency=0.0)

    # Assuring that print statements are correct
    # Modify if print behavior changes
    for n in range(6):
        assert f'Executing Trial {n}' in tester.print_log
        assert f'Finished Trial {n}' in tester.print_log

    lines = tester.print_log.split('\n')
    latency_line = None
    rep_rate_line = None
    for line in lines:
        if 'Latency' in line:
            latency_line = line
        if 'Rep Rate' in line:
            rep_rate_line = line
    assert latency_line
    assert rep_rate_line

    stats = _get_stats_from_print_line(latency_line)
    assert stats['Median']
    assert stats['p05']
    assert stats['p10']
    assert stats['p90']
    assert stats['p95']
    assert stats['p05'] <= stats['p10'] <= stats['Median']
    assert stats['Median'] <= stats['p90'] <= stats['p95']

    stats = _get_stats_from_print_line(rep_rate_line)
    assert stats['Median']
    assert stats['p05']
    assert stats['p10']
    assert stats['p90']
    assert stats['p95']
    assert stats['p05'] >= stats['p10'] >= stats['Median']
    assert stats['Median'] >= stats['p90'] >= stats['p95']


def test_create_rep_rate_circuit():
    """Testing an internal method to make sure that the circuit
  construction is correct.
  """
    circuit, sweep = rep_rate._create_rep_rate_circuit(
        False,
        cirq.ISWAP,
        qubits=cirq.GridQubit.rect(11, 1),
        depth=14,
        num_sweeps=1)
    # 14 layers plus one moment for measurement
    assert len(circuit) == 15
    assert len(circuit.all_qubits()) == 11
    assert sweep is None

    circuit, sweep = rep_rate._create_rep_rate_circuit(
        True, cirq.CZ, qubits=cirq.GridQubit.rect(1, 2), depth=4, num_sweeps=3)
    # 4 layers plus one moment for measurement
    assert len(circuit) == 5
    assert len(circuit.all_qubits()) == 2
    assert len(sweep) == 3
