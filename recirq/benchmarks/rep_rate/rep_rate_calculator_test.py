import cirq
import cirq_google as cg
import recirq.benchmarks.rep_rate.rep_rate_calculator as rep_rate

def test_latency_with_simulator():
    tester = rep_rate.RepRateCalculator(
        project_id='test_project',
        processor_id='qproc',
        device=cg.Foxtail,
        sampler=cirq.Simulator()
    )
    latencies = tester.get_latency_samples(50)
    assert len(latencies) == 50

    # All of these samll circuits should finish in less
    # than 1 second
    assert all(0.0 < latency < 1.0 for latency in latencies)
