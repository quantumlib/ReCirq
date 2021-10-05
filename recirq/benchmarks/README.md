Benchmarks in ReCirq

This package is home to benchmarks used for testing and measuring
end-to-end performance on hardware, specifically Google's Quantum Computing
Service.

Benchmarks here are designed to test the performance of the system in various
aspects and to give an estimate of the processor's performance.  These metrics
can also be used to replicate the processor's data sheet.

Benchmarks here include:

*   Repetition Rate calculator (rep_rate):  This benchmark will time circuits
to estimate time taken for circuits of given width, depth, and number of sweeps.
Due to the complexity of the stack and various bottlenecks at different stages,
the repetition rate (number of circuits executed per second) can vary widely
depending on the nature of the circuit being run.  This calculator can help
figure out how long a circuit execution will take.

Note that this package does not include methods for characterization and
calibration of individual qubits and gates.  These can be found in the
cirq repository (cirq.qcvv).
