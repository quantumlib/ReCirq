# Measurement-Induced Entanglement Experiment

This module contains code for studying measurement-induced entanglement in quantum systems using Google's Cirq framework.

## Circuit Construction

The experiment uses a 6x6 grid of qubits with ancilla qubits for error mitigation. The circuit construction process involves:

1. **Grid Setup**: 
   - Creates a 6x6 grid of physical qubits
   - Adds ancilla qubits for error mitigation
   - Positions probe qubits at specific distances (3-6 units apart)

2. **Circuit Generation**:
   - Applies CZ gates to create cluster state entanglement
   - Implements single-qubit rotations (Rz and Ry gates)
   - Adds basis rotations for measurement in X, Y, or Z basis
   - Includes ancilla qubit operations for error mitigation

### Implementation Details

This experiment is implemented on the **Willow** superconducting qubits platform. Let $L$ be the system linear dimension and initialize all qubits in the product state $|0\rangle^{\otimes N}$. The protocol proceeds as follows:

1. **Apply Hadamard gates:** $\bigotimes_j H_j$
2. **Apply nearest-neighbor $ZZ$ gates for $t = \pi/4$:**
   $e^{i(\pi/4)\sum_{\langle j,k \rangle} Z_j Z_k}$
3. **Apply single-qubit rotations:** $\bigotimes_j e^{i(\theta/2) Y_j} e^{i(\phi/2) Z_j}$

After these steps, projective measurement in the $Z$ basis is performed on the preparation qubits (solid red in the figure below) to prepare the probe qubits' states (solid green). Shadow measurements are then performed on the probe qubits using the $X$, $Y$, and $Z$ bases with equal probability to collect data.

The nearest-neighbor gate $e^{i(\pi/4)\sum_{\langle j,k \rangle} Z_j Z_k}$ is experimentally implemented by first applying a controlled-$Z$ ($\mathrm{CZ}$) gate between each pair of qubits, followed by local $Z^{-1/2}$ on both qubits, i.e., $\mathrm{CZ}_{jk}Z_j^{-1/2}Z_k^{-1/2}$. All nearest-neighbor two-qubit gates are applied in the following sequence to maintain constant circuit depth as the system size scales up: first, odd horizontal links; then, even horizontal links; followed by odd vertical links and finally even vertical links.

![Experimental realization of the circuit on quantum chip with 6x6 lattice at d=5.](figs/app/shallow_lattice.pdf)

*Figure: Experimental realization of the circuit on a quantum chip with $6\times6$ lattice at $d=5$. Solid red: preparation qubits; solid green: probe qubits A and B. Faded qubits are ancillaries for readout error mitigation.*

#### Error Mitigation

Since the dominant source of error in this experiment is readout error, a replication-based error mitigation strategy is employed. After all gates (including basis rotations on probe qubits for shadow measurement), an additional CNOT gate is inserted from each perimeter qubit to its corresponding ancillary qubit (represented as faded boxes in the figure above, indicated by yellow bonds). Experimentally, this $\mathrm{CNOT}_{\text{control},\,\text{target}}$ is implemented as a $(\mathbb{I} \otimes H) \cdot \mathrm{CZ} \cdot (\mathbb{I} \otimes H)$ gate sequence. This operation replicates the measurement outcome of the physical qubit onto the ancillary qubit. In post-processing, only measurement outcomes with matching values on each physical-ancillary pair are retained (post-selection), further enhancing the reliability of the experimental results on the mitigated qubits.

## Data Processing Pipeline

The data processing pipeline handles quantum measurement results through several steps:

1. **Measurement Processing**:
   - Loads raw measurement data for different basis combinations
   - Performs post-selection on 2-qubit mitigation
   - Extracts preparation sequences and probe qubit measurements

2. **State Reconstruction**:
   - Constructs shadow states from measurement outcomes
   - Builds density matrices for the probe qubits
   - Handles basis transformations and state preparation

3. **Data Organization**:
   - Organizes data by experiment parameters (distance, theta, etc.)
   - Saves processed data in a structured format
   - Supports data shuffling for statistical analysis

## Tensor Network Analysis

The analysis uses tensor network methods to detect and quantify entanglement:

1. **State Analysis**:
   - Uses tensor contractions to process the quantum state
   - Implements cross, side, and corner measurements
   - Applies noise mitigation through the `eps` function

2. **Entanglement Measures**:
   - Calculates negativity (`Neg`) between states
   - Computes squashed concurrence (`bSqc`)
   - Evaluates von Neumann entropy (`Sa`)

3. **Error Handling**:
   - Includes numerical stability measures
   - Validates density matrix properties
   - Handles device and dtype compatibility

## Usage

To run the experiment:

1. Set up the configuration in `config.py`
2. Run the experiment using `experiment.py`
3. Process the data using `data_processing.py`
4. Analyze results using `data_analysis.py`

## Dependencies

- Cirq
- PyTorch
- NumPy
- Cirq-Google (for hardware access)
