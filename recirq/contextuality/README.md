# Quantum Contextuality Experiments

This module contains Cirq implementations of the quantum contextuality experiments described in
"Quantum-classical separation in bounded-resource tasks arising from measurement contextuality,"
[(Google Quantum AI and Collaborators, _Science_ 394, 870–874, 2026)][paper]. In quantum mechanics,
the outcome of a measurement can depend on which other compatible measurements are made alongside it
(its _context_). The experiments use this property, known as contextuality, to perform tasks on a
superconducting quantum processor with success probabilities beyond the limits of classical
strategies. The code for the four experiments are provided in this directory: two few-qubit
protocols (the Mermin–Peres magic square game and a Kochen–Specker–Bell inequality test) and two
scalable many-body tasks (the $N$-player GHZ parity game and the 2D hidden linear function problem).

[paper]: https://www.science.org/doi/10.1126/science.aec3042

## Contents

Each of the four experiments in the paper has its own module, with unit tests in the matching
`*_test.py` file.

*   `magic_square_game.py`: the magic square game (Fig. 1 in the
    paper). Alice and Bob share two Bell pairs and, without communicating, fill
    in a row and a column of a 3×3 table of ±1 values. The best classical
    strategy wins with probability 8/9, while the ideal quantum strategy always
    wins. The function `run_contextuality_experiment()` plays all nine
    row–column combinations and returns a `ContextualityResult`, which computes
    how often the players satisfy the parity rules and agree on the cell that
    they share.

*   `kochen_specker.py`: a test of the Bell–Kochen–Specker (BKS) inequality (Fig. 2). A
    long random sequence of rows and columns of a Mermin–Peres square is
    measured on two data qubits, using an ancilla qubit to make quantum
    nondemolition measurements. `MagicSquare.kochen_specker_expt()` builds the
    experiment, and `MagicSquareResult.estimate_chi()` estimates the quantity χ,
    which is at most 4 in any noncontextual hidden-variable theory and equals 6
    in ideal quantum theory.

*   `ghz_game_2d.py`: the N-player GHZ parity game (Fig. 3). `GHZ2dExperiment`
    prepares GHZ states of a chosen size on a 2D grid of qubits and plays the
    game with random questions. `GHZ2dResults` reports the winning probability,
    along with the X- and Z-stabilizer expectation values used to estimate the
    fidelity of the GHZ state. The best classical winning probability approaches
    1/2 as N grows, while the ideal quantum strategy always wins.

*   `shallow_circuits_experiment.py`: the 2D hidden linear function (HLF) problem
    of Bravyi et al. (Fig. 4). `ShallowCircuitExperiment` generates and runs
    random constant-depth Clifford circuits on a 2D grid of qubits.
    `ShallowCircuitResults` uses Stim to check whether each measured bitstring
    is a valid solution, and computes the fraction of valid outcomes and the
    paper's conjectured lower bound on the classical circuit depth.

## Installation

Follow the [installation instructions](../../README.md#installation-and-documentation) for ReCirq.
Doing so also installs [Stim](https://github.com/quantumlib/Stim) and `stimcirq`, which the HLF
module needs and which are listed in `extra-requirements.txt`.

```shell
pip install -r requirements.txt
pip install -r recirq/contextuality/extra-requirements.txt
```

## Usage

The experiments accept any `cirq.Sampler`, so you can try them on a simulator before running them on
hardware. For example, the following code estimates χ on a noiseless simulator:

```python
import cirq
from recirq.contextuality import kochen_specker as ks

q0, anc, q1 = cirq.LineQubit.range(3)
experiment = ks.EASY_SQUARE.kochen_specker_expt(q0, q1, anc, num_contexts=36, seed=0)
result = experiment.run(cirq.Simulator(), repetitions=100)
chi, chi_error = result.estimate_chi()
print(chi)  # Prints 6.0, the ideal quantum value. The classical limit is 4.
```

The unit tests show how to use the other modules. To run the tests, use this command from the top
level of the repository:

```shell
pytest recirq/contextuality
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{GoogleQuantumAI2026contextuality,
  title     = {Quantum-classical separation in bounded-resource tasks arising from
               measurement contextuality},
  author    = {{Google Quantum AI and Collaborators}},
  journal   = {Science},
  year      = {2026},
  month     = nov,
  volume    = {394},
  number    = {6826},
  pages     = {870--874},
  doi       = {10.1126/science.aec3042},
  url       = {https://www.science.org/doi/10.1126/science.aec3042},
  publisher = {American Association for the Advancement of Science}
}
```

A preprint version is available on arXiv ([arXiv:2512.02284](https://arxiv.org/abs/2512.02284)).
