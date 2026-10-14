# ReCirq

Research using Cirq!

This project contains modules for running quantum computing applications and
experiments through Cirq. By way of example, we also document best practices
for setting up robust experimental pipelines.

The Python package in the `recirq/` directory contains a subdirectory for each
experiment, such as `qaoa`, `hfvqe`, `fermi_hubbard`, `otoc`, `toric_code`,
`kpz`, `qcqmc`, `qec_memory`, `time_crystals`, and `random_circuit_sampling`,
along with the `cirqflow` workflow utilities and other shared helpers. The
`docs/` directory contains tutorials and Colab/Jupyter notebooks that accompany
the experiments; they are published at https://quantumai.google/cirq/experiments.

## Installation and Documentation

ReCirq requires Python 3.10 or later. It is not available as a PyPI package, so
please clone this repository and install from source. We recommend doing this
inside a virtual Python environment (for example, one created with Python's
built-in [`venv`](https://docs.python.org/3/library/venv.html) module). Once
you have a Python virtual environment, then do these steps:

```shell
git clone https://github.com/quantumlib/ReCirq.git
cd ReCirq
pip install .
```

This installs the dependencies for all of the experiments. If you only want
the dependencies for a specific experiment, install `requirements.txt` and the
`extra-requirements.txt` file in that experiment's subdirectory instead, as
described in the [contribution guidelines].

Documentation is available at https://quantumai.google/cirq/experiments.

## See Also

ReCirq leverages [Cirq] as a quantum programming language and SDK.

Google Quantum AI has a suite of open-source software.
From high-performance simulators, to novel tools for expressing and
analyzing fault-tolerant quantum algorithms, our software stack lets you
develop quantum programs for a variety of applications.

<div align="center">

| Your interests | Software to explore  |
|----------------|----------------------|
| Programming NISQ quantum computers? | [Cirq] |
| Quantum algorithms?<br>Fault-tolerant quantum computing (FTQC)? | [Qualtran] |
| Large circuits and/or a lot of simulations? | [qsim] |
| Circuits with thousands of qubits and millions of Clifford operations? | [Stim] |
| Quantum error correction (QEC)? | [Stim]<br>[Chromobius]<br>[Tesseract Decoder] |
| Chemistry and/or material science? | [OpenFermion]<br>[OpenFermion-FQE]<br>[OpenFermion-PySCF]<br>[OpenFermion-Psi4] |
| Quantum machine learning (QML)? | [TensorFlow Quantum] |

</div>

[Cirq]: https://github.com/quantumlib/Cirq
[Qualtran]: https://github.com/quantumlib/Qualtran
[qsim]: https://github.com/quantumlib/qsim
[Stim]: https://github.com/quantumlib/Stim
[Chromobius]: https://github.com/quantumlib/chromobius
[OpenFermion]: https://github.com/quantumlib/OpenFermion
[OpenFermion-FQE]: https://github.com/quantumlib/OpenFermion-FQE
[OpenFermion-PySCF]: https://github.com/quantumlib/OpenFermion-PySCF
[OpenFermion-Psi4]: https://github.com/quantumlib/OpenFermion-Psi4
[Tesseract Decoder]: https://github.com/quantumlib/Tesseract-decoder
[TensorFlow Quantum]: https://github.com/tensorflow/quantum

## How to cite ReCirq

<a href="https://doi.org/10.5281/zenodo.4091470">
  <img align="right" alt="DOI 10.5281/zenodo.4091470"
       src="https://img.shields.io/badge/10.5281%2Fzenodo.4091470-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1">
</a>

Releases of ReCirq are archived on [Zenodo]. Click the badge at right to see all
citation formats for all versions, or use the following BibTeX:

```bibtex
@software{quantum_ai_team_and_collaborators_2020_4091470,
  author       = {{Quantum AI team and collaborators}},
  title        = {ReCirq},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4091470},
  url          = {https://doi.org/10.5281/zenodo.4091470}
}
```

[Zenodo]: https://doi.org/10.5281/zenodo.4091470

## Community and Support

*   Have a question or problem? See the [support guide] for suggestions on how
    to get help.
*   Found a bug or have a feature request? [Open an issue on GitHub].
*   Want to contribute code? Read the [contribution guidelines] and our
    community [code of conduct].

[support guide]: https://github.com/quantumlib/ReCirq/blob/main/SUPPORT.md
[Open an issue on GitHub]: https://github.com/quantumlib/ReCirq/issues
[contribution guidelines]: https://github.com/quantumlib/ReCirq/blob/main/CONTRIBUTING.md
[code of conduct]: https://github.com/quantumlib/ReCirq/blob/main/CODE_OF_CONDUCT.md

## Contact

For any questions or concerns not addressed here, please email
quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2025 Google LLC.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/ReCirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
