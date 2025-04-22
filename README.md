# ReCirq

Research using Cirq!

This project contains modules for running quantum computing applications and
experiments through Cirq. By way of example, we also document best practices
for setting up robust experimental pipelines.

## Installation and Documentation

ReCirq is not available as a PyPI package. Please clone this repository and
install from source:

```shell
cd recirq/
pip install .
```

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
| Quantum error correction (QEC)? | [Stim]<br>[Tesseract Decoder] |
| Chemistry and/or material science? | [OpenFermion]<br>[OpenFermion-FQE]<br>[OpenFermion-PySCF]<br>[OpenFermion-Psi4] |
| Quantum machine learning (QML)? | [TensorFlow Quantum] |

</div>

[Cirq]: https://github.com/quantumlib/Cirq
[Qualtran]: https://github.com/quantumlib/Qualtran
[qsim]: https://github.com/quantumlib/qsim
[Stim]: https://github.com/quantumlib/Stim
[OpenFermion]: https://github.com/quantumlib/OpenFermion
[OpenFermion-FQE]: https://github.com/quantumlib/OpenFermion-FQE
[OpenFermion-PySCF]: https://github.com/quantumlib/OpenFermion-PySCF
[OpenFermion-Psi4]: https://github.com/quantumlib/OpenFermion-Psi4
[Tesseract Decoder]: https://github.com/quantumlib/Tesseract-decoder
[TensorFlow Quantum]: https://github.com/tensorflow/quantum
[ReCirq]: https://github.com/quantumlib/ReCirq

## How to cite ReCirq

<a href="https://doi.org/10.5281/zenodo.4091470">
  <img align="right" src="https://img.shields.io/badge/10.5281%2Fzenodo.4091470-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1">
</a>

New releases of ReCirq are uploaded to [Zenodo] automatically. Click the badge at right to see all
citation formats for all versions, or use the following BibTeX:

```bibtex
@software{quantum_ai_team_and_collaborators_2020_4091470,
  author       = {{Quantum AI team and collaborators}},
  title        = {ReCirq},
  month        = Oct,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4091470},
  url          = {https://doi.org/10.5281/zenodo.4091470}
}
```

[Zenodo]: https://doi.org/10.5281/zenodo.4091470

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
    <img align="center" width="15%" alt="Google Quantum AI"
         src="./docs/images/quantum-ai-vertical.svg">
  </a>
</div>
