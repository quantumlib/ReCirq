# Hartree-Fock on a superconducting qubit quantum processor

This examples describes how to set up molecular data and perform the experiment in [arXiv:2004.04174](https://arxiv.org/abs/2004.04174).
The goal in providing this code is transparency and reproducibility.
This is a living code base and various pieces may be integrated into OpenFermion
over time. 

The HFVQE experiment seeks to error mitigate the basis rotation circuit primitive.  This is achieved
by post-selection, variational relaxation of the parameters, and purification.  The tutorial demonstrates
how to collect data and perform analysis such as extracting fidelity and performing optimization.
 

## Quickstart
The quickstart tutorial provided with this module describes how
to initialize and run a Hartree-Fock VQE calculation. It steps through
estimating the 1-RDM given a set of parameters for the basis transformation
unitary and then provides an example of variational relaxation of the
parameters.

Utilities for estimating all quantities described in
[arXiv:2004.04174](https://arxiv.org/abs/2004.04174) such as fidelities,
fidelity witness values, absolute errors, and error bars are also provided.

All software for running the experiment is in the `recirq.hfvqe` submodule.
The  molecular data used in the experiment can be found in the 
`recirq.hfvqe.molecular_data` directory.  

## Molecular Data
The paper describes the performance of VQE-HF for four hydrogen chain systems
and diazene. We provide molecular data files and utilities for generating the
hydrogen chain inputs using OpenFermion, Psi4, and OpenFermion-Psi4.
The Diazene data can be found in the 
[openfermion-cloud](https://github.com/quantumlib/OpenFermion/tree/master/cloud_library)
repository.  A tutorial on how the data is generated can be found in this [ipython notebook](molecular_data.ipynb).
