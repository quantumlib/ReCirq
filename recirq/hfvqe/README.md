Hartree-Fock on a superconducting qubit quantum processor
---------------------------------------------------------

This module contains the necessary code to run
the Hartree-Fock VQE experiment described in [arXiv:2004.04174](https://arxiv.org/abs/2004.04174).  The goal in providing this code is
transparency of the experiment and so other researchers can 
use this as a basis for molecular simulations.  This is a living code base 
and various pieces may be integrated into OpenFermion over time.  

Quickstart
----------
An [ipython notebook](quickstart.ipynb) provided with this module describes how to initialize and run
a Hartree-Fock VQE calculation.  It steps through estimating the 1-RDM
given a set of parameters for the basis transformation unitary and then provides an example of
variational relaxation of the parameters.  

Utilities for estimating all quantities described in [arXiv:2004.04174](https://arxiv.org/abs/2004.04174) such as fidelities,
fidelity witness values, absolute errors, and error bars are also provided.

All software for running the experiment is in the `openfermioncirq.experiments.hfvqe` subfolder.  The 
molecular data used in the experiment can be found in the 
`openfermioncirq.experiments.hfvqe.molecular_data` directory.  

Molecular Data
--------------
The paper describes the performance of VQE-HF for four Hydrogen chain systems and Diazene.  We provide
molecular data files and utilities for generating the Hydrogen chain inputs using OpenFermion, 
OpenFermion-Psi4. and Psi4. The Diazene data can be found in the 
[openfermion-cloud](https://github.com/quantumlib/OpenFermion/tree/master/cloud_library) repository.
