# QCQMC

Notebooks outlining how to reproduce the results of [Unbiasing fermionic quantum Monte Carlo with a quantum computer](https://www.nature.com/articles/s41586-021-04351-z).
Quantum Monte Carlo methods are a class of classical algorithms that can offer
an efficient solution to the many-electron schroedinger equation but are plagued
by the `fermion sign problem`. Typically, a constraint is introduced to overcome
this problem at the cost of introducing an uncontrolled bias in the results. In
this paper, it was shown that a quantum computer could be used to prepare
complicated trial wavefunctions in order to unbias the classical results. 

## Code Overview
The [Code Overview](./high_level.ipynb) notebook introduces the basic structure
of the code provided in recirq to generate the trial wavefunction's prepare in the [QCQMC paper](https://www.nature.com/articles/s41586-021-04351-z).

## End-to-End
The [End-to-End](./full_workflow.ipynb) notebook provides and end-to-end example
for the H4 molecule and interfaces with ipie to produce numbers similar to those
reported in the [QCQMC](https://www.nature.com/articles/s41586-021-04351-z)
paper.


## Experimental Wavefunctions

The [Experimental Wavefunctions](./experimental_wavefunctions.ipynb) notebook demonstrates
how to download and analyze the experimental wavefunctions, which can reproduce
the results in [QCQMC](https://www.nature.com/articles/s41586-021-04351-z).
