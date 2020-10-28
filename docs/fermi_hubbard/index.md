# Fermi-Hubbard

Strongly correlated quantum systems give rise to many exotic physical phenomena, 
including high-temperature superconductivity. 
Simulating these systems on quantum computers may avoid the prohibitively 
high computational cost incurred in classical approaches. 
However, systematic errors and decoherence effects presented in current quantum 
devices make it difficult to achieve this. 
We simulated the dynamics of the one-dimensional Fermi-Hubbard model 
using 16 qubits on Google's quantum processor in 
[arxiv:2010.07965](https://arxiv.org/abs/2010.07965).
We observed separations in the 
spreading velocities of charge and spin densities in the highly excited regime, 
a regime that is beyond the conventional quasiparticle picture. 
We employed a sequence of error-mitigation techniques to reduce decoherence 
effects and residual systematic errors. 
These procedures allowed us to simulate the time evolution of the model 
faithfully despite having over 600 two-qubit gates in our circuits.

This module contains the code used to generate Fermi-Hubbard circuits,
execute the experiments, and analyze the results. Please read on to see
an example experiment and analysis of the published data.

