# Simulation of Many Body Local Discrete Time Crystal circuits

This module contains code that runs experiments and generates graphs to replicate Figures 2d through 3d as presented in the paper: Observation of Time-Crystalline Eigenstate Order on a Quantum Processor ([Nature](https://www.nature.com/articles/s41586-021-04257-w)).

A "crystal" is something with stable structure that is resistant to change in spite of external perturbation. For example, a diamond is a "space-crystal" in the sense that it's physical molecular structure spans space and refuses to change in spite of interference. Within reason, one can heat or toss about a diamond and it stays a diamond, with the same crystalline structure. A "time crystal" is then something with stable and resilient time structure. Perhaps the simplest time structure is that of oscillation, where some signal demonstrates a repeating pattern after some fixed time period.

The Many Body Local Discrete Time Crystal circuit list presented in the paper and the following notebooks serve to demonstrate stable and consistent oscillation over discrete time steps (U-cycles). It is implemented on a connected chain of qubits that compose a many body (qubit) system with local interaction (qubit connections). The key is that the circuits demonstrate oscillation behavior in spite of interference by random variance. The following notebooks demonstrate how the circuit list is generated and tested, and how the resulting graphs support the existence of time-crystalline behavior.

## Table of Contents

* The [Time Crystal Circuit Generation](time_crystal_circuit_generation.ipynb) notebook covers the provided function `recirq.time_crystals.symbolic_dtc_circuit_list()`, which creates a list of symbolic time crystal circuits with increasingly many $U$-cycles. This circuit list is used in each of the experiments to model crystalline behavior over time.

* The [Time Crystal Data Collection](time_crystal_data_collection.ipynb) notebook details each of the five experiments necessary to generate Figures 2d through 3d of [the paper](https://arxiv.org/abs/2107.13571), including the parameter options to be compared, the methodology for doing so, and some of the data postprocessing.

* The [Time Crystal Data Analysis](time_crystal_data_analysis.ipynb) notebook primarily demonstrates use of the collected data to generate the figures of the paper. It also discusses some of the conclusions that can be drawn about the results, and how they support the existence of time-crystalline behavior.
