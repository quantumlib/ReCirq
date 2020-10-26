# Simulation of spin and charge dynamics in the Fermi-Hubbard model

This module contains code necessary to run and analyse results of the 
publication [arXiv:2010.07965](https://arxiv.org/abs/2010.07965) **[quant-ph]**.

## Installation

The ReCirq package can be cloned directly or added to the local Python
environment with the command:
```
pip install git+https://github.com/quantumlib/ReCirq
```
To import and use:
```
import recirq.fermi_hubbard
```

## Usage

This package allows to study the Fermi-Hubbard dynamics with wide range of 
parameters (see ```parameters.py``` for a range of possible choices) on a 
quantum simulator or on the Google quantum hardware.

To analyze and reproduce the publication results, there are two jupyter 
notebooks provided:

  * The [experiment_example.ipynb](
  ../../docs/fermi_hubbard/experiment_example.ipynb) notebook guides on how the
  Fermi-Hubbard code is structured, how to simulate the Fermi-Hubbard problem 
  and how to execute the experiment with Google's Quantum Computing Service.
  * The [publication_results.ipynb](
  ../../docs/fermi_hubbard/publication_results.ipynb) notebook guides through 
  the analysis of results from the data set which was released together with the
  publication. The data set consists of four zip files and is accessible at 
  [https://doi.org/10.5061/dryad.crjdfn32v](
  https://doi.org/10.5061/dryad.crjdfn32v). The files need to be downloaded and
  extracted into the ```docs/fermi_hubbard/fermi_hubbard_data``` directory. 
 
 ## Package Contents
 
The ```fermi-hubbard``` package is split in four separate parts:
 
  * Data model: the ```layouts.py``` and ```parameters.py``` files contain data
  classes which encode a Fermi-Hubbard problem parameters in a self-contained
  way. They are used during execution to construct the problem circuits and 
  analysis to post-process the results. 
  * Execution: the ```execution.py``` file backed by ```circuits.py```,
  ```fermionic_circuits.py``` and ```decomposition.py``` contain all the code 
  necessary to create circuits, execute them and persist the execution results.
  * Analysis: experiment results are post-processed with the set of helper
  functions in ```post_procesing.py``` and plotted with functions in 
  ```data_plotting.py```.
  * Publication data: the ```publication.py``` file contains a set of helper
  functions and pre-defined data which is specific to the publication.
