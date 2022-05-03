# QAOA

Combinatorial optimization problems can be solved with the quantum approximate
optimization algorithm (QAOA; also known as "quantum alternating operator ansatz").

We demonstrated the application of the Google Sycamore superconducting qubit
quantum processor to combinatorial optimization problems with the QAOA
[arxiv:2004.04197](https://arxiv.org/abs/2004.04197).
Like past QAOA experiments, we studied performance for problems defined on the
(planar) connectivity graph of our hardware; however, we also applied the QAOA
to the Sherrington-Kirkpatrick model and MaxCut, both high dimensional graph
problems for which the QAOA requires significant compilation.

This module contains the code to generate QAOA circuits, execute demonstrations,
and analyze the results. Please read on to learn about:

*  [Example problems](./example_problems.ipynb)
*  [Tasks](./tasks.ipynb) and compilation
*  [Analysis](./precomputed_analysis.ipynb)

See the left navigation for more information about the QAOA experiment
and how to run it end-to-end.


