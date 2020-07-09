.. _data-collection-idioms:

Data Collection Idioms
======================

Running meaningful experiments on quantum hardware requires the structured
collection of experimental data and flexible analysis pipelines to draw
conclusions from raw data.

This documentation describes a collection of idioms to reference when developing
experimental pipelines. Honed via real experience, studies in this project use
these ideas, and we encourage users to adopt these conventions in their own experiments.
We introduce the concepts by way of example.

Design philosophy
-----------------


In the first tutorial, we run a simple readout-like experiment in two steps:

 1. `Data Collection <Readout-Data-Collection.ipynb>`_
 2. `Analysis <Readout-Analysis.ipynb>`_

In the future, we can explore documenting more complicated set-ups like

 a. Separate problem generation step
 b. More complicated dependencies between tasks
 c. Multiple analysis routines for one data-collection
 d. Computationally intensive analysis
 e. and more!