# Simulation and data analysis code for "Quantum advantage in learning from experiments"

Code to generate main text data for [Quantum advantage in learning from experiments](https://arxiv.org/abs/2112.00778)

Running `python3 -m recirq.qml_lfe.learn_***_* --help` will give a list of all available command line
flags for each experiment file as well as a short description of what each experiment
module does.

To generate bitstring data for an 8-qubit depth-5 experiment on learning 1D physical dynamics:

`python3 -m recirq.qml_lfe.learn_dynamics_c --n=8 --depth=5`

`python3 -m recirq.qml_lfe.learn_dynamics_q.py --n=8 --depth=5`

To generate bitstring data for an 8-qubit experiment on learning physical states:

`python3 -m recirq.qml_lfe.learn_states_c.py --n=8`

`python3 -m recirq.qml_lfe.learn_states_q.py --n=8`


