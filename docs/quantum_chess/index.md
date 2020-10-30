# Quantum Chess

Quantum phenomena have remained largely inaccessible to the general public. 
This can be attributed to the fact that we do not experience quantum mechanics 
on a tangible level in our daily lives. Games can provide an environment in 
which people can experience the strange behavior of the quantum world in a 
fun and mentally engaging way. Games can also offer an interesting test 
bed for near term quantum devices because they can be tailored to support 
varying amounts of quantum behavior through simple rule changes, 
which can be useful when dealing with limited resources. 

Quantum Chess is a variant of Chess, which is built on top of unitary 
dynamics and includes non-trivial quantum effects such as superposition, 
entanglement, and interference. With the introduction of the split move, players
can place their pieces in superposition on multiple squares at once. For a full
description of the rules, and to learn more about the game, check out the 
[Quantum Chess website](https://www.quantumchess.net). And you can see this [paper on arxiv](https://arxiv.org/abs/1906.05836) 
to learn more about the math behind the game.

This module explores an implementation of Quantum Chess that runs using Cirq. In
[Concepts](./concepts) we explore various quantum computing concepts, such as error
mitigation, post-selection, and qubit mapping, without requiring specific domain
knowledge, like quantum chemistry. And in [Quantum Chess REST API] we will implement
a fully functional server, that defines the specific REST endpoints required by the
Quantum Chess Engine to offload its quantum state handling to an external resource.
A similar server was used in [this presentation](https://youtu.be/ec-Mb8OJuRg) to run Quantum Chess on Google's 
Rainbow chip during the Google Quantum AI Summer symposium.