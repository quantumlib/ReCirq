# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Containers to represent Fermi-Hubbard problem."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union

import abc
from itertools import product
from numbers import Number, Real as NumbersReal

import cirq
import numpy as np
import openfermion

from recirq.fermi_hubbard.layouts import (
    LineLayout,
    QubitsLayout,
    ZigZagLayout
)

# Type alias for real numbers (supports int, float, np.float32, np.float64, etc.)
RealNumber = Union[int, float]


@dataclass(init=False)
class Hamiltonian:
    """Single spin-chain Fermi-Hubbard Hamiltonian description.

        H =  - Σ_i Σ_ν (J_i c_{i,ν}^† c_{i+1,ν} + h.c.)
             + Σ_i U_i n_{i,↑} n_{i,↓}
             + Σ_i Σ_ν V_i n_{i,ν} n_{i+1,ν}
             + Σ_i Σ_ν (ε_{i,ν} - μ_ν) n_{i,ν},

        where:
            i = 1, 2, ..., sites_count are the site indices,
            ν = ↑,↓ denote the two spin states,
            c_{i,ν} are the fermionic annihilation operators,
            n_{i,ν} = c_{i,ν}^† c_{i,ν} are the number operators.

    Attributes:
        sites_count: Total number of sites, where each site has spin-up and
            spin-down occupancy.
        j: The hopping coefficients J_i.
        u: The on-site interaction coefficients U_i.
        v: Same-spin nearest-neighbour interaction coefficients V_i.
        local_charge: Local electric field coefficients 0.5 * (ε_{i,↑} +
            ε_{i,↓}).
        local_spin: Local magnetic field coefficients 0.5 * (ε_{i,↑} - ε_{i,↓}).
        mu_up: Local chemical potential for spin-up states μ_↑.
        mu_down: Local chemical potential for spin-down states μ_↓.
    """
    sites_count: int
    j: Union[RealNumber, Tuple[RealNumber]]
    u: Union[RealNumber, Tuple[RealNumber]]
    v: Union[RealNumber, Tuple[RealNumber]]
    local_charge: Union[RealNumber, Tuple[RealNumber]]
    local_spin: Union[RealNumber, Tuple[RealNumber]]
    mu_up: Union[RealNumber, Tuple[RealNumber]]
    mu_down: Union[RealNumber, Tuple[RealNumber]]

    def __init__(self, *,
                 sites_count: int,
                 j: Union[RealNumber, Iterable[RealNumber]],
                 u: Union[RealNumber, Iterable[RealNumber]],
                 v: Union[RealNumber, Iterable[RealNumber]] = 0,
                 local_charge: Union[RealNumber, Iterable[RealNumber]] = 0,
                 local_spin: Union[RealNumber, Iterable[RealNumber]] = 0,
                 mu_up: Union[RealNumber, Iterable[RealNumber]] = 0,
                 mu_down: Union[RealNumber, Iterable[RealNumber]] = 0,
                 ) -> None:
        self.sites_count = sites_count
        self.j = _iterable_to_tuple(j)
        self.u = _iterable_to_tuple(u)
        self.v = _iterable_to_tuple(v)
        self.local_charge = _iterable_to_tuple(local_charge)
        self.local_spin = _iterable_to_tuple(local_spin)
        self.mu_up = _iterable_to_tuple(mu_up)
        self.mu_down = _iterable_to_tuple(mu_down)

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls, f'recirq.{cls.__name__}': cls}

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    @property
    def interactions_count(self) -> int:
        return self.sites_count - 1

    @property
    def j_array(self) -> np.ndarray:
        if isinstance(self.j, NumbersReal):
            return np.full(self.interactions_count, self.j)
        return np.array(self.j)

    @property
    def u_array(self) -> np.ndarray:
        if isinstance(self.u, NumbersReal):
            return np.full(self.sites_count, self.u)
        return np.array(self.u)

    @property
    def v_array(self) -> np.ndarray:
        if isinstance(self.v, NumbersReal):
            return np.full(self.interactions_count, self.v)
        return np.array(self.v)

    @property
    def local_charge_array(self) -> np.ndarray:
        if isinstance(self.local_charge, NumbersReal):
            return np.full(self.sites_count, self.local_charge)
        return np.array(self.local_charge)

    @property
    def local_spin_array(self) -> np.ndarray:
        if isinstance(self.local_spin, NumbersReal):
            return np.full(self.sites_count, self.local_spin)
        return np.array(self.local_spin)

    @property
    def mu_up_array(self) -> np.ndarray:
        if isinstance(self.mu_up, NumbersReal):
            return np.full(self.sites_count, self.mu_up)
        return np.array(self.mu_up)

    @property
    def mu_down_array(self) -> np.ndarray:
        if isinstance(self.mu_down, NumbersReal):
            return np.full(self.sites_count, self.mu_down)
        return np.array(self.mu_down)

    @property
    def local_up_array(self):
        return (self.local_charge_array -
                self.local_spin_array -
                self.mu_up_array)

    @property
    def local_down_array(self):
        return (self.local_charge_array +
                self.local_spin_array -
                self.mu_down_array)

    def as_diagonal_coulomb_hamiltonian(
            self
    ) -> openfermion.DiagonalCoulombHamiltonian:
        """Exports Fermi Hubbard Hamiltonian as DiagonalCoulombHamiltonian.

        Returns: Description of Fermi Hubbard problem as
            openfermion.DiagonalCoulombHamiltonian.
        """

        def spin_map(j: int, spin: int) -> int:
            """Mapping from site and spin to index (separated in spin sectors).

            Assigns indices 0 through self.sites_count - 1 when spin = 0 and
            indices self.sites_count through 2 * self.sites_count - 1 when
            spin = 1.
            """
            return j + spin * self.sites_count

        modes = 2 * self.sites_count

        # Prepare one-body matrix T_ij.
        one_body = np.zeros((modes, modes))

        # Nearest-neighbor hopping terms.
        t = self.j_array
        for j, s in product(range(self.interactions_count), [0, 1]):
            j1 = (j + 1) % self.sites_count
            one_body[spin_map(j, s), spin_map(j1, s)] += -t[j]
            one_body[spin_map(j1, s), spin_map(j, s)] += -t[j]

        # Local interaction terms.
        local_up = self.local_up_array
        local_down = self.local_down_array
        for j in range(self.sites_count):
            one_body[spin_map(j, 0), spin_map(j, 0)] += local_up[j]
            one_body[spin_map(j, 1), spin_map(j, 1)] += local_down[j]

        # Prepare the two-body matrix V_ij.
        two_body = np.zeros((modes, modes))

        # On-site interaction terms.
        u = self.u_array
        for j in range(self.sites_count):
            two_body[spin_map(j, 0), spin_map(j, 1)] += u[j] / 2.
            two_body[spin_map(j, 1), spin_map(j, 0)] += u[j] / 2.

        # Nearest-neighbor interaction terms.
        v = self.v_array
        for j, (s0, s1) in product(range(self.interactions_count),
                                   np.ndindex((2, 2))):
            j1 = (j + 1) % self.sites_count
            two_body[spin_map(j, s0), spin_map(j1, s1)] += v[j] / 2.
            two_body[spin_map(j1, s1), spin_map(j, s0)] += v[j] / 2.

        return openfermion.DiagonalCoulombHamiltonian(one_body, two_body)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

class SingleParticle:
    """Base class for initial states that define single particle on a chain."""

    @property
    def particles(self) -> int:
        return 1

    @abc.abstractmethod
    def get_amplitudes(self, sites_count: int) -> np.ndarray:
        """Calculates fermionic amplitudes for each site.

        Args:
            sites_count: Number of fermionic sites.

        Returns:
            Complex array of size sites_count with amplitude for each fermion
            on each site. Must be normalized to 1.
        """


@dataclass(init=False)
class FixedSingleParticle(SingleParticle):
    """Fixed array of particle amplitudes.

    This initial state is fixed to constant number of sites.
    """
    amplitudes: Tuple[Number]

    def __init__(self, *, amplitudes: Iterable[Number]) -> None:
        _check_normalized(amplitudes)
        self.amplitudes = tuple(amplitudes)

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls, f'recirq.{cls.__name__}': cls}

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    def get_amplitudes(self, sites_count: int) -> np.ndarray:
        if sites_count != len(self.amplitudes):
            raise ValueError(f'Fixed single particle not compatible with '
                             f'{sites_count} sites')
        return np.array(self.potential)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass
class UniformSingleParticle(SingleParticle):
    """Uniform single particle amplitudes initial state."""
    pass

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls, f'recirq.{cls.__name__}': cls}

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    def get_amplitudes(self, sites_count: int) -> np.ndarray:
        return np.full(sites_count, 1.0 / np.sqrt(sites_count))

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass(init=False)
class PhasedGaussianSingleParticle(SingleParticle):
    """Gaussian shaped particle density distribution initial state.

    We first define the local density function for site i = 1, 2, ...,
    sites_count as:

        d_i =  e^{-0.5 * [(i - m) / σ]^2} / N,

    where

        N = Σ_i e^{-0.5 * [(i - m) / σ]^2}

    is the normalization factor. The amplitudes of the traveling Gaussian
    wavepacket are:

        a_i = √d_i * e^{1j * (i - m) * k}.

    The sites_count is an argument to the get_potential method of this class.

    Attributes:
        k: the phase gradient (velocity) of the Gaussian wavepacket.
        sigma: the standard deviation of the Gaussian density distribution
            spanning [0, 1] interval, σ = sigma * sites_count.
        position: the mean position of the Gaussian density spanning [0, 1]
        interval, m = position * (sites_count - 1) + 1.
    """

    k: RealNumber
    sigma: RealNumber
    position: RealNumber

    def __init__(self, *, k: RealNumber, sigma: RealNumber, position: RealNumber) -> None:
        self.k = k
        self.sigma = sigma
        self.position = position

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls, f'recirq.{cls.__name__}': cls}

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    def get_amplitudes(self, sites_count: int) -> np.ndarray:
        return _phased_gaussian_amplitudes(sites_count,
                                           self.k,
                                           self.sigma,
                                           self.position)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


class TrappingPotential:
    """Base class for initial states defined by trapping potential on a chain.
    """

    @abc.abstractmethod
    def get_potential(self, sites_count: int) -> np.ndarray:
        """Calculates trapping potential to use.

        Args:
            sites_count: Number of fermionic sites.

        Returns:
            Real array of size sites_count with trapping field magnitude at each
            site.
        """

    def as_quadratic_hamiltonian(self,
                                 sites_count: int,
                                 j: Union[RealNumber, Iterable[RealNumber]],
                                 ) -> openfermion.QuadraticHamiltonian:
        """Creates a nonintercting Hamiltonian H_0 that describes particle
        in a trapping potential:

            H_0 = - Σ_i (J_i c_i^† c_{i+1} + h.c.) + Σ_i ε_i n_i,

        where:
            i = 1, 2, ..., sites_count are the site indices,
            c_i are the fermionic annihilation operators,
            n_i = c_i^† c_i are the number operators,
            ε_i is the i-th element of potential array obtained through
              get_potential(sites_count) call.

        Attributes:
            sites_count: Total number of sites.
            j: The hopping coefficients J_i. If Iterable is passed, its size
                must be equal to sites_count - 1.
        """
        return _potential_to_quadratic_hamiltonian(
            self.get_potential(sites_count), j)


@dataclass(init=False)
class FixedTrappingPotential(TrappingPotential):
    """Fixed array describing trapping potential."""
    particles: int
    potential: Tuple[RealNumber]

    def __init__(self, *, particles: int, potential: Iterable[RealNumber]) -> None:
        self.particles = particles
        self.potential = tuple(potential)

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls, f'recirq.{cls.__name__}': cls}

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    def get_potential(self, sites_count: int) -> np.ndarray:
        if sites_count != len(self.potential):
            raise ValueError(f'Fixed potential not compatible with '
                             f'{sites_count} sites')
        return np.array(self.potential)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass(init=False)
class UniformTrappingPotential(TrappingPotential):
    """Uniform trapping potential initial state."""
    particles: int

    def __init__(self, *, particles: int) -> None:
        self.particles = particles

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls, f'recirq.{cls.__name__}': cls}

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    def get_potential(self, sites_count: int) -> np.ndarray:
        return np.zeros(sites_count)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass(init=False)
class GaussianTrappingPotential(TrappingPotential):
    """Gaussian shaped trapping potential for creating the initial state.

    The coefficient ε_i at each site i = 1, 2, ..., sites_count is equal:

        ε_i = scale * e^{−0.5 * (i − m)^2 / σ^2}.

    The sites_count is an argument to the get_potential method of this class.

    Args:
        particles: the number of particles in the potential.
        center: the center position of the Gaussian potential spanning [0, 1]
            interval, m = center * (sites_count - 1) + 1.
        sigma: the standard deviation of the Gaussian spanning [0, 1] interval,
            σ = sigma * (sites_count - 1).
        scale: the scale of the potential s.
    """

    particles: int
    center: RealNumber
    sigma: RealNumber
    scale: RealNumber

    def __init__(self, *,
                 particles: int,
                 center: RealNumber,
                 sigma: RealNumber,
                 scale: RealNumber) -> None:
        self.particles = particles
        self.center = center
        self.sigma = sigma
        self.scale = scale

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls, f'recirq.{cls.__name__}': cls}

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    def get_potential(self, sites_count: int) -> np.ndarray:
        def gaussian(x: int) -> float:
            return self.scale * np.exp(-0.5 * ((x - self.center) /
                                                    self.sigma) ** 2)
        return np.array([gaussian(x) for x in np.linspace(0, 1, sites_count)])

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


ChainInitialState = Union[FixedSingleParticle,
                          PhasedGaussianSingleParticle,
                          FixedTrappingPotential,
                          GaussianTrappingPotential,
                          UniformSingleParticle,
                          UniformTrappingPotential]
"""Initial state that describes independent, noninteracting chain."""


@dataclass(init=False)
class IndependentChainsInitialState:
    """Initial state with two independent, noninteracting chains."""

    up: ChainInitialState
    down: ChainInitialState

    def __init__(self, *,
                 up: ChainInitialState,
                 down: ChainInitialState) -> None:
        self.up = up
        self.down = down

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {
            cls.__name__: cls,
            f'recirq.{cls.__name__}': cls,
            **FixedSingleParticle.cirq_resolvers(),
            **PhasedGaussianSingleParticle.cirq_resolvers(),
            **FixedTrappingPotential.cirq_resolvers(),
            **GaussianTrappingPotential.cirq_resolvers(),
            **UniformSingleParticle.cirq_resolvers(),
            **UniformTrappingPotential.cirq_resolvers()
        }

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    @property
    def up_particles(self) -> int:
        return self.up.particles

    @property
    def down_particles(self) -> int:
        return self.down.particles

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


InitialState = IndependentChainsInitialState
"""Arbitrary initial state type."""


@dataclass
class FermiHubbardParameters:
    """Parameters of a Fermi-Hubbard problem instance.

    This container uniquely defines all the parameters necessary to simulate
    the Fermi-Hubbard problem, including mapping of Fermions on the chip.

    Attributes:
        hamiltonian: Hamiltonian used for dynamics evolution.
        initial_state: Initial state description used to prepare quantum state
            for dynamics evolution.
        layout: Description of mapping of fermions on a quantum processor.
        dt: Time evolution constant that defies Trotter step length.
    """
    hamiltonian: Hamiltonian
    initial_state: InitialState
    layout: QubitsLayout
    dt: float

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {
            cls.__name__: cls,
            f'recirq.{cls.__name__}': cls,
            **Hamiltonian.cirq_resolvers(),
            **IndependentChainsInitialState.cirq_resolvers(),
            **LineLayout.cirq_resolvers(),
            **ZigZagLayout.cirq_resolvers(),
        }

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    @property
    def sites_count(self) -> int:
        return self.layout.size

    @property
    def up_particles(self) -> int:
        return self.initial_state.up_particles

    @property
    def down_particles(self) -> int:
        return self.initial_state.down_particles

    @property
    def up_down_particles(self) -> Tuple[int, int]:
        return self.up_particles, self.down_particles

    def representative_parameters(self) -> 'FermiHubbardParameters':
        return FermiHubbardParameters(self.hamiltonian,
                                      self.initial_state,
                                      self.layout.default_layout(),
                                      self.dt)

    def equals_for_rescaling(self, other: 'FermiHubbardParameters') -> bool:
        if not isinstance(other, FermiHubbardParameters):
            return False
        if isinstance(self.layout, ZigZagLayout) and isinstance(other.layout, ZigZagLayout):
            interacting = not np.allclose(self.hamiltonian.u, 0.0)
            other_interacting = not np.allclose(other.hamiltonian.u, 0.0)
            return interacting == other_interacting
        return False

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'


def _check_normalized(array: Iterable[Number]) -> None:
    if not np.isclose(np.linalg.norm(array), 1):
        raise ValueError('Array is not normalized')


def _phased_gaussian_amplitudes(sites_count: int,
                                k_t0: float,
                                sigma: float,
                                position: float) -> np.ndarray:

    def gaussian(x):
        return np.exp(-0.5 * ((x - position) / sigma) ** 2)

    densities = np.array(
        [gaussian(x) for x in np.linspace(0, 1, sites_count)], dtype=float)
    densities /= np.sum(densities)

    return np.array(
        [np.sqrt(densities[i]) * np.exp(1j * (x - position) * k_t0)
         for i, x in enumerate(np.linspace(0, 1, sites_count))], dtype=complex)


def _potential_to_quadratic_hamiltonian(
        potential: np.ndarray,
        j: Union[RealNumber, Iterable[RealNumber]]
) -> openfermion.QuadraticHamiltonian:
    sites_count = len(potential)

    if isinstance(j, NumbersReal):
        j = np.full(sites_count - 1, j)
    else:
        j = np.array(j)

    if len(j) != sites_count - 1:
        raise ValueError('Hopping coefficient size incompatible with potential')

    # Prepare one-body matrix T_ij.
    one_body = np.zeros((sites_count, sites_count))

    # Nearest-neighbor hopping terms.
    for i in range(sites_count - 1):
        one_body[i, i + 1] += -j[i]
        one_body[i + 1, i] += -j[i]

    # Local interaction terms.
    for i in range(sites_count):
        one_body[i, i] += potential[i]

    return openfermion.QuadraticHamiltonian(one_body)


def _iterable_to_tuple(value: Any) -> Any:
    return value if isinstance(value, NumbersReal) else tuple(value)
