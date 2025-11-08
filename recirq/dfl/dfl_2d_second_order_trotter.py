import os
import pickle
from typing import cast, List, Sequence, Set, Tuple

import cirq
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


class LGTDFL:
    """
    A class for simulating Disorder-free localization (DFL) on a 2-dimensional Lattice Gauge
    Theory (LGT) with  Second order trotter dynamics.

    Attributes:
        qubit_grid: A list of cirq.GridQubit objects representing the 2D grid.
        origin: Top right matter qubit
        dt: The time step for the simulation.
        h: The gauge field strength.
        mu: The matter field strength.
        matter_qubits: A sorted list of cirq.GridQubit objects representing matter qubits.
        gauge_qubits: A sorted list of cirq.GridQubit objects representing gauge qubits.
        all_qubits: A sorted list of all qubits (matter + gauge).
    """

    def __init__(
        self,
        qubit_grid: Sequence[cirq.GridQubit],
        origin: cirq.GridQubit,
        dt: float,
        h: float,
        mu: float,
    ):
        """Initializes the LGTDFL class.

        Args:
            qubit_grid: A list of cirq.GridQubit objects representing the 2D grid.
            origin: The starting cirq.GridQubit object for grid generation.
            dt: The time step for the simulation.
            h: The gauge field strength.
            mu: The matter field strength.
        """
        self.qubit_grid = qubit_grid
        self.origin = origin
        self.dt = dt
        self.h = h
        self.mu = mu
        self.matter_qubits, self.gauge_qubits = self.create_lgt_grid()
        self.all_qubits = sorted(self.matter_qubits + self.gauge_qubits)

    def get_y_neighbors(self, qubit: cirq.GridQubit) -> Sequence[cirq.GridQubit]:
        """Returns the vertical (row-axis) neighbors of a qubit.
        If a matter qubit is provided, this returns the neighboring gauge qubits,
        and vice-versa.

        Args:
            qubit: The central qubit.

        Returns:
            A sequence of neighboring qubits of the opposite type that are
            within the defined grid.
        """
        if qubit in self.gauge_qubits:
            return [
                q
                for q in [
                    cirq.GridQubit(qubit.row + 1, qubit.col),
                    cirq.GridQubit(qubit.row - 1, qubit.col),
                ]
                if q in self.matter_qubits
            ]
        else:
            return [
                q
                for q in [
                    cirq.GridQubit(qubit.row + 1, qubit.col),
                    cirq.GridQubit(qubit.row - 1, qubit.col),
                ]
                if q in self.gauge_qubits
            ]

    def get_x_neighbors(self, qubit: cirq.GridQubit) -> Sequence[cirq.GridQubit]:
        """Returns the horizontal (column-axis) neighbors of a qubit.
        If a matter qubit is provided, this returns the neighboring gauge qubits,
        and vice-versa.

        Args:
            qubit: The central qubit.

        Returns:
            A sequence of neighboring qubits of the opposite type that are
            within the defined grid.
        """
        if qubit in self.gauge_qubits:
            return [
                q
                for q in [
                    cirq.GridQubit(qubit.row, qubit.col + 1),
                    cirq.GridQubit(qubit.row, qubit.col - 1),
                ]
                if q in self.matter_qubits
            ]
        else:
            return [
                q
                for q in [
                    cirq.GridQubit(qubit.row, qubit.col + 1),
                    cirq.GridQubit(qubit.row, qubit.col - 1),
                ]
                if q in self.gauge_qubits
            ]

    def create_lgt_grid(
        self,
    ) -> Tuple[Tuple[cirq.GridQubit, ...], Tuple[cirq.GridQubit, ...]]:
        """Creates a lattice gauge theory (LGT) grid with simplified logic.

        Args:
            qubit_grid: A list of cirq.GridQubit objects.
            origin: The starting cirq.GridQubit object.

        Returns:
            A tuple containing two lists: (matter_qubits, gauge_qubits).
        """

        qubit_set = self.qubit_grid
        matter_qubits = [self.origin]
        gauge_qubits = []
        queue = [self.origin]

        while queue:
            current_matter = queue.pop(0)
            neighbors = current_matter.neighbors()

            for neighbor in neighbors:
                if neighbor in qubit_set and neighbor not in gauge_qubits:
                    gauge_qubits.append(neighbor)

                    # Find matter neighbors for gauge qubits
                    if neighbor.row == current_matter.row:  # Horizontal gauge
                        horizontal_neighbor = cirq.GridQubit(
                            neighbor.row,
                            neighbor.col + (neighbor.col - current_matter.col),
                        )
                        if (
                            horizontal_neighbor in self.qubit_grid
                            and horizontal_neighbor not in matter_qubits
                        ):
                            matter_qubits.append(horizontal_neighbor)
                            queue.append(horizontal_neighbor)
                    else:  # Vertical gauge
                        vertical_neighbor = cirq.GridQubit(
                            neighbor.row + (neighbor.row - current_matter.row),
                            neighbor.col,
                        )
                        if (
                            vertical_neighbor in self.qubit_grid
                            and vertical_neighbor not in matter_qubits
                        ):
                            matter_qubits.append(vertical_neighbor)
                            queue.append(vertical_neighbor)

        return tuple(sorted(matter_qubits)), tuple(sorted(gauge_qubits))  # type: ignore

    def draw_lgt_grid(self):
        """Draw the LGT grid."""
        d = {q: 1 for q in self.gauge_qubits}
        d.update({q: 0 for q in self.matter_qubits})
        cirq.Heatmap(d).plot()

    def _layer_matter_gauge_x(self) -> cirq.Circuit:
        """Single qubit rotations i.e., the mu and h terms."""
        moment = []
        for q in self.matter_qubits:
            moment.append(cirq.rx(2 * self.mu * self.dt).on(q))
        for q in self.gauge_qubits:
            moment.append(cirq.rx(2 * self.h * self.dt).on(q))
        return cirq.Circuit.from_moments(cirq.Moment(moment))

    def layer_hadamard(self, which_qubits="all") -> cirq.Circuit:
        """Creates a circuit layer containing Hadamard gates on the specified qubits.

        Args:
            which_qubits: Specifies which qubits get the Hadamard gates.
                Valid values are "all", "gauge", or "matter".

        Returns:
            A cirq.Circuit containing a single Moment of Hadamard operations.
        """
        moment = []
        if which_qubits == "gauge":
            for q in self.gauge_qubits:
                moment.append(cirq.H(q))
        elif which_qubits == "matter":
            for q in self.matter_qubits:
                moment.append(cirq.H(q))
        elif which_qubits == "all":
            for q in self.all_qubits:
                moment.append(cirq.H(q))
        return cirq.Circuit.from_moments(cirq.Moment(moment))

    def layer_measure(self) -> cirq.Circuit:
        """Creates a circuit layer containing measurement operations on all qubits.

        Returns:
            A cirq.Circuit containing a single Moment of measurement operations
            with the key "m".
        """
        moment = []
        for q in self.all_qubits:
            moment.append(cirq.measure(q, key="m"))
        return cirq.Circuit.from_moments(cirq.Moment(moment))

    def trotter_circuit(self, n_cycles, two_qubit_gate="cz_simultaneous") -> cirq.Circuit:
        """Constructs the second-order Trotter circuit for the DFL Hamiltonian.

        Args:
            n_cycles: The number of Trotter steps/cycles to include in the circuit.
            two_qubit_gate: The type of two-qubit gate to use in the layers.
                Valid values are "cz_simultaneous" or "cphase_simultaneous".

        Returns:
            The complete cirq.Circuit for the Trotter evolution.
        """
        if two_qubit_gate == "cz_simultaneous":
            return self._layer_floquet_cz_simultaneous() * n_cycles

        elif two_qubit_gate == "cphase_simultaneous":
            if n_cycles == 0:
                return cirq.Circuit()
            if n_cycles == 1:
                return (
                    self._layer_floquet_cphase_simultaneous_first()
                    + self._layer_floquet_cphase_simultaneous_last_missing_piece()
                )
            else:
                return (
                    self._layer_floquet_cphase_simultaneous_first()
                    + (n_cycles - 1) * self._layer_floquet_cphase_simultaneous_middle()
                    + self._layer_floquet_cphase_simultaneous_last_missing_piece()
                )

    def layer_floquet(
        self, two_qubit_gate="cz_simultaneous", layer="middle"
    ) -> cirq.Circuit:

        """Constructs a layer of the Trotter circuit.

        Args:
            two_qubit_gate: The type of two-qubit gate to use.
                Valid values are "cz_simultaneous" or "cphase_simultaneous".
            layer: Specifies which piece of the sequence to return.
                Valid values are "middle", "last", or "first".

        Returns:
            The cirq.Circuit representing the requested Trotter layer.

        Raises:
            ValueError: If an invalid option for `two_qubit_gate` or `layer` is given.
        """

        if two_qubit_gate == "cz_simultaneous":
            return self._layer_floquet_cz_simultaneous()

        elif two_qubit_gate == "cphase_simultaneous":
            if layer == "middle":
                return self._layer_floquet_cphase_simultaneous_middle()
            elif layer == "last":
                return self._layer_floquet_cphase_simultaneous_last_missing_piece()
            elif layer == "first":
                return self._layer_floquet_cphase_simultaneous_first()
            else:
                raise ValueError("Invalid layer option")
        else:
            raise ValueError("Invalid two_qubit_gate option")

    def _energy_bump_initial_state(
        self,
        matter_config: str,
        excited_qubits: Sequence[cirq.GridQubit],
    ) -> cirq.Circuit:
        """Circuit for energy bump initial state.
        It typically consists of single qubit gates and the basis change circuit U_B.
        But in this second order implementation, I am removing U_B since
        it cancels out with the U_B in the second order trotter circuit."""
        theta = np.arctan(self.h)
        moment = []
        for q in self.gauge_qubits:
            if q in excited_qubits:
                moment.append(cirq.ry(np.pi + theta).on(q))
            else:
                moment.append(cirq.ry(theta).on(q))

        for q in self.matter_qubits:
            if matter_config == "single_sector":
                moment.append(cirq.H(q))

        return cirq.Circuit.from_moments(moment)

    def _layer_floquet_cz_simultaneous(self) -> cirq.Circuit:
        """Circuit for a trotter step of the DFL Hamiltonian
        Second order Trotter U(t) = e^(-itA/2) e^(-itB) e^(-iA/2).
        We take A = zZz term and B = Rx terms, so the trotter layer
        should look like UB Rz(t/2) UB Rx(t) UB Rz(t/2) UB.
        However, UB from previous layer cancels out with UB in the
        following layer so the trotter layer now looks like
        ....Rz(t/2) UB Rx(t) UB Rz(t/2)....
        """

        moment_rz = []
        moment_rx = []
        moment_h = []
        for q in self.gauge_qubits:
            moment_rz.append(cirq.rz(self.dt).on(q))
            moment_h.append(cirq.H(q))
            moment_rx.append(cirq.rx(2 * self.h * self.dt).on(q))

        for q in self.matter_qubits:
            moment_rx.append(cirq.rx(2 * self.mu * self.dt).on(q))

        return (
            cirq.Circuit.from_moments(cirq.Moment(moment_rz))
            + self._change_basis()
            + cirq.Circuit.from_moments(cirq.Moment(moment_rx))
            + self._change_basis()
            + cirq.Circuit.from_moments(cirq.Moment(moment_rz))
        )

    def _layer_floquet_cphase_simultaneous_middle(self) -> cirq.Circuit:
        """Circuit for a trotter step of the DFL Hamiltonian in terms of CPhase
        Second order Trotter U(t) = e^(-itA/2) e^(-itB) e^(-iA/2).
        After cancelling U_B's the circuit for the trotter layer looks like
        Rz(t/2) UB Rx(t) UB Rz(t/2). However we can condense UB Rz(t) UB of two
        consecutive layers in terms of Cphase gates bringing 8 CZ gates down to
        4 CZ and 2 CPhase gates. Let's take a repeating layer in the middle of the form
        ...UB Rz(t/2) Rz(t/2) UB Rx(t)..... = ...UB Rz(t) UB Rx(t)...
        Thus, the first layer should look like Rz(t/2)UB Rx(t).....
        and the last layer is missing ....UB Rz(T/2)
        """

        moment_0x = []
        moment_1x = []

        moment_0y = []
        moment_1y = []

        moment_h = []

        moment_rz1 = []
        moment_rz2 = []

        moment_rx = []

        for q in self.gauge_qubits:
            nbrs = self.get_x_neighbors(q)
            if len(nbrs) == 2:
                q0, q1 = nbrs[0], nbrs[1]
                moment_0x.append(cirq.CZ(q0, q))  # left to right
                moment_1x.append(cirq.cphase(-4 * self.dt).on(q1, q))  # right to left

                moment_h.append(cirq.H(q))

                moment_rz1.append(cirq.rz(2 * self.dt).on(q))
                moment_rz1.append(cirq.rz(2 * self.dt).on(q1))

        for q in self.gauge_qubits:
            nbrs = self.get_y_neighbors(q)
            if len(nbrs) == 2:
                q0, q1 = nbrs[0], nbrs[1]
                moment_0y.append(cirq.CZ(q0, q))  # top to bottom
                moment_1y.append(cirq.cphase(-4 * self.dt).on(q1, q))  # bottom to top

                moment_h.append(cirq.H(q))

                moment_rz2.append(cirq.rz(2 * self.dt).on(q))
                moment_rz2.append(cirq.rz(2 * self.dt).on(q1))

        for q in self.gauge_qubits:
            moment_rx.append(cirq.rx(2 * self.h * self.dt).on(q))

        for q in self.matter_qubits:
            moment_rx.append(cirq.rx(2 * self.mu * self.dt).on(q))

        return cirq.Circuit.from_moments(
            cirq.Moment(moment_h),
            cirq.Moment(moment_0x),
            cirq.Moment(moment_0y),
            cirq.Moment(moment_h),
            cirq.Moment(moment_1x),
            cirq.Moment(moment_rz1),
            cirq.Moment(moment_1y),
            cirq.Moment(moment_rz2),
            cirq.Moment(moment_h),
            cirq.Moment(moment_0x),
            cirq.Moment(moment_0y),
            cirq.Moment(moment_h),
            cirq.Moment(moment_rx),
        )

    def _layer_floquet_cphase_simultaneous_first(self) -> cirq.Circuit:
        """Circuit for a trotter step of the DFL Hamiltonian in terms of CPhase.
        After cancelling U_B's the circuit for the middle trotter layer looks like
        ...Rz(t/2) UB Rx(t) UB Rz(t/2).... However we can condense UB Rz(t) UB of two
        consecutive layers in terms of Cphase gates bringing 8 CZ gates down to
        4 CZ and 2 CPhase gates. A layer in the middle then takes the form
        ...UB Rz(t/2) Rz(t/2) UB Rx(t)..... = ...UB Rz(t) UB Rx(t)...

        However,the first floquet layer for the cphase implementation looks like
        Rz(t/2)UB Rx(t).... Here, we will use the CZ implementation of UB.
        """

        moment_rz = []
        moment_rx = []

        for q in self.gauge_qubits:
            moment_rz.append(cirq.rz(self.dt).on(q))
            moment_rx.append(cirq.rx(2 * self.h * self.dt).on(q))

        for q in self.matter_qubits:
            moment_rx.append(cirq.rx(2 * self.mu * self.dt).on(q))

        return (
            cirq.Circuit.from_moments(cirq.Moment(moment_rz))
            + self._change_basis()
            + cirq.Circuit.from_moments(cirq.Moment(moment_rx))
        )

    def _layer_floquet_cphase_simultaneous_last_missing_piece(self) -> cirq.Circuit:
        """Circuit for a trotter step of the DFL Hamiltonian in terms of CPhase.
        After cancelling U_B's the circuit for the trotter layer looks like
        Rz(t/2) UB Rx(t) UB Rz(t/2). However we can condense UB Rz(t) UB of two
        consecutive layers in terms of Cphase gates bringing 8 CZ gates down to
        4 CZ and 2 CPhase gates. A layer in the middle then takes the form
        ...UB Rz(t/2) Rz(t/2) UB Rx(t)..... = ...UB Rz(t) UB Rx(t)...

        However,the last floquet layer for the cphase implementation looks like
        ......UB Rz(t/2) Here, we will use the CZ implementation of UB.
        """
        moment_rz = []
        for q in self.gauge_qubits:
            moment_rz.append(cirq.rz(self.dt).on(q))

        return self._change_basis() + cirq.Circuit.from_moments(cirq.Moment(moment_rz))

    def _change_basis(self) -> cirq.Circuit:
        """Transform the LGT basis to the dual basis."""
        moment_0x = []
        moment_1x = []

        moment_0y = []
        moment_1y = []

        moment_h = []

        for q in self.gauge_qubits:
            nbrs = self.get_x_neighbors(q)
            if len(nbrs) == 2:
                q0, q1 = nbrs[0], nbrs[1]
                moment_0x.append(cirq.CZ(q0, q))
                moment_1x.append(cirq.CZ(q1, q))
                moment_h.append(cirq.H(q))

        for q in self.gauge_qubits:
            nbrs = self.get_y_neighbors(q)
            if len(nbrs) == 2:
                q0, q1 = nbrs[0], nbrs[1]
                moment_0y.append(cirq.CZ(q0, q))
                moment_1y.append(cirq.CZ(q1, q))
                moment_h.append(cirq.H(q))

        return cirq.Circuit.from_moments(
            cirq.Moment(moment_h),
            cirq.Moment(moment_0x),
            cirq.Moment(moment_1x),
            cirq.Moment(moment_0y),
            cirq.Moment(moment_1y),
            cirq.Moment(moment_h),
        )

    def _interaction_indices(self) -> Sequence[Tuple[int, int, int]]:
        indices = []
        for q in self.gauge_qubits:
            nbrs = self.get_x_neighbors(q)
            if nbrs:
                q0, q1 = nbrs[0], nbrs[1]
                i0, i, i1 = (
                    self.all_qubits.index(q0),
                    self.all_qubits.index(q),
                    self.all_qubits.index(q1),
                )
                indices.append((i0, i, i1))

            nbrs = self.get_y_neighbors(q)
            if nbrs:
                q0, q1 = nbrs[0], nbrs[1]
                i0, i, i1 = (
                    self.all_qubits.index(q0),
                    self.all_qubits.index(q),
                    self.all_qubits.index(q1),
                )
                indices.append((i0, i, i1))
        return indices

    def _matter_indices(self) -> Sequence[int]:
        indices = []
        for i, q in enumerate(self.all_qubits):
            if q in self.matter_qubits:
                indices.append(i)
        return indices

    def _gauge_indices(self) -> Sequence[int]:
        indices = []
        for i, q in enumerate(self.all_qubits):
            if q in self.gauge_qubits:
                indices.append(i)
        return indices

    def _compute_observables_one_instance_dual_basis(
        self, bits_z: npt.NDArray[np.int8], bits_x: npt.NDArray[np.int8]
    ) -> Sequence[np.ndarray]:
        bits_x_rescaled = 1 - 2 * bits_x
        bits_z_rescaled = 1 - 2 * bits_z

        gauge_inds = [self.all_qubits.index(q) for q in self.gauge_qubits]

        matter_inds = [self.all_qubits.index(q) for q in self.matter_qubits]
        gauge_inds = [self.all_qubits.index(q) for q in self.gauge_qubits]

        exp_gauge_x = np.mean(bits_x_rescaled[:, gauge_inds], axis=0)
        var_gauge_x = np.var(bits_x_rescaled[:, gauge_inds], axis=0)

        exp_interaction = np.mean(bits_z_rescaled[:, gauge_inds], axis=0)
        var_interaction = np.mean(bits_z_rescaled[:, gauge_inds], axis=0)

        matter_x_terms = []
        # in the dual basis the matter x takes the form
        # \sigma^x_j -> \sigma^x_j \prod_{k \in N(j)} X_{jk}

        for q in self.matter_qubits:
            prod_x = bits_x_rescaled[:, self.all_qubits.index(q)]
            q_nbrs = [q_n for q_n in q.neighbors() if q_n in self.all_qubits]
            for q_n in q_nbrs:
                prod_x *= bits_x_rescaled[
                          :, self.all_qubits.index(cast(cirq.GridQubit, q_n))
                          ]
            matter_x_terms.append(prod_x)

        exp_matter_x = np.mean(matter_x_terms, axis=1)
        var_matter_x = np.var(matter_x_terms, axis=1)

        exp_energy = []
        var_energy = []
        for _, idx in enumerate(self._interaction_indices()):
            q0 = self.all_qubits[idx[0]]
            q1 = self.all_qubits[idx[2]]

            w0 = 1 / len([n for n in q0.neighbors() if n in self.gauge_qubits])
            w1 = 1 / len([n for n in q1.neighbors() if n in self.gauge_qubits])

            mat_1_index = self.matter_qubits.index(self.all_qubits[idx[0]])
            gauge_index = self.gauge_qubits.index(self.all_qubits[idx[1]])
            mat_2_index = self.matter_qubits.index(self.all_qubits[idx[2]])

            em = (
                exp_interaction[gauge_index]
                + self.h * exp_gauge_x[gauge_index]
                + self.mu
                * (w0 * exp_matter_x[mat_1_index] + w1 * exp_matter_x[mat_2_index])
            )
            ev = (
                var_interaction[gauge_index]
                + self.h * var_gauge_x[gauge_index]
                + self.mu
                * (w0 * var_matter_x[mat_1_index] + w1 * var_matter_x[mat_2_index])
            )
            exp_energy.append(em)
            var_energy.append(ev)

        return [
            np.stack([exp_matter_x, var_matter_x], axis=-1),
            np.stack([exp_gauge_x, var_gauge_x], axis=-1),
            np.stack([exp_interaction, var_interaction], axis=-1),
            np.stack([exp_energy, var_energy], axis=-1),
        ]

    def compute_observables_dual_basis(
        self,
        bits_z: Sequence[npt.NDArray[np.int8]],
        bits_x: Sequence[npt.NDArray[np.int8]],
    ) -> Sequence[np.ndarray]:
        """This calculates the mean and variance over multiple instances
        of measurement outcomes for the matter X, gauge X, interaction
        and energy terms.

        Args:
            bits_z: Measurement outcomes in the Z basis.
            bits_x: Measurement outcomes in the X basis.

        Returns:
            A sequence of NumPy arrays, where each array contains the mean and
            variance for the respective observable.
        """
        num_instances = len(bits_z)
        if num_instances == 1:
            return self._compute_observables_one_instance_dual_basis(
                bits_z[0], bits_x[0]
            )
        else:
            # compute mean and variance of the mean over randomized instances
            matter_x = []
            gauge_x = []
            interaction = []
            energy = []
            for ins in range(num_instances):
                mx, gx, intr, e = self._compute_observables_one_instance_dual_basis(
                    bits_z[ins], bits_x[ins]
                )
                matter_x.append(mx[:, 0])
                gauge_x.append(gx[:, 0])
                interaction.append(intr[:, 0])
                energy.append(e[:, 0])

            return [
                np.stack(
                    [
                        np.mean(np.array(matter_x), axis=0),
                        np.var(np.array(matter_x), axis=0) / num_instances,
                    ],
                    axis=-1,
                ),
                np.stack(
                    [
                        np.mean(np.array(gauge_x), axis=0),
                        np.var(np.array(gauge_x), axis=0) / num_instances,
                    ],
                    axis=-1,
                ),
                np.stack(
                    [
                        np.mean(np.array(interaction), axis=0),
                        np.var(np.array(interaction), axis=0) / num_instances,
                    ],
                    axis=-1,
                ),
                np.stack(
                    [
                        np.mean(np.array(energy), axis=0),
                        np.var(np.array(energy), axis=0) / num_instances,
                    ],
                    axis=-1,
                ),
            ]

    def _postselect_on_charge_one_instance_dual_basis(
        self, bits: npt.NDArray[np.int8]
    ) -> npt.NDArray[np.int8]:
        q_measured = []
        for q in self.matter_qubits:
            q_measured.append(1 - 2 * bits[self.all_qubits.index(q)])
        q_measured_array = np.array(q_measured)
        q_measured_array = np.transpose(q_measured_array)
        charges = np.repeat(1, q_measured_array.shape[0])
        selected = np.all(q_measured_array == charges, axis=0)
        return bits[selected]

    def postselect_on_charge_dual_basis(self, bits: npt.NDArray[np.int8]) -> np.ndarray:
        """Performs charge post-selection on measurement outcomes.

        Args:
            bits: The raw measurement outcomes.

        Returns:
            A NumPy array containing only the post-selected measurement instances.
        """
        num_instances = bits.shape[0]
        bits_ps = []
        for ins in range(num_instances):
            bits_ps_i = self._postselect_on_charge_one_instance_dual_basis(bits[ins])
            if len(bits_ps_i) > 0:
                bits_ps.append(bits_ps_i)
        return np.array(bits_ps)

    def get_2d_dfl_experiment_circuits(
        self,
        initial_state: str,
        n_cycles: Sequence[int] | npt.NDArray,
        excited_qubits: Sequence[cirq.GridQubit],
        n_instances: int = 10,
        two_qubit_gate: str = "cz_simultaneous",
        basis="dual",
    ) -> List[cirq.Circuit]:
        """Generates the set of circuits needed for the 2D DFL experiment.

        Args:
            initial_state: The initial state preparation.
                Valid values are "single_sector" or "superposition".
            n_cycles: The number of Trotter steps (cycles) to simulate.
            excited_qubits: Qubits to be excited in the initial state.
            n_instances: The number of instances to generate
            two_qubit_gate: The type of two-qubit gate to use in the Trotter step.
                Valid values are "cz_simultaneous" or "cphase_simultaneous".
            basis: The basis for the final circuit structure.
                Valid values are "lgt" or "dual".

        Returns:
            A list of all generated cirq.Circuit objects.

        Raises:
            ValueError: If an invalid option for `initial_state`
                or `basis` is given.
        """
        if initial_state == "single_sector":
            initial_circuit = self._energy_bump_initial_state(
                "single_sector", excited_qubits
            )
        elif initial_state == "superposition":
            initial_circuit = self._energy_bump_initial_state(
                "superposition", excited_qubits
            )
        else:
            raise ValueError("Invalid initial state")
        circuits = []
        for n_cycle in tqdm(n_cycles):
            print(int(np.max([0, n_cycle - 1])))
            circ = initial_circuit + self.trotter_circuit(n_cycle, two_qubit_gate)
            if basis == "lgt":
                circ += self._change_basis()
            elif basis == "dual":
                pass
            else:
                raise ValueError("Invalid option for basis")
            for _ in range(n_instances):
                if basis == "lgt":
                    circ_z = circ + cirq.measure([q for q in self.all_qubits], key="m")
                elif basis == "dual":
                    circ_z = (
                        circ
                        + self.layer_hadamard("matter")
                        + cirq.measure([q for q in self.all_qubits], key="m")
                    )
                else:
                    raise ValueError("Invalid option for basis")
                circ_x = (
                    circ
                    + self.layer_hadamard("all")
                    + cirq.measure([q for q in self.all_qubits], key="m")
                )
                circuits.append(circ_z)
                circuits.append(circ_x)
        return circuits
