import os
from typing import Sequence, Optional, Iterable, List

import cirq
import numpy as np
from matplotlib import pyplot as plt
import cirq_google as cg

SQRT_ISWAP = cirq.ISWAP ** 0.5


def create_example_circuit(
        line_length: int = 20,
        segment_length: int = 5,
        num_trotter_steps: int = 10,
) -> cirq.Circuit:
    """Returns a linear chain circuit to demonstrate Floquet calibration on."""

    line = cirq.LineQubit.range(line_length)
    segments = [line[i: i + segment_length]
                for i in range(0, line_length - segment_length + 1, segment_length)]

    circuit = cirq.Circuit()

    # Initial state preparation.
    for segment in segments:
        circuit += [cirq.X.on(segment[len(segment) // 2])]

    # Trotter steps.
    for step in range(num_trotter_steps):
        offset = step % 2
        moment = cirq.Moment()
        for segment in segments:
            moment += cirq.Moment(
                [SQRT_ISWAP(a, b) for a, b in zip(segment[offset::2],
                                                  segment[offset + 1::2])])
        circuit += moment

    # Measurement.
    circuit += cirq.measure(*line, key='z')
    return circuit


def z_density_from_measurements(
        measurements: np.ndarray,
        post_select_filling: Optional[int] = 1
) -> np.ndarray:
    """Returns density for one segment on the line."""
    counts = np.sum(measurements, axis=1, dtype=int)

    if post_select_filling is not None:
        errors = np.abs(counts - post_select_filling)
        counts = measurements[errors == 0]

    return np.average(counts, axis=0)


def z_densities_from_result(
        result: cirq.Result,
        segment_lens: Iterable[int],
        post_select_filling: Optional[int] = 1
) -> List[np.ndarray]:
    """Returns densities for each segment on the line."""
    measurements = result.measurements['z']
    z_densities = []

    offset = 0
    for segment_len in segment_lens:
        z_densities.append(z_density_from_measurements(
            measurements[:, offset: offset + segment_len],
            post_select_filling)
        )
        offset += segment_len
    return z_densities


def plot_density(
        ax: plt.Axes,
        sim_density: np.ndarray,
        raw_density: np.ndarray,
        cal_density: Optional[np.ndarray] = None,
        raw_errors: Optional[np.ndarray] = None,
        cal_errors: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        show_legend: bool = True,
        show_ylabel: bool = True,
) -> None:
    """Plots the density of a single segment for simulated, raw, and calibrated
    results.
    """
    colors = ["grey", "orange", "green"]
    alphas = [0.5, 0.8, 0.8]
    labels = ["sim", "raw", "cal"]

    # Plot densities.
    for i, density in enumerate([sim_density, raw_density, cal_density]):
        if density is not None:
            ax.plot(
                range(len(density)),
                density,
                "-o" if i == 0 else "o",
                markersize=11,
                color=colors[i],
                alpha=alphas[i],
                label=labels[i]
            )

    # Plot errors if provided.
    errors = [raw_errors, cal_errors]
    densities = [raw_density, cal_density]
    for i, (errs, dens) in enumerate(zip(errors, densities)):
        if errs is not None:
            ax.errorbar(
                range(len(errs)),
                dens,
                errs,
                linestyle='',
                color=colors[i + 1],
                capsize=8,
                elinewidth=2,
                markeredgewidth=2
            )

    # Titles, axes, and legend.
    ax.set_xticks(list(range(len(sim_density))))
    ax.set_xlabel("Qubit index in segment")
    if show_ylabel:
        ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend()


def plot_densities(
        sim_density: np.ndarray,
        raw_densities: Sequence[np.ndarray],
        cal_densities: Optional[Sequence[np.ndarray]] = None,
        rows: int = 3
) -> None:
    """Plots densities for simulated, raw, and calibrated results on all segments.
    """
    if not cal_densities:
        cal_densities = [None] * len(raw_densities)

    cols = (len(raw_densities) + rows - 1) // rows

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 3.5), sharey=True
    )
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows > 1 and cols > 1:
        axes = [axes[row, col] for row in range(rows) for col in range(cols)]

    for i, (ax, raw, cal) in enumerate(zip(axes, raw_densities, cal_densities)):
        plot_density(
            ax,
            sim_density,
            raw,
            cal,
            title=f"Segment {i + 1}",
            show_legend=False,
            show_ylabel=i % cols == 0
        )

    # Common legend for all subplots.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)

    plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=3.0)


def main():
    os.remove('simple.png')
    segment_length = 5
    n_segments = 4
    n_segments = 2
    segment_lens = [5] * n_segments
    line_length = segment_length * n_segments
    num_trotter_steps = 20

    circuit_on_segment = create_example_circuit(
        line_length=segment_length,
        segment_length=segment_length,
        num_trotter_steps=num_trotter_steps,
    )

    nreps = 20_000
    sim_result = cirq.Simulator().run(circuit_on_segment, repetitions=nreps)

    circuit = create_example_circuit(
        line_length=line_length,
        segment_length=segment_length,
        num_trotter_steps=num_trotter_steps
    )
    sampler = cg.PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(
        mean=cg.SQRT_ISWAP_INV_PARAMETERS,
        sigma=cg.PhasedFSimCharacterization(
            theta=0.01, zeta=0.10, chi=0.01, gamma=0.10, phi=0.02
        ),
    )
    sampler = cirq.DensityMatrixSimulator(noise=cirq.depolarize(5e-3))
    raw_results = sampler.run(circuit, repetitions=nreps)

    # Simulator density.
    sim_density, = z_densities_from_result(sim_result, [segment_length])

    # Processor densities without Floquet calibration.
    raw_densities = z_densities_from_result(raw_results, segment_lens)

    raw_avg = np.average(raw_densities, axis=0)
    raw_std = np.std(raw_densities, axis=0, ddof=1)

    plot_density(
        plt.gca(),
        sim_density,
        raw_density=raw_avg,
        raw_errors=raw_std,
        title="Average over segments"
    )
    plt.savefig('simple.png')


if __name__ == '__main__':
    main()
