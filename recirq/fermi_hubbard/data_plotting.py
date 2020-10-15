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
"""Create plots out of experiment results."""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recirq.fermi_hubbard.post_processing import (
    AggregatedQuantity,
    InstanceBundle,
    PerSiteQuantity,
    Rescaling
)


def default_top_label(
        ax: plt.Axes, u: float, step: Optional[int], dt: float) -> None:
    """Function that puts a label on top of axis with standard text."""

    if step is not None:
        label = f'u={u}, η={step}, τ={round(step * dt, 1)}'
    else:
        label = f'u={u}'

    ax.text(0.5, 1.02, label, ha='center', va='bottom', transform=ax.transAxes)


def plot_quantity(
        bundle: Union[InstanceBundle, Sequence[InstanceBundle]],
        quantity_name: str,
        *,
        plot_file: Optional[str] = None,
        us: Optional[Iterable[float]] = None,
        steps: Optional[Iterable[int]] = None,
        show_numerics: bool = True,
        show_std_dev: bool = False,
        show_std_error: bool = False,
        axes: Optional[Sequence[Sequence[plt.Axes]]] = None,
        colors: Optional[Sequence[str]] = None,
        label_top_func: Optional[Callable[
            [plt.Axes, float, Optional[int], float], None]] = default_top_label,
        space_horizontal: float = 0.2,
        space_vertical: float = 0.4,
) -> Tuple[Optional[plt.Figure], Sequence[Sequence[plt.Axes]]]:
    """Pre-defined plotting for Fermi-Hubbard quantities.

    Args:
        bundle: Bundle, or iterable of bundles to plot the data from.
        quantity_name: Name of quantity to plot, the list of supported names
            can be retrieved by listing keys of quantities property of
             InstanceBundle class.
        plot_file: File name where plot image should be rendered to. If not
            provided, no file will be written to.
        us: List of interaction strengths U to plot. If not provided, all
            bundles will be plotted.
        steps: List of Trotter steps to plot. If not provided, the steps will
            be extracted from bundles provided that all of them have the same
            range.
        show_numerics: If true, the exact numerical simulation will be plotted
            as lines.
        show_std_dev: If true, the data standard deviation will be plotted as
            error bars.
        show_std_error: If true, the standard error of a mean will be plotted
            as error bars.
        axes: Axes to render onto. The shape of axes must be compatible with the
            data plotted. If not provided, an appropriate axes will be created.
        colors: List of colors to use for each plotted chain. If not provided,
            a set of predefined colors for each quantity will be used.
        label_top_func: The function that renders text on top of each axes.
        space_horizontal: The horizontal spacing between axes.
        space_vertical: The vertical spacing between axes.

    Returns:
        Tuple of:
         - Matplotlib figure created for plotting or None if axes were passed in
           as input.
         - Axes created for the purpose of plotting.
    """

    if isinstance(bundle, InstanceBundle):
        bundles = bundle,
    else:
        bundles = bundle

    if not len(bundles):
        return None, [[]]

    if not steps:
        steps = bundles[0].steps
        for bundle in bundles[1:]:
            if steps != bundle.steps:
                raise ValueError(
                    'Unequal Trotter steps between experiment bundles')
    else:
        steps = list(steps)

    if not us:
        us = [bundle.u for bundle in bundles]
    else:
        us = list(us)

    averages, errors, dts, data_shape, bundles = _extract_bundle_data(
        bundles, quantity_name, us, steps,
        show_std_dev=show_std_dev,
        show_std_error=show_std_error)

    allow_numerics = quantity_name != 'scaling'

    if allow_numerics and show_numerics:
        numerics, *_ = _extract_bundle_data(
            bundles, quantity_name, us, steps,
            show_std_dev=show_std_dev,
            show_std_error=show_std_error,
            simulated=True)
    else:
        numerics = None

    if not colors:
        if (quantity_name.startswith('up_down') or
                quantity_name.startswith('scaling')):
            colors = 'tab:orange', 'tab:green'
        elif quantity_name.startswith('charge_spin'):
            colors = 'tab:blue', 'tab:red'
        elif quantity_name.startswith('post_selection'):
            colors = 'tab:purple',

    split_yaxes = (quantity_name.startswith('charge_spin') and
                   not quantity_name.endswith('_dt'))

    left_label, right_label = _find_left_right_labels(quantity_name)

    if len(data_shape) == 1:
        sites_count, = data_shape
        sites = list(range(1, sites_count + 1))
        axis_width = max(2.5, 0.3 * len(sites))
        axis_aspect_ratio = axis_width / 2.5
        label_bottom = 'Site'

        fig, axes = _plot_density_quantity(
            averages, errors, numerics, us, dts, steps, sites,
            colors=colors,
            axes=axes,
            split_yaxes=split_yaxes,
            label_top_func=label_top_func,
            label_left=left_label,
            label_right=right_label,
            label_bottom=label_bottom,
            axis_width=axis_width,
            axis_aspect_ratio=axis_aspect_ratio,
            space_horizontal=space_horizontal,
            space_vertical=space_vertical)
    else:
        if axes:
            if len(axes) != 1:
                raise ValueError('Incompatible number of axes rows')
            axes, = axes

        axis_width = max(2.5, 0.3 * len(steps))
        axis_aspect_ratio = axis_width / 2.5
        label_bottom = 'Time'

        fig, axes = _plot_aggregated_quantity(
            averages, errors, numerics, us, dts, steps,
            colors=colors,
            axes=axes,
            split_yaxes=split_yaxes,
            label_top_func=label_top_func,
            label_left=left_label,
            label_right=right_label,
            label_bottom=label_bottom,
            axis_width=axis_width,
            axis_aspect_ratio=axis_aspect_ratio,
            space_horizontal=space_horizontal,
            space_vertical=space_vertical)

        if quantity_name == 'scaling':
            _plot_rescaling_lines(bundles, axes)

        axes = axes,

    if plot_file:
        plt.savefig(plot_file)

    return fig, axes


def _find_left_right_labels(name: str) -> Tuple[Optional[str], Optional[str]]:
    if name.startswith('up_down_'):
        left_chain, right_chain = '', None
    elif name.startswith('charge_spin_'):
        left_chain, right_chain = 'Charge ', 'Spin '
    else:
        left_chain, right_chain = '', None

    if name.endswith('_norescale'):
        name = name[:-len('_norescale')]

    if name.endswith('_dt'):
        name = name[:-len('_dt')]
        suffix = ' rate'
    else:
        suffix = ''

    if name.endswith('density'):
        text = 'density'
    elif name.endswith('spreading'):
        text = 'spreading'
    elif name.endswith('position_average'):
        text = 'average position'
    elif name.endswith('scaling'):
        text = 'scaling'
    elif name.endswith('post_selection'):
        text = 'success rate'
    else:
        return None, None

    def format_chain(chain: Optional[str], text: str) -> Optional[str]:
        if chain is None:
            return None
        return f'{chain}{text}{ suffix}'.capitalize()

    return format_chain(left_chain, text), format_chain(right_chain, text)


def quantity_data_frame(bundle: Union[InstanceBundle, Sequence[InstanceBundle]],
                        quantity_name: str,
                        us: Optional[Iterable[float]] = None,
                        steps: Optional[Iterable[int]] = None,
                        simulated: bool = False
                        ) -> Tuple[pd.DataFrame, type, List[InstanceBundle]]:
    """Extracts given quantity from InstanceBundle(s) as Panda's data frame.

    Args:
        bundle: Bundle, or iterable of bundles to plot the data from.
        quantity_name: Name of quantity to plot, the list of supported names
            can be retrieved by listing keys of quantities property of
             InstanceBundle class.
        us: List of interaction strengths U to plot. If not provided, all
            bundles will be plotted.
        steps: List of Trotter steps to plot. If not provided, the steps will
            be extracted from bundles provided that all of them have the same
            range.
        simulated: When true, extracts the exact numerical simulation that
            matches the instance problem parameters instead of the bundle
            experimental data.

    Returns:
        Tuple of:
         - Panda's data frame with accumulated data. The columns of returned
           data frame depends whether the quantity is of AggregatedQuantity or
           PerSiteQuantity.
         - Type of selected quantity, either AggregatedQuantity or
           PerSiteQuantity.
         - List of InstanceBundles used, when us list selects a subset of
           provided bundles.
    """

    def create_aggregated_data(data: List[Tuple[float,
                                          int,
                                          int,
                                          float,
                                          float,
                                          Optional[float],
                                          Optional[float]]]
                               ) -> pd.DataFrame:
        return pd.DataFrame(data, columns=[
            'u', 'chain', 'step', 'time', 'value', 'std_error', 'std_dev'])

    def generate_aggregated_entries(dt: float,
                                    u: float,
                                    quantity: AggregatedQuantity,
                                    step: int,
                                    data_step: int
                                    ) -> Tuple[float,
                                               int,
                                               int,
                                               float,
                                               float,
                                               Optional[float],
                                               Optional[float]]:

        for chain in range(quantity.chains_count):
            if quantity.std_error is not None:
                std_error = float(quantity.std_error[chain][step])
            else:
                std_error = None

            if quantity.std_dev is not None:
                std_dev = float(quantity.std_dev[chain][step])
            else:
                std_dev = None

            yield (u,
                   chain,
                   data_step,
                   data_step * dt,
                   float(quantity.average[chain][step]),
                   std_error,
                   std_dev)

    def create_per_site_data(data: List[Tuple[float,
                                              int,
                                              int,
                                              int,
                                              float,
                                              float,
                                              Optional[float],
                                              Optional[float]]]
                             ) -> pd.DataFrame:
        return pd.DataFrame(data, columns=[
            'u', 'chain', 'step', 'time', 'site', 'value', 'std_error',
            'std_dev'])

    def generate_per_site_entries(dt: float,
                                  u: float,
                                  quantity: AggregatedQuantity,
                                  step: int,
                                  data_step: int
                                  ) -> Tuple[float,
                                             int,
                                             int,
                                             int,
                                             float,
                                             float,
                                             Optional[float],
                                             Optional[float]]:

        for chain in range(quantity.chains_count):
            for site in range(len(quantity.average[chain][step])):
                if quantity.std_error is not None:
                    std_error = quantity.std_error[chain][step][site]
                else:
                    std_error = pd.NA

                if quantity.std_dev is not None:
                    std_dev = quantity.std_dev[chain][step][site]
                else:
                    std_dev = pd.NA

                yield (u,
                       chain,
                       data_step,
                       data_step * dt,
                       site + 1,
                       quantity.average[chain][step][site],
                       std_error,
                       std_dev)

    if isinstance(bundle, InstanceBundle):
        bundles = bundle,
    else:
        bundles = bundle

    if not len(bundles):
        raise ValueError('At least one InstanceBundle is mandatory')

    if not steps:
        steps = bundles[0].steps
        for bundle in bundles[1:]:
            if steps != bundle.steps:
                raise ValueError(
                    'Unequal Trotter steps between experiment bundles')
    else:
        steps = list(steps)

    if not us:
        us = [bundle.u for bundle in bundles]
    else:
        us = list(us)

    steps_map = {step: index for index, step in enumerate(steps)}
    us_map = {u: index for index, u in enumerate(us)}
    us_bundles = []

    data = []
    quantity_type = None

    missing_u = set(us)
    for bundle in bundles:
        if bundle.u not in us_map:
            continue

        if simulated:
            bundle = bundle.exact_numerics_bundle()

        us_bundles.append(bundle)
        missing_u.remove(bundle.u)

        quantity = bundle.quantities[quantity_name]()

        if quantity_type is None:
            if isinstance(quantity, AggregatedQuantity):
                quantity_type = AggregatedQuantity
            elif isinstance(quantity, PerSiteQuantity):
                quantity_type = PerSiteQuantity
            else:
                raise ValueError(f'Unknown quantity type {quantity_type}')
        else:
            assert isinstance(quantity, quantity_type), (
                f'Inconsistent quantity types {type(quantity)} and '
                f'{quantity_type}')

        missing_steps = set(steps)
        for s, step in enumerate(bundle.steps):
            if step not in steps_map:
                continue

            missing_steps.remove(step)
            step_index = steps_map[step]

            if quantity_type == AggregatedQuantity:
                data += generate_aggregated_entries(
                    bundle.dt, bundle.u, quantity, s, step_index)
            elif quantity_type == PerSiteQuantity:
                data += generate_per_site_entries(
                    bundle.dt, bundle.u, quantity, s, step_index)
            else:
                raise RuntimeError(f'Unexpected quantity type {quantity_type}')

        if missing_steps:
            raise ValueError(
                f'No experiments to cover Trotter steps {missing_steps}')

    if missing_u:
        raise ValueError(f'No experiments to cover U values {missing_u}')

    if quantity_type == AggregatedQuantity:
        pandas = create_aggregated_data(data)
    elif quantity_type == PerSiteQuantity:
        pandas = create_per_site_data(data)
    else:
        raise RuntimeError(f'Unexpected quantity type {quantity_type}')

    return pandas, quantity_type, us_bundles


def _extract_bundle_data(bundles: Sequence[InstanceBundle],
                         quantity_name: str,
                         us: List[float],
                         steps: List[int],
                         show_std_dev: bool,
                         show_std_error: bool,
                         simulated: bool = False
                         ) -> Tuple[List[List[List[np.ndarray]]],
                                    List[List[List[np.ndarray]]],
                                    List[float],
                                    Tuple[int, ...],
                                    List[InstanceBundle]]:

    def init_list(shape: Tuple[int, ...]) -> List:
        if len(shape) == 1:
            return [None] * shape[0]
        return [init_list(shape[1:]) for _ in range(shape[0])]

    if show_std_dev and show_std_error:
        raise ValueError('Either standard deviation or standard error of a '
                         'mean can be shown, not both.')

    steps_map = {step: index for index, step in enumerate(steps)}
    us_map = {u: index for index, u in enumerate(us)}
    us_bundles = []
    dts = [0.0] * len(us)

    averages = None
    errors = None
    chains_count = None
    data_shape = None

    missing_u = set(us)
    for bundle in bundles:
        if bundle.u not in us_map:
            continue

        if simulated:
            bundle = bundle.exact_numerics_bundle()

        us_bundles.append(bundle)
        u_index = us_map[bundle.u]
        missing_u.remove(bundle.u)

        dts[u_index] = bundle.dt

        quantity = bundle.quantities[quantity_name]()
        missing_steps = set(steps)
        for s, step in enumerate(bundle.steps):
            if step not in steps_map:
                continue

            missing_steps.remove(step)
            step_index = steps_map[step]

            if chains_count is None:
                chains_count = quantity.chains_count
                shape = chains_count, len(us), len(steps)
                averages = init_list(shape)
                errors = init_list(shape)
            elif chains_count != quantity.chains_count:
                raise ValueError(
                    'Incompatible quantity between bundles')

            for chain in range(chains_count):
                average = quantity.average[chain][s]
                if show_std_error and quantity.std_error:
                    error = quantity.std_error[chain][s]
                elif show_std_dev and quantity.std_dev:
                    error = quantity.std_dev[chain][s]
                else:
                    error = None

                if data_shape is None:
                    data_shape = average.shape
                if data_shape != average.shape or (
                        error is not None and data_shape != error.shape):
                    raise ValueError(
                        'Incompatible quantity shapes between bundles')

                if error is None:
                    error = np.nan

                averages[chain][u_index][step_index] = average
                errors[chain][u_index][step_index] = error

        if missing_steps:
            raise ValueError(
                f'No experiments to cover Trotter steps {missing_steps}')

    if missing_u:
        raise ValueError(f'No experiments to cover U values {missing_u}')

    return averages, errors, dts, data_shape, us_bundles


def _plot_density_quantity(
        values: List[List[List[np.ndarray]]],
        errors: List[List[List[np.ndarray]]],
        numerics: Optional[List[List[List[np.ndarray]]]],
        us: Sequence[float],
        dts: Sequence[float],
        steps: Sequence[int],
        sites: Sequence[int],
        colors: Optional[Sequence[str]],
        axes: Optional[Sequence[Sequence[plt.Axes]]] = None,
        split_yaxes: bool = True,
        label_top_func: Optional[
            Callable[[plt.Axes, float, Optional[int], float], None]] = None,
        label_left: Optional[str] = None,
        label_right: Optional[str] = None,
        label_bottom: Optional[str] = None,
        axis_width: Optional[float] = 4.0,
        axis_aspect_ratio: Optional[float] = 4.0 / 3.0,
        space_left: float = 0.7,
        space_right: float = 0.7,
        space_top: float = 0.3,
        space_bottom: float = 0.5,
        space_horizontal: float = 0.15,
        space_vertical: float = 0.15,
        span_extension: float = 0.05
) -> Tuple[Optional[plt.Figure], Sequence[Sequence[plt.Axes]]]:

    if axes is None:
        fig, axes = _create_default_axes(len(us),
                                         len(steps),
                                         axis_width=axis_width,
                                         axis_aspect_ratio=axis_aspect_ratio,
                                         space_left=space_left,
                                         space_right=space_right,
                                         space_top=space_top,
                                         space_bottom=space_bottom,
                                         space_horizontal=space_horizontal,
                                         space_vertical=space_vertical)
    else:
        fig = None

    if len(axes) != len(us):
        raise ValueError('Incompatible number of axes rows')

    spans = _calculate_spans(values, 1, span_extension)
    if numerics:
        numerics_spans = _calculate_spans(numerics, 1, span_extension)
        spans = [_merge_spans(pair) for pair in zip(spans, numerics_spans)]
    if not split_yaxes:
        spans = [_merge_spans(spans)] * len(spans)

    xlim = min(sites) - 0.5, max(sites) + 0.5

    for chain, (values_chain, errors_chain) in enumerate(zip(values, errors)):
        for index_u, (u, dt_u, values_u, errors_u, axes_u) in enumerate(
                zip(us, dts, values_chain, errors_chain, axes)):

            if len(axes_u) != len(steps):
                raise ValueError('Incompatible number of axes columns')

            for index_step, (step, values_step, errors_step, ax_step) in \
                    enumerate(zip(steps, values_u, errors_u, axes_u)):

                color = colors[chain] if colors else None

                # Disable labels for inner subplots.
                if index_u != len(us) - 1:
                    ax_step.set_xticklabels([])
                if index_step:
                    ax_step.set_yticklabels([])

                # Determine plotting axes for each chain.
                if chain == 0:
                    ax = ax_step
                    if index_step == 0 and label_left:
                        ax.set_ylabel(label_left,
                                      color=color if split_yaxes else 'k')
                    if label_top_func:
                        label_top_func(ax, u, step, dt_u)
                    if label_bottom and index_u == len(us) - 1:
                        ax.set_xlabel(label_bottom)
                elif chain == 1:
                    if split_yaxes:
                        ax = ax_step.twinx()
                        if index_step != len(steps) - 1:
                            ax.set_yticklabels([])
                        elif label_right:
                            ax.set_ylabel(label_right, color=color)
                    else:
                        ax = ax_step
                else:
                    raise ValueError(f'Unsupported number of chains '
                                     f'{len(values)}')

                if numerics:
                    ax.plot(sites,
                            numerics[chain][index_u][index_step],
                            color=color,
                            alpha=0.75)

                ax.scatter(sites,
                           values_step,
                           color=color)

                ax.errorbar(
                    sites,
                    values_step,
                    errors_step,
                    linestyle='',
                    color=color,
                    capsize=5,
                    elinewidth=1,
                    markeredgewidth=1)

                ax.set_xlim(*xlim)
                ax.set_ylim(*spans[chain])
                if split_yaxes:
                    ax.tick_params('y', colors=color)

    return fig, axes


def _plot_aggregated_quantity(
        values: List[List[List[np.ndarray]]],
        errors: List[List[List[np.ndarray]]],
        numerics: Optional[List[List[List[np.ndarray]]]],
        us: Sequence[float],
        dts: Sequence[float],
        steps: Sequence[int],
        colors: Optional[Sequence[str]],
        axes: Optional[Sequence[Sequence[plt.Axes]]] = None,
        split_yaxes: bool = True,
        label_top_func: Optional[
            Callable[[plt.Axes, float, Optional[int], float], None]] = None,
        label_left: Optional[str] = None,
        label_right: Optional[str] = None,
        label_bottom: Optional[str] = None,
        axis_width: Optional[float] = 4.0,
        axis_aspect_ratio: Optional[float] = 4.0 / 3.0,
        space_left: float = 0.7,
        space_right: float = 0.7,
        space_top: float = 0.3,
        space_bottom: float = 0.5,
        space_horizontal: float = 0.15,
        space_vertical: float = 0.15,
        span_extension: float = 0.05
) -> Tuple[Optional[plt.Figure], Sequence[plt.Axes]]:

    if axes is None:
        fig, (axes,) = _create_default_axes(1,
                                            len(us),
                                            axis_width=axis_width,
                                            axis_aspect_ratio=axis_aspect_ratio,
                                            space_left=space_left,
                                            space_right=space_right,
                                            space_top=space_top,
                                            space_bottom=space_bottom,
                                            space_horizontal=space_horizontal,
                                            space_vertical=space_vertical)
    else:
        fig = None

    if len(axes) != len(us):
        raise ValueError('Incompatible number of axes columns')

    spans = _calculate_spans(values, 0, span_extension)
    if numerics:
        numerics_spans = _calculate_spans(numerics, 0, span_extension)
        spans = [_merge_spans(pair) for pair in zip(spans, numerics_spans)]
    if not split_yaxes:
        spans = [_merge_spans(spans)] * len(spans)

    for chain, (values_chain, errors_chain) in enumerate(zip(values, errors)):

        for index_u, (u, dt_u, values_u, errors_u, ax_u) in enumerate(
                zip(us, dts, values_chain, errors_chain, axes)):

            times = np.array(steps) * dt_u
            color = colors[chain] if colors else None

            # Disable labels for inner subplots.
            if index_u:
                ax_u.set_yticklabels([])

            # Determine plotting axes for each chain.
            if chain == 0:
                ax = ax_u
                if index_u == 0 and label_left:
                    ax.set_ylabel(label_left,
                                  color=color if split_yaxes else 'k')
                if label_top_func:
                    label_top_func(ax, u, None, dt_u)
                if label_bottom:
                    ax.set_xlabel(label_bottom)
            elif chain == 1:
                if split_yaxes:
                    ax = ax_u.twinx()
                    if index_u != len(us) - 1:
                        ax.set_yticklabels([])
                    elif label_right:
                        ax.set_ylabel(label_right, color=color)
                else:
                    ax = ax_u
            else:
                raise ValueError(f'Unsupported number of chains '
                                 f'{len(values)}')

            if numerics:
                ax.plot(times,
                        numerics[chain][index_u],
                        color=color,
                        alpha=0.75)

            ax.scatter(times,
                       values_u,
                       color=color)

            ax.errorbar(
                times,
                values_u,
                errors_u,
                linestyle='',
                color=color,
                capsize=5,
                elinewidth=1,
                markeredgewidth=1)

            ax.set_xlim(np.min(times) - dt_u, np.max(times) + dt_u)
            ax.set_ylim(*spans[chain])
            if split_yaxes:
                ax.tick_params('y', colors=color)

    return fig, axes


def _plot_rescaling_lines(bundles: Sequence[InstanceBundle],
                          axes: Sequence[plt.Axes]) -> None:

    def plot_rescaling(ax: plt.Axes,
                       steps: List[int],
                       dt: float,
                       rescaling: Rescaling,
                       color: str) -> None:
        x = np.array([steps[0], steps[-1]])
        y = x * rescaling.slope + rescaling.intercept
        time = x * dt
        ax.plot(time, y, color=color)

    for bundle, ax in zip(bundles, axes):
        plot_rescaling(ax, bundle.steps, bundle.dt, bundle.rescaling, 'purple')
        plot_rescaling(
            ax, bundle.steps, bundle.dt, bundle.intrinsic_rescaling, 'indigo')


def _calculate_spans(values: List[List[List[np.ndarray]]],
                     value_dim: int,
                     span_extension: float) -> List[Tuple[float, float]]:

    def expand_span(span_min: float, span_max: float) -> Tuple[float, float]:
        span = span_max - span_min
        return (span_min - span * span_extension,
                span_max + span * span_extension)

    axes = tuple(range(1, 3 + value_dim))
    spans = zip(np.min(values, axes), np.max(values, axes))
    return [expand_span(span_min, span_max) for span_min, span_max in spans]


def _merge_spans(spans: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    merged_min, merged_max = np.inf, -np.inf
    for span_min, span_max in spans:
        merged_min = min(merged_min, span_min)
        merged_max = max(merged_max, span_max)
    return merged_min, merged_max


def _create_default_axes(rows: int,
                         columns: int,
                         *,
                         axis_width: Optional[float] = 4.0,
                         total_width: Optional[float] = None,
                         axis_aspect_ratio: Optional[float] = 4.0 / 3.0,
                         total_height: Optional[float] = None,
                         space_left: float = 0.5,
                         space_right: float = 0.5,
                         space_top: float = 0.3,
                         space_bottom: float = 0.3,
                         space_horizontal: float = 0.15,
                         space_vertical: float = 0.15
                         ) -> Tuple[plt.Figure, List[List[plt.Axes]]]:

    if total_width is None:
        if axis_width is None:
            raise ValueError('Either axis width or total width must be '
                             'provided')
        total_width = (space_left + space_right +
                       axis_width * columns +
                       space_horizontal * (columns - 1))
    elif axis_width is None:
        axis_width = (total_width - space_left - space_right -
                      space_horizontal * (columns - 1)) / columns
    else:
        raise ValueError('Axis width and total width can not be provided at '
                         'the same time')

    if total_height is None:
        if axis_aspect_ratio is None:
            raise ValueError('Either axis aspect ratio or total height must be '
                             'provided')
        axis_height = axis_width / axis_aspect_ratio
        total_height = (space_top + space_bottom +
                        axis_height * rows + space_vertical * (rows - 1))
    elif axis_aspect_ratio is None:
        axis_height = (total_height - space_top - space_bottom -
                       space_vertical * (rows - 1)) / rows
    else:
        raise ValueError('Axis aspect ratio and total height can not be '
                         'provided at the same time')

    fig: plt.Figure = plt.figure(figsize=(total_width, total_height))
    axes = []

    offset_bottom = space_bottom
    for _ in range(rows):
        row_axes = []
        offset_left = space_left
        for _ in range(columns):
            row_axes.append(plt.axes([offset_left / total_width,
                                      offset_bottom / total_height,
                                      axis_width / total_width,
                                      axis_height/ total_height]))
            offset_left += axis_width + space_horizontal
        axes.append(row_axes)
        offset_bottom += axis_height + space_vertical

    axes = list(reversed(axes))

    return fig, axes
