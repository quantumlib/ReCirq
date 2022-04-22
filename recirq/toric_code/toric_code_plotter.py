# Copyright 2022 Google
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
"""Module for plotting toric code rectangles in matplotlib."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.colorbar
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt

from . import toric_code_plaquettes as tcp
from . import toric_code_rectangle as tcr

# Custom diverging colormaps
GRAY = "#bdc1c6"  # X, Z = 0, "Grey 400"
BLUE = "#174ea6"  # Z = +1, "Blue 900"
RED = "#ea4335"  # Z = -1, "Red 500"
PURPLE = "#681da8"  # X = +1, "Purple 900"
ORANGE = "#fbbc04"  # X = -1, "Yellow 500"
RED_GRAY_BLUE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "RedGrayBlue", [(0.0, RED), (0.5, GRAY), (1.0, BLUE)]
)
ORANGE_GRAY_PURPLE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "OrangeGrayPurple", [(0.0, ORANGE), (0.5, GRAY), (1.0, PURPLE)]
)


class ToricCodePlotter:
    """Make plots of toric code rectangles and parity data."""

    def __init__(
        self,
        x_cmap: matplotlib.colors.Colormap = ORANGE_GRAY_PURPLE,
        z_cmap: matplotlib.colors.Colormap = RED_GRAY_BLUE,
    ):
        self.x_cmap = x_cmap
        self.z_cmap = z_cmap

    def plot_expectation_values(
        self,
        data: tcp.ToricCodePlaquettes,
        ax: Optional[plt.Axes] = None,
        patch_kwargs: Optional[Dict[str, Any]] = None,
        set_axis_off: bool = True,
        force_equal_aspect: bool = True,
    ) -> plt.Axes:
        """Plot toric code plaquette expectation values as colored tiles."""
        if patch_kwargs is None:
            patch_kwargs = {"edgecolor": "white", "linewidth": 3.0}

        rows = data.code.rows
        cols = data.code.cols

        # Set up axis
        if ax is None:
            _fig, ax = plt.subplots()
        if set_axis_off:  # Remove axis frame
            ax.set_axis_off()
        if force_equal_aspect:
            ax.set_aspect("equal")
        ax.set_xlim(-1, cols)
        ax.set_ylim(rows, -1)

        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

        # Plot X plaquettes, rows * cols array of diamonds
        for row, col in data.code.x_plaquette_indices():
            ax.add_patch(self.get_patch(data, row, col, x_basis=True, **patch_kwargs))

        # Plot Z plaquettes, (rows + 1) * (cols + 1) array of diamonds going off the edge
        for row, col in data.code.z_plaquette_indices():
            ax.add_patch(self.get_patch(data, row, col, x_basis=False, **patch_kwargs))

        return ax

    def plot_code(
        self,
        code: tcr.ToricCodeRectangle,
        x_value: float = 1.0,
        z_value: float = 1.0,
        **kwargs,
    ) -> plt.Axes:
        """Plot toric code plaquettes with uniform parity values."""
        data = tcp.ToricCodePlaquettes.for_uniform_parity(code, x_value, z_value)
        return self.plot_expectation_values(data, **kwargs)

    def get_patch(
        self, data: tcp.ToricCodePlaquettes, row: int, col: int, x_basis: bool, **kwargs
    ) -> mpatches.Patch:
        """Generate a single patch polygon for a plaquette."""
        if not x_basis:  # z basis, includes special boundary cases
            coordinates = [
                (col - 0.5, row - 1),
                (col, row - 0.5),
                (col - 0.5, row),
                (col - 1, row - 0.5),
            ]
            value = data.z_plaquettes[row, col]

            # Handle special cases for corner and edge
            if data.code.is_corner(row, col):
                return self.get_z_corner_patch(row, col, value, **kwargs)
            if data.code.is_edge(row, col):
                return self.get_z_edge_patch(row, col, value, data.code, **kwargs)

        else:  # x basis
            coordinates = [
                (col, row - 0.5),
                (col + 0.5, row),
                (col, row + 0.5),
                (col - 0.5, row),
            ]
            value = data.x_plaquettes[row, col]

        # At this point it's definitely an interior tile
        return mpatches.Polygon(
            coordinates, closed=True, facecolor=self.color(value, x_basis), **kwargs
        )

    def get_z_corner_patch(
        self, row: int, col: int, value: float, **kwargs
    ) -> mpatches.Patch:
        """Handle special case for a corner z plaquette."""
        coordinates = [
            (col - 0.5, row - 1),
            (col, row - 0.5),
            (col - 0.5, row),
            (col - 1, row - 0.5),
        ]
        if (row, col) == (0, 0):  # top left
            path_data = [
                (mpath.Path.MOVETO, coordinates[1]),
                (mpath.Path.LINETO, coordinates[2]),
                (mpath.Path.CURVE4, (-0.9, -0.4)),
                (mpath.Path.CURVE4, (-0.4, -0.9)),
                (mpath.Path.MOVETO, coordinates[1]),
            ]
        elif row == 0:  # top right
            path_data = [
                (mpath.Path.MOVETO, coordinates[2]),
                (mpath.Path.LINETO, coordinates[3]),
                (mpath.Path.CURVE4, (coordinates[3][0] + 0.4, coordinates[3][1] - 0.4)),
                (mpath.Path.CURVE4, (coordinates[2][0] + 0.4, coordinates[2][1] - 0.4)),
                (mpath.Path.MOVETO, coordinates[2]),
            ]
        elif col == 0:  # bottom left
            path_data = [
                (mpath.Path.MOVETO, coordinates[0]),
                (mpath.Path.LINETO, coordinates[1]),
                (mpath.Path.CURVE4, (coordinates[1][0] - 0.4, coordinates[1][1] + 0.4)),
                (mpath.Path.CURVE4, (coordinates[0][0] - 0.4, coordinates[0][1] + 0.4)),
                (mpath.Path.MOVETO, coordinates[0]),
            ]
        else:  # bottom right
            path_data = [
                (mpath.Path.MOVETO, coordinates[0]),
                (mpath.Path.LINETO, coordinates[3]),
                (mpath.Path.CURVE4, (coordinates[3][0] + 0.4, coordinates[3][1] + 0.4)),
                (mpath.Path.CURVE4, (coordinates[0][0] + 0.4, coordinates[0][1] + 0.4)),
                (mpath.Path.MOVETO, coordinates[0]),
            ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        return mpatches.PathPatch(
            path, facecolor=self.color(value, x_basis=False), **kwargs
        )

    def get_z_edge_patch(
        self, row: int, col: int, value: float, code: tcr.ToricCodeRectangle, **kwargs
    ) -> mpatches.Patch:
        """Handle special case for an edge (non-corner) z plaquette."""
        if not code.is_edge(row, col):
            raise ValueError(f"({row}, {col}) is not an edge for code.")
        coordinates = [
            (col - 0.5, row - 1),
            (col, row - 0.5),
            (col - 0.5, row),
            (col - 1, row - 0.5),
        ]
        if row == 0:  # top
            path_data = [
                (mpath.Path.MOVETO, coordinates[1]),
                (mpath.Path.LINETO, coordinates[2]),
                (mpath.Path.LINETO, coordinates[3]),
                (mpath.Path.CURVE4, (coordinates[3][0] + 0.4, coordinates[3][1] - 0.4)),
                (mpath.Path.CURVE4, (coordinates[1][0] - 0.4, coordinates[1][1] - 0.4)),
                (mpath.Path.MOVETO, coordinates[1]),
            ]
        elif col == 0:  # left
            path_data = [
                (mpath.Path.MOVETO, coordinates[0]),
                (mpath.Path.LINETO, coordinates[1]),
                (mpath.Path.LINETO, coordinates[2]),
                (mpath.Path.CURVE4, (coordinates[2][0] - 0.4, coordinates[2][1] - 0.4)),
                (mpath.Path.CURVE4, (coordinates[0][0] - 0.4, coordinates[0][1] + 0.4)),
                (mpath.Path.MOVETO, coordinates[0]),
            ]
        elif col == code.cols:  # right
            path_data = [
                (mpath.Path.MOVETO, coordinates[0]),
                (mpath.Path.LINETO, coordinates[3]),
                (mpath.Path.LINETO, coordinates[2]),
                (mpath.Path.CURVE4, (coordinates[2][0] + 0.4, coordinates[2][1] - 0.4)),
                (mpath.Path.CURVE4, (coordinates[0][0] + 0.4, coordinates[0][1] + 0.4)),
                (mpath.Path.MOVETO, coordinates[0]),
            ]
        else:  # bottom
            path_data = [
                (mpath.Path.MOVETO, coordinates[3]),
                (mpath.Path.LINETO, coordinates[0]),
                (mpath.Path.LINETO, coordinates[1]),
                (mpath.Path.CURVE4, (coordinates[1][0] - 0.4, coordinates[1][1] + 0.4)),
                (mpath.Path.CURVE4, (coordinates[3][0] + 0.4, coordinates[3][1] + 0.4)),
                (mpath.Path.MOVETO, coordinates[3]),
            ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        return mpatches.PathPatch(
            path, facecolor=self.color(value, x_basis=False), **kwargs
        )

    def cmap(self, x_basis: bool) -> matplotlib.colors.Colormap:
        return self.x_cmap if x_basis else self.z_cmap

    def color(self, parity: float, x_basis: bool) -> Tuple[int, int, int, int]:
        """Evaluate the color of a plaquette with a given parity."""
        parity_mapped_to_unit_interval = (parity + 1) / 2  # Color maps use [0, 1]
        return self.cmap(x_basis)(parity_mapped_to_unit_interval)

    def make_colorbar(
        self,
        x_basis: bool,
        orientation: str = "vertical",
        ax: Optional[plt.Axes] = None,
    ) -> matplotlib.colorbar.ColorbarBase:
        if ax is None:
            figsize: Tuple[float, float]
            if orientation == "vertical":
                figsize = (0.1, 2.0)
            elif orientation == "horizontal":
                figsize = (2.0, 0.1)
            else:
                raise ValueError(
                    f'Invalid orientation={orientation}, expected "vertical" or "horizontal"'
                )
            _fig, ax = plt.subplots(figsize=figsize)

        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        return matplotlib.colorbar.ColorbarBase(
            ax, cmap=self.cmap(x_basis), norm=norm, orientation=orientation
        )
