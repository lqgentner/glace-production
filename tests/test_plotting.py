"""Tests of the plotting helpers."""

from __future__ import annotations

from io import BytesIO
import re
from typing import Literal, NamedTuple

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
import pytest

from glace_production.plotting import add_grid, add_scalebar, set_context, set_style

MM = 1 / 25.4  # inches per mm


@pytest.mark.parametrize(
    ("axis", "x_on", "y_on"), [("x", True, False), ("y", False, True), ("both", True, True)]
)
def test_add_grid_unclips_the_requested_gridlines(
    axis: Literal["x", "y", "both"], x_on: bool, y_on: bool
) -> None:
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    add_grid(ax, axis=axis, linewidth=0.4)
    for lines, on in ((ax.get_xgridlines(), x_on), (ax.get_ygridlines(), y_on)):
        assert all(line.get_visible() == on for line in lines)
        if on:
            assert all(not line.get_clip_on() and line.get_linewidth() == 0.4 for line in lines)
    plt.close(fig)


def test_paper_context_sets_text_to_8_pt_and_titles_to_9_pt() -> None:
    with plt.rc_context():
        set_context("paper")
        for key in ("font.size", "axes.labelsize", "xtick.labelsize", "legend.fontsize"):
            assert plt.rcParams[key] == 8, key
        assert plt.rcParams["axes.titlesize"] == 9
        assert plt.rcParams["figure.titlesize"] == 9
        assert plt.rcParams["axes.linewidth"] == pytest.approx(0.5)


def test_style_saves_a_figure_at_its_size_although_text_overhangs() -> None:
    with plt.rc_context():
        set_style()
        fig = plt.figure(figsize=(180 * MM, 60 * MM))
        # An axes filling the figure: its tick labels lie outside it
        fig.add_axes((0, 0, 1, 1))
        buf = BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches=None)
        plt.close(fig)
    media_box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", buf.getvalue())
    assert media_box is not None
    width, height = (float(v) for v in media_box.groups())
    assert (width, height) == pytest.approx((180 * MM * 72, 60 * MM * 72), abs=0.01)


class ScalebarRows(NamedTuple):
    """Rows of a drawn filled scalebar, in points."""

    height: float  # of the white inside the outline
    above: float  # white above the label's ink
    below: float  # white below the label's ink, down to the bottom line
    overhang: float  # of the end ticks above the white


def filled_scalebar_rows(
    size: float, height: float | None = None, linewidth: float = 0.0
) -> ScalebarRows:
    """Measure a filled scalebar drawn white with a black label and outline on grey.

    The white rows are the box as it is seen: inside the outline, above the bottom line.
    """
    dpi = 600
    fig, ax = plt.subplots(figsize=(3, 3), dpi=dpi, facecolor="0.5")
    ax.set_facecolor("0.5")
    ax.set(xlim=(0, 10_000), ylim=(0, 10_000), xticks=[], yticks=[])
    add_scalebar(
        ax,
        style="filled",
        crs=CRS.from_epsg(32632),
        color="black",
        fill_color="white",
        fill_alpha=1.0,
        linewidth=linewidth,
        prop=FontProperties(size=size),
        height=height,
    )
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].astype(int)
    plt.close(fig)
    white = (rgb > 250).all(axis=2)
    black = (rgb < 128).all(axis=2)
    box_rows = np.flatnonzero(white.sum(axis=1) > 20)
    box_cols = np.flatnonzero(white[box_rows].any(axis=0))
    inside = black[box_rows[0] : box_rows[-1] + 1, box_cols[0] : box_cols[-1] + 1]
    ink_rows = np.flatnonzero(inside.any(axis=1))
    # The left end tick: black just left of the white, followed up from the box's top
    tick, top = black[:, box_cols[0] - 2], box_rows[0]
    while linewidth and tick[top - 1]:
        top -= 1
    px_per_pt = dpi / 72
    return ScalebarRows(
        height=len(box_rows) / px_per_pt,
        above=ink_rows[0] / px_per_pt,
        below=(len(box_rows) - 1 - ink_rows[-1]) / px_per_pt,
        overhang=(box_rows[0] - top) / px_per_pt,
    )


@pytest.mark.parametrize("height", [None, 0.08])
@pytest.mark.parametrize("linewidth", [0.0, 2.0])
def test_filled_scalebar_centres_its_label_ink_in_the_white(
    height: float | None, linewidth: float
) -> None:
    rows = filled_scalebar_rows(8, height, linewidth)
    assert rows.above == pytest.approx(rows.below, abs=0.25)


def test_filled_scalebar_end_ticks_stop_at_the_top_of_the_fill() -> None:
    rows = filled_scalebar_rows(8, linewidth=2.0)
    assert rows.overhang == pytest.approx(0, abs=0.25)


def test_filled_scalebar_box_grows_with_its_label() -> None:
    small = filled_scalebar_rows(8).height
    large = filled_scalebar_rows(16).height
    assert large == pytest.approx(2 * small, rel=0.05)
