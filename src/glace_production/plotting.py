"""Helper functions for plotting."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, cast, get_args, override

from freezebase.utils import depends_on_optional
import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, TextArea, VPacker
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from pyproj import CRS

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase

ContextType = Literal["paper", "notebook", "talk", "poster"]

# Text sizes in points per context, (body, title): steps of a type scale rather than
# multiples of one size. Body is every text but the titles: labels, ticks, legends.
TEXT_SIZES: dict[ContextType, tuple[float, float]] = {
    "paper": (8, 9),
    "notebook": (10, 12),
    "talk": (14, 18),
    "poster": (20, 24),
}

Location = Literal[
    "lower left",
    "lower center",
    "lower right",
    "center left",
    "center right",
    "upper left",
    "upper center",
    "upper right",
]
ScalebarStyle = Literal["thin", "filled"]
ScaleUnit = Literal["auto", "metric", "imperial"]
_UnitSystem = Literal["metric", "imperial"]

_NICE_NUMBERS = [1, 2, 5]
# Luminance threshold above which a colour is considered "light"
_LIGHTNESS_THRESHOLD = 0.5


def set_style() -> None:
    """
    Set the general style of Matplotlib plotting elements.

    Sets stylistic defaults such as background colour, spine visibility, and
    font family.
    Inspired by ``seaborn.set_style()``.
    """
    # Common parameters
    style_dict = {
        # Figure
        "figure.facecolor": "white",
        "figure.titleweight": "bold",
        # Nature maximum figure size: [7.2, 6.7]
        "figure.figsize": [7.2, 5.0],
        # Axes
        "axes.labelcolor": "black",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.axisbelow": True,
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.top": False,
        "ytick.right": False,
        # Grid
        "grid.linestyle": "-",
        "grid.color": "lightgray",
        # Text
        "text.color": "black",
        "font.family": ["sans-serif"],
        "font.sans-serif": [
            "Arial",
            "TeX Gyre Heros",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "DejaVu Sans",
            "sans-serif",
        ],
        # Type 42 (TrueType) is a Nature requirement
        "pdf.fonttype": 42,
        # Plot elements
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "white",
        # Legend
        "legend.frameon": False,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "white",
        # Handles 1.2 font sizes long instead of 2, so patch handles read as swatches
        "legend.handlelength": 1.2,
        # Save Figure
        # At the figure size, so a figure laid out to the journal's width prints at it;
        # pass bbox_inches="tight" to crop to the content. bbox_inches=None in savefig
        # means this setting, not "no cropping".
        "savefig.bbox": "standard",
        # Used with bbox_inches="tight" only
        "savefig.pad_inches": 0.01,
        "savefig.dpi": 450,
        # LaTeX packages
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",
    }
    # Update RC parameters
    plt.rcParams.update(style_dict)


def set_context(context: ContextType = "notebook") -> None:
    """
    Set the context-dependent sizes of Matplotlib plotting elements.

    Text takes the body and title sizes of the context from ``TEXT_SIZES``; lines,
    ticks and markers are scaled by one factor per context.
    Inspired by ``seaborn.set_context()``.

    Parameters
    ----------
    context : {"paper", "notebook", "talk", "poster"}, default "notebook"
        Defines the sizes of plot elements.

    Raises
    ------
    ValueError
        If *context* is not one of the valid options.
    """
    valid_contexts = get_args(ContextType)
    if context not in valid_contexts:
        msg = f"Invalid context '{context}'. Must be one of {list(valid_contexts)}."
        raise ValueError(msg)

    # Set up dictionary of default parameters
    base_context = {
        # Axes
        "axes.linewidth": 0.625,
        # Grid
        "grid.linewidth": 0.625,
        # Plot elements
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "patch.linewidth": 1,
        # Ticks
        "xtick.major.width": 0.625,
        "ytick.major.width": 0.625,
        "xtick.minor.width": 0.625,
        "ytick.minor.width": 0.625,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        # Distance between the ticks and their labels
        "xtick.major.pad": 3.5,
        "ytick.major.pad": 3.5,
        "xtick.minor.pad": 3.4,
        "ytick.minor.pad": 3.4,
    }

    # Scale all the parameters by the same factor depending on the context
    scaling = {"paper": 0.8, "notebook": 1, "talk": 1.5, "poster": 2}[context]
    scaled_context = {k: v * scaling for k, v in base_context.items()}
    body, title = TEXT_SIZES[context]
    text = {
        "font.size": body,
        "axes.labelsize": body,
        "xtick.labelsize": body,
        "ytick.labelsize": body,
        "legend.fontsize": body,
        "axes.titlesize": title,
        "figure.titlesize": title,
    }
    # Update RC parameters
    plt.rcParams.update(scaled_context | text)


def add_grid(ax: Axes, axis: Literal["x", "y", "both"] = "both", **kwargs: Any) -> None:
    """
    Draw gridlines that are not cut at the edges of the axes.

    The axes clip would draw a gridline on its edge, such as one at the upper
    axis limit, at half its width. Call after setting the axis limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the grid on.
    axis : {"x", "y", "both"}, default "both"
        Which gridlines to draw.
    **kwargs : Any
        Line properties passed to ``Axes.grid``, e.g. ``linewidth``.
    """
    ax.grid(visible=True, axis=axis, **kwargs)
    lines = []
    if axis in ("x", "both"):
        lines += ax.get_xgridlines()
    if axis in ("y", "both"):
        lines += ax.get_ygridlines()
    for line in lines:
        line.set_clip_on(False)


def _nice_length(target: float) -> float:
    """Return the largest 'nice' length (1, 2, or 5 x 10^n) not exceeding *target*."""
    if target <= 0:
        msg = "Target length must be positive"
        raise ValueError(msg)
    exponent = math.floor(math.log10(target))
    base = 10**exponent
    for n in reversed(_NICE_NUMBERS):
        candidate = n * base
        if candidate <= target:
            return candidate
    return base


# Metres per display unit
_M_PER_KM = 1000.0
_M_PER_MI = 1609.344
_M_PER_FT = 0.3048

# CRS unit names (from pyproj axis_info) that belong to the imperial family
_IMPERIAL_UNIT_NAMES = frozenset(
    {"foot", "US survey foot", "Clarke's foot", "Gold Coast foot"},
)


def _crs_unit_factor(crs: CRS) -> float:
    """Return the conversion factor from CRS axis units to metres."""
    return crs.axis_info[0].unit_conversion_factor


def _is_imperial_crs(crs: CRS) -> bool:
    """Return ``True`` when the CRS axis unit belongs to the imperial family."""
    return crs.axis_info[0].unit_name in _IMPERIAL_UNIT_NAMES


def _resolve_system(unit: ScaleUnit, crs: CRS) -> _UnitSystem:
    """Return ``"metric"`` or ``"imperial"`` for the effective display system."""
    if unit == "auto":
        return "imperial" if _is_imperial_crs(crs) else "metric"
    return unit


def _pick_unit(length_m: float, system: _UnitSystem) -> tuple[str, float]:
    """Return ``(unit_label, metres_per_unit)`` for *length_m* in *system*.

    Picks the larger unit (km / mi) when the value is large enough to
    produce a whole number or clean fraction, matching typical map conventions:
    metric switches at 1 km, imperial at 1 mi.
    """
    if system == "imperial":
        if length_m >= _M_PER_MI:
            return "mi", _M_PER_MI
        return "ft", _M_PER_FT
    # metric
    if length_m >= _M_PER_KM:
        return "km", _M_PER_KM
    return "m", 1.0


def _format_label(value: float, unit_label: str) -> str:
    """Return ``"<value:g> <unit>"``."""
    return f"{value:g} {unit_label}"


@depends_on_optional("cartopy", "pip install Cartopy")
def _geoaxes_crs(ax: Axes) -> CRS:
    """Return the CRS of a cartopy ``GeoAxes``.

    Raises
    ------
    TypeError
        If *ax* is not a cartopy ``GeoAxes``.
    """
    # Lazy: cartopy is a heavy optional dependency, needed only without an explicit CRS.
    from cartopy.mpl.geoaxes import GeoAxes  # noqa: PLC0415

    if not isinstance(ax, GeoAxes):
        msg = "crs must be provided for non-GeoAxes axes"
        raise TypeError(msg)
    # GeoAxes is `Any` in the environment pyrefly runs against, so the isinstance above
    # narrows nothing.
    # pyrefly: ignore[missing-attribute]
    return CRS.from_user_input(ax.projection.to_wkt())


def _resolve_crs(ax: Axes, crs: CRS | None) -> CRS:
    """Resolve and validate the projected CRS for scalebar calculations.

    Parameters
    ----------
    ax : Axes | GeoAxes
        A matplotlib ``Axes`` or cartopy ``GeoAxes`` to resolve the
        CRS from.
    crs : CRS or None
        An explicit pyproj CRS, or ``None`` to extract from a ``GeoAxes``.

    Returns
    -------
    CRS
        A validated projected CRS.

    Raises
    ------
    TypeError
        If *crs* is ``None`` and *ax* is not a cartopy ``GeoAxes``.
    ValueError
        If the resolved CRS is geographic.
    """
    resolved = crs or _geoaxes_crs(ax)
    if resolved.is_geographic:
        msg = f"Geographic CRS '{resolved.name}' is not supported; supply a projected CRS"
        raise ValueError(msg)

    return resolved


def _perceived_lightness(color: str) -> float:
    """Return ITU-R BT.709 relative luminance of *color* in [0, 1]."""
    r, g, b, *_ = mcolors.to_rgba(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrasting_fill(color: str) -> str:
    """Return ``"black"`` for light colours and ``"white"`` for dark colours."""
    return "black" if _perceived_lightness(color) >= _LIGHTNESS_THRESHOLD else "white"


class ThinScaleBar(AnchoredOffsetbox):
    """
    A scalebar artist for matplotlib ``Axes`` with a metric projected CRS (``"thin"`` style).

    The bar length is recomputed from the current axes limits on every
    ``draw()`` call, so the scalebar remains correct after zoom or pan.
    Draws a U-shape with the label above (or below) the bar.

    Parameters
    ----------
    ax : Axes | GeoAxes
        A matplotlib ``Axes`` or cartopy ``GeoAxes``.
        For plain ``Axes`` an explicit *crs* must be supplied.
    loc : Location or int, default "lower right"
        Anchor position, using matplotlib legend vocabulary
        (e.g. ``"lower right"``, ``"upper center"``) or the equivalent
        integer code (1-10, same as ``AnchoredOffsetbox``).
    max_extent : float, default 0.4
        Maximum fraction of the x-axis extent the bar should cover.
        A 'nice' length in that range is chosen automatically.
    color : str, default "black"
        Colour of the bar segments and label text.
    linewidth : float, default 1.4
        Line width of the scalebar segments.
    outline_color : str or None, default None
        Colour of the outline (path effect) behind bar and text.
        Pass ``None`` to disable the outline.
    outline_linewidth : float, default 2.0
        Stroke width of the outline effect.
    prop : FontProperties or None, optional
        ``FontProperties`` of the label, also used to scale ``pad`` and ``borderpad``.
        If ``None``, the label uses the default font and the padding scales with
        ``legend.fontsize`` from rcParams.
    pad : float, default 0.4
        Padding around the child box, as a fraction of ``prop`` font size.
    borderpad : float, default 0.5
        Padding between the frame and the anchor point, as a fraction of
        ``prop`` font size.
    sep : float, default 2.0
        Separation between the bar and the label, in points.
    frameon : bool, default False
        Whether to draw a ``FancyBboxPatch`` background behind the scalebar.
    crs : CRS or None, default None
        A projected ``pyproj.CRS``. When ``None`` and *ax*
        is a cartopy ``GeoAxes``, the CRS is extracted from ``ax.projection``.
        Required for plain ``Axes``.
    unit : {"auto", "metric", "imperial"}, default "auto"
        Display unit system for the label. ``"auto"`` matches the CRS: metric
        (m / km) for metre-based CRS, imperial (ft / mi) for foot-based CRS.
    **kwargs
        Forwarded to ``AnchoredOffsetbox.__init__``.
    """

    def __init__(
        self,
        ax: Axes,
        *,
        loc: Location | int = "lower right",
        max_extent: float = 0.4,
        color: str = "black",
        linewidth: float = 1.4,
        outline_color: str | None = None,
        outline_linewidth: float = 2.0,
        prop: FontProperties | None = None,
        pad: float = 0.4,
        borderpad: float = 0.5,
        sep: float = 2.0,
        frameon: bool = False,
        crs: CRS | None = None,
        unit: ScaleUnit = "auto",
        **kwargs: Any,
    ) -> None:
        self._ax = ax
        self._max_extent = max_extent
        self._crs = _resolve_crs(ax, crs)
        self._unit_pref = unit

        self._label_top = True

        # path effects
        bar_effects: list[patheffects.AbstractPathEffect] = []
        text_effects: list[patheffects.AbstractPathEffect] = []
        if outline_color is not None:
            bar_effects = [
                patheffects.Stroke(
                    linewidth=linewidth + outline_linewidth,
                    foreground=outline_color,
                ),
                patheffects.Normal(),
            ]
            text_effects = [
                patheffects.Stroke(linewidth=outline_linewidth, foreground=outline_color),
                patheffects.Normal(),
            ]

        # bar geometry in data coordinates
        self._size_bar = AuxTransformBox(ax.transData)
        self._bar_line = Line2D(
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            color=color,
            linewidth=linewidth,
            solid_capstyle="projecting",
        )
        if outline_color is not None:
            self._bar_line.set_path_effects(bar_effects)
        self._size_bar.add_artist(self._bar_line)

        # label
        label_props: dict[str, Any] = {"color": color, "ha": "center"}
        if prop is not None:
            label_props["fontproperties"] = prop
        self._txt_label = TextArea("", textprops=label_props)
        if outline_color is not None:
            # TextArea exposes no public handle on its Text, so path effects
            # have to be set on the private one.
            cast("Any", self._txt_label)._text.set_path_effects(text_effects)  # noqa: SLF001

        # layout: label above bar for upper locations, below otherwise
        children: list[Artist] = (
            [self._txt_label, self._size_bar]
            if self._label_top
            else [self._size_bar, self._txt_label]
        )
        self._box = VPacker(children=children, align="center", pad=0, sep=sep)

        _fp = prop if prop is not None else FontProperties(size=mpl.rcParams["legend.fontsize"])

        super().__init__(
            cast("str", loc),
            pad=pad,
            borderpad=borderpad,
            child=self._box,
            prop=_fp,
            frameon=frameon,
            **kwargs,
        )
        self.patch.set_alpha(0.5)
        self.patch.set_edgecolor("none")

    def _compute_bar_geometry(self) -> tuple[float, float, str]:
        """Return bar length (native CRS units), tick height (native), and label."""
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        x_extent = x1 - x0
        y_extent = y1 - y0
        factor = _crs_unit_factor(self._crs)
        system = _resolve_system(self._unit_pref, self._crs)
        unit_label, m_per_unit = _pick_unit(x_extent * factor * self._max_extent, system)
        target_display = x_extent * factor / m_per_unit * self._max_extent
        bar_display = _nice_length(target_display)
        bar_length = bar_display * m_per_unit / factor
        tick_height = 0.012 * y_extent
        label = _format_label(bar_display, unit_label)
        return bar_length, tick_height, label

    @override
    def draw(self, renderer: RendererBase) -> None:
        """Draw the scalebar, recomputing geometry from current axes limits."""
        if not self.get_visible():
            return
        x0, x1 = self._ax.get_xlim()
        if x1 - x0 <= 0:
            return

        bar_length, tick_height, label = self._compute_bar_geometry()
        self._bar_line.set_xdata([0.0, 0.0, bar_length, bar_length])
        self._bar_line.set_ydata(
            [tick_height, 0.0, 0.0, tick_height],
        )
        self._txt_label.set_text(label)

        super().draw(renderer)


class FilledScaleBar(AnchoredOffsetbox):
    """
    A scalebar artist for matplotlib ``Axes`` with a metric projected CRS (``"filled"`` style).

    The bar length is recomputed from the current axes limits on every
    ``draw()`` call, so the scalebar remains correct after zoom or pan.
    Draws a U-shape outline (left, bottom, right edges) with a semi-transparent
    fill and the label text centred inside the box: the ink of the label, not its
    line box, in the fill above the bottom line, so the gaps above and below it are
    equal.

    The fill colour defaults to black when *color* is light and to white when
    *color* is dark, ensuring maximum contrast. Pass *fill_color* explicitly to
    override this behaviour.

    Parameters
    ----------
    ax : Axes | GeoAxes
        A matplotlib ``Axes`` or cartopy ``GeoAxes``.
        For plain ``Axes`` an explicit *crs* must be supplied.
    loc : Location or int, default "lower right"
        Anchor position, using matplotlib legend vocabulary
        (e.g. ``"lower right"``, ``"upper center"``) or the equivalent
        integer code (1-10, same as ``AnchoredOffsetbox``).
    max_extent : float, default 0.4
        Maximum fraction of the x-axis extent the bar should cover.
        A 'nice' length in that range is chosen automatically.
    color : str, default "black"
        Colour of the bar outline and label text.
    linewidth : float, default 1.4
        Line width of the scalebar U outline.
    fill_color : str or None, default None
        Fill colour of the rectangle behind the label. When ``None`` the fill
        colour is chosen automatically for maximum contrast against *color*:
        black for light colours, white for dark colours.
    fill_alpha : float, default 0.8
        Alpha of the rectangle fill behind the label.
    outline_color : str or None, default None
        Colour of the outline (path effect) behind bar and text.
        Pass ``None`` to disable the outline.
    outline_linewidth : float, default 2.0
        Stroke width of the outline effect.
    prop : FontProperties or None, optional
        ``FontProperties`` of the label, also used to scale ``pad`` and ``borderpad``.
        If ``None``, the label uses the default font and the padding scales with
        ``legend.fontsize`` from rcParams.
    pad : float, default 0.4
        Padding around the child box, as a fraction of ``prop`` font size.
    borderpad : float, default 0.5
        Padding between the frame and the anchor point, as a fraction of
        ``prop`` font size.
    crs : CRS or None, default None
        A projected ``pyproj.CRS``. When ``None`` and *ax*
        is a cartopy ``GeoAxes``, the CRS is extracted from ``ax.projection``.
        Required for plain ``Axes``.
    unit : {"auto", "metric", "imperial"}, default "auto"
        Display unit system for the label. ``"auto"`` matches the CRS: metric
        (m / km) for metre-based CRS, imperial (ft / mi) for foot-based CRS.
    height : float or None, default None
        Height of the box inside the outline, as a fraction of the y-axis extent.
        ``None`` fits the box to the label: its ink height plus *label_pad* above
        and below.
    label_pad : float, default 0.25
        Space above and below the label's ink, as a fraction of the label's font
        size. Only used when *height* is ``None``.
    **kwargs
        Forwarded to ``AnchoredOffsetbox.__init__``.
    """

    def __init__(
        self,
        ax: Axes,
        *,
        loc: Location | int = "lower right",
        max_extent: float = 0.4,
        color: str = "black",
        linewidth: float = 1.4,
        fill_color: str | None = None,
        fill_alpha: float = 0.8,
        outline_color: str | None = None,
        outline_linewidth: float = 2.0,
        prop: FontProperties | None = None,
        pad: float = 0.4,
        borderpad: float = 0.5,
        crs: CRS | None = None,
        unit: ScaleUnit = "auto",
        height: float | None = None,
        label_pad: float = 0.25,
        **kwargs: Any,
    ) -> None:
        self._ax = ax
        self._max_extent = max_extent
        self._crs = _resolve_crs(ax, crs)
        self._unit_pref = unit
        self._height = height
        self._label_pad = label_pad

        resolved_fill_color = fill_color if fill_color is not None else _contrasting_fill(color)

        # path effects
        bar_effects: list[patheffects.AbstractPathEffect] = []
        text_effects: list[patheffects.AbstractPathEffect] = []
        if outline_color is not None:
            bar_effects = [
                patheffects.Stroke(
                    linewidth=linewidth + outline_linewidth,
                    foreground=outline_color,
                ),
                patheffects.Normal(),
            ]
            text_effects = [
                patheffects.Stroke(linewidth=outline_linewidth, foreground=outline_color),
                patheffects.Normal(),
            ]

        # single AuxTransformBox holds all geometry
        self._size_bar = AuxTransformBox(ax.transData)

        # fill rectangle (placeholder size, updated in draw())
        self._fill = Rectangle(
            (0, 0),
            1,
            1,
            facecolor=resolved_fill_color,
            edgecolor="none",
            alpha=fill_alpha,
            zorder=1,
        )

        # U-shaped outline (placeholder data, updated in draw())
        self._bar_line = Line2D(
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            color=color,
            linewidth=linewidth,
            # the ticks end flush with the fill, the bottom corners are square
            solid_capstyle="butt",
            solid_joinstyle="miter",
            zorder=2,
        )
        if outline_color is not None:
            self._bar_line.set_path_effects(bar_effects)

        # label text centred inside box by its ink (placeholder position, updated in draw())
        self._label_text = Text(
            0.5,
            0.5,
            "",
            ha="center",
            va="baseline",
            color=color,
            fontproperties=prop,
            zorder=3,
        )
        if outline_color is not None:
            self._label_text.set_path_effects(text_effects)

        self._size_bar.add_artist(self._fill)
        self._size_bar.add_artist(self._bar_line)
        self._size_bar.add_artist(self._label_text)

        _fp = prop if prop is not None else FontProperties(size=mpl.rcParams["legend.fontsize"])

        super().__init__(
            cast("str", loc),
            pad=pad,
            borderpad=borderpad,
            child=self._size_bar,
            prop=_fp,
            frameon=False,
            **kwargs,
        )
        self.patch.set_alpha(0.5)
        self.patch.set_edgecolor("none")

    def _compute_bar_geometry(self) -> tuple[float, str]:
        """Return bar length (native CRS units) and label."""
        x0, x1 = self._ax.get_xlim()
        x_extent = x1 - x0
        factor = _crs_unit_factor(self._crs)
        system = _resolve_system(self._unit_pref, self._crs)
        unit_label, m_per_unit = _pick_unit(x_extent * factor * self._max_extent, system)
        target_display = x_extent * factor / m_per_unit * self._max_extent
        bar_display = _nice_length(target_display)
        bar_length = bar_display * m_per_unit / factor
        label = _format_label(bar_display, unit_label)
        return bar_length, label

    @override
    def draw(self, renderer: RendererBase) -> None:
        """Draw the scalebar, recomputing geometry from current axes limits."""
        if not self.get_visible():
            return
        x0, x1 = self._ax.get_xlim()
        if x1 - x0 <= 0:
            return

        bar_length, label = self._compute_bar_geometry()
        self._label_text.set_text(label)

        # The label's ink, from its baseline, in points; and points in y data units
        prop = self._label_text.get_fontproperties()
        ink = TextPath((0, 0), label, prop=prop).get_extents()
        y0, y1 = self._ax.get_ylim()
        # points_to_pixels is typed for arrays too; a scalar in gives a scalar out
        px_per_pt = cast("float", renderer.points_to_pixels(1.0))
        data_per_pt = px_per_pt * (y1 - y0) / self._ax.bbox.height
        if self._height is None:
            pad = self._label_pad * prop.get_size_in_points()
            inner = (ink.height + 2 * pad) * data_per_pt
        else:
            inner = self._height * (y1 - y0)
        # The bottom line is centred on y = 0, so its upper half covers the fill: the
        # white seen starts above it
        half_line = self._bar_line.get_linewidth() / 2 * data_per_pt
        top = half_line + inner

        # update fill
        self._fill.set_width(bar_length)
        self._fill.set_height(top)

        # update U outline: left↓ bottom→ right↑
        self._bar_line.set_xdata([0.0, 0.0, bar_length, bar_length])
        self._bar_line.set_ydata([top, 0.0, 0.0, top])

        # baseline placed so that the ink's centre is that of the white seen
        baseline = half_line + inner / 2 - (ink.y0 + ink.y1) / 2 * data_per_pt
        self._label_text.set_position((bar_length / 2, baseline))

        super().draw(renderer)


def add_scalebar(
    ax: Axes,
    style: ScalebarStyle = "thin",
    *,
    loc: Location | int = "lower right",
    max_extent: float = 0.4,
    color: str = "black",
    linewidth: float = 1.4,
    fill_color: str | None = None,
    fill_alpha: float = 0.8,
    outline_color: str | None = None,
    outline_linewidth: float = 2.0,
    prop: FontProperties | None = None,
    pad: float = 0.4,
    borderpad: float = 0.5,
    sep: float = 2.0,
    frameon: bool = False,
    crs: CRS | None = None,
    unit: ScaleUnit = "auto",
    **kwargs: Any,
) -> ThinScaleBar | FilledScaleBar:
    """
    Add a scalebar to a matplotlib ``Axes``.

    Parameters
    ----------
    ax : Axes | GeoAxes
        A matplotlib ``Axes`` or cartopy ``GeoAxes``.
        For plain ``Axes`` an explicit *crs* must be supplied.
    style : {"thin", "filled"}, default "thin"
        Scalebar style: ``"thin"`` draws a U-shape with label above;
        ``"filled"`` draws a U-shape outline with a contrasting fill and label inside.
    loc : Location or int, default "lower right"
        Anchor position (e.g. ``"lower right"`` or integer code 1-10).
    max_extent : float, default 0.4
        Maximum fraction of the x-axis extent the bar should cover.
    color : str, default "black"
        Colour of the bar segments and label text.
    linewidth : float, default 1.4
        Line width of the scalebar segments.
    fill_color : str or None, default None
        Fill colour for the ``"filled"`` style. When ``None`` the colour is
        chosen automatically for maximum contrast against *color*.
        Only used for ``"filled"`` style.
    fill_alpha : float, default 0.8
        Alpha of the rectangle fill. Only used for ``"filled"`` style.
    outline_color : str or None, default None
        Colour of the outline behind bar and text. ``None`` disables it.
    outline_linewidth : float, default 2.0
        Stroke width of the outline effect.
    prop : FontProperties or None, optional
        ``FontProperties`` of the label, also used to scale padding. ``None``
        uses the default font for the label and ``legend.fontsize`` for padding.
    pad : float, default 0.4
        Padding around the child box, as a fraction of ``prop`` font size.
    borderpad : float, default 0.5
        Padding between the frame and the anchor point, as a fraction of
        ``prop`` font size.
    sep : float, default 2.0
        Separation between bar and label, in points. Only used for ``"thin"`` style.
    frameon : bool, default False
        Whether to draw a background box behind the scalebar.
        Only used for ``"thin"`` style.
    crs : CRS or None, default None
        A projected ``pyproj.CRS``. When ``None`` and *ax*
        is a cartopy ``GeoAxes``, the CRS is extracted from ``ax.projection``.
        Required for plain ``Axes``.
    unit : {"auto", "metric", "imperial"}, default "auto"
        Display unit system for the label. ``"auto"`` matches the CRS: metric
        (m / km) for metre-based CRS, imperial (ft / mi) for foot-based CRS.
    **kwargs
        Forwarded to the scalebar class constructor.

    Returns
    -------
    ThinScaleBar or FilledScaleBar
        The artist that was added to the axes.
    """
    sb: ThinScaleBar | FilledScaleBar
    if style == "filled":
        sb = FilledScaleBar(
            ax,
            loc=loc,
            max_extent=max_extent,
            color=color,
            linewidth=linewidth,
            fill_color=fill_color,
            fill_alpha=fill_alpha,
            outline_color=outline_color,
            outline_linewidth=outline_linewidth,
            prop=prop,
            pad=pad,
            borderpad=borderpad,
            crs=crs,
            unit=unit,
            **kwargs,
        )
    else:
        sb = ThinScaleBar(
            ax,
            loc=loc,
            max_extent=max_extent,
            color=color,
            linewidth=linewidth,
            outline_color=outline_color,
            outline_linewidth=outline_linewidth,
            prop=prop,
            pad=pad,
            borderpad=borderpad,
            sep=sep,
            frameon=frameon,
            crs=crs,
            unit=unit,
            **kwargs,
        )
    ax.add_artist(sb)
    return sb
