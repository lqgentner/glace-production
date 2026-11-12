"""Composite pipelines: configuration, AOI resolution, and per-cell processing."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Literal

from freezebase.mgrs import MGRSGrid
from omegaconf import OmegaConf
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from s1bursts import (
    BurstCatalog,
    BurstProcessor,
    CoherenceConfig,
    FileManager,
    ResolutionWeighter,
    RTCConfig,
    SearchQuery,
    SNAPConfig,
    TimeRange,
    download_burst_map,
)
from s1bursts.burst import Burst, BurstCollection, BurstPairCollection
from s1bursts.burstmap import create_s1_geometry_lut
from s1bursts.errors import (
    BurstMapUnavailableError,
    CompositeError,
    InconsistentInputsError,
    ProductSettingsMismatchError,
)

# Runtime imports: pydantic resolves these annotations when building _BaseConfig.
from s1bursts.query import Mode, Polarization  # noqa: TC002
import shapely
from shapely import Polygon
from upath import UPath
import yaml

from glace_production.config import settings
from glace_production.datasets import SGI2023, AlpineConventionPerimeter, PaulEtAl2020
from glace_production.manifest import RunManifest, asset_key, written_paths
from glace_production.staging import stage_dem_for_snap

if TYPE_CHECKING:
    import geopandas as gpd
    from s1bursts.product import CoherenceProduct, RTCProduct

logger = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
ETRS89 = "EPSG:3035"

Scope = Literal["alps", "alps-total", "switzerland"]

# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


_DEFAULT_PRODUCT_PROFILE: dict[str, Any] = {
    "compress": "zstd",
    "zstd_level": 1,
    "predictor": 3,
}
_DEFAULT_COMPOSITE_PROFILES: dict[str, dict[str, Any]] = {
    "default": {"compress": "lerc_zstd", "max_z_error": 1.0e-5},
    "COH_LRW": {"max_z_error": 1.0e-4},
    "RTC_LRW_QA-CQM": {"max_z_error": 1.0e-3},
    "COH_LRW_QA-CQM": {"max_z_error": 1.0e-3},
    "RTC_LRW_QA-NUM": {"max_z_error": 0},
    "COH_LRW_QA-NUM": {"max_z_error": 0},
}

# Product types a composite profile may be keyed by, as spelled by s1bursts'
# ``Product.product_type``. Matched rather than enumerated so new pass
# directions do not need listing, but strict enough to catch the easy typo of
# an underscore where the QA suffix uses a hyphen.
_PRODUCT_TYPE_RE = re.compile(r"^(RTC|COH)_(LRW|COMP_[AD])(_QA-(NUM|CQM))?$")


class CompressionConfig(BaseModel):
    """GeoTIFF codec settings for the two output trees.

    ``products`` is one profile for every per-burst intermediate; they are
    written losslessly, so no per-product distinction is needed.

    ``composites`` is keyed by ``product_type`` with an optional ``"default"``
    entry. Per-product entries are merged *over* the default, so they need only
    state what differs -- a bound without a codec still inherits the codec.
    """

    model_config = ConfigDict(extra="forbid")

    products: dict[str, Any] = Field(
        default_factory=lambda: dict(_DEFAULT_PRODUCT_PROFILE),
    )
    composites: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {k: dict(v) for k, v in _DEFAULT_COMPOSITE_PROFILES.items()},
    )

    @model_validator(mode="after")
    def _check_composite_keys(self) -> CompressionConfig:
        """Reject product types that would silently never match.

        An unrecognised key is always a mistake: the lookup falls back to
        ``"default"``, so the intended setting is dropped without any error.
        """
        unknown = [
            key for key in self.composites if key != "default" and not _PRODUCT_TYPE_RE.match(key)
        ]
        if unknown:
            listed = ", ".join(repr(k) for k in sorted(unknown))
            msg = (
                f"Unknown composite product type(s): {listed}. Expected 'default' or a "
                "product_type such as 'RTC_LRW', 'COH_LRW', 'RTC_LRW_QA-CQM', "
                "'COH_LRW_QA-NUM' (note the hyphen before NUM/CQM)."
            )
            raise ValueError(msg)
        return self


class DirectoryConfig(BaseModel):
    """Dirs relative to settings.data_dir, or absolute paths and ``s3://`` URIs taken as given.

    Supports {scope} and {year} as template placeholders.
    """

    base_dir: str
    slc_dir: str | None = None
    prod_dir: str | None = None
    comp_dir: str | None = None


class DemConfig(BaseModel):
    """The single DEM every SNAP operator of a run uses.

    ``name``, ``nodata`` and ``apply_egm`` describe an external DEM file and are
    ignored when ``source`` names one of SNAP's auto-download DEMs. ``name`` is
    the stable, machine-independent identity written to product metadata.

    ``nodata`` must sit outside the DEM's value range: SNAP ignores the file's own
    nodata tag and discards every pixel equal to the declared value. Its default of
    0 would drop real ellipsoidal heights of 0 m, and NaN cannot be declared (SNAP
    rejects ``nan`` in the graph XML). NaN voids in the file are still treated as
    nodata, since SNAP maps any NaN it resamples onto the declared value.
    """

    source: str = "Copernicus 30m Global DEM"
    name: str | None = None
    nodata: float = -32768.0
    apply_egm: bool = False

    @property
    def is_external(self) -> bool:
        """Whether ``source`` is a file rather than a SNAP built-in DEM name.

        Uses the same rule as ``s1bursts``: a name with a file suffix is a path.
        """
        return bool(Path(self.source).suffix)

    @model_validator(mode="after")
    def _require_external_name(self) -> DemConfig:
        if self.is_external and not self.name:
            msg = "name is required for an external DEM so products carry stable provenance"
            raise ValueError(msg)
        return self


# DEM parameters are configured once per run, not per operator.
_HOISTED_DEM_FIELDS = ("dem", "external_dem_nodata", "external_dem_apply_egm")

# Fields of earlier configs that s1bursts 0.8 dropped, with what replaces them.
_LINEAR_ONLY = "products and composites are written in linear units only; delete the key"
_REMOVED_TOP_LEVEL = {
    "rtc_decibel": _LINEAR_ONLY,
    "cqm_decibel": _LINEAR_ONLY,
    "max_items": "the burst search returns every match; delete the key",
}
_REMOVED_PER_SECTION = {
    "backscatter": {
        "flatten_terrain": "the backscatter block is an RTCConfig, always terrain-flattened",
    },
    "coherence": {"square_pixel": "SNAP's coherence window is set by 'window' only"},
}


WINDOWS_DIR = "windows"
"""Subdirectory of the config directory holding one windows file per scope it was computed over."""

WINDOW_SCRIPT = "scripts/s1-composites/compute-snow-window.py"


def default_config_dir() -> Path:
    """Return the repository's ``config/`` directory, where the committed windows live."""
    return Path(__file__).resolve().parents[2] / "config"


def windows_file(config_dir: Path, window: str) -> Path:
    """Return the committed windows file ``window`` that applies to ``config_dir``.

    ``{config_dir}/windows/{window}.yml`` when it exists, else the repository's
    own. A saved ``config.yaml`` lives beside its archive, not under ``config/``,
    and must still be held to the committed table.
    """
    beside = config_dir / WINDOWS_DIR / f"{window.lower()}.yml"
    if beside.exists():
        return beside
    return default_config_dir() / WINDOWS_DIR / f"{window.lower()}.yml"


def resolve_time_range(
    inline: TimeRange | None,
    *,
    window: str,
    year: int,
    config_dir: Path,
) -> TimeRange:
    """Settle the compositing window from the config and the committed windows file.

    The window is a decision that defines the dataset, so it is a committed
    input: ``compute-snow-window.py`` writes one row per year into the windows
    file of a scope, and a run reads it back from the file its config names. A
    config may also carry the window inline -- a saved run configuration does,
    so it can be rerun as it was.

    Parameters
    ----------
    inline
        The ``time_range`` of the config, if it sets one.
    window
        The windows file to read, ``{config_dir}/windows/{window}.yml``.
    year
        The row of the windows file.
    config_dir
        The directory of the config; see :func:`windows_file`.

    Raises
    ------
    ValueError
        If neither source has a window for ``year``, or both have one and they
        disagree. A disagreement means the table was recomputed after the run
        the inline value came from; which one is right is a decision, not
        something to resolve silently.
    """
    path = windows_file(config_dir, window)
    committed: TimeRange | None = None
    if path.exists():
        rows = yaml.safe_load(path.read_text()) or {}
        row = rows.get(year)
        if row is not None:
            committed = TimeRange(start=row["start"], end=row["end"])

    if inline is not None and committed is not None and inline != committed:
        msg = (
            f"time_range {inline.start}..{inline.end} in the config disagrees with "
            f"{committed.start}..{committed.end} in {path} for {year}. The committed window "
            "changed since this configuration was saved; decide which one stands."
        )
        raise ValueError(msg)
    resolved = inline or committed
    if resolved is None:
        msg = (
            f"No compositing window {window!r} for {year}: {path} has no row for it and the "
            f"config sets no time_range. Run {WINDOW_SCRIPT} with a config of scope "
            f"{window!r} for that year, review the row it writes, and commit it."
        )
        raise ValueError(msg)
    return resolved


class _BaseConfig(BaseModel):
    """Shared fields for all composite experiment configs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scope: Scope
    year: int
    window: Scope
    """The windows file the compositing window is read from, ``config/windows/{window}.yml``.

    Separate from ``scope`` so that runs over different areas can share one window,
    and with it composites of the same names and contents. Defaults to ``scope``."""
    dirs: DirectoryConfig
    dem: DemConfig = Field(default_factory=DemConfig)
    skip_cdse: bool = False
    polarizations: list[Polarization] = ["VV", "VH"]
    mode: Mode = "IW"
    baseline_days: int = 12
    pair_tolerance_hours: float = 6
    """How far an acquisition pair may deviate from ``baseline_days`` and still count.

    A genuine repeat pass lands within seconds of the baseline; a CDSE burst
    timestamp has been seen two hours off its product's sensing time; and two
    satellites a day apart in phase (June 2026) give pairs a full day off. The
    default keeps the hours and drops the day. s1bursts refuses half a day or
    more, since from there a pair's spacing no longer rounds to
    ``baseline_days``."""
    pixel_size_m: int = 10
    glacier_buffer_m: int = 250
    simplify_m: int = 20
    time_range: TimeRange | None = None
    """The compositing window, inclusive calendar days. Resolved by :meth:`from_yaml`; see
    :func:`resolve_time_range`. The same window goes to the search and to the composite."""
    backscatter: RTCConfig = Field(default_factory=RTCConfig)
    coherence: CoherenceConfig = Field(default_factory=CoherenceConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)

    @model_validator(mode="before")
    @classmethod
    def _default_window_to_scope(cls, data: Any) -> Any:
        """Read the window of the scope itself when the config names none."""
        if isinstance(data, dict) and data.get("window") is None and "scope" in data:
            return {**data, "window": data["scope"]}
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, data: Any) -> Any:
        """Name the fields s1bursts 0.8 removed, instead of a bare extra-field error."""
        if not isinstance(data, dict):
            return data
        found = [(key, hint) for key, hint in _REMOVED_TOP_LEVEL.items() if key in data]
        for section, fields in _REMOVED_PER_SECTION.items():
            block = data.get(section)
            if isinstance(block, dict):
                found += [
                    (f"{section}.{key}", hint) for key, hint in fields.items() if key in block
                ]
        if found:
            msg = "; ".join(f"'{key}' was removed in s1bursts 0.8: {hint}" for key, hint in found)
            raise ValueError(msg)
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_per_operator_dem(cls, data: Any) -> Any:
        """Point configs that still set the DEM per operator at the ``dem`` block."""
        if not isinstance(data, dict):
            return data
        for section in ("backscatter", "coherence"):
            block = data.get(section)
            if not isinstance(block, dict):
                continue
            found = [f for f in _HOISTED_DEM_FIELDS if f in block]
            if found:
                msg = (
                    f"{', '.join(found)} may not be set under '{section}'. "
                    "The DEM is configured once per run in the top-level 'dem' block "
                    "(source, nodata, apply_egm)."
                )
                raise ValueError(msg)
        return data

    def expand_template(self, template: str) -> str:
        """Substitute the ``{scope}`` and ``{year}`` placeholders of a template."""
        return template.format(scope=self.scope.lower(), year=self.year)

    def resolve_dem_source(self) -> str | Path | UPath:
        """Return the template-expanded DEM source, as a path if it is a file.

        A relative file path is resolved under ``settings.data_dir``.
        """
        source = self.expand_template(self.dem.source)
        if not self.dem.is_external:
            return source
        return settings.resolve_data_path(source)

    def resolve_dem(self) -> str | Path:
        """Return the DEM as SNAP needs it: a built-in name or a local file path.

        A remote external DEM is staged into the local cache, then half-pixel
        compensated so that SNAP samples it where its georeferencing says.
        """
        source = self.resolve_dem_source()
        if isinstance(source, str):
            return source
        return stage_dem_for_snap(source)

    def to_snap_runtime_config(self) -> SNAPConfig:
        """Return the SNAP runtime config carrying the per-burst codec settings.

        Distinct from :meth:`to_snap_configs`, which returns the per-operator
        *graph* configs. This one governs how GPT's output is rewritten to its
        final product, and is where ``compression.products`` takes effect.
        """
        return SNAPConfig(rasterio_profile=dict(self.compression.products))

    def to_snap_configs(self) -> tuple[RTCConfig, CoherenceConfig]:
        """Return the per-operator configs with the resolved DEM injected."""
        dem = {
            "dem": self.resolve_dem(),
            "dem_name": self.dem.name,
            "external_dem_nodata": self.dem.nodata,
            "external_dem_apply_egm": self.dem.apply_egm,
        }
        return (
            self.backscatter.model_copy(update=dem),
            self.coherence.model_copy(update=dem),
        )

    def to_file_manager(self) -> FileManager:
        """Return the run's directories, template-expanded and resolved.

        A relative directory sits under ``settings.data_dir``; an absolute path
        or ``s3://`` URI is taken as given.

        Returns
        -------
        FileManager
            The file manager over the resolved directories.
        """

        def expand(template: str) -> Path | UPath:
            return settings.resolve_data_path(self.expand_template(template))

        d = self.dirs
        return FileManager(
            base_dir=expand(d.base_dir),
            slc_dir=expand(d.slc_dir) if d.slc_dir else None,
            prod_dir=expand(d.prod_dir) if d.prod_dir else None,
            comp_dir=expand(d.comp_dir) if d.comp_dir else None,
        )

    def save_yaml(self, path: Path | UPath) -> None:
        data = _config_to_dict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


class CompositeConfig(_BaseConfig):
    """Full experiment configuration for a composite run.

    ``method`` selects the compositing strategy: locally resolution weighted
    (``"lrw"``) or unweighted per orbit-pass direction (``"pass"``).
    """

    method: Literal["lrw", "pass"] = "lrw"
    aggregation: Literal["mean", "median"] | float = "median"
    clip_max_resolution: float | None = None
    write_qa: bool = False
    overwrite: bool = False
    """Rewrite composites that already exist.

    Off, a run resumes: a tile whose composite and QA files the manifest records,
    at the sizes a listing shows, is skipped before any search, and the manifest
    carries its rows. Its burst products are not checked, so intermediates
    deleted since are not recreated. On, every tile is recomposed -- which
    is what an archive written by an older ``s1bursts`` needs before it can be
    published, since band statistics and digests only exist for files written
    by the current one.
    """

    @model_validator(mode="after")
    def _check_method_fields(self) -> CompositeConfig:
        if self.method == "pass" and self.clip_max_resolution is not None:
            msg = "clip_max_resolution only applies to method='lrw'"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(
        cls,
        path: Path | UPath,
        overrides: list[str] | None = None,
        *,
        resolve_window: bool = True,
    ) -> CompositeConfig:
        """Load config from a YAML file with optional dot-notation CLI overrides.

        Parameters
        ----------
        path:
            Path to a YAML config file, local or in an object store.
        overrides:
            Optional list of ``key=value`` strings (OmegaConf dot-notation),
            e.g. ``["scope=switzerland", "coherence.window=[12,3]"]``.
        resolve_window:
            Settle ``time_range`` against the committed windows file named by
            ``window`` (see :func:`resolve_time_range`) and refuse to load without one.
            Only the script that computes windows turns this off.
        """
        # Parsed from its text, since OmegaConf.load opens local files only.
        cfg = OmegaConf.create(path.read_text())
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        config = cls.model_validate(OmegaConf.to_container(cfg, resolve=True))
        if not resolve_window:
            return config
        return config.model_copy(
            update={
                "time_range": resolve_time_range(
                    config.time_range,
                    window=config.window,
                    year=config.year,
                    # A config read from an object store has no windows file beside it
                    config_dir=(
                        path.resolve().parent if isinstance(path, Path) else default_config_dir()
                    ),
                ),
            },
        )


# ---------------------------------------------------------------------------
# YAML serialisation helpers
# ---------------------------------------------------------------------------


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert a nested dict/list structure for clean YAML output."""
    if isinstance(obj, BaseModel):
        return _make_serialisable(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    if isinstance(obj, (Path, UPath)):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _config_to_dict(config: _BaseConfig) -> dict[str, Any]:
    """Convert a config to a plain dict suitable for YAML.

    ``dem.source`` is written template-expanded, and the per-operator DEM fields
    and computed fields are dropped, so the dump records the DEM actually used
    and stays loadable by :meth:`CompositeConfig.from_yaml`.
    """
    data = _make_serialisable(config.model_dump())
    data["dem"]["source"] = str(config.resolve_dem_source())
    for section, model in (("backscatter", config.backscatter), ("coherence", config.coherence)):
        for field in (*_HOISTED_DEM_FIELDS, *type(model).model_computed_fields):
            data[section].pop(field, None)
    return data


# ---------------------------------------------------------------------------
# AOI resolution
# ---------------------------------------------------------------------------


def resolve_aoi(
    scope: Scope,
    *,
    simplify_m: int = 20,
    glacier_buffer_m: int = 250,
) -> gpd.GeoSeries:
    """Return a WGS-84 GeoSeries of geometries to cover for the given scope."""
    match scope.lower():
        case "alps":
            s = (
                PaulEtAl2020()
                .get_data(download=True)
                .geometry.to_crs(ETRS89)
                .simplify(simplify_m)
                .buffer(glacier_buffer_m)
                .to_crs(WGS84)
            )

        case "alps-total":
            s = AlpineConventionPerimeter().get_data(download=True).geometry.to_crs(WGS84)
        case "switzerland":
            s = (
                SGI2023()
                .get_data(download=True)
                .geometry.simplify(simplify_m)
                .buffer(glacier_buffer_m)
                .to_crs(WGS84)
            )
        case _:
            msg = f"Unknown scope: {scope}"
            raise ValueError(msg)

    return s


def ensure_burst_map(mode: Mode) -> None:
    """Load the Sentinel-1 burst-ID map, downloading it first if it is not cached.

    A fresh fleet VM has no cached map, and s1bursts reports that with
    :class:`~s1bursts.errors.BurstMapUnavailableError` rather than downloading it.

    Parameters
    ----------
    mode
        Acquisition mode whose map is needed, e.g. ``"IW"``.
    """
    try:
        create_s1_geometry_lut(mode=mode)
    except BurstMapUnavailableError:
        logger.info("Sentinel-1 %s burst-ID map not cached, downloading...", mode)
        download_burst_map(mode=mode)
        create_s1_geometry_lut(mode=mode)


def _index_local_bursts(
    file_manager: FileManager,
    config: _BaseConfig,
    time_range: TimeRange,
) -> dict[Polarization, BurstCollection]:
    """Index the SLC bursts already present in ``slc_dir``, one entry per polarization.

    An SLC filename carries no footprint, so the geometry a CDSE search would
    have returned is taken from the Sentinel-1 burst-ID map, which is downloaded
    if it is not cached yet. The SLC directory is listed once per polarization
    here rather than once per cell, which matters when it lives on S3.

    Parameters
    ----------
    file_manager:
        File manager whose ``slc_dir`` is searched.
    config:
        Run configuration; ``mode`` and ``polarizations`` act as filters.
    time_range:
        The compositing window; a burst acquired on any of its days is kept.
    """
    # Load the map up front, so the download happens once and before any listing.
    ensure_burst_map(config.mode)

    # A filename dates a burst to the day, and the window is days, so this
    # selects the same bursts an online run of this config selects from CDSE.
    index: dict[Polarization, BurstCollection] = {}
    for pol in config.polarizations:
        bursts = [
            # The footprint is materialised here so cells can be selected spatially.
            replace(burst, _geometry=burst.geometry)
            for burst in file_manager.find(Burst, polarization=pol, time_range=time_range)
            if burst.mode == config.mode
        ]
        logger.info(
            "Found %d local SLC bursts (pol=%s) in %s",
            len(bursts),
            pol,
            file_manager.slc_dir,
        )
        index[pol] = BurstCollection.from_bursts(bursts)
    return index


def _select_local_bursts(local_bursts: BurstCollection, polygon: Polygon) -> BurstCollection:
    """Select the indexed local bursts whose footprint intersects ``polygon``."""
    return local_bursts.intersecting(polygon)


# ---------------------------------------------------------------------------
# Per-cell processing (shared between LRW and pass pipelines)
# ---------------------------------------------------------------------------


class IncompleteCellError(RuntimeError):
    """A cell's burst products are not all there, so no composite may be written.

    A composite that exists is taken as complete by every later run — the
    compose call skips it — so one written from a partial set of bursts would
    survive every rerun. Refusing here leaves the tile without a composite, and
    the rerun that produces the missing bursts composes it.
    """


def require_complete(step: str, summary: Any, *, geobox: Any, pol: str) -> None:
    """Raise :class:`IncompleteCellError` if any operation of ``step`` failed.

    Parameters
    ----------
    step
        Which stage the summary describes: ``"download"``, ``"backscatter"`` or
        ``"coherence"``.
    summary
        The :class:`s1bursts.summary.OperationSummary` that stage returned. Its
        ``products`` list holds only what succeeded or already existed, so a
        non-zero ``n_failed`` is the only record that anything is missing.
    geobox, pol
        Named in the error so the log says which cell was left out.
    """
    if summary.n_failed:
        msg = (
            f"{step}: {summary.n_failed} of {summary.n_total} failed for geobox {geobox} "
            f"(pol={pol}); not compositing an incomplete cell. Rerun once the products exist."
        )
        raise IncompleteCellError(msg)


def expected_assets(config: CompositeConfig) -> frozenset[str]:
    """Return the asset keys of one complete tile of this run.

    Empty for ``method="pass"``, so no tile is skipped before its search: a
    pass composite exists only for the passes a tile was seen from.

    Parameters
    ----------
    config
        Supplies ``method``, ``polarizations``, ``baseline_days`` and ``write_qa``.

    Returns
    -------
    frozenset[str]
        Asset keys such as ``rtc_vv`` or ``coh12_vh_qa_cqm``.
    """
    if config.method != "lrw":
        return frozenset()
    qas = (None, "NUM", "CQM") if config.write_qa else (None,)
    return frozenset(
        asset_key(product, pol, qa=qa)
        for product in ("RTC", f"COH{config.baseline_days:02d}")
        for pol in config.polarizations
        for qa in qas
    )


def find_pairs(bursts: BurstCollection, config: _BaseConfig) -> BurstPairCollection:
    """Return the burst pairs at the config's temporal baseline.

    Parameters
    ----------
    bursts
        The acquisitions of one cell and polarization.
    config
        Supplies ``baseline_days`` and ``pair_tolerance_hours``.

    Returns
    -------
    BurstPairCollection
        Pairs whose spacing is within the tolerance of the baseline.
    """
    return bursts.find_pairs(
        baseline=config.baseline_days,
        max_delta=pd.Timedelta(hours=config.pair_tolerance_hours),
    )


def _process_cell(
    *,
    catalog: BurstCatalog,
    local_bursts: BurstCollection | None,
    processor: BurstProcessor,
    geobox: Any,
    pol: Polarization,
    config: _BaseConfig,
    snap_configs: tuple[RTCConfig, CoherenceConfig],
    time_range: TimeRange,
) -> tuple[list[RTCProduct], list[CoherenceProduct]]:
    """Download and process bursts for one (geobox, polarization) combination.

    ``snap_configs`` are the backscatter and coherence configs with the resolved
    DEM injected, as returned by :meth:`_BaseConfig.to_snap_configs`.
    ``local_bursts`` are the indexed local SLCs of this polarization, used when
    the CDSE catalog is skipped; when it is ``None``, the catalog is searched and
    the bursts are downloaded instead.

    Returns (rtc_products, coh_products).
    """
    backscatter, coherence = snap_configs
    polygon = shapely.box(*geobox.geographic_extent.boundingbox)
    if geobox.crs is None:
        msg = "GeoBox has no CRS."
        raise TypeError(msg)

    if local_bursts is None:
        query = SearchQuery(
            time_range=time_range,
            polarization=pol,
            intersects=polygon,
            mode=config.mode,
        )
        bursts = catalog.search(query)
    else:
        bursts = _select_local_bursts(local_bursts, polygon)

    if not bursts:
        logger.warning("No bursts found for geobox %s (pol=%s), skipping.", geobox, pol)
        return [], []

    pairs = find_pairs(bursts, config)

    if local_bursts is None:
        dl_summary = catalog.download(bursts, max_workers=3)
        logger.info(
            "Download: %d/%d processed, %d skipped, %d failed",
            dl_summary.n_processed,
            dl_summary.n_total,
            dl_summary.n_skipped,
            dl_summary.n_failed,
        )
        require_complete("download", dl_summary, geobox=geobox, pol=pol)

    rtc_summary = processor.create_backscatter(bursts, crs=geobox.crs, config=backscatter)
    logger.info(
        "Backscatter: %d/%d processed, %d skipped, %d failed",
        rtc_summary.n_processed,
        rtc_summary.n_total,
        rtc_summary.n_skipped,
        rtc_summary.n_failed,
    )
    require_complete("backscatter", rtc_summary, geobox=geobox, pol=pol)

    coh_summary = processor.create_coherences(pairs, crs=geobox.crs, config=coherence)
    logger.info(
        "Coherence: %d/%d processed, %d skipped, %d failed",
        coh_summary.n_processed,
        coh_summary.n_total,
        coh_summary.n_skipped,
        coh_summary.n_failed,
    )
    require_complete("coherence", coh_summary, geobox=geobox, pol=pol)

    return rtc_summary.products, coh_summary.products


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def _compose_rtc(
    weighter: ResolutionWeighter,
    rtc_prods: list[RTCProduct],
    geobox: Any,
    config: CompositeConfig,
    *,
    time_range: TimeRange,
) -> Any:
    """Composite RTC products with the configured method; returns what was written."""
    if config.method == "lrw":
        return weighter.compose_rtc_lrw(
            rtc_prods,
            geobox,
            aggregation=config.aggregation,
            clip_max_resolution=config.clip_max_resolution,
            write_qa=config.write_qa,
            overwrite=config.overwrite,
            time_range=time_range,
        )
    return weighter.compose_rtc_by_pass(
        rtc_prods,
        geobox,
        aggregation=config.aggregation,
        write_qa=config.write_qa,
        overwrite=config.overwrite,
        time_range=time_range,
    )


def _compose_coherence(
    weighter: ResolutionWeighter,
    coh_prods: list[CoherenceProduct],
    rtc_prods: list[RTCProduct],
    geobox: Any,
    config: CompositeConfig,
    *,
    time_range: TimeRange,
) -> Any:
    """Composite coherence products with the configured method; returns what was written."""
    if config.method == "lrw":
        return weighter.compose_coherence_lrw(
            coh_prods,
            rtc_prods,
            geobox,
            aggregation=config.aggregation,
            clip_max_resolution=config.clip_max_resolution,
            write_qa=config.write_qa,
            overwrite=config.overwrite,
            time_range=time_range,
        )
    return weighter.compose_coherence_by_pass(
        coh_prods,
        geobox,
        aggregation=config.aggregation,
        write_qa=config.write_qa,
        overwrite=config.overwrite,
        time_range=time_range,
    )


def run_composite(config: CompositeConfig) -> None:
    """Execute the composite pipeline (LRW or per-pass, per ``config.method``)."""
    fm = config.to_file_manager()
    # Stage the DEM once up front: a missing or unreachable one should fail here,
    # not hours into the run on the first SNAP call.
    snap_configs = config.to_snap_configs()

    aoi = resolve_aoi(
        config.scope,
        simplify_m=config.simplify_m,
        glacier_buffer_m=config.glacier_buffer_m,
    )
    grid = MGRSGrid(aoi, resolution=config.pixel_size_m)
    logger.info("Processing %d geoboxes (method=%s)", len(grid), config.method)

    if config.time_range is None:
        msg = "config.time_range is unset; load the config through CompositeConfig.from_yaml"
        raise ValueError(msg)
    time_range = config.time_range
    logger.info("Compositing window: %s (windows file %r)", time_range, config.window)

    catalog = BurstCatalog(file_manager=fm, show_progress=False)
    processor = BurstProcessor(fm, config.to_snap_runtime_config(), show_progress=False)
    # checksum=True hashes the encoded bytes before upload, which is the only
    # moment a digest can be had without reading the object back; the manifest
    # publishes it as ``file:checksum``.
    weighter = ResolutionWeighter(
        fm,
        rasterio_profile=dict(config.compression.composites),
        checksum=True,
    )
    manifest = RunManifest.load(fm.comp_dir, weighter.checksums)
    # A tile whose every composite the manifest and a listing agree on needs no
    # search, no burst checks and no checkpoint.
    complete = (
        set()
        if config.overwrite
        else manifest.complete_tiles(expected_assets(config), time_range.start, time_range.end)
    )
    logger.info("%d of %d geoboxes are complete and are skipped", len(complete), len(grid))

    # Index the local SLCs once up front, so an empty directory or a missing
    # burst map fails here rather than inside the per-cell exception handler.
    local_index = _index_local_bursts(fm, config, time_range) if config.skip_cdse else {}

    for i, geobox in enumerate(grid, 1):
        if geobox.mgrs_code in complete:
            logger.info("[%d/%d] %s is complete, skipping", i, len(grid), geobox)
            continue
        logger.info("[%d/%d] Processing geobox %s", i, len(grid), geobox)
        written: list[Any] = []
        for pol in config.polarizations:
            try:
                rtc_prods, coh_prods = _process_cell(
                    catalog=catalog,
                    local_bursts=local_index[pol] if config.skip_cdse else None,
                    processor=processor,
                    geobox=geobox,
                    pol=pol,
                    config=config,
                    snap_configs=snap_configs,
                    time_range=time_range,
                )

                if not rtc_prods:
                    logger.warning(
                        "[%d/%d] No RTC products for geobox %s (pol=%s), skipping RTC.",
                        i,
                        len(grid),
                        geobox,
                        pol,
                    )
                else:
                    written.append(
                        _compose_rtc(weighter, rtc_prods, geobox, config, time_range=time_range),
                    )

                if not coh_prods:
                    logger.warning(
                        "[%d/%d] No coherence products for geobox %s (pol=%s), skipping.",
                        i,
                        len(grid),
                        geobox,
                        pol,
                    )
                else:
                    written.append(
                        _compose_coherence(
                            weighter,
                            coh_prods,
                            rtc_prods,
                            geobox,
                            config,
                            time_range=time_range,
                        ),
                    )
            except (IncompleteCellError, CompositeError) as error:
                # Expected outcomes, not faults: an outage left the cell incomplete,
                # or the cell has nothing valid to compose. The message says which,
                # and a traceback would only bury it.
                logger.error("[%d/%d] %s (pol=%s): %s", i, len(grid), geobox, pol, error)  # noqa: TRY400
                continue
            except (InconsistentInputsError, ProductSettingsMismatchError):
                # Every cell would hit these: the products disagree with each other
                # or with the config, so the run stops rather than log 900 failures.
                raise
            except Exception:
                logger.exception(
                    "[%d/%d] Failed to process geobox %s (pol=%s), skipping.",
                    i,
                    len(grid),
                    geobox,
                    pol,
                )
                continue
        # Checkpointed per tile so a crash loses at most this tile's digests. The
        # skipped files of a resumed run are carried from the previous manifest.
        manifest.checkpoint(written_paths(fm, written))

    # Sizes come from a listing of what is actually there, taken last.
    manifest.reconcile()
    manifest.write()
    config.save_yaml(fm.comp_dir / "config.yaml")
