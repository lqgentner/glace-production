# GLACE production

The code that produced GLACE Alps (Sentinel-1 coherence and backscatter composites of the
European Alps, 2015–2026, on Source Cooperative as `giuz/glace-alps`), the inputs of its
catalog (`glace-catalog`) and the results of the accompanying paper. It is a snapshot of the
pipeline as it ran, not a development repository: keep changes to what reproduces those
results. `README.md` describes the stages and their pixi tasks.

## Environment

pixi manages the environments; `pyproject.toml` holds the Python dependencies and the
`[tool.pixi.*]` tables, and `pixi.lock` pins every package for
`linux-64`, `osx-arm64` and `win-64` (only Linux is tested).

| Environment | Contents | Use |
| --- | --- | --- |
| `default` | runtime + `dev` + `deploy` | pipeline, tests, linting, Ansible |
| `notebooks` | `default` + plotting + Quarto | rendering the paper notebooks |
| `deploy` | runtime + `deploy` | Ansible and the OpenStack CLI |
| `prod` | runtime only | what the container installs |

ESA SNAP's `gpt` (conda package `esa-snap-s1tbx-gpt`, `sarforge` channel) and Quarto come
from conda; everything else from PyPI. Raster I/O goes through rasterio and the GDAL inside
its wheel: there are no `osgeo` bindings. That GDAL does not see grids installed by
`pyproj sync`, so the environment sets `PROJ_NETWORK=ON` and PROJ fetches the grids it needs
(EGM2008 for the DEM) from cdn.proj.org; the tests switch it off. s1bursts is pinned to `>=0.9,<0.10` and freezebase
to `>=0.11,<0.12`.

Configuration lives in `.env` (template: `.env.example`). Every `pixi run` and `pixi shell`
sources it through `scripts/utils/pixi-activate.sh`; running `.pixi/envs/*/bin/python`
directly bypasses it. `glace_production.config` builds its settings at import time and
needs `GLACIERS_DATA_DIR`, so it only imports under `pixi run` (or in the tests, whose
`conftest.py` sets placeholders). Object stores authenticate with named AWS profiles
(`GLACIERS_S3_PROFILE`, `CDSE_S3_PROFILE`); keys stay in `~/.aws/credentials`.

```bash
pixi run python scripts/s1-composites/process-composites.py config/lrw-alps.yml year=2023
```

## Data paths

Paths in `config/` (DEM, `dirs:`, label and composite directories) are relative to
`settings.data_dir` and resolved by `settings.resolve_data_path`; an absolute path or
`s3://` URI is taken as given. The deploy VMs get their data directory from
`glaciers_data_dir` in `deploy/group_vars/all.yml`, not from the local `.env`.

The paper notebooks take every directory from the configs through `notebooks/data_paths.py`
(`run_dirs(year, config)`, `run_scope()`), the label masks from
`config/validation/locations-ch.yml` and the maps from the validation store. They never name
a directory themselves. `dataset-overview.qmd` reads the published mosaics from Source Coop.

## Compositing

`config/lrw-alps.yml` is GLACE (local resolution-weighted, scope `alps-total`);
`config/pass-alps.yml` the per-pass comparison composites of the Alpine glaciers
(2021–2024, `s1-comp-pass`), which the validation and the LRW-effect figure use.

**The compositing window is a committed input.** `config/windows/alps-total.yml` holds one
row per year: the least snow-covered 90 days from MODIS NDSI over the Alpine Convention
perimeter, written by `scripts/s1-composites/compute-snow-window.py`, reviewed and
committed. A run reads its window from the file its config's `window` names and refuses to
start without one; it never queries Earth Engine.

A run leaves `manifest.parquet` and `config.yaml` beside its composites
(`glace_production/manifest.py`). The manifest has one row per composite (filename, size,
multihash sha256, header fields, band statistics, GeoTIFF tags) and is the contract
`glace-catalog` publishes from; that repository holds the column list in a gate.

**Restarting a run is rerunning the same command.** A tile whose files the manifest records
with digests, at the sizes a listing shows, is skipped. A missing tile is composed; a tile
the previous process died on has its missing layers written. `overwrite=true` recomposes
existing tiles. A composite is only written from a complete cell: if a download,
backscatter or coherence step of a tile failed, the tile is left without a composite and the
rerun composes it.

`scripts/dem/download-glo30.py --config config/dem/glo30-alps.yml` builds the DEM as
`freezebase.vrt` mosaic and warp (`glace_production.dem.ellipsoidal_heights`) and refuses to
replace an existing file without `--overwrite`.

## Labels and validation

`glace_production.labels` rasterizes the Swiss Glacier Inventory 2023 (preliminary release
R2026) into per-tile uint8 masks (1 glacier, 0 background, 255 ignore) under the retreat
rule: in the mask of year *Y*, earlier years' outlines are ignore without a buffer and later
years' outlines ignore with a 250 m buffer; everything outside Switzerland is ignore.

`glace_production.validation` holds the two-step validation of the paper: separability
(AUC, JM) and a spatial cross-validation of random forests. The stages are scripts under
`scripts/validation/`, each driven by a YAML in `config/validation/`, writing under
`settings.models_dir / "validation"`. Every stage stores its resolved `config.yaml` and
refuses to resume when the config changed (`--overwrite` replaces the outputs); downstream
stages identify their inputs by generation ids. The folds CSV is committed and never
recomputed by a run. The validation scope is every glacier pixel plus a 250 m ring outside
the outlines; compared layers share one support, the rows where all of them are finite.

## Checks

```bash
pixi run check   # ruff check, ruff format --check, pyrefly, pytest
```

The tests are offline: they reach neither S3 nor Earth Engine. CI runs the same checks;
`containers.yml` builds `ghcr.io/lqgentner/glace-production`.

## Code style

- NumPy-style docstrings for public functions, classes and methods; no `Examples` or
  `Notes` sections.
- Python 3.13+ annotations; Pyrefly runs under the `strict` preset on `src/`, `tests/` and
  `scripts/`. Keep casts and `# pyrefly: ignore[code]` suppressions rare, each with a reason.
- Every module starts with `from __future__ import annotations`.
