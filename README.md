<p align="center">
  <br />
  <img src="https://data.source.coop/giuz/glace-alps/_assets/glace-wordmark.svg" alt="GLACE" width="350">
  <br />
</p>

<h1 align="center">GLACE production</h1>

<p align="center">
  Production pipeline of GLACE
  <br />
  Sentinel-1 interferometric coherence and backscatter composites of the European Alps
  <br />
  <br />
  <a href="https://lqgentner.github.io/glace-viewer">🗺️ Web map</a>
  ·
  <a href="https://source.coop/giuz/glace-alps">💾 Dataset on Source Coop</a>
  ·
  <a href="https://browser.portolan-sdi.org/#/external/data.source.coop/giuz/glace-alps/catalog.json">🧭 Portolan/STAC browser</a>
  ·
  <a href="https://github.com/lqgentner/glace-catalog">🏷️ Metadata repository</a>
</p>

## About

This repository contains the code that produced the
[GLACE Alps](https://source.coop/giuz/glace-alps) dataset: annual Sentinel-1 composites
of 12-day interferometric coherence and terrain-corrected backscatter of the European
Alps from 2015 to 2026.

It reproduces three things:

1. **The dataset**: for every year, the least snow-covered 90 days, the Sentinel-1
   bursts acquired in this period, their backscatter and coherence products, and the
   local resolution-weighted composites on the 10 m MGRS tile grid.
2. **The inputs of the catalog**: the run manifest beside each year's composites
   and the glacier fraction of every tile, from which
   [glace-catalog](https://github.com/lqgentner/glace-catalog) builds the STAC
   catalog, the mosaics and the web map.
3. **The results of the paper** *Annual Sentinel-1 coherence and backscatter
   composites of the European Alps for glacier mapping* (Gentner, Piermattei and
   Wulf): the validation against the Swiss Glacier Inventory 2023, with all of its
   numbers, tables and figures.

The lockfile [`pixi.lock`](pixi.lock) pins the exact version of each dependency, so the
whole pipeline can be reproduced. It also runs from the container image
[`ghcr.io/lqgentner/glace-production`](https://github.com/lqgentner/glace-production/pkgs/container/glace-production).
Most of the work happens in two first-party libraries:

- [s1bursts](https://pypi.org/project/s1bursts/) 0.9: download, SAR processing and
  compositing
- [freezebase](https://pypi.org/project/freezebase/) 0.11: geospatial helpers and I/O
  on S3

> [!NOTE]
> This repository is a snapshot and is not developed further.

## Setup

The environments are managed with [pixi](https://pixi.sh). `pixi.lock` pins every
package for Linux, macOS (Apple silicon) and Windows, although GLACE Alps was only
produced and tested on Linux. ESA SNAP's `gpt` comes with every environment, Quarto
with `notebooks`.

```bash
# pipeline, tests and deploy tools
pixi install
# adds Quarto and the plotting libraries
pixi install -e notebooks
# copy, then fill in the environment variables
cp .env.example .env
```

These keys in `.env` are required:

| Key | Required for |
| --- | --- |
| `GLACIERS_DATA_DIR` | every step: a local directory or an `s3://` URI, to which every path in [`config/`](config/) is relative |
| `GLACIERS_S3_PROFILE` | every step, when `GLACIERS_DATA_DIR` is an `s3://` URI, plus `GLACIERS_S3_ENDPOINT_URL` unless the profile sets `endpoint_url` |
| `CDSE_USERNAME`, `CDSE_PASSWORD` | composites: the Sentinel-1 bursts, from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) |
| `CDSE_S3_PROFILE` | composites: the SLCs of bursts the on-demand service fails to extract |
| `EOSDIS_USERNAME`, `EOSDIS_PASSWORD` | glacier fractions, Global Seasonal dataset download, web map inputs |
| `GLACIERS_MODELS_DIR` | validation |
| `GLACIERS_REPORTS_DIR`, `GLACIERS_FIGURES_DIR` | paper |
| `EE_SERVICE_ACC_EMAIL`, `EE_SERVICE_ACC_KEY_FILE` | only a new compositing window, from [Google Earth Engine](https://earthengine.google.com) |

[`.env.example`](.env.example) also lists the optional settings, such as the SNAP
workers and memory and the cache directory. Object stores authenticate with a named
AWS profile from `~/.aws/credentials`, never with keys in `.env`.

## Running

Every production stage is a pixi task, and `pixi task list` shows all of them. The
composite and validation stages resume where they stopped, so rerunning an unfinished
task continues it.

| Step | Task | Writes |
| --- | --- | --- |
| DEM | `pixi run dem` | `dem/glo30-alps_ellipsoid.tif`, Copernicus GLO-30 in ellipsoidal heights |
| Compositing window | `pixi run snow-window 2023` | a row for `config/windows/alps-total.yml` (committed, review before adding) |
| Composites | `pixi run fleet jobs-lrw.yml` or `pixi run composites 2023` | `alps-total/{year}/s1-slc`, `s1-prod`, `s1-comp` with `manifest.parquet` |
| Per-pass composites | `pixi run fleet jobs-pass.yml` or `pixi run composites-pass 2023` | `alps-total/{year}/s1-comp-pass`, 2021–2024 |
| Catalog input | `pixi run glacier-fractions` | the glacier fraction per tile, read by glace-catalog |
| Web map input | `pixi run viewer-inventories` | the inventory overlays of glace-viewer, in `../glace-viewer/data` |
| Label masks | `pixi run labels` | `labels/sgi2023/`, one mask per tile and year |
| Comparison dataset | `pixi run global-seasonal` | the Global Seasonal coherence on the validation tiles (after `pixi run locations`) |
| Validation | `pixi run validation` | the store under `GLACIERS_MODELS_DIR/validation` |
| Paper | `pixi run paper` | tables under `GLACIERS_REPORTS_DIR`, figures under `GLACIERS_FIGURES_DIR` |

The compositing windows of 2015–2026 are committed, so the composites can run
without Earth Engine. A composite run reads its window from
`config/windows/alps-total.yml`.

Producing the composites took a lot of processing and storage: 15 TB of bursts and
35 TB of intermediate products. GLACE Alps was produced on one virtual machine per year
(15 vCPUs, 60 GB of memory) on the University of Zurich's ScienceCloud, through the
Ansible playbooks in [`deploy/`](deploy/) (see [`deploy/README.md`](deploy/README.md)).

`validation` runs the validation chain: sample extraction, separability, the spatial
cross-validation, its metrics and the prediction maps. The spatial folds are committed
([`config/validation/folds-ch.csv`](config/validation/folds-ch.csv)), and
`pixi run folds` recomputes them.

## Layout

| Path | Contents |
| --- | --- |
| `src/glace_production/` | compositing runs, manifest, snow window, labels, datasets, validation |
| `scripts/` | the entry points the tasks run |
| `config/` | run configs, compositing windows, DEM and validation configs |
| `notebooks/` | the paper notebooks (Quarto), with `data_paths.py` reading their paths from the configs |
| `deploy/` | Ansible playbooks and job manifests for the OpenStack fleet |
| `containers/` | the container image the fleet runs |
| `tests/` | offline tests: `pixi run check` runs Ruff, Pyrefly and pytest |

## Citing

If you use this code, cite it as:

> Gentner, L. Q. (2026). *GLACE production pipeline* (Version 1.0) [Computer software].
> https://github.com/lqgentner/glace-production

Its Zenodo DOI follows with the first release. If you use the dataset, cite
version 1.0 of GLACE Alps as:

> Gentner, L. Q., Piermattei, L., & Wulf, H. (2026). *GLACE Alps* (Version 1.0)
> [Data set]. LaRS – Language Repository of Switzerland.
> https://doi.org/10.48656/60er-3j55

To refer to the dataset across versions, use its canonical DOI,
https://doi.org/10.48656/d4r9-fj97. The paper is under review, and its
citation follows here once it is published.

## License

The code is released under the [MIT License](LICENSE). GLACE Alps itself is
published on [Source Cooperative](https://source.coop/giuz/glace-alps) under its
own license.
