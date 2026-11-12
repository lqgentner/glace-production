"""Holds project related constants and configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from freezebase.settings import S3Settings
from freezebase.utils import get_cache_dir
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from pathlib import Path

    from upath import UPath


class DirSettings(BaseSettings):
    """Directory paths from ``GLACIERS_*`` env vars (stored as raw strings).

    Only ``GLACIERS_DATA_DIR`` is required. The others are optional, so the
    deploy workers, which only produce composites, do not need them; reading
    one that is unset raises.
    """

    model_config = SettingsConfigDict(
        env_prefix="GLACIERS_",
        extra="ignore",
    )

    data_dir: str
    reports_dir: str | None = None
    figures_dir: str | None = None
    models_dir: str | None = None


class EEServiceAccSettings(BaseSettings):
    """Google Earth Engine service account credentials.

    ``EE_SERVICE_ACC_KEY_FILE`` points at the key file, the key itself never
    passes through the environment.
    """

    model_config = SettingsConfigDict(
        env_prefix="EE_SERVICE_ACC_",
        extra="ignore",
    )

    email: str | None = None
    key_file: str | None = None


class Settings:
    """
    Project configuration registry.

    Composes :class:`freezebase.settings.S3Settings` and :class:`DirSettings` and exposes
    directory paths as :class:`~pathlib.Path` (local) or
    :class:`~upath.UPath` (S3) depending on the configured value.
    """

    def __init__(self) -> None:
        # GLACIERS_S3_PROFILE selects a named profile, so no access key passes
        # through the environment; GLACIERS_S3_ENDPOINT_URL is optional when the
        # profile's config section carries the endpoint. The checksum policies
        # default to "when_required", which the UZH Ceph gateway needs.
        self.s3 = S3Settings.from_env("GLACIERS_S3_")
        self.dirs = DirSettings()
        self.ee_service_acc = EEServiceAccSettings()

    def _resolve(self, raw: str) -> Path | UPath:
        return self.s3.resolve_path(raw)

    def _resolve_optional(self, name: str, raw: str | None) -> Path | UPath:
        if raw is None:
            msg = f"GLACIERS_{name.upper()} is not set; this environment has no {name}."
            raise ValueError(msg)
        return self._resolve(raw)

    def resolve_path(self, raw: str) -> Path | UPath:
        """Resolve a local path or ``s3://`` URI using the configured S3 credentials."""
        return self._resolve(raw)

    def resolve_data_path(self, raw: str) -> Path | UPath:
        """Resolve a config path: absolute paths and URIs as given, otherwise under ``data_dir``.

        Configs name their inputs relative to the data directory, so the same file
        serves a local mirror and the object store alike.
        """
        if "://" in raw or raw.startswith("/"):
            return self.resolve_path(raw)
        return self.data_dir / raw

    @property
    def cache_dir(self) -> Path:
        """Cache directory for derived data such as MODIS NDSI series.

        Honours ``FREEZEBASE_CACHE`` from the environment, falling back to the
        platform cache directory.
        """
        return get_cache_dir()

    @property
    def data_dir(self) -> Path | UPath:
        """Base directory for raw, interim, and processed data."""
        return self._resolve(self.dirs.data_dir)

    @property
    def reports_dir(self) -> Path | UPath:
        """Output directory for analysis."""
        return self._resolve_optional("reports_dir", self.dirs.reports_dir)

    @property
    def figures_dir(self) -> Path | UPath:
        """Sub-directory for generated plots."""
        return self._resolve_optional("figures_dir", self.dirs.figures_dir)

    @property
    def models_dir(self) -> Path | UPath:
        """Storage for trained model binaries."""
        return self._resolve_optional("models_dir", self.dirs.models_dir)


settings = Settings()
