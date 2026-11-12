#!/usr/bin/env bash
# Sourced by every `pixi run`/`pixi shell` (see [tool.pixi.activation]).
# The pydantic settings classes in glace_production.config read their keys from
# the process environment only, and .env is gitignored and machine-specific, so
# exporting it here is what makes project code runnable without an extra flag.
if [ -f "$PIXI_PROJECT_ROOT/.env" ]; then
    set -a
    . "$PIXI_PROJECT_ROOT/.env"
    set +a
fi
