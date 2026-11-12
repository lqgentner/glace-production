# syntax=docker/dockerfile:1

FROM ghcr.io/prefix-dev/pixi:0.82.1 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer, cached until pyproject.toml or pixi.lock change. The empty
# README.md and LICENSE only satisfy manifest validation. The cache is removed in
# the same RUN because a cache mount would turn pixi's hardlinks into copies.
COPY pyproject.toml pixi.lock ./
COPY scripts/utils/pixi-activate.sh ./scripts/utils/pixi-activate.sh
RUN touch README.md LICENSE \
    && pixi install --locked -e prod --skip glace-production \
    && rm -rf /root/.cache

# .gitignore is needed by hatchling when selecting wheel files.
COPY .gitignore README.md LICENSE ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# .git is excluded from the context (.dockerignore), so the version is passed in.
# The unscoped variable is the only one hatch-vcs honours; set on this command
# only so it cannot affect other packages built from source.
ARG VERSION=0.0.0
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    pixi install --locked -e prod \
    && rm -rf /root/.cache

RUN pixi shell-hook -e prod -s bash > /shell-hook.sh


FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The environment must stay at the path it was solved at.
COPY --link --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY --link --from=build /shell-hook.sh /shell-hook.sh
# pixi.lock records the exact environment the image was built from.
COPY --link --from=build /app/pyproject.toml /app/pixi.lock /app/README.md /app/LICENSE ./
COPY --link --from=build /app/src ./src
COPY --link --from=build /app/scripts ./scripts
COPY --link --from=build /app/config ./config

COPY --chmod=0755 <<'SH' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -e
. /shell-hook.sh
exec "$@"
SH

ENV PATH="/app/.pixi/envs/prod/bin:${PATH}"
ENV GDAL_CACHEMAX=512MB
ENV AWS_SHARED_CREDENTIALS_FILE=/app/secrets/aws-credentials
ENV EE_SERVICE_ACC_KEY_FILE=/app/secrets/ee-service-account.json

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
