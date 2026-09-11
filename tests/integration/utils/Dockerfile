# Build using the documented minimal archive. Only the runner and tests enter the image;
# pass a release version or mount a prebuilt wheel at runtime.
FROM ghcr.io/astral-sh/uv:0.9.8 AS uv
FROM node:22.19.0-bookworm-slim AS node
FROM python:3.12.10-slim-bookworm

ARG DATABRICKS_VERSION=1.9.0
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip git libstdc++6 \
    && curl -fL "https://github.com/databricks/cli/releases/download/v${DATABRICKS_VERSION}/databricks_cli_${DATABRICKS_VERSION}_linux_${TARGETARCH:-$(dpkg --print-architecture)}.zip" -o /tmp/databricks.zip \
    && unzip /tmp/databricks.zip -d /tmp/databricks \
    && install /tmp/databricks/databricks /usr/local/bin/databricks \
    && rm -rf /tmp/databricks* /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && useradd --create-home --uid 10001 integration

WORKDIR /opt/ug-integration
COPY scripts/run_integration.py scripts/run_integration.py
COPY tests/integration tests/integration
RUN mkdir /results && chown integration:integration /results
USER integration
ENV HOME=/home/integration
ENTRYPOINT ["python", "scripts/run_integration.py", "--output", "/results/run"]
