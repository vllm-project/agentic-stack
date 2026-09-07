# syntax=docker/dockerfile:1.7

ARG RUST_VERSION=1.96.0
ARG DEBIAN_VERSION=bookworm
ARG CARGO_CHEF_VERSION=0.1.77
ARG CARGO_CHEF_IMAGE_DIGEST=sha256:fa7281503a177bd5af6261f4041ca6b36d9f0de8d3090886c33cbd8e65b88ca9
ARG DEBIAN_IMAGE_DIGEST=sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818

FROM lukemathwalker/cargo-chef:${CARGO_CHEF_VERSION}-rust-${RUST_VERSION}-${DEBIAN_VERSION}@${CARGO_CHEF_IMAGE_DIGEST} AS chef

WORKDIR /workspace

FROM chef AS planner

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS rust-build

ARG CARGO_BUILD_JOBS=4
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}

WORKDIR /workspace

COPY --from=planner /workspace/recipe.json recipe.json
RUN cargo chef cook --locked --release --recipe-path recipe.json

RUN rm -rf crates
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo clean \
      -p agentic-server-core \
      -p agentic-server \
      -p agentic-praxis \
      -p agentic-llm-d && \
    cargo build --locked --release -p agentic-server -p agentic-llm-d && \
    install -Dm755 -s target/release/agentic-server /out/agentic-server && \
    install -Dm755 -s target/release/agentic-llm-d /out/agentic-llm-d

FROM debian:${DEBIAN_VERSION}-slim@${DEBIAN_IMAGE_DIGEST} AS runtime

ARG RUNTIME_GID=0
ARG RUNTIME_UID=10001

RUN apt-get update && \
    apt-get install --yes --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/lib/agentic-api && \
    chown "${RUNTIME_UID}:${RUNTIME_GID}" /var/lib/agentic-api && \
    chmod g=u,g+s /var/lib/agentic-api

COPY --from=rust-build /out/agentic-server /usr/local/bin/agentic-server
COPY --from=rust-build /out/agentic-llm-d /usr/local/bin/agentic-llm-d
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ARG OCI_CREATED=""
ARG OCI_BUILD_PIPELINE=local
ARG OCI_BUILD_URL=""
ARG OCI_REVISION=""
ARG OCI_SOURCE="https://github.com/vllm-project/agentic-api"
ARG OCI_VERSION=""

LABEL org.opencontainers.image.created="${OCI_CREATED}" \
      org.opencontainers.image.description="Rust gateway for stateful agentic APIs backed by vLLM" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${OCI_REVISION}" \
      org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.title="agentic-api" \
      org.opencontainers.image.url="${OCI_BUILD_URL}" \
      org.opencontainers.image.version="${OCI_VERSION}" \
      ai.vllm.build.commit="${OCI_REVISION}" \
      ai.vllm.build.pipeline="${OCI_BUILD_PIPELINE}" \
      ai.vllm.build.url="${OCI_BUILD_URL}" \
      ai.vllm.image.tag="${OCI_VERSION}"

WORKDIR /var/lib/agentic-api
USER ${RUNTIME_UID}:${RUNTIME_GID}

ENV GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=9000 \
    AGENTIC_API_HOME=/var/lib/agentic-api

EXPOSE 9000
ENTRYPOINT ["docker-entrypoint.sh"]
