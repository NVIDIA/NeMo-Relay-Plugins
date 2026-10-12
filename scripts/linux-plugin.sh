# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run all plugin-owned stages, including the installed-bundle smoke test, in
# the same Linux environment used to compile the artifact.
set -euo pipefail
repo_dir="$(pwd)"
mkdir -p .cache
cp "$1" .cache/linux-plugin-plan.json
docker run --rm \
  --volume "$repo_dir:$repo_dir" \
  --workdir "$repo_dir" \
  --env GH_TOKEN \
  --env PLUGIN_NAME \
  --env PLUGIN_PLATFORM \
  --env PLUGIN_BUILD_IMAGE \
  --env PLUGIN_RUST \
  --env PLUGIN_RUST_TARGET \
  --env PLUGIN_PYTHON_VERSION \
  "$PLUGIN_BUILD_IMAGE" /bin/bash -euc '
    set -o pipefail
    git config --global --add safe.directory "$PWD"
    # Vendored protoc is a glibc program even when Rust targets musl.
    if command -v apk >/dev/null; then
      apk add --no-cache gcompat
    fi
    IFS=. read -r python_major python_minor _ <<< "$PLUGIN_PYTHON_VERSION"
    python_tag="cp${python_major}${python_minor}"
    export PATH="/opt/python/$python_tag-$python_tag/bin:$PATH"
    python --version
    python -m pip install --target /tmp/relay-tools uv==0.11.32
    export PATH="/tmp/relay-tools/bin:$PATH"
    curl --proto "=https" --tlsv1.2 --silent --show-error --fail https://sh.rustup.rs |
      sh -s -- -y --profile minimal --default-host "$PLUGIN_RUST_TARGET" \
        --default-toolchain "$PLUGIN_RUST"
    export PATH="$HOME/.cargo/bin:$PATH"
    cargo install cargo-about --version 0.9.1 --locked --features cli
    # Keep the container environment separate from the runner environment.
    export UV_PROJECT_ENVIRONMENT=/tmp/relay-plugin-venv
    uv sync --locked --python "$(command -v python)"
    uv run --locked python -m scripts.plugins run "$PLUGIN_NAME" \
      --platform "$PLUGIN_PLATFORM" --plan .cache/linux-plugin-plan.json
  '
