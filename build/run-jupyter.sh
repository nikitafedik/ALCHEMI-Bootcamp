#!/usr/bin/env bash
# Launch the ALCHEMI Playbook JupyterLab container on a lab VM.
#
# Prerequisites:
#   1. Docker installed with the NVIDIA Container Toolkit.
#   2. `docker load -i alchemi-bundle.tar` already run so the four images
#      (alchemi-playbook, alchemi-bgr, prom/prometheus, grafana/grafana)
#      are present locally.
#   3. Repo checked out so part-1-nim/ and part-2-toolkit/ can be bind-mounted
#      and edited live during the workshop.
#
# What this container can do:
#   - JupyterLab on host port 8888.
#   - From inside the notebook, `!docker run …` spawns sibling containers on
#     the host (via the mounted /var/run/docker.sock) so participants can
#     start BGR/Prometheus/Grafana on their own random ports from
#     00_container_setup.ipynb.
#   - Reaches host-published ports as `host.docker.internal:<port>`.
#
# Usage:
#   ./build/run-jupyter.sh           # foreground
#   ./build/run-jupyter.sh -d        # detached

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pass through any flags (e.g. -d) to `docker run`.
DOCKER_FLAGS=("$@")

mkdir -p "${HOME}/.cache/nim"

# HOST_REPO_ROOT / HOST_NIM_CACHE are passed in so 00_container_setup.ipynb can
# bind-mount the right host paths into the sibling NIM/Prometheus/Grafana
# containers it spawns via the shared docker socket. From inside this jupyter
# container, only host-side paths work in `-v` flags of those sibling runs.
docker run --rm "${DOCKER_FLAGS[@]}" \
    --name alchemi-jupyter \
    --gpus all \
    --platform linux/amd64 \
    --shm-size 8g \
    -p 8888:8888 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    --add-host=host.docker.internal:host-gateway \
    -e HOST_REPO_ROOT="${REPO_ROOT}" \
    -e HOST_NIM_CACHE="${HOME}/.cache/nim" \
    -v "${HOME}/.cache/nim:/root/.cache/nim" \
    -v "${REPO_ROOT}/part-1-nim:/workspace/part-1-nim" \
    -v "${REPO_ROOT}/part-2-toolkit:/workspace/part-2-toolkit" \
    alchemi-playbook:latest
