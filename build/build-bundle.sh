#!/usr/bin/env bash
# Build the offline distribution bundle for the ALCHEMI Playbook lab.
#
# Pulls the three runtime images (BGR NIM, Prometheus, Grafana), builds the
# jupyter env image from build/Dockerfile, then saves all four into a single
# tar that lab admins load with `docker load -i alchemi-bundle.tar`.
#
# Requirements:
#   - Docker with the buildx / amd64 builder
#   - `docker login nvcr.io` completed (needed to pull the BGR NIM)
#
# Usage:
#   ./build/build-bundle.sh [output_path]
#
# Default output: ./alchemi-bundle.tar (~40-50 GB)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-${REPO_ROOT}/alchemi-bundle.tar}"

BGR_IMAGE="nvcr.io/nim/nvidia/alchemi-bgr:1.0.0"
PROM_IMAGE="prom/prometheus:latest"
GRAFANA_IMAGE="grafana/grafana:latest"
ENV_IMAGE="alchemi-playbook:latest"

# OVITO ships only x86_64 wheels, so the bundle targets linux/amd64
# regardless of the host architecture you build on.
export DOCKER_DEFAULT_PLATFORM=linux/amd64

echo "==> Pulling runtime images"
docker pull "${BGR_IMAGE}"
docker pull "${PROM_IMAGE}"
docker pull "${GRAFANA_IMAGE}"

echo "==> Building jupyter env image (${ENV_IMAGE})"
docker build \
    -t "${ENV_IMAGE}" \
    -f "${REPO_ROOT}/build/Dockerfile" \
    "${REPO_ROOT}"

echo "==> Saving bundle to ${OUTPUT}"
docker save -o "${OUTPUT}" \
    "${BGR_IMAGE}" \
    "${PROM_IMAGE}" \
    "${GRAFANA_IMAGE}" \
    "${ENV_IMAGE}"

SIZE=$(du -h "${OUTPUT}" | cut -f1)
echo "==> Done. Bundle: ${OUTPUT} (${SIZE})"
echo "    Lab admin loads with: docker load -i $(basename "${OUTPUT}")"
