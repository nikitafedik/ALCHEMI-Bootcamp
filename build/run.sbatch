#!/bin/bash
# Slurm launcher for the ALCHEMI Playbook JupyterLab container.
#
# Usage:
#   sbatch build/run.sbatch
#
# After it starts, `cat ~/port_forwarding_command` prints the SSH tunnel for
# the JupyterLab port. The setup notebook (00_container_setup.ipynb) picks
# three more random ports for BGR / Prometheus / Grafana inside the job;
# add extra `-L` flags to the tunnel for any of those you want to open in a
# local browser.

#SBATCH --job-name=alchemi
#SBATCH --nodes=1
#SBATCH --partition=primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=12:00:00

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
# Override BUNDLE_TAR / REPO_ROOT in your environment to match the cluster.
BUNDLE_TAR="${BUNDLE_TAR:-/lustre/fs01/docker-containers/alchemi-bootcamp/alchemi-bundle.tar}"
REPO_ROOT="${REPO_ROOT:-/lustre/fs01/bootcamps/bootcamp_workspaces/${USER}/ALCHEMI-Bootcamp}"

SERVER="$(hostname)"
EXTPORT=9999

# ─── Load the image bundle (idempotent — `docker load` skips present images) ──
echo "Loading image bundle from ${BUNDLE_TAR}"
docker load < "${BUNDLE_TAR}"

# ─── Pick a free JupyterLab port in 9000-9999 ─────────────────────────────────
while true; do
    PORT=$(shuf -n 1 -i 9000-9999)
    if ! netstat -atun | grep -q ":${PORT}\b"; then
        break
    fi
done

# ─── Write the SSH-forwarding hint ────────────────────────────────────────────
cat > ~/port_forwarding_command <<EOF
# JupyterLab (Part 1, Part 2)
ssh -L localhost:${EXTPORT}:${SERVER}:${PORT} ssh.axisapps.io -l \${HASHNUMBER}

# After 00_container_setup.ipynb finishes, add extra -L flags for the random
# BGR / Prometheus / Grafana ports it prints, e.g.:
#   ssh -L localhost:${EXTPORT}:${SERVER}:${PORT} \\
#       -L localhost:11234:${SERVER}:11234 \\
#       -L localhost:11567:${SERVER}:11567 \\
#       -L localhost:11890:${SERVER}:11890 \\
#       ssh.axisapps.io -l \${HASHNUMBER}
EOF
echo "Port-forward command written to ~/port_forwarding_command"
echo "JupyterLab will be at http://localhost:${EXTPORT} once the tunnel is up."

# ─── NIM cache + workspace prep ───────────────────────────────────────────────
mkdir -p "${HOME}/.cache/nim"
mkdir -p "${REPO_ROOT}/part-1-nim" "${REPO_ROOT}/part-2-toolkit"

# ─── Launch JupyterLab ────────────────────────────────────────────────────────
# - /var/run/docker.sock mount lets 00_container_setup.ipynb spawn sibling
#   BGR / Prometheus / Grafana containers on the host. Requires rootless or
#   group-docker access on the compute node.
# - host.docker.internal lets the notebook reach the random host ports those
#   sibling containers publish.
GPU_IDS="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-all}}"

docker run --rm \
    --name "alchemi-jupyter-${SLURM_JOB_ID:-$$}" \
    --gpus "device=${GPU_IDS}" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size 8g \
    -p "${PORT}":8888 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    --add-host=host.docker.internal:host-gateway \
    -e HOST_REPO_ROOT="${REPO_ROOT}" \
    -e HOST_NIM_CACHE="${HOME}/.cache/nim" \
    -v "${HOME}/.cache/nim:/root/.cache/nim" \
    -v "${REPO_ROOT}/part-1-nim:/workspace/part-1-nim" \
    -v "${REPO_ROOT}/part-2-toolkit:/workspace/part-2-toolkit" \
    alchemi-playbook:latest
