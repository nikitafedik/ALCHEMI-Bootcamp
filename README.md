# ALCHEMI Playbook

This playbook provides researchers hands-on approaches for GPU-accelerated computational chemistry and materials discovery with NVIDIA ALCHEMI. Across two tutorials, participants screen oxide catalysts for the oxygen-evolution reaction and predict the melting point of a molecular crystal — running from a JupyterLab container that spawns the BGR NIM and monitoring stack as sibling containers on per-participant random ports.

## Playbook contents

The content is structured in two parts:

- **[Part 1: OER Catalyst Screening with ALCHEMI NIMs](part-1-nim/)** — screen rutile oxide catalyst surfaces (IrO₂, RuO₂, TiO₂) for oxygen-evolution activity using the ALCHEMI BGR (Batch Geometry Relaxation) NIM and the MACE-MP-0 foundation model.
- **[Part 2: OLED Melting Point predictions with ALCHEMI Toolkit](part-2-toolkit/)** — predict molecular-crystal melting points via the Solid-Liquid Coexistence (SLC) pipeline using the ALCHEMI Toolkit Python library and the AIMNet2 neural network potential, with naphthalene as a model OLED material.

## Tools and frameworks

The tools and frameworks used in this playbook:

- [NVIDIA ALCHEMI BGR NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/alchemi-bgr?version=1.0.0) — batch geometry-relaxation microservice
- [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) — Python library for atomistic dynamics
- [MACE-MP-0](https://github.com/ACEsuit/mace) — materials foundation model used by the BGR NIM
- [AIMNet2](https://github.com/isayevlab/AIMNet2) — neural network potential for molecular dynamics
- [OVITO](https://www.ovito.org/) — atomistic visualization
- [JupyterLab](https://jupyterlab.readthedocs.io/) — interactive notebook environment

## Playbook duration

Approximately **90–120 minutes per part** (~3–4 hours total).

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Background | Python proficiency; basic familiarity with computational chemistry / atomistic simulation. |
| GPU host | NVIDIA x86_64 GPU. Tested on A100, H100, B200, L40S, RTX 6000 Ada. |
| Docker | Latest [Docker Engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and the Docker Compose v2 plugin. |
| NGC API key | Required for Part 1's live BGR NIM service. Generate one at [build.nvidia.com](https://build.nvidia.com). Part 2 needs no NGC entitlement. |
| Internet | Needed during initial container build and image pulls. |

## Deployment

The playbook ships as an offline **image bundle**. The lab admin loads it onto each VM once; participants then launch JupyterLab and use `00_container_setup.ipynb` to spin up their own per-participant NIM and monitoring stack.

### Bundle contents

`build/build-bundle.sh` produces a single `alchemi-bundle.tar` (~40–50 GB) containing four images:

| Image | Purpose |
|-------|---------|
| `alchemi-playbook:latest` | JupyterLab + conda env + tutorials (built from `build/Dockerfile`) |
| `nvcr.io/nim/nvidia/alchemi-bgr:1.0.0` | BGR NIM (MACE-MP-0) |
| `prom/prometheus:latest` | Metrics scraping |
| `grafana/grafana:latest` | Metrics dashboards |

### Building the bundle (one-time, by the lab admin)

```bash
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
./build/build-bundle.sh                    # → alchemi-bundle.tar
```

### Per-VM provisioning

Pick one launcher depending on where the workshop runs:

**Standalone VM** — interactive shell on a single GPU box:

```bash
docker load -i alchemi-bundle.tar          # loads all four images
./build/run-jupyter.sh                     # starts JupyterLab on :8888
```

**Slurm cluster** — submit a job that picks a random jupyter port and writes the SSH-forward hint to `~/port_forwarding_command`:

```bash
sbatch build/run.sbatch                    # adjust BUNDLE_TAR / REPO_ROOT for your cluster
cat ~/port_forwarding_command
```

Both launchers mount the host's docker socket and expose `host.docker.internal` so the setup notebook can spawn sibling BGR/Prom/Grafana containers on random ports.

### Participant workflow

1. Open JupyterLab:
   - **Standalone VM:** `http://<lab-vm-host>:8888/lab`
   - **Slurm:** open the SSH tunnel from `~/port_forwarding_command`, then `http://localhost:9999/lab`
2. Run **[`part-1-nim/00_container_setup.ipynb`](part-1-nim/00_container_setup.ipynb)** end-to-end. It allocates three random ports (BGR, Prometheus, Grafana), writes `part-1-nim/config.py` with the choice, and launches the three sibling containers.
3. Open **[`part-1-nim/alchemi-oer-catalyst-screening.ipynb`](part-1-nim/alchemi-oer-catalyst-screening.ipynb)** — it imports `config.BGR_SERVER` automatically.
4. Part 2 (`part-2-toolkit/melting-point-slc.ipynb`) needs no setup notebook; it runs entirely inside JupyterLab with no NIM dependency.

### FAST_DEMO mode (offline, no live BGR)

Part 1's control-panel cell exposes a `FAST_DEMO = True` toggle. When enabled, the notebook reads pre-cached BGR responses from `part-1-nim/cached_responses/oer-catalyst-screening/` instead of calling the live BGR NIM. Useful for sessions without an NGC entitlement or with limited GPU availability — you can skip `00_container_setup.ipynb` entirely.

## License

Apache 2.0 — see [LICENSE](LICENSE).
