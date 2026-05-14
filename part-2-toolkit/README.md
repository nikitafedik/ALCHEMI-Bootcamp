# Part 2: OLED Melting Point predictions with ALCHEMI Toolkit

Interactive Jupyter notebook that predicts the melting point of a molecular crystal using the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) Python library and the AIMNet2 neural network potential. Naphthalene serves as the model OLED-material system, walked through the Solid-Liquid Coexistence (SLC) pipeline end-to-end.

## Deployment

This notebook runs from the same JupyterLab container as Part 1 — see the [repo README](../README.md) for the bundle-load and launcher (`run-jupyter.sh` or `run.sbatch`) instructions. Part 2 is **standalone** — no NGC API key, no BGR NIM, no `00_container_setup.ipynb`. Open the notebook (`part-2-toolkit/melting-point-slc.ipynb`) directly in JupyterLab once the container is up.
