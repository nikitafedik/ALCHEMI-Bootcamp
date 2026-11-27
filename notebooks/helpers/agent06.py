# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 06, Hooked in at every step.

The three molecular complexes and the composed model of the setup cell, the
session lines and the step-check entry point. The result views live in
``agent06_views``, the step contracts in ``agent06_check``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .agent06_views import plot_displacement, plot_interaction, plot_separation, show_retired, show_run
from .common import configure_tutorial as _configure_tutorial

__all__ = [
    "configure_tutorial",
    "make_work_dir",
    "plot_displacement",
    "plot_interaction",
    "plot_separation",
    "prepare_complexes",
    "show_retired",
    "show_run",
    "show_session",
]

WORK_DIR = Path(__file__).resolve().parents[1] / "agent06_work"
NOTEBOOK = "notebooks/alchemi-agent-06-hooked-in-at-every-step.ipynb"

# The three default complexes, in the order of ``complexes``: saved, verified structures
# (see data/agent06/README.md). To change a complex, change its file name here.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "agent06"
DEFAULT_COMPLEX_FILES = (
    "complex-1-phenol-n-methylacetamide.xyz",
    "complex-2-propyne-methyl-azide.xyz",
    # the selection trial's recommendation (marked DOUBLE CHECK for the educator); saved alternative:
    # "complex-3-alternative-pyrrole-n-methylpropan-2-imine.xyz"
    "complex-3-phenol-n-methylpropan-2-imine.xyz",
)

# Names of the three complexes, in the order of ``complexes``; filled by prepare_complexes().
COMPLEX_NAMES: list[str] = []
# Interaction energy of each default complex at its starting structure with the composed
# model, in eV, as recorded in the data files; the views read the learner's unit from it.
START_INTERACTION_EV: list[float] = []


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    WORK_DIR.mkdir(exist_ok=True)
    return WORK_DIR


make_work_dir()


def configure_tutorial() -> None:
    """Apply the course plot and display style (light only)."""

    _configure_tutorial()


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(
        "Give these to your agent when you start the session:\n"
        f"  notebook    {NOTEBOOK}\n"
        "  helpers     from helpers import agent06 as helpers\n"
        "  work in     notebooks/agent06_work/\n"
        "  check with  python -m helpers.agent06 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
    )


# --------------------------------------------------------------------------
# Saved frames on a CPU
# --------------------------------------------------------------------------


CPU_THREADS = 4  # three complexes, 66 atoms: more threads slow a CPU run down


def _cpu_settings(device) -> None:
    """Settings for the CPU fallback; on a GPU nothing changes.

    The model's neighbor hook is compiled with ``torch.compile``. On a CPU the
    compilation takes longer than the short runs of this notebook, so compiled
    kernels run eagerly. The systems are small, so PyTorch uses at most
    ``CPU_THREADS`` threads: with one thread per core a run takes several
    times longer.
    """

    import torch

    if device.type == "cpu":
        torch.compiler.set_stance("force_eager")
        torch.set_num_threads(min(CPU_THREADS, torch.get_num_threads()))


def _copy_saved_frames_on_cpu() -> None:
    """Make ``HostMemory`` store a copy of each frame when the run is on a CPU.

    ``HostMemory.write`` moves every saved frame to the CPU. From a GPU that is
    a copy. On a CPU the move returns the live tensors, so every saved frame
    would show the last state of the run. With this wrapper a CPU batch is
    cloned before it is stored; a GPU batch goes through unchanged.
    """

    from nvalchemi.dynamics.sinks import HostMemory

    if getattr(HostMemory.write, "_agent06_copies", False):
        return
    original = HostMemory.write

    def write(self, batch, mask=None):
        if batch.device.type == "cpu":
            batch = batch.clone()
        return original(self, batch, mask)

    write._agent06_copies = True
    write.__doc__ = original.__doc__
    HostMemory.write = write


# --------------------------------------------------------------------------
# The complexes and the model
# --------------------------------------------------------------------------


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _composed_model(device):
    """AIMNet2 with a Coulomb term fed by its charges and D3, as composed in "Made to order"."""

    from nvalchemi.models import (
        AIMNet2Wrapper,
        DFTD3ModelWrapper,
        PipelineGroup,
        PipelineModelWrapper,
        PipelineStep,
    )

    from . import agent04  # the Coulomb term and the D3 settings of notebook 04

    aimnet = AIMNet2Wrapper.from_checkpoint(agent04.model_checkpoint(), device=device).eval()
    coulomb = agent04.DirectCoulombAdapter().to(device).eval()
    d3 = DFTD3ModelWrapper(**agent04.d3_settings(), auto_download=False).to(device).eval()
    model = (
        PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[PipelineStep(aimnet, wire={"charges": "partial_charges"}), coulomb],
                    use_autograd=True,
                ),
                PipelineGroup(steps=[d3], use_autograd=False),
            ],
            neighbor_adaptation="always",
        )
        .to(device)
        .eval()
    )
    aimnet.set_config("active_outputs", {"energy", "forces", "charges"})
    coulomb.set_config("active_outputs", {"energy", "forces"})
    d3.set_config("active_outputs", {"energy", "forces"})
    model.set_config("active_outputs", {"energy", "forces", "charges"})
    return model


def load_default_complexes() -> tuple[list, list[str], list[int]]:
    """Read the three saved default complexes: ``(structures, names, fragment_a_sizes)``.

    ``structures`` are ASE ``Atoms`` at the equilibrium geometry of the NCI Atlas,
    positions in Å. Each pair is neutral, and so is each of its two fragments.
    The first ``fragment_a_sizes[i]`` atoms of structure ``i`` are fragment A.
    """

    from ase.io import read

    structures, names, sizes, recorded = [], [], [], []
    for file in DEFAULT_COMPLEX_FILES:
        atoms = read(DATA_DIR / file, format="extxyz")
        info = atoms.info
        if int(info["charge"]) or int(info["charge_a"]) or int(info["charge_b"]):
            raise ValueError(f"{file}: this notebook uses neutral pairs of neutral fragments")
        split = int(info["fragment_a_size"])
        if not 0 < split < len(atoms):
            raise ValueError(f"{file}: fragment_a_size {split} does not split {len(atoms)} atoms")
        atoms.info = {"charge": 0, "system_name": str(info["name"]), "source": str(info["source"])}
        atoms.pbc = False
        structures.append(atoms)
        names.append(str(info["name"]))
        sizes.append(split)
        recorded.append(float(info["composed_model_ev"]))
    START_INTERACTION_EV[:] = recorded
    return structures, names, sizes


def prepare_complexes():
    """Return ``(complexes, composed_model, fragment_a_sizes)``.

    ``complexes`` is a list of three ``AtomicData``, one molecular complex each,
    at its equilibrium geometry and at rest, on the GPU when there is one. Each
    has an ``energy`` and a ``forces`` field (zeros) and its total ``charge``.
    Positions are in Å. Every complex is neutral, and so is each of its two
    fragments.

    ``composed_model`` is AIMNet2 with its Coulomb and D3 terms, as composed in
    "Made to order". It returns ``energy`` (eV) and ``forces`` (eV/Å).

    ``fragment_a_sizes`` is a list of three ints: the first ``n`` atoms of
    complex ``i`` are fragment A, the rest are fragment B.

    In this notebook a per-complex step index uses ``-1`` for "did not happen":
    ``stop_step`` holds the MD step at which a complex was retired, and ``-1``
    for a complex that ran to the end.
    """

    import torch
    from nvalchemi.data import AtomicData

    device = _device()
    _cpu_settings(device)
    _copy_saved_frames_on_cpu()
    structures, names, fragment_a_sizes = load_default_complexes()
    COMPLEX_NAMES[:] = names
    complexes = []
    for structure in structures:
        data = AtomicData.from_atoms(structure, device=device)
        data.add_system_property(
            "energy", torch.zeros((1, 1), dtype=data.positions.dtype, device=device)
        )
        data.add_node_property("forces", torch.zeros_like(data.positions))
        complexes.append(data)
    return complexes, _composed_model(device), [int(n) for n in fragment_a_sizes]


if __name__ == "__main__":
    from .agent06_check import main

    sys.exit(main(sys.argv[1:]))
