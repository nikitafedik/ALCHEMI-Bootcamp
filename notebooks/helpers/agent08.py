# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 08, Paths of least resistance, batched.

The model and the four reactions of the notebook live here. The result views
are in ``agent08_views``; the step contracts and the step check
(``python -m helpers.agent08 check <step>``, run from ``notebooks/``) are in
``agent08_check``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

from .agent08_views import (
    plot_energy_profiles,
    show_neb_log,
    show_paths,
    show_reactions,
    show_saddle_table,
)
from .common import configure_tutorial

__all__ = [
    "REACTIONS",
    "RELAX_FMAX",
    "configure_tutorial",
    "largest_forces",
    "load_model",
    "load_reactions",
    "make_work_dir",
    "plot_energy_profiles",
    "save_default_reactions",
    "show_neb_log",
    "show_paths",
    "show_reactions",
    "show_saddle_table",
    "show_session",
]

# The folder of the step files. ALCHEMI_AGENT08_WORK names another folder.
WORK_DIR = Path(os.environ.get("ALCHEMI_AGENT08_WORK") or Path(__file__).resolve().parents[1] / "agent08_work")
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "agent08"
# The saved default ends, two frames per reaction, relaxed with the notebook's model (AIMNet2 + Coulomb).
REACTIONS_FILE = DATA_DIR / "reactions-v2.extxyz"
NOTEBOOK = "notebooks/alchemi-agent-08-paths-of-least-resistance.ipynb"
# The notebook's model is AIMNet2 with its Coulomb term: Toolkit's
# model.compute_hessian works on it.
MODEL_NAME = "aimnet2-wb97m-d3_0"
COULOMB_CUTOFF = 15.0  # Å
# Largest per-atom force norm, in eV/Å, that the loader accepts on the two ends of a path. It is
# six times below the NEB stop of neb_settings (0.003) and ten times below the saddle tolerance of
# the step check (0.005), so an end reads as a stationary point in every later test.
RELAX_FMAX = 0.0005
SAVE_FMAX = 0.0002  # eV/Å, the saved ends are relaxed to this, so that they pass RELAX_FMAX on any machine
RELAX_MAX_STEPS = 3000

# Lines this module can print or raise to the learner.
MESSAGES = {
    "ends_not_relaxed": (
        "the two ends of the paths did not relax: the largest force on an atom is {force:.4f} eV/Å "
        "after {steps} steps, and the loader asks for {fmax:g} eV/Å"
    ),
    "unknown_reaction": "Unknown reaction {name!r}; expected one of {known}",
}


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return WORK_DIR


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(
        "Give these to your agent when you start the session:\n"
        f"  notebook    {NOTEBOOK}\n"
        "  helpers     from helpers import agent08 as helpers\n"
        "  work in     notebooks/agent08_work/\n"
        "  check with  python -m helpers.agent08 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
    )


# ---------------------------------------------------------------------------
# Model: AIMNet2 with its Coulomb term
# ---------------------------------------------------------------------------


def _coulomb_adapter(cutoff: float):
    """The course Coulomb term, in a form that also gives second derivatives.

    The course adapter in ``model_support`` computes its energy in plain
    PyTorch, so it can be differentiated twice; it only lacks the declaration
    that Toolkit's ``compute_hessian`` asks for. The subclass adds it.
    """

    from .model_support import DirectCoulombAdapter

    class TwiceDifferentiableCoulomb(DirectCoulombAdapter):
        """Finite Coulomb term for isolated molecules, fed by AIMNet2's partial charges."""

        def _validate_derivative_request(self, request) -> None:
            """Accept second derivatives."""

    return TwiceDifferentiableCoulomb(cutoff=cutoff)


def load_model(device):
    """Return the model of the notebook, ready for energies and forces.

    AIMNet2 with its Coulomb term, as composed in "Made to order". One
    autograd group: AIMNet2 hands its partial charges to the Coulomb
    term (cutoff 15 Å), and the forces come from the summed energy. Energies
    in eV, forces in eV/Å. ``model.compute_hessian`` works on this model.
    """

    import torch
    from nvalchemi.models import (
        AIMNet2Wrapper,
        PipelineGroup,
        PipelineModelWrapper,
        PipelineStep,
    )

    device = torch.device(device)
    aimnet = AIMNet2Wrapper.from_checkpoint(MODEL_NAME, device=device).eval()
    coulomb = _coulomb_adapter(COULOMB_CUTOFF).to(device).eval()
    groups = [
        PipelineGroup(
            steps=[PipelineStep(aimnet, wire={"charges": "partial_charges"}), coulomb],
            use_autograd=True,
        )
    ]
    model = PipelineModelWrapper(groups=groups, neighbor_adaptation="always").to(device).eval()
    aimnet.set_config("active_outputs", {"energy", "forces", "charges"})
    coulomb.set_config("active_outputs", {"energy", "forces"})
    model.set_config("active_outputs", {"energy", "forces", "charges"})
    return model


# ---------------------------------------------------------------------------
# Reactions: two ends per path, atoms matched by index
# ---------------------------------------------------------------------------


def _rotate_about(points, origin, axis, degrees):
    """Turn ``points`` by ``degrees`` about the line through ``origin`` along ``axis``."""

    axis = axis / np.linalg.norm(axis)
    t = np.deg2rad(degrees)
    v = points - origin
    return origin + (
        v * np.cos(t) + np.cross(axis, v) * np.sin(t) + np.outer(v @ axis, axis) * (1 - np.cos(t))
    )


def _ammonia():
    """NH3 and its mirror image: the N goes through the H3 plane, H indices kept."""

    from ase.build import molecule

    a = molecule("NH3")
    b = a.copy()
    p = b.positions.copy()
    hs = p[1:]
    normal = np.cross(hs[1] - hs[0], hs[2] - hs[0])
    normal /= np.linalg.norm(normal)
    p[0] = p[0] - 2.0 * ((p[0] - hs.mean(0)) @ normal) * normal
    b.positions = p
    return a, b


def _ethane():
    """Staggered ethane and the next staggered minimum: the turn is split between the methyls."""

    from ase.build import molecule

    a = molecule("C2H6")
    b = a.copy()
    p = b.positions.copy()
    axis = p[1] - p[0]
    p[2:5] = _rotate_about(p[2:5], p[0], axis, -60.0)
    p[5:8] = _rotate_about(p[5:8], p[1], axis, +60.0)
    b.positions = p
    return a, b


def _hcn():
    """HCN and HNC. The C-N axis is turned by 60 degrees between the ends, so the H moves
    around the bond; a straight line between collinear ends would pass through it."""

    a = Atoms("HCN", positions=[[-1.07, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]])
    b = Atoms("HCN", positions=[[2.17, 0.0, 0.0], [0.0, 0.0, 0.0], [1.17, 0.0, 0.0]])
    mid = np.array([0.58, 0.0, 0.0])
    b.positions = _rotate_about(b.positions, mid, np.array([0.0, 0.0, 1.0]), 60.0)
    return a, b


def _malonaldehyde():
    """The chelated enol of malonaldehyde and the same molecule after the proton moved.

    Atoms: O1 C1 C2 C3 O2 H(C1) H(C2) H(C3) H(O). The other end is the same
    shape mirrored through the plane x = 0, with the partner atoms exchanged.
    """

    pos = np.array(
        [
            [-1.28, -1.20, 0.0],  # O1 carbonyl
            [-1.22, 0.03, 0.0],  # C1
            [0.00, 0.76, 0.0],  # C2
            [1.19, 0.11, 0.0],  # C3
            [1.29, -1.21, 0.0],  # O2 hydroxyl
            [-2.17, 0.60, 0.0],  # H on C1
            [0.00, 1.84, 0.0],  # H on C2
            [2.13, 0.66, 0.0],  # H on C3
            [0.37, -1.57, 0.0],  # H on O2
        ]
    )
    a = Atoms("OCCCOHHHH", positions=pos)
    partner = [4, 3, 2, 1, 0, 7, 6, 5, 8]
    b = a.copy()
    b.positions = pos[partner] * np.array([-1.0, 1.0, 1.0])
    return a, b


# name -> builder of the two unrelaxed ends, atoms matched by index
REACTIONS = {
    "ammonia inversion": _ammonia,
    "ethane rotation": _ethane,
    "HCN to HNC": _hcn,
    "malonaldehyde proton transfer": _malonaldehyde,
}
_KEEPS_ITS_TURN = ("HCN to HNC",)  # the product keeps the orientation its builder gave it


def largest_forces(model, atoms_list) -> list[float]:
    """Largest per-atom force norm on every structure, in eV/Å, from a fresh model call."""

    from nvalchemi.data import AtomicData, Batch

    from .model_support import prepare_model_batch

    device = next(model.parameters()).device
    batch = Batch.from_data_list([AtomicData.from_atoms(a, device=device) for a in atoms_list], device=device)
    forces = model(prepare_model_batch(batch, model))["forces"].detach().cpu().numpy().astype(float)
    counts = [len(a) for a in atoms_list]
    return [float(np.linalg.norm(f, axis=1).max()) for f in np.split(forces, np.cumsum(counts)[:-1])]


def _relax(model, atoms_list, fmax: float = RELAX_FMAX, max_steps: int = RELAX_MAX_STEPS):
    """Relax every structure in one batch with L-BFGS; return ASE ``Atoms`` in the same order.

    The result is measured with a fresh model call. A structure that is still
    above ``fmax`` after ``max_steps`` raises: an end that is not a minimum
    would spoil every later test of a path.
    """

    import torch
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import LBFGS, ConvergenceHook, FusedStage

    device = next(model.parameters()).device
    records = []
    for atoms in atoms_list:
        data = AtomicData.from_atoms(atoms, device=device)
        data.add_system_property(
            "energy", torch.zeros((1, 1), dtype=data.positions.dtype, device=device)
        )
        data.add_node_property("forces", torch.zeros_like(data.positions))
        records.append(data)
    batch = Batch.from_data_list(records, device=device)
    engine = LBFGS(
        model=model, n_steps=max_steps, convergence_hook=ConvergenceHook.from_fmax(fmax)
    )
    stage = FusedStage(sub_stages=[(0, engine)], device_type=device.type)
    for hook in model.make_neighbor_hooks():
        stage.register_hook(hook)
    with stage:
        batch = stage.run(batch, n_steps=max_steps + 10)
    counts = batch.num_nodes_per_graph.tolist()
    relaxed = []
    for atoms, positions in zip(atoms_list, torch.split(batch.positions.detach().cpu(), counts)):
        new = atoms.copy()
        new.positions = positions.numpy()
        relaxed.append(new)
    worst = max(largest_forces(model, relaxed))
    if not worst <= fmax:
        raise RuntimeError(MESSAGES["ends_not_relaxed"].format(force=worst, fmax=fmax, steps=max_steps))
    return relaxed


def _lay_over(p, q):
    """Return ``p`` turned and moved onto ``q`` (same atom order, proper rotation)."""

    pc, qc = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(u @ vt))
    return pc @ (u @ np.diag([1.0, 1.0, d]) @ vt) + q.mean(0)


def _bare(atoms) -> Atoms:
    """A copy with the symbols and positions only, without what the file added."""

    return Atoms(atoms.get_chemical_symbols(), positions=atoms.positions.copy())


def _lay_product_over(name: str, reactant, product) -> None:
    """Lay the product of reaction ``name`` over its reactant, unless it keeps the turn of its builder."""

    if name not in _KEEPS_ITS_TURN:
        product.positions = _lay_over(product.positions, reactant.positions)


def _saved_ends(names: list[str]):
    """The saved default ends of ``names`` as ``(reactants, products)``, or None when a frame is missing."""

    from ase.io import read

    if not REACTIONS_FILE.is_file():
        return None
    frames = {(a.info.get("reaction"), a.info.get("end")): a for a in read(REACTIONS_FILE, index=":")}
    if any((name, end) not in frames for name in names for end in ("reactant", "product")):
        return None
    return [_bare(frames[(n, "reactant")]) for n in names], [_bare(frames[(n, "product")]) for n in names]


def _built_ends(names: list[str]):
    """The unrelaxed ends of ``names`` from the builders in ``REACTIONS``."""

    pairs = [REACTIONS[name]() for name in names]
    return [a for a, _ in pairs], [b for _, b in pairs]


def load_reactions(model, names=None):
    """Return ``(reactants, products, reaction_names)``.

    ``reactants[i]`` and ``products[i]`` are the two ends of path ``i`` as ASE
    ``Atoms``: minima of ``model`` with a largest force at or below
    0.0005 eV/Å, with atoms matched by index and the two ends
    already laid over each other, so a straight line between them is a fair
    first path. Do not align them again.

    The ends are read from the saved default file ``data/agent08/reactions-v2.extxyz``
    and measured with ``model``. Where a force is above the limit (another
    model, for example), the ends are relaxed again with
    ``model``; the loader raises when they do not get there.

    ``names`` picks reactions from ``REACTIONS``; the default is all four.
    """

    names = list(names or REACTIONS)
    unknown = [name for name in names if name not in REACTIONS]
    if unknown:
        raise ValueError(MESSAGES["unknown_reaction"].format(name=unknown[0], known=", ".join(REACTIONS)))
    ends = _saved_ends(names) or _built_ends(names)
    reactants, products = ends
    if max(largest_forces(model, reactants + products)) > RELAX_FMAX:
        relaxed = _relax(model, reactants + products)
        reactants, products = relaxed[: len(names)], relaxed[len(names) :]
    for name, a, b in zip(names, reactants, products):
        _lay_product_over(name, a, b)
        a.info["name"] = b.info["name"] = name
    return reactants, products, names


def save_default_reactions(model, path: Path | None = None) -> Path:
    """Build the four reactions with ASE, relax the ends with ``model`` and save them.

    This is how ``data/agent08/reactions-v2.extxyz`` was made: the builders in
    ``REACTIONS`` (ASE only), relaxed to 0.0002 eV/Å, the product laid over
    the reactant. Each frame carries ``reaction`` and ``end`` in its comment
    line. Returns the path written.
    """

    from ase.io import write

    names = list(REACTIONS)
    reactants, products = _built_ends(names)
    relaxed = _relax(model, reactants + products, fmax=SAVE_FMAX)
    frames = []
    for k, name in enumerate(names):
        a, b = relaxed[k], relaxed[len(names) + k]
        _lay_product_over(name, a, b)
        for atoms, end in ((a, "reactant"), (b, "product")):
            atoms.info = {"reaction": name, "end": end}
            frames.append(atoms)
    path = Path(path or REACTIONS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, frames, format="extxyz")
    return path


if __name__ == "__main__":
    from .agent08_check import main

    sys.exit(main(sys.argv))
