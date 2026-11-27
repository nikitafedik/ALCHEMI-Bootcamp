# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 05, Guest model.

Defaults for the "Try it" pieces: ``load_pair_scans()`` (the molecule pairs of Made to order and
the composed model's energies on them) and ``load_molecules()`` (five small molecules within the
30-atom range of AceFF 2.0). Views: ``check_wrapped_model``, ``plot_interaction_curves``,
``show_relaxed_forces``. Step check: ``python -m helpers.agent05 check <step>``.

Every line a view or a check prints to the learner is a draft and lives in ``_WORDS``.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms

from .common import configure_tutorial as _configure_tutorial

__all__ = [
    "check_wrapped_model",
    "configure_tutorial",
    "load_aspirin",
    "load_malonaldehyde",
    "load_molecules",
    "load_pair_scans",
    "make_work_dir",
    "plot_interaction_curves",
    "plot_timestep_scan",
    "show_proton_hop",
    "show_relaxed_forces",
    "show_session",
]

_NOTEBOOKS = Path(__file__).resolve().parents[1]
WORK_DIR = _NOTEBOOKS / "agent05_work"
_DATA = _NOTEBOOKS / "data" / "agent05"
_COMPOSED_SCANS = _DATA / "pair_scans_composed.npz"  # composed AIMNet2 + Coulomb + D3 energies, eV
_MOLECULES = _DATA / "molecules.extxyz"  # five small molecules, each within 30 atoms
_KCAL_PER_EV = 23.060548  # 1 eV in kcal/mol
_STYLE = Path(__file__).resolve().parents[2] / "shared" / "alchemi-figure.mplstyle"
_GREEN, _GREY, _LIGHT_GREY = "#76B900", "#6F6F6F", "#B8B8B8"

# The word of "Done when" in the relax piece, the board's text: "below the threshold" means strictly
# below (revision of 26 Nov): a value equal to the threshold, or above it by any amount, is not
# below. The column head of the forces view and the line of the check use this word.
_THRESHOLD_WORD = "below"
_TRAINED_ATOMS = 30  # largest molecules in AceFF 2.0's training data (model card)

# All learner-visible lines of the views and the checks, in one place (drafts for the educator).
_WORDS = {
    "session": (
        "Give these to your agent when you start the session:\n"
        "  notebook    notebooks/alchemi-agent-05-guest-model.ipynb\n"
        "  helpers     from helpers import agent05 as helpers\n"
        "  work in     notebooks/agent05_work/\n"
        "  check with  python -m helpers.agent05 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
    ),
    "table_heads": ("", "value", "unit"),
    "table_outputs": "wrapper outputs",
    "table_device": "device",
    "table_energy": "energy",
    "table_forces": "forces",
    "forces_heads": ("molecule", "atoms", "largest force [eV/Å]", "threshold [eV/Å]", f"{_THRESHOLD_WORD} the threshold"),
    "yes": "yes",
    "no": "no",
    "curves_ylabel": "Interaction energy [kcal/mol]",
    "curves_xlabel": "Intermolecular coordinate",
    "curves_names": {"composed": "AIMNet2 + Coulomb + D3", "wrapped": "AceFF 2.0 (wrapped)"},
    "check_ok": "OK",
    "check_problems": "{n} problem(s)",
    "check_usage": "usage: python -m helpers.agent05 check <step>   steps: {steps}",
    "check_unknown": "unknown step {step!r}; steps: {steps}",
    "check_nofile": "{file} not found in {folder}",
    "check_failed": "{file} failed: fix the error above and run the check again",
    "script_name": "{step}: {file} uses {name}, which does not exist in a notebook cell; remove that line",
    "unexpected": "unexpected: {text} (the step still passes)",
    "above_threshold": (
        "every molecule's largest force should be " + _THRESHOLD_WORD + " force_threshold ({limit:g} eV/Å); "
        "not below it: {names}"
    ),
    "note_size": (
        "note: {names} larger than the {limit} atoms of AceFF 2.0's training molecules, so the model is "
        "untested at that size (the step still passes)"
    ),
    "left_out": "relaxed_batch holds {n} molecules and {name} holds {m}, so {k} did not reach relaxed_batch",
}


def configure_tutorial() -> None:
    """Apply the course style (light only) and quiet torchmdnet's checkpoint-format notice.

    TorchMD-Net warns "Old-format checkpoint detected" when it loads the AceFF checkpoint and remaps it
    itself; the filter is narrow, by message and module (fix pass 27 Nov, R4). The course setup imports
    Toolkit's models first, so this filter sits in front of the "once" rule they add.
    """

    _configure_tutorial()
    warnings.filterwarnings(
        "ignore",
        message=r"Old-format checkpoint detected",
        category=UserWarning,
        module=r"torchmdnet\.models\.model",
    )


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    WORK_DIR.mkdir(exist_ok=True)
    return WORK_DIR


make_work_dir()


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(_WORDS["session"])


def load_molecules() -> list[Atoms]:
    """Return five small molecules as ASE ``Atoms``: ethanol, acetate (charge -1), aspirin,
    paracetamol and caffeine. Each has at most 24 atoms and carries its name and total charge
    in ``info``. Built once with RDKit (see ``data/agent05/README.md``)."""

    from ase.io import read

    return [atoms for atoms in read(_MOLECULES, index=":")]


def load_pair_scans():
    """Return ``(scan_batch, scan_labels, composed_energy)`` for the pair scans.

    scan_batch: Batch of 90 structures (AB, A, B for each of 30 scan points, total charges kept).
    scan_labels: 30 dicts, one per scan point, with the structure indices and the reference
    interaction energies in kcal/mol. composed_energy: (90, 1) tensor in eV from the composed
    AIMNet2 + Coulomb + D3 model of the previous notebook.
    """

    import torch
    from nvalchemi.data import AtomicData, Batch

    from .nci import load_nci_atlas

    atoms, rows, refs = load_nci_atlas()
    labels = []
    for r in refs.itertuples(index=False):
        sel = rows[(rows["system_id"] == r.system_id) & (rows["scale"] == r.scale)]
        idx = {f: int(sel[sel["fragment"] == f]["graph_index"].iloc[0]) for f in ("AB", "A", "B")}
        labels.append(
            {
                "system_id": str(r.system_id),
                "system_name": str(r.system_name),
                "interaction_class": str(r.interaction_class),
                "scale": float(r.scale),
                "idx_ab": idx["AB"],
                "idx_a": idx["A"],
                "idx_b": idx["B"],
                "ref_ccsd_t_cbs": float(r.ccsd_t_cbs_kcal_mol),
                "ref_dft_d3": float(r.dft_d3_kcal_mol),
            }
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_batch = Batch.from_data_list([AtomicData.from_atoms(a, device=device) for a in atoms]).to(device)
    composed_energy = torch.tensor(np.load(_COMPOSED_SCANS)["composed_energy"], device=device)
    if len(atoms) != 90 or len(labels) != 30 or tuple(composed_energy.shape) != (90, 1):
        raise RuntimeError("pair-scan data does not match the stored composed energies")
    return scan_batch, labels, composed_energy


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------
def _show_table(rows, heads) -> None:
    import pandas as pd
    from IPython.display import display

    display(pd.DataFrame(rows, columns=list(heads)).style.hide(axis="index"))


def check_wrapped_model(aceff_model) -> None:
    """Show a small table of what the wrapped model returns on the pair scans."""

    scan_batch, _, _ = load_pair_scans()
    out = aceff_model(scan_batch.clone())
    shape = lambda value: str(tuple(value.shape))  # noqa: E731
    rows = [
        (_WORDS["table_outputs"], ", ".join(sorted(aceff_model.model_config.active_outputs)), ""),
        (_WORDS["table_device"], str(next(aceff_model.parameters()).device), ""),
        (_WORDS["table_energy"], shape(out["energy"]), "eV"),
        (_WORDS["table_forces"], shape(out["forces"]), "eV/Å"),
    ]
    _show_table(rows, _WORDS["table_heads"])


def _below(value: float, limit: float) -> bool:
    """True when a largest force is strictly below the threshold, with no allowance (the view and
    the check share this one comparison)."""

    return bool(float(value) < float(limit))


def _force_text(value: float, limit: float) -> str:
    """The value with at least four decimals, and more until the printed number reads on the same
    side of the threshold as the value itself (0.04997 shows as 0.04997, not 0.0500)."""

    value, limit = float(value), float(limit)
    if value == limit:
        return f"{value:.4f}"
    for decimals in range(4, 13):
        text = f"{value:.{decimals}f}"
        if float(text) != limit and (float(text) < limit) == (value < limit):
            return text
    return repr(value)


def _formula(batch) -> list[str]:
    """One Hill formula per graph of a Batch, from its atomic numbers."""

    from ase.data import chemical_symbols

    numbers = batch.atomic_numbers.detach().cpu().long().numpy()
    owner = batch.batch_idx.detach().cpu().long().numpy()
    names = []
    for graph in range(batch.num_graphs):
        counts = Counter(chemical_symbols[z] for z in numbers[owner == graph])
        order = [s for s in ("C", "H") if s in counts] + sorted(s for s in counts if s not in ("C", "H"))
        names.append("".join(s + (str(counts[s]) if counts[s] > 1 else "") for s in order))
    return names


def show_relaxed_forces(relaxed_batch, relaxed_max_force, force_threshold) -> None:
    """One row per molecule: its largest force after the relaxation and whether it meets the
    threshold. ``relaxed_max_force`` holds one value per molecule, in eV/Å."""

    import torch

    force = torch.as_tensor(relaxed_max_force).detach().cpu().double().flatten()
    if len(force) != relaxed_batch.num_graphs:
        raise ValueError(
            f"relaxed_max_force should hold one value per molecule ({relaxed_batch.num_graphs}); "
            f"found {len(force)}. Run the step check."
        )
    threshold = float(force_threshold)
    sizes = relaxed_batch.num_nodes_per_graph.detach().cpu().long().tolist()
    rows = [
        (name, size, _force_text(value, threshold), f"{threshold:g}", _WORDS["yes"] if _below(value, threshold) else _WORDS["no"])
        for name, size, value in zip(_formula(relaxed_batch), sizes, force.tolist())
    ]
    _show_table(rows, _WORDS["forces_heads"])


def _show(figure, alt: str) -> None:
    """Show a figure drawn in the course figure style as one 920 px picture."""

    from base64 import b64encode
    from html import escape
    from io import BytesIO

    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)
    payload = b64encode(buffer.getvalue()).decode("ascii")
    display(
        HTML(
            f'<img src="data:image/png;base64,{payload}" alt="{escape(alt, quote=True)}" '
            'style="display:block;box-sizing:border-box;width:100%;max-width:920px;height:auto;margin:0">'
        )
    )


def _interaction(energy, scan_labels) -> np.ndarray:
    """E(AB) - E(A) - E(B) in kcal/mol for each scan point, from a (90, 1) energy tensor in eV."""

    import torch

    e = torch.as_tensor(energy).detach().cpu().double().flatten().numpy() * _KCAL_PER_EV
    ab, a, b = (np.array([lab[k] for lab in scan_labels]) for k in ("idx_ab", "idx_a", "idx_b"))
    return e[ab] - e[a] - e[b]


def plot_interaction_curves(scan_labels, **energies) -> None:
    """Plot E(AB) - E(A) - E(B) in kcal/mol for each pair scan, one line per model.

    Each keyword is a (90, 1) energy tensor in eV (``composed=``, ``wrapped=``). The wrapped model
    is the green line and the others are grey. No reference points, no error numbers.
    """

    import matplotlib.pyplot as plt

    from . import common

    colors = {"wrapped": _GREEN}
    curves = {key: _interaction(value, scan_labels) for key, value in energies.items()}
    scale = np.array([lab["scale"] for lab in scan_labels])
    systems = tuple(dict.fromkeys(lab["system_name"] for lab in scan_labels))
    system_of = np.array([lab["system_name"] for lab in scan_labels])
    common.register_course_fonts()
    with plt.style.context(str(_STYLE)):
        figure, axes = plt.subplots(1, len(systems), figsize=(9.2, 4.0), sharex=True, squeeze=False)
        for axis, system in zip(axes.ravel(), systems):
            mask = system_of == system
            order = np.argsort(scale[mask])
            for key, values in curves.items():
                axis.plot(
                    scale[mask][order],
                    values[mask][order],
                    color=colors.get(key, _GREY),
                    linewidth=2.4 if key in colors else 2.0,
                    label=_WORDS["curves_names"].get(key, key),
                    zorder=4 if key in colors else 3,
                )
            axis.axhline(0.0, color=_LIGHT_GREY, linewidth=0.8)
            axis.set_title(system)
        axes[0, 0].set_ylabel(_WORDS["curves_ylabel"])
        figure.supxlabel(_WORDS["curves_xlabel"])
        handles, labels = axes[0, -1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=len(labels))
        _show(
            figure,
            "Interaction curves of the wrapped and the composed model across three dissociation scans.",
        )


# --------------------------------------------------------------------------
# Parked: the views and checks of the pieces that left this notebook with plan 3 (the timestep scan,
# the proton hop, the two-name check of the wrapped model). Nothing in the notebook calls them and
# `check <step>` does not list them. They are kept, unchanged, for the notebook that takes the
# timestep piece (or for option C of the board question a05q-shape):
# `load_aspirin`, `load_malonaldehyde`, `plot_timestep_scan`, `show_proton_hop`, `_parked_problems`.
# --------------------------------------------------------------------------
_SEED = 7  # fixed embedding seed: the same geometry on every machine


def _embed(smiles: str, name: str, charge: int = 0) -> Atoms:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=_SEED) != 0:
        raise RuntimeError(f"Could not build a 3D structure for {name}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    atoms = Atoms(
        [atom.GetSymbol() for atom in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
        pbc=False,
    )
    atoms.info["name"] = name
    atoms.info["charge"] = charge  # AceFF takes the total charge of each structure
    return atoms


def load_aspirin() -> Atoms:
    """Return aspirin (21 atoms, total charge 0) as an ASE ``Atoms`` object."""

    return _embed("CC(=O)Oc1ccccc1C(=O)O", "aspirin")


def load_malonaldehyde() -> Atoms:
    """Return enol malonaldehyde, O=CH-CH=CH-OH, with its O-H...O hydrogen bond.

    ``info["hop_pair"]`` holds the indices of the transferring hydrogen and the far oxygen.
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    for seed in range(_SEED, _SEED + 200):
        mol = Chem.AddHs(Chem.MolFromSmiles(r"O=C/C=C\O"))  # atoms 0 O(carbonyl) ... 4 O(hydroxyl)
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        pos = mol.GetConformer().GetPositions()
        hydroxyl_h = next(a.GetIdx() for a in mol.GetAtomWithIdx(4).GetNeighbors() if a.GetAtomicNum() == 1)
        if np.linalg.norm(pos[hydroxyl_h] - pos[0]) < 2.3:  # hydrogen bond to the far oxygen
            atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=pos, pbc=False)
            atoms.info.update(name="malonaldehyde", charge=0, hop_pair=(int(hydroxyl_h), 0))
            return atoms
    raise RuntimeError("Could not build a hydrogen-bonded malonaldehyde")


def plot_timestep_scan(replica_dt, saved_step, total_energy) -> None:
    """Plot the change of total energy in time, one line per timestep.

    replica_dt: (n_replicas,) fs. saved_step: (n_saved,) steps. total_energy: (n_saved,
    n_replicas) eV. Every line is grey, darker for larger timesteps, named at its end; a copy
    whose energy becomes non-finite is drawn to its last finite point and marked.
    """

    import matplotlib.pyplot as plt
    import torch

    from . import common

    dt = torch.as_tensor(replica_dt).detach().cpu().double().flatten().numpy()
    step = torch.as_tensor(saved_step).detach().cpu().double().flatten().numpy()
    energy = torch.as_tensor(total_energy).detach().cpu().double().numpy()
    if energy.shape != (len(step), len(dt)):
        raise ValueError("total_energy must have shape (n_saved, n_replicas)")
    change = (energy - energy[0]) * 1000.0  # meV from each copy's first saved value
    common.register_course_fonts()
    with plt.style.context(str(_STYLE)):
        figure, axis = plt.subplots(figsize=(9.2, 4.0))
        order = np.argsort(dt)
        ends = []   # (label, x, y, colour) at each line's last point
        for rank, c in enumerate(order):
            t_fs = step * dt[c]
            ok = np.isfinite(change[:, c])
            last = int(np.where(ok)[0].max()) if ok.any() else 0
            shade = plt.cm.Greys(0.5 + 0.45 * rank / max(len(order) - 1, 1))
            axis.plot(t_fs[: last + 1], change[: last + 1, c], color=shade, linewidth=1.8)
            ends.append((f"{dt[c]:g} fs", t_fs[last], change[last, c], shade))
            if not ok.all():
                axis.scatter([t_fs[last]], [change[last, c]], marker="x", color=shade, zorder=5)
        axis.set_yscale("symlog", linthresh=1.0)
        axis.axhline(0.0, color=_LIGHT_GREY, linewidth=0.8)
        axis.set(xlabel="time [fs]", ylabel="change of total energy [meV]")
        axis.set_xlim(right=axis.get_xlim()[1] * 1.08)
        _label_line_ends(axis, ends)
        _show(
            figure,
            "Change of total energy over time for copies of one molecule, each with its own timestep.",
        )


def _label_line_ends(axis, ends, gap_pt=4.0, steps_pt=(0, 8, -8, 14, -14, 20, -20)):
    """Name each line at its end, nudged up or down until no other line runs under the name."""

    import numpy as np

    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_px = axis.transData.transform
    lines = [np.column_stack(ln.get_data()) for ln in axis.get_lines() if len(ln.get_xdata()) > 1]
    lines = [to_px(xy[np.isfinite(xy).all(axis=1)]) for xy in lines]
    placed = []
    for label, x, y, colour in ends:
        own = to_px([[x, y]])[0]
        for dy in steps_pt:
            note = axis.annotate(label, (x, y), xytext=(gap_pt, dy), textcoords="offset points",
                                 color=colour, va="center", ha="left", fontsize=7.9)
            box = note.get_window_extent(renderer).padded(2)
            hit = any(box.overlaps(other) for other in placed)
            for px in lines:
                if hit:
                    break
                if abs(px[-1][0] - own[0]) < 0.5 and abs(px[-1][1] - own[1]) < 0.5:
                    continue   # the label's own line
                inside = (px[:, 0] >= box.x0) & (px[:, 0] <= box.x1)
                if inside.any():
                    ys = np.interp(np.linspace(box.x0, box.x1, 12), px[:, 0], px[:, 1], left=np.nan, right=np.nan)
                    hit = bool(np.any((ys >= box.y0) & (ys <= box.y1)))
            if not hit:
                break
            note.remove()
        else:
            note = axis.annotate(label, (x, y), xytext=(gap_pt, 0), textcoords="offset points",
                                 color=colour, va="center", ha="left", fontsize=7.9)
        placed.append(note.get_window_extent(renderer))


def show_proton_hop(hop_store, hop_distance, cv_grid, free_energy) -> None:
    """Plot the hop distance of each walker over time, beside the free-energy profile.

    hop_distance: (n_saved, n_walkers) in Å. cv_grid: (n_bins,) in Å. free_energy: (n_bins,)
    in eV. hop_store holds the saved frames of one walker (kept for a structure player).
    The walkers are grey; the free-energy profile is green.
    """

    import matplotlib.pyplot as plt
    import torch

    from . import common

    distance = torch.as_tensor(hop_distance).detach().cpu().double().numpy()
    grid = torch.as_tensor(cv_grid).detach().cpu().double().numpy()
    profile = torch.as_tensor(free_energy).detach().cpu().double().numpy()
    common.register_course_fonts()
    with plt.style.context(str(_STYLE)):
        figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
        for walker in range(distance.shape[1]):
            left.plot(distance[:, walker], linewidth=1.0, alpha=0.8, color=_GREY if walker == 0 else _LIGHT_GREY)
        left.set(xlabel="time, in saved frames", ylabel="distance, hydrogen to far oxygen [Å]")
        left.set_title("Each walker")
        right.plot(grid, profile, color=_GREEN, linewidth=2.6)
        right.axhline(0.0, color=_LIGHT_GREY, linewidth=0.8)
        right.set(xlabel="distance [Å]", ylabel="free energy [eV]")
        right.set_title("Free-energy profile")
        _show(
            figure,
            "Hop distance of each walker over time, and the free-energy profile along that distance.",
        )


_PARKED_STEPS = {
    "aceff_check": "aceff_check.py",
    "timestep_scan": "timestep_scan.py",
    "proton_hop": "proton_hop.py",
}


def _parked_problems(step: str, ns: dict) -> list[str]:
    """One plain line per problem for a parked step (the checks as they stood on 25 Nov)."""

    import torch

    found: list[str] = []

    def need(name):
        if name not in ns:
            found.append(f"{name} is missing: the file must create it")
            return None
        return ns[name]

    def finite(name, value):
        if not bool(torch.isfinite(torch.as_tensor(value).float()).all()):
            found.append(f"{name} has values that are not finite")

    if step == "aceff_check":
        energy, force = need("aceff_energy"), need("aceff_max_force")
        if energy is not None:
            if _shape(energy) != (90, 1):
                found.append(f"aceff_energy has shape {_shape(energy)}, expected (90, 1)")
            finite("aceff_energy", energy)
        if force is not None:
            value = torch.as_tensor(force).float()
            if value.numel() != 1:
                found.append("aceff_max_force must be one number")
            elif not 0.0 <= float(value) < 50.0:
                found.append(f"aceff_max_force is {float(value):.3g} eV/Å, outside the range of a force")
    elif step == "timestep_scan":
        dt, saved, energy = need("replica_dt"), need("saved_step"), need("total_energy")
        if dt is not None and saved is not None and energy is not None:
            n_replicas, n_saved = len(torch.as_tensor(dt).flatten()), len(torch.as_tensor(saved).flatten())
            if n_replicas < 2 or len(set(torch.as_tensor(dt).flatten().tolist())) != n_replicas:
                found.append("replica_dt must hold two or more different timesteps, in fs")
            if _shape(energy) != (n_saved, n_replicas):
                found.append(f"total_energy has shape {_shape(energy)}, expected (n_saved, n_replicas) = ({n_saved}, {n_replicas})")
    elif step == "proton_hop":
        distance, grid, free, store = need("hop_distance"), need("cv_grid"), need("free_energy"), need("hop_store")
        if distance is not None:
            if len(_shape(distance) or ()) != 2 or _shape(distance)[1] < 2:
                found.append("hop_distance must have shape (n_saved, n_walkers) with several walkers")
            elif not (0.5 < float(torch.as_tensor(distance).min()) and float(torch.as_tensor(distance).max()) < 6.0):
                found.append("hop_distance is outside the range of an O-H distance, in Å")
        if grid is not None and free is not None and _shape(grid) != _shape(free):
            found.append("cv_grid and free_energy must have the same shape (n_bins,)")
        if store is not None and distance is not None and len(store) != _shape(distance)[0]:
            found.append("hop_store must hold one frame for each saved step of one walker")
    return found


# --------------------------------------------------------------------------
# The step check: python -m helpers.agent05 check <step>
#
# The check runs the setup cell, the files of the earlier steps, then the step's file, all in one
# namespace as the notebook does after %load. Then it checks exactly what the step promises and
# prints one plain line per problem. A result off the expected relation is reported as
# "unexpected" and passes. Exit code 0 passes, 1 fails.
# --------------------------------------------------------------------------
_STEPS = {
    "aceff_wrapper": {"file": "aceff_wrapper.py", "after": []},
    "aceff_pairs": {"file": "aceff_pairs.py", "after": ["aceff_wrapper"]},
    "aceff_relax": {"file": "aceff_relax.py", "after": ["aceff_wrapper"]},
}
# Names a script has and a notebook cell does not; the check runs a file without them, as %load does.
_SCRIPT_NAMES = {"__file__"}
_PAIR_RELATION_LIMIT = 5.0  # kcal/mol: mean |wrapped - composed| interaction energy on the scans


def _shape(value):
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


def _unexpected_pairs(energy, ns) -> list[str]:
    """Lines for a wrapped-model curve that departs from the composed model's curve by a lot."""

    import torch

    labels, composed = ns.get("scan_labels"), ns.get("composed_energy")
    if labels is None or composed is None or _shape(energy) != _shape(composed):
        return []
    if not bool(torch.isfinite(torch.as_tensor(energy)).all()):
        return []
    gap = float(np.mean(np.abs(_interaction(energy, labels) - _interaction(composed, labels))))
    if gap <= _PAIR_RELATION_LIMIT:
        return []
    return [
        _WORDS["unexpected"].format(
            text=(
                f"the interaction energies of aceff_energy differ from the composed model's by "
                f"{gap:.1f} kcal/mol on average, so the unit may not be eV"
            )
        )
    ]


def _largest_start(ns: dict, relaxed) -> tuple[str | None, int]:
    """Name and size of the largest set of molecules the step's file made: a Batch, or a list of
    ASE Atoms or AtomicData. Only names the step's own file created count (``_step_names``, set
    by ``_check``); without that record every name but ``scan_batch`` counts."""

    from nvalchemi.data import AtomicData, Batch

    names = ns.get("_step_names")
    best: tuple[str | None, int] = (None, 0)
    for name, value in ns.items():
        if name.startswith("_") or name == "scan_batch" or value is relaxed:
            continue
        if names is not None and name not in names:
            continue
        if isinstance(value, Batch):
            count = int(value.num_graphs)
        elif isinstance(value, (list, tuple)) and value and all(isinstance(v, (Atoms, AtomicData)) for v in value):
            count = len(value)
        else:
            continue
        if count > best[1]:
            best = (name, count)
    return best


def _problems(step: str, ns: dict) -> tuple[list[str], list[str]]:
    """Return (problems, notes): one plain line per problem, and the 'unexpected' lines."""

    import torch

    problems: list[str] = []
    notes: list[str] = []

    def need(name):
        if name not in ns:
            problems.append(f"{step}: {name} is missing; the step file should create it")
            return None
        return ns[name]

    def finite(name, value):
        if not bool(torch.isfinite(torch.as_tensor(value).double()).all()):
            problems.append(f"{step}: {name} should be finite everywhere; found values that are not (nan or inf)")

    if step == "aceff_wrapper":
        model = need("aceff_model")
        if model is not None:
            config = getattr(model, "model_config", None)
            if config is None or not {"energy", "forces"} <= set(config.active_outputs):
                problems.append(f"{step}: aceff_model should return energy and forces; its outputs are different")
            elif torch.cuda.is_available() and next(model.parameters()).device.type != "cuda":
                problems.append(f"{step}: aceff_model should be on the gpu; found {next(model.parameters()).device}")
            else:
                out = model(ns["scan_batch"].clone())
                n = ns["scan_batch"].num_graphs
                if _shape(out["energy"]) != (n, 1):
                    problems.append(
                        f"{step}: the energy of aceff_model on scan_batch should have shape ({n}, 1); "
                        f"found {_shape(out['energy'])}"
                    )
                if _shape(out["forces"]) != (ns["scan_batch"].num_nodes, 3):
                    problems.append(
                        f"{step}: the forces of aceff_model should have shape (n_atoms, 3) = "
                        f"({ns['scan_batch'].num_nodes}, 3); found {_shape(out['forces'])}"
                    )
                finite("the energy of aceff_model", out["energy"])
                notes += _unexpected_pairs(out["energy"].detach(), ns)
    elif step == "aceff_pairs":
        energy = need("aceff_energy")
        if energy is not None:
            want = _shape(ns.get("composed_energy"))
            if _shape(energy) != want:
                problems.append(
                    f"{step}: aceff_energy should have shape {want} in eV (one row for each structure of "
                    f"scan_batch); found {_shape(energy)}"
                )
            else:
                finite("aceff_energy", energy)
                notes += _unexpected_pairs(energy, ns)
    elif step == "aceff_relax":
        batch, force, threshold = need("relaxed_batch"), need("relaxed_max_force"), need("force_threshold")
        model = ns.get("aceff_model")
        if batch is not None and force is not None and threshold is not None:
            n = getattr(batch, "num_graphs", None)
            f = torch.as_tensor(force).detach().cpu().double()
            if n is None:
                problems.append(f"{step}: relaxed_batch should be a Batch of molecules; found {type(batch).__name__}")
            elif tuple(f.shape) != (n,):
                problems.append(
                    f"{step}: relaxed_max_force should have one value for each molecule, shape ({n},) in "
                    f"eV/Å; found shape {tuple(f.shape)}"
                )
            elif torch.as_tensor(threshold).numel() != 1 or not float(threshold) > 0:
                problems.append(f"{step}: force_threshold should be one number above 0, in eV/Å")
            else:
                limit = float(threshold)
                finite("relaxed_max_force", f)
                names = _formula(batch)
                above = [f"{name} ({_force_text(v, limit)})" for name, v in zip(names, f.tolist()) if not _below(v, limit)]
                if above:
                    problems.append(f"{step}: " + _WORDS["above_threshold"].format(limit=limit, names=", ".join(above)))
                if model is not None:
                    fresh = model(batch.clone())["forces"].detach().norm(dim=-1)
                    owner = batch.batch_idx.long()
                    per = torch.zeros(n, device=fresh.device, dtype=fresh.dtype).scatter_reduce(
                        0, owner, fresh, reduce="amax", include_self=False
                    )
                    off = (per.cpu().double() - f).abs().max().item()
                    if off > 1e-3:
                        problems.append(
                            f"{step}: relaxed_max_force should be the largest force on an atom of each molecule "
                            f"after the relaxation; a fresh call of aceff_model gives values up to {off:.4f} eV/Å away"
                        )
                sizes = batch.num_nodes_per_graph.detach().cpu().long().tolist()
                large = [f"{name} ({size} atoms)" for name, size in zip(names, sizes) if size > _TRAINED_ATOMS]
                if large:
                    verb = " is" if len(large) == 1 else " are"
                    notes.append(_WORDS["note_size"].format(names=", ".join(large) + verb, limit=_TRAINED_ATOMS))
                start, count = _largest_start(ns, batch)
                if count > n:
                    text = _WORDS["left_out"].format(n=n, name=start, m=count, k=count - n)
                    notes.append(_WORDS["unexpected"].format(text=text))
    return problems, notes


def _check(step: str) -> int:
    import traceback

    if step not in _STEPS:
        print(_WORDS["check_unknown"].format(step=step, steps=", ".join(_STEPS)))
        return 2
    ns: dict = {"__name__": "__main__"}
    exec(
        "from helpers import agent05 as helpers\n"
        "helpers.configure_tutorial()\n"
        "work_dir = helpers.make_work_dir()\n"
        "scan_batch, scan_labels, composed_energy = helpers.load_pair_scans()\n",
        ns,
    )
    for name in [*_STEPS[step]["after"], step]:
        path = WORK_DIR / _STEPS[name]["file"]
        if not path.exists():
            print(_WORDS["check_nofile"].format(file=path.name, folder=WORK_DIR))
            return 1
        before = set(ns)
        try:
            exec(compile(path.read_text(), str(path), "exec"), ns)
        except NameError as error:
            if getattr(error, "name", None) not in _SCRIPT_NAMES:
                traceback.print_exc()
                print(_WORDS["check_failed"].format(file=path.name))
                return 1
            print("problem:", _WORDS["script_name"].format(step=step, file=path.name, name=error.name))
            print(_WORDS["check_problems"].format(n=1))
            return 1
        except Exception:
            traceback.print_exc()
            print(_WORDS["check_failed"].format(file=path.name))
            return 1
    ns["_step_names"] = set(ns) - before  # the names the step's own file created
    found, notes = _problems(step, ns)
    for line in found:
        print("problem:", line)
    for line in notes:
        print(line)
    print(_WORDS["check_ok"] if not found else _WORDS["check_problems"].format(n=len(found)))
    return 0 if not found else 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "check":
        raise SystemExit(_check(sys.argv[2]))
    print(_WORDS["check_usage"].format(steps=", ".join(_STEPS)))
    raise SystemExit(2)
