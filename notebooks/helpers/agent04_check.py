# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step contracts and the agent's own check for Agent 04, Made to order.

``python -m helpers.agent04 check <step>`` runs the setup cell and the files of
the earlier steps, then the step's file twice (a learner reruns a cell), in one
namespace, as the notebook does after ``%load``. It then checks the names the
step must leave behind and what the step's "Done when" promises, and prints one
plain line per problem, or one OK line. A result off the expected relation is
printed as "unexpected" after the OK line and the step still passes. The
contracts hold the shapes and units, so the learner's prompt only needs the
names.

Every line the check prints to the learner is in ``LINES``, in one place.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["LINES", "STEPS", "check_step", "main"]

KCAL_PER_EV = 23.0605
MIN_COULOMB_RANGE = 12.0  # Å: 5 Å drops pairs (4.2 kcal/mol on the scans), 12 Å and the 15 Å default hold (0.34)
# True: the conformer step names its force threshold, helpers.relax_settings()["fmax"], in its input row, and
# a conformer above it is a problem. False: the threshold is not part of the step, and a conformer above it is
# an "unexpected" note that passes.
THRESHOLD_IN_CONTRACT = True
FORCE_TOLERANCE = 1.01  # the check recomputes the forces; 1 % covers the float32 noise of a second model call
NOT_IN_CELL = ("__file__",)  # names a script has and a notebook cell does not
ELECTROSTATICS_WORDS = ("Coulomb", "Ewald", "PME")  # class names of electrostatics terms: the course's and Toolkit's
MAX_SCAN_MAE = 8.0  # kcal/mol: a composed model further than this from the scan references is miswired
MAX_NCIA_MAE = 6.0  # kcal/mol: the same for the NCIA complexes
MAX_ENERGY_GAP = 50.0  # eV: composed and bare energies further apart than this are in different units

# Every line printed to the learner.
LINES = {
    "ok": "{step}: OK ({n} names)",
    "usage": "usage: python -m helpers.agent04 check <step>\nsteps: {steps}",
    "file_missing": "{step}: {file} is missing in {folder}",
    "file_stops": "{step}: {file} stops with {error}",
    "file_stops_again": "{step}: {file} stops when it runs a second time, with {error}; a rerun starts from the names the first run made",
    "edge_ptr": (
        "{step}: Toolkit 0.3.0 cannot reset the neighbor list of a pipeline without a D3 group when the "
        "electrostatics term asks for pairs; use helpers.DirectCoulombAdapter, or build the pipeline with "
        "neighbor_adaptation=\"always\""
    ),
    "name_missing": "{step}: {name} is missing",
    "plot_data": "the plot cannot draw {name}: {error}",
    "energy_shape": "{name} has shape {shape}; the contract says ({n}, 1) or ({n},) in eV, one energy per structure",
    "not_finite": "{name} holds values that are not finite",
    "scan_batch_count": "scan_batch holds {got} structures; the contract says all {n} in one batch",
    "scan_batch_gpu": "scan_batch is not on the GPU",
    "scan_labels_count": "scan_labels has {got} entries; it should stay as the loader gave it ({n})",
    "charge_lost": "scan_batch changed the total charge of {k} of {n} structures; each structure keeps its own total charge",
    "coulomb_range": "{name} has a Coulomb term with a range of {cutoff:g} Å; use at least {floor:g} Å",
    "term_missing": "composed_model has no {term} term",
    "no_d3_has_d3": "no_d3_model still has a D3 term",
    "no_d3_no_coulomb": "no_d3_model has no electrostatics term",
    "not_in_cell": "{step}: {file} uses {name}, which does not exist in a notebook cell; remove that line",
    "scan_error": (
        "composed_energy is {mae:.1f} kcal/mol from the reference on average, an unexpected error; "
        "check the wiring and the units"
    ),
    "ncia_error": (
        "ncia_energy is {mae:.1f} kcal/mol from the reference on average, an unexpected error; "
        "check the order of structures and the units"
    ),
    "energy_gap": "composed_energy differs from base_energy by up to {gap:.0f} eV; check the units",
    "conformer_counts": "{name} holds {got} conformers; conformer_batch and relaxed_conformers must hold the same conformers",
    "not_batch": (
        "{name} is a {got}, not a batch; give conformer_batch and relaxed_conformers as batches with the "
        "conformers in the loader's order, and conformer_energy in the same order"
    ),
    "conformer_loader": "the batches hold {got} conformers; the loader gives {n}, all in one batch",
    "not_moved": "no atom moved between conformer_batch and relaxed_conformers; relaxed_conformers holds the starting structures",
    "not_relaxed": (
        "{k} of {n} conformers still have a force above {threshold:g} eV/Å; relax every conformer to "
        "helpers.relax_settings()[\"fmax\"] and give the run enough steps"
    ),
    "forces_fail": "composed_model could not be run on relaxed_conformers: {error}",
    "autograd_group": (
        "{name} takes the forces of the electrostatics term outside the summed energy; put AIMNet2 and the "
        "Coulomb term in one PipelineGroup with use_autograd=True"
    ),
    "unexpected_order": (
        "{step}: unexpected: the error without D3 ({no_d3:.2f} kcal/mol) is not between the complete model "
        "({composed:.2f}) and the bare model ({base:.2f}); the step still passes"
    ),
    "unexpected_composed": (
        "{step}: unexpected: the complete model ({composed:.2f} kcal/mol) is not closer to the reference than "
        "the bare model ({base:.2f}); the step still passes"
    ),
    "unexpected_higher": (
        "{step}: unexpected: {k} of {n} conformers have a higher energy after relaxing; the step still passes"
    ),
    "unexpected_force": (
        "{step}: unexpected: {k} of {n} conformers still have a force above {threshold:g} eV/Å; "
        "their end points are not minima; the step still passes"
    ),
}


@dataclass(frozen=True)
class Step:
    """One ``%load`` step: its file, the steps before it, the names it creates, its checks and notes."""

    file: str
    after: tuple[str, ...]
    names: tuple[str, ...]
    checks: tuple[Callable[[dict], list[str]], ...] = field(default_factory=tuple)
    notes: tuple[Callable[[dict, str], list[str]], ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else np.asarray(value).shape


def _energy_problems(ns: dict, name: str, n: int) -> list[str]:
    """Energy of ``n`` structures, shape (n, 1) or (n,), finite, in eV."""

    if name not in ns:
        return []
    shape = _shape(ns[name])
    if shape not in {(n, 1), (n,)}:
        return [LINES["energy_shape"].format(name=name, shape=shape, n=n)]
    if not np.isfinite(_array(ns[name])).all():
        return [LINES["not_finite"].format(name=name)]
    return []


def _terms(model: Any) -> list[Any]:
    """The terms of a model: the step models of a pipeline, taken apart down to plain models.

    The bare AIMNet2 wrapper is one term (its network holds a short-range Coulomb module of its own, which is
    not an electrostatics term of the composed model). Any other model that is not a pipeline is searched
    through all its modules, so a hand-written wrapper still shows its parts.
    """

    groups = getattr(model, "groups", None)
    if groups:
        out: list[Any] = []
        for group in groups:
            for step in getattr(group, "steps", ()):
                out += _terms(getattr(step, "model", step))
        return out
    if "AIMNet2" in type(model).__name__:
        return [model]
    modules = getattr(model, "modules", None)
    return list(modules()) if callable(modules) else [model]


def _class_names(model: Any) -> list[str]:
    return [type(m).__name__ for m in _terms(model)]


def _has_electrostatics(model: Any) -> bool:
    return any(word in name for name in _class_names(model) for word in ELECTROSTATICS_WORDS)


def _coulomb_problems(ns: dict, name: str) -> list[str]:
    """A Coulomb term needs a range of at least ``MIN_COULOMB_RANGE`` Å, or long-range pairs drop out."""

    if name not in ns:
        return []
    out = []
    for part in _terms(ns[name]):
        if "Coulomb" in type(part).__name__ and hasattr(part, "cutoff"):
            cutoff = float(part.cutoff)
            if cutoff < MIN_COULOMB_RANGE:
                out.append(LINES["coulomb_range"].format(name=name, cutoff=cutoff, floor=MIN_COULOMB_RANGE))
    return out


def _autograd_problems(ns: dict, name: str) -> list[str]:
    """Forces follow the summed energy: the group that holds the Coulomb term uses autograd."""

    groups = getattr(ns.get(name), "groups", None)
    if not groups:
        return []
    for group in groups:
        models = [getattr(step, "model", step) for step in getattr(group, "steps", ())]
        electrostatics = any(word in type(m).__name__ for m in models for word in ELECTROSTATICS_WORDS)
        if electrostatics and not getattr(group, "use_autograd", True):
            return [LINES["autograd_group"].format(name=name)]
    return []


def _interaction_kcal(energy: Any) -> np.ndarray:
    e = _array(energy).reshape(-1, 3)
    return (e[:, 0] - e[:, 1] - e[:, 2]) * KCAL_PER_EV


def _scan_mae(energy: Any) -> float:
    from . import agent04 as helpers

    _, labels = helpers.load_nci_scans()
    ref = np.array([label["ref_ccsd_t_cbs"] for label in labels])
    return float(np.abs(_interaction_kcal(energy) - ref).mean())


# --------------------------------------------------------------------------
# Problems per step (the step fails)
# --------------------------------------------------------------------------


def _scan_batch_problems(batch: Any) -> list[str]:
    """All 90 scan structures in one batch, on the GPU when there is one, each with its own total charge."""

    import torch

    out = []
    got = getattr(batch, "num_graphs", None)
    if got != 90:
        out.append(LINES["scan_batch_count"].format(got=got if got is not None else "?", n=90))
    positions = getattr(batch, "positions", None)
    if positions is not None and torch.cuda.is_available() and positions.device.type != "cuda":
        out.append(LINES["scan_batch_gpu"])
    elif got == 90:
        out += _charge_problems(batch)
    return out


def _charge_problems(batch: Any) -> list[str]:
    """The total charge of each scan structure is the one the loader gave."""

    from . import agent04 as helpers

    have = getattr(batch, "charge", None)
    if have is None:
        return []
    atoms, _ = helpers.load_nci_scans()
    want = np.array([a.info.get("charge", 0) for a in atoms], dtype=float)
    lost = int((~np.isclose(_array(have).reshape(-1), want)).sum())
    return [LINES["charge_lost"].format(k=lost, n=len(want))] if lost else []


def _check_compose_build(ns: dict) -> list[str]:
    out = _energy_problems(ns, "base_energy", 90)
    if "scan_batch" in ns:
        out += _scan_batch_problems(ns["scan_batch"])
    if "scan_labels" in ns and len(ns["scan_labels"]) != 30:
        out.append(LINES["scan_labels_count"].format(got=len(ns["scan_labels"]), n=30))
    return out


def _check_compose_run(ns: dict) -> list[str]:
    out = _energy_problems(ns, "composed_energy", 90) + _energy_problems(ns, "ncia_energy", 615)
    out += _coulomb_problems(ns, "composed_model") + _autograd_problems(ns, "composed_model")
    if "composed_model" in ns:
        if not _has_electrostatics(ns["composed_model"]):
            out.append(LINES["term_missing"].format(term="electrostatics"))
        if not any("D3" in n for n in _class_names(ns["composed_model"])):
            out.append(LINES["term_missing"].format(term="D3 dispersion"))
    if not out:
        from . import agent04 as helpers

        mae = _scan_mae(ns["composed_energy"])
        if mae > MAX_SCAN_MAE:
            out.append(LINES["scan_error"].format(mae=mae))
        ncia = float(np.abs(_interaction_kcal(ns["ncia_energy"]) - helpers.load_ncia_references()).mean())
        if ncia > MAX_NCIA_MAE:
            out.append(LINES["ncia_error"].format(mae=ncia))
        if "base_energy" in ns:
            gap = float(np.abs(_array(ns["composed_energy"]).reshape(-1) - _array(ns["base_energy"]).reshape(-1)).max())
            if gap > MAX_ENERGY_GAP:
                out.append(LINES["energy_gap"].format(gap=gap))
    return out


def _check_no_d3(ns: dict) -> list[str]:
    out = _energy_problems(ns, "no_d3_energy", 90) + _coulomb_problems(ns, "no_d3_model")
    out += _autograd_problems(ns, "no_d3_model")
    if "no_d3_model" in ns:
        names = _class_names(ns["no_d3_model"])
        if any("D3" in n for n in names):
            out.append(LINES["no_d3_has_d3"])
        if not _has_electrostatics(ns["no_d3_model"]):
            out.append(LINES["no_d3_no_coulomb"])
    if not out and all(k in ns for k in ("base_energy", "composed_energy", "no_d3_energy")):
        out += _no_d3_plot_problems(ns)
    return out


def _no_d3_plot_problems(ns: dict) -> list[str]:
    """The step promises that the plot adds the curve to the three panels: the plot's data layer takes it."""

    from . import agent04 as helpers
    from .agent04_plots import interaction_curves_data

    try:
        _, labels = helpers.load_nci_scans()
        data = interaction_curves_data(labels, base=ns["base_energy"], composed=ns["composed_energy"], no_d3=ns["no_d3_energy"])
        if "no_d3" not in data["lines"]:
            raise ValueError("no line for no_d3_energy")
    except Exception as error:
        return [LINES["plot_data"].format(name="no_d3_energy", error=str(error)[:120])]
    return []


def _conformer_count(ns: dict) -> tuple[int | None, list[str]]:
    """The number of conformers in the two batches, and a line when a name is not a batch or the counts differ.

    A list of ASE ``Atoms`` has no ``num_graphs``: it is reported alone, and the count is None.
    """

    out = []
    n = None
    for name in ("conformer_batch", "relaxed_conformers"):
        if name in ns:
            graphs = getattr(ns[name], "num_graphs", None)
            if graphs is None:
                return None, [LINES["not_batch"].format(name=name, got=type(ns[name]).__name__)]
            n = n or graphs
            if graphs != n:
                out.append(LINES["conformer_counts"].format(name=name, got=graphs))
    return n, out


def _conformer_map_problems(ns: dict) -> list[str]:
    """The step promises that the map shows where each conformer ended: the map's data layer takes the names."""

    from .agent04_plots import conformers_data

    try:
        data = conformers_data(ns["conformer_batch"], ns["relaxed_conformers"], ns["conformer_energy"])
        if not np.isfinite(np.asarray(data["end"], dtype=float)).all():
            raise ValueError("an end point is not finite")
    except Exception as error:
        return [LINES["plot_data"].format(name="the conformers", error=str(error)[:120])]
    return []


def _check_conformers(ns: dict) -> list[str]:
    from . import agent04 as helpers

    n, out = _conformer_count(ns)
    if n is None and out:
        return out
    expected = len(helpers.load_dipeptide_grid())
    if n is not None and n != expected:
        out.append(LINES["conformer_loader"].format(got=n, n=expected))
    if n:
        out += _energy_problems(ns, "conformer_energy", n)
    if not out and "conformer_batch" in ns and "relaxed_conformers" in ns:
        moved = np.abs(_array(ns["relaxed_conformers"].positions) - _array(ns["conformer_batch"].positions)).max()
        if moved < 1e-4:
            out.append(LINES["not_moved"])
    if not out and all(k in ns for k in ("conformer_batch", "relaxed_conformers", "conformer_energy")):
        out += _conformer_map_problems(ns)
    if not out and THRESHOLD_IN_CONTRACT and "composed_model" in ns:
        # the step promises that every conformer was relaxed: each one is below the step's force threshold
        try:
            _, _, peak = _conformer_evaluation(ns)
        except Exception as error:
            return [LINES["forces_fail"].format(error=str(error)[:120])]
        threshold = float(helpers.relax_settings()["fmax"])
        above = int((peak > threshold * FORCE_TOLERANCE).sum())
        if above:
            out.append(LINES["not_relaxed"].format(k=above, n=len(peak), threshold=threshold))
    return out


def _conformer_evaluation(ns: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energies of the starts and of the relaxed conformers (eV) and each relaxed conformer's largest force (eV/Å).

    Two calls of ``composed_model``; kept in the namespace so the check and its notes share them.
    """

    if "_a04_conformer_evaluation" in ns:
        return ns["_a04_conformer_evaluation"]
    import torch

    from . import agent04 as helpers

    model = ns["composed_model"]
    results = []
    for key in ("conformer_batch", "relaxed_conformers"):
        batch = helpers.prepare_model_batch(ns[key].clone(), model)
        with torch.enable_grad():
            results.append((model(batch), batch))
    (start, _), (end, relaxed) = results
    e0 = start["energy"].detach().reshape(-1)
    e1 = end["energy"].detach().reshape(-1)
    norms = end["forces"].detach().norm(dim=-1)
    peak = torch.zeros(len(e1), device=norms.device, dtype=norms.dtype).index_reduce_(
        0, relaxed.batch_idx.long(), norms, "amax", include_self=False
    )
    ns["_a04_conformer_evaluation"] = tuple(_array(x) for x in (e0, e1, peak))
    return ns["_a04_conformer_evaluation"]


# --------------------------------------------------------------------------
# Notes per step (a result off the expected relation: printed, the step passes)
# --------------------------------------------------------------------------


def _notes_no_d3(ns: dict, step: str) -> list[str]:
    if not all(k in ns for k in ("no_d3_energy", "composed_energy", "base_energy")):
        return []
    try:
        no_d3, composed, base = (_scan_mae(ns[k]) for k in ("no_d3_energy", "composed_energy", "base_energy"))
    except Exception:
        return []
    if composed >= base:
        return [LINES["unexpected_composed"].format(step=step, composed=composed, base=base)]
    if not (composed < no_d3 < base):
        return [LINES["unexpected_order"].format(step=step, no_d3=no_d3, composed=composed, base=base)]
    return []


def _notes_conformers(ns: dict, step: str) -> list[str]:
    if not all(k in ns for k in ("conformer_batch", "relaxed_conformers", "composed_model")):
        return []
    from . import agent04 as helpers

    try:
        e0, e1, peak = _conformer_evaluation(ns)
    except Exception:
        return []
    out = []
    higher = int((e1 > e0 + 1e-6).sum())
    if higher:
        out.append(LINES["unexpected_higher"].format(step=step, k=higher, n=len(e0)))
    if not THRESHOLD_IN_CONTRACT:
        threshold = float(helpers.relax_settings()["fmax"])
        above = int((peak > threshold * FORCE_TOLERANCE).sum())
        if above:
            out.append(LINES["unexpected_force"].format(step=step, k=above, n=len(e1), threshold=threshold))
    return out


STEPS: dict[str, Step] = {
    "compose_build": Step(
        "compose_build.py",
        (),
        ("aimnet_model", "scan_batch", "scan_labels", "base_energy"),
        (_check_compose_build,),
    ),
    "compose_run": Step(
        "compose_run.py",
        ("compose_build",),
        ("composed_model", "composed_energy", "ncia_energy"),
        (_check_compose_run,),
    ),
    "no_d3": Step(
        "no_d3.py",
        ("compose_build", "compose_run"),
        ("no_d3_model", "no_d3_energy"),
        (_check_no_d3,),
        (_notes_no_d3,),
    ),
    "conformers": Step(
        "conformers.py",
        ("compose_build", "compose_run"),
        ("conformer_batch", "relaxed_conformers", "conformer_energy"),
        (_check_conformers,),
        (_notes_conformers,),
    ),
}


# --------------------------------------------------------------------------
# Running a check
# --------------------------------------------------------------------------


def _order(step: str) -> list[str]:
    """The steps to run, earlier ones first, each once, ending with ``step``."""

    seen: list[str] = []

    def visit(name: str) -> None:
        for earlier in STEPS[name].after:
            visit(earlier)
        if name not in seen:
            seen.append(name)

    visit(step)
    return seen


def _run_file(path: Path, ns: dict) -> str | None:
    """Run one file in the namespace; return the last line of its error, or None."""

    try:
        exec(compile(path.read_text(), str(path), "exec"), ns)
    except Exception:
        return traceback.format_exc().strip().splitlines()[-1]
    return None


def _stop_lines(step: str, path: Path, error: str, again: bool = False) -> list[str]:
    """The line for a file that stopped with ``error``, on its first run or when it ran ``again``."""

    if "edge_ptr" in error:
        return [LINES["edge_ptr"].format(step=step)]
    for missing in NOT_IN_CELL:  # a script name such as __file__: a %load cell has none
        if error.startswith(f"NameError: name '{missing}' is not defined"):
            return [LINES["not_in_cell"].format(step=step, file=path.name, name=missing)]
    key = "file_stops_again" if again else "file_stops"
    return [LINES[key].format(step=step, file=path.name, error=error)]


def check_step(step: str, work_dir: Path | None = None) -> list[str]:
    """Run ``step`` after its earlier steps in one namespace; return the lines to print.

    The lines are the problems, one each, or the OK line followed by any "unexpected" notes.
    """

    from . import agent04 as helpers

    work_dir = Path(work_dir or helpers.WORK_DIR)
    ns: dict[str, Any] = {"__name__": "__main__"}
    exec(
        "from helpers import agent04 as helpers\n"
        "helpers.configure_tutorial()\n"
        "work_dir = helpers.make_work_dir()\n",
        ns,
    )

    for name in _order(step):
        path = work_dir / STEPS[name].file
        if not path.exists():
            return [LINES["file_missing"].format(step=name, file=path.name, folder=work_dir)]
        error = _run_file(path, ns)
        if error:
            return _stop_lines(name, path, error)
    path = work_dir / STEPS[step].file
    error = _run_file(path, ns)  # a learner reruns a cell: the file must work a second time
    if error:
        return _stop_lines(step, path, error, again=True)
    spec = STEPS[step]
    problems = [LINES["name_missing"].format(step=step, name=name) for name in spec.names if name not in ns]
    if not problems:
        for check in spec.checks:
            problems += [f"{step}: {line}" for line in check(ns)]
    if problems:
        return problems
    lines = [LINES["ok"].format(step=step, n=len(spec.names))]
    for note in spec.notes:
        lines += note(ns, step)
    return lines


def main(argv: list[str]) -> int:
    """Entry point of ``python -m helpers.agent04 check <step>``."""

    if len(argv) != 2 or argv[0] != "check" or argv[1] not in STEPS:
        print(LINES["usage"].format(steps=", ".join(STEPS)))
        return 2
    lines = check_step(argv[1])
    print("\n".join(lines))
    return 0 if ": OK (" in lines[0] else 1
