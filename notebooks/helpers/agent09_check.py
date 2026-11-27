# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step contracts and the agent's own check for Agent 09, Small-batch distilled (liquid SiC).

Run from ``notebooks/``::

    python -m helpers.agent09 check <step>

The check runs the setup cell, brings in the results of the earlier steps,
then runs the step's file twice in one namespace (a learner runs a cell again),
and prints one plain line per problem, or ``<step>: PASS``. The contracts hold
the shapes and units, so the learner's prompt only needs the names.

Two kinds of line. A problem means the step is not done: the check fails. A
line that starts ``<step>: unexpected:`` means the step is done and its result
is off the expected relation (an error did not fall, a copy did not stay
liquid, the student's structure departs from the teacher's): the check prints
it and still passes.

A step that passes leaves its results in ``agent09_work/.check_cache``. A later
step's check loads them when the earlier file has not changed since.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["STEPS", "check_step", "main", "resolve_step"]

MIN_STRUCTURES = 24  # a smaller store is reported as unexpected
SAMPLE_FRAMES = 8  # frames on which the check recomputes forces
FORCE_TOLERANCE = 1e-2  # eV/Å: two calls of one model on one structure agree to this
FINETUNE_LEAST_FALL = 0.02  # a force error that falls by less than 2% hardly changed
# The small CPU setting shrinks the five budgets of the setup cell with this variable:
# "optimizer steps,copies,MD steps,training steps,MD steps of the student's run" (for example "4,2,30,20,40").
_BUDGETS_VARIABLE = "ALCHEMI_AGENT09_BUDGETS"
_CACHE_FOLDER = ".check_cache"
_SCRIPT_NAMES = {"__file__"}  # names a script has and a notebook cell does not


@dataclass(frozen=True)
class Step:
    """One ``%load`` step: its file, the names it leaves behind, the steps it builds on."""

    file: str
    names: tuple[str, ...]
    after: tuple[str, ...] = ()


STEPS: dict[str, Step] = {
    "finetune": Step("finetune.py", ("finetuned_model", "ft_loss_history")),
    "sample_label": Step("sample_label.py", ("labeled_path",), after=("finetune",)),
    "distill": Step("distill.py", ("student_model", "loss_history"), after=("finetune", "sample_label")),
    "run_student": Step(
        "run_student.py", ("student_frames_path", "stability"), after=("finetune", "sample_label", "distill")
    ),
    "structure": Step(
        "structure.py", ("teacher_rdf", "student_rdf"), after=("finetune", "sample_label", "distill", "run_student")
    ),
}


def _helpers():
    from . import agent09

    return agent09


# ---------------------------------------------------------------------------
# Shared tests
# ---------------------------------------------------------------------------


def _first_line(exc: BaseException) -> str:
    text = str(exc).splitlines()[0][:200] if str(exc) else ""
    return f"{type(exc).__name__}: {text}".rstrip(": ").strip()


def _is_path(value) -> bool:
    return isinstance(value, (str, os.PathLike))


def _same_path(a, b) -> bool:
    return _is_path(a) and _is_path(b) and Path(a).resolve() == Path(b).resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weight_digest(model) -> str:
    """A fingerprint of a model's weights, byte for byte."""

    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _shape_and_finite(step, where, tensor, shape, unit, problems) -> bool:
    import torch

    if tuple(tensor.shape) != shape:
        problems.append(f"{step}: {where} has shape {tuple(tensor.shape)}; it should be {shape}, in {unit}")
        return False
    if not bool(torch.isfinite(tensor).all()):
        problems.append(f"{step}: {where} has values that are not finite")
        return False
    return True


def _sample_of(batch, count: int = SAMPLE_FRAMES):
    """``count`` structures of a batch, spread evenly over it, and their indices."""

    import torch

    n = batch.num_graphs
    picks = torch.linspace(0, n - 1, min(count, n)).round().long().unique()
    return batch.index_select(picks), picks


def _atoms_of(batch, picks):
    """Row indices of the atoms that belong to the picked structures."""

    import torch

    ptr = batch.batch_ptr.cpu().long()
    return torch.cat([torch.arange(int(ptr[i]), int(ptr[i + 1])) for i in picks.tolist()])


def _force_gap(model, batch, stored_forces) -> tuple[float, float]:
    """Mean length of (stored force minus the model's force) on a sample, and the model's mean force, eV/Å."""

    sample, picks = _sample_of(batch)
    rows = _atoms_of(batch, picks)
    fresh = _helpers().model_forces(model, sample).double().cpu()
    stored = stored_forces.detach().double().cpu()[rows]
    return float((stored - fresh).norm(dim=-1).mean()), float(fresh.norm(dim=-1).mean())


def _open_store(step: str, name: str, ns: dict, problems: list[str]):
    """Return ``(reader, batch)`` of the store named ``name``, or ``None`` after adding a problem line."""

    path = ns[name]
    if not _is_path(path):
        problems.append(f"{step}: {name} is {type(path).__name__}; it should be the path of a Zarr store")
        return None
    if not Path(path).exists():
        problems.append(f"{step}: {name} {path} does not exist")
        return None
    try:
        return _helpers().read_store(path)
    except Exception as exc:
        problems.append(f"{step}: {name} {path} could not be read as a Zarr store: {_first_line(exc)}")
        return None


def _has_fields(step, name, reader, wanted, problems) -> bool:
    fields = set(reader.field_levels)
    missing = [f for f in wanted if f not in fields]
    for f in missing:
        problems.append(f"{step}: the store at {name} has no {f}; found {', '.join(sorted(fields))}")
    return not missing


def _enough(step, name, reader, problems, notes) -> None:
    """Fewer than two frames is not a run; a small store is unexpected, and the step still passes."""

    if len(reader) < 2:
        problems.append(f"{step}: the store at {name} holds {len(reader)} structure; an MD saves many frames")
    elif len(reader) < MIN_STRUCTURES:
        notes.append(
            f"{step}: unexpected: the store at {name} holds only {len(reader)} structures; save frames more "
            f"often for a dataset of at least {MIN_STRUCTURES}"
        )


def _are_liquid_cells(step, name, batch, ns, problems) -> bool:
    """Every structure of the store is a liquid cell of the starts: 216 atoms, Si108C108, periodic, a course cell."""

    import torch

    reference = ns["starts"][0]
    atoms = len(reference)
    ptr = batch.batch_ptr.cpu().long()
    sizes = ptr[1:] - ptr[:-1]
    if not bool((sizes == atoms).all()):
        problems.append(f"{step}: the structures in the store at {name} are not whole liquid cells of {atoms} atoms")
        return False
    numbers = batch.atomic_numbers.detach().cpu().long().reshape(-1, atoms)
    if not bool(((numbers == 14).sum(1) == atoms // 2).all()) or not bool(((numbers == 6).sum(1) == atoms // 2).all()):
        problems.append(f"{step}: the structures in the store at {name} are not Si{atoms // 2}C{atoms // 2} cells")
        return False
    pbc = getattr(batch, "pbc", None)
    cell = getattr(batch, "cell", None)
    if pbc is None or cell is None or not bool(pbc.all()):
        problems.append(
            f"{step}: the structures in the store at {name} are not periodic; the liquid cells are periodic in all "
            "three directions"
        )
        return False
    cells = {tuple(round(float(x), 3) for x in a.cell.array.reshape(-1)) for a in [*ns["starts"], *ns["test_starts"]]}
    wanted = torch.tensor(sorted(cells), dtype=torch.float64)
    stored = cell.detach().cpu().double().reshape(-1, 9)
    close = (stored[:, None, :] - wanted[None, :, :]).abs().amax(-1).min(dim=1).values
    if float(close.max()) > 1e-3:
        problems.append(
            f"{step}: some structures in the store at {name} are not in the cell of any start; the MD runs at "
            "constant volume, in the cells of the starts"
        )
        return False
    return True


def _frames_differ(step, name, batch, problems) -> None:
    """The store holds frames of a run, not copies of one structure."""

    atoms = int(batch.num_nodes // batch.num_graphs)
    positions = batch.positions.detach().cpu().double().reshape(-1, atoms, 3)
    if positions.shape[0] > 1 and float((positions - positions[0]).abs().max()) < 1e-6:
        problems.append(f"{step}: every structure in the store at {name} is the same; the frames of an MD differ")


def _nearest_start(batch, copies: int, starts, test_starts) -> list[str]:
    """For each copy of the first saved snapshot: "test" or "sample", whichever start lies nearest (same cell)."""

    import numpy as np

    atoms = int(batch.num_nodes // batch.num_graphs)
    positions = batch.positions.detach().cpu().double().numpy().reshape(-1, atoms, 3)[:copies]
    cells = batch.cell.detach().cpu().double().numpy().reshape(-1, 3, 3)[:copies]
    out = []
    for pos, cell in zip(positions, cells):
        best, kind = np.inf, "none"
        inverse = np.linalg.inv(cell)
        for label, pool in (("sample", starts), ("test", test_starts)):
            for start in pool:
                if not np.allclose(start.cell.array, cell, atol=1e-3):
                    continue
                fractional = (pos - start.positions) @ inverse
                fractional -= np.round(fractional)
                rms = float(np.sqrt(((fractional @ cell) ** 2).sum(-1).mean()))
                if rms < best:
                    best, kind = rms, label
        out.append(kind)
    return out


def _is_model(step, name, ns, problems) -> bool:
    import torch
    from nvalchemi.models.base import BaseModelMixin

    model = ns[name]
    if not isinstance(model, BaseModelMixin) or not isinstance(model, torch.nn.Module):
        problems.append(f"{step}: {name} is {type(model).__name__}; it should be a Toolkit model")
        return False
    return True


def _loss_lines(step, name, ns, problems, notes, expect_fall: bool = True) -> None:
    """A loss history is a list of losses or of (step, loss) pairs; a loss that does not fall is unexpected."""

    import numpy as np

    try:
        _, losses = _helpers().loss_points(ns[name])
    except Exception:
        problems.append(
            f"{step}: {name} should be a list of losses in the order of training, or of (step, loss) "
            f"pairs; found {type(ns[name]).__name__}"
        )
        return
    if losses.size < 2 or not np.isfinite(losses).all():
        problems.append(f"{step}: {name} needs at least two finite loss values; found {losses.size}")
        return
    if not expect_fall:
        return
    # batches of different frames give losses that jump from step to step: compare thirds, not single values
    third = max(1, losses.size // 3)
    first, last = float(np.mean(losses[:third])), float(np.mean(losses[-third:]))
    if not last < first:
        notes.append(
            f"{step}: unexpected: the loss in {name} did not fall: {first:.3g} on average over the first third, "
            f"{last:.3g} over the last third"
        )


# ---------------------------------------------------------------------------
# The contracts
# ---------------------------------------------------------------------------


def _check_finetune(ns, problems, notes, ctx) -> None:
    step = "finetune"
    if not _is_model(step, "finetuned_model", ns, problems):
        return
    model, base = ns["finetuned_model"], ns["base_model"]
    if model is base:
        problems.append(
            f"{step}: finetuned_model is base_model itself; fine-tune a copy, so that base_model stays as published"
        )
        return
    if _weight_digest(base) != ctx["base_weights"]:
        problems.append(
            f"{step}: base_model's weights changed during the training; fine-tune a copy, so that "
            "base_model stays as published"
        )
        return
    ours, theirs = dict(model.named_parameters()), dict(base.named_parameters())
    changed = [
        n for n, p in ours.items()
        if n not in theirs or p.shape != theirs[n].shape
        or not bool((p.detach().cpu() == theirs[n].detach().cpu()).all())
    ]
    if not changed:
        problems.append(f"{step}: finetuned_model has the weights of base_model; the training did not update it")
        return
    if not any("interactions.1." in n or "products.1." in n for n in changed):
        problems.append(
            f"{step}: the last interaction block of finetuned_model did not train (weights named interactions.1 and "
            "products.1); the step trains it together with the readouts"
        )
        return
    if not any("readouts" in n for n in changed):
        problems.append(f"{step}: the readouts of finetuned_model did not train; the step trains them too")
        return
    _loss_lines(step, "ft_loss_history", ns, problems, notes, expect_fall=False)
    if problems:
        return
    helpers = _helpers()
    before, after = helpers.score(base, ns["test_path"]), helpers.score(model, ns["test_path"])
    if not (after["force_error"] < (1 - FINETUNE_LEAST_FALL) * before["force_error"]
            and after["energy_error"] < before["energy_error"]):
        notes.append(
            f"{step}: unexpected: on the frames of test_path the errors did not both fall: energy "
            f"{before['energy_error']:.1f} to {after['energy_error']:.1f} meV/atom, force "
            f"{before['force_error']:.3f} to {after['force_error']:.3f} eV/Å"
        )


def _liquid_note(step, name, verdict, notes) -> None:
    if verdict["judged"] and verdict["n_liquid"] < verdict["copies"]:
        notes.append(
            f"{step}: unexpected: {verdict['n_liquid']} of {verdict['copies']} copies in the store at {name} are "
            f"still liquid ({'; '.join(verdict['reasons'])})"
        )


def _check_sample_label(ns, problems, notes, ctx) -> None:
    step, name = "sample_label", "labeled_path"
    for other in ("train_path", "test_path"):
        if _same_path(ns[name], ns[other]):
            problems.append(f"{step}: {name} is the store of {other}; the sampled frames go to a new store")
            return
    opened = _open_store(step, name, ns, problems)
    if opened is None:
        return
    reader, batch = opened
    wanted = ("positions", "atomic_numbers", "cell", "pbc", "teacher_energy", "teacher_forces")
    if not _has_fields(step, name, reader, wanted, problems):
        return
    _enough(step, name, reader, problems, notes)
    if not _are_liquid_cells(step, name, batch, ns, problems):
        return
    _frames_differ(step, name, batch, problems)
    energy, forces = batch["teacher_energy"].reshape(-1), batch["teacher_forces"]
    ok = _shape_and_finite(
        step, f"teacher_energy in the store at {name}", energy, (batch.num_graphs,), "eV", problems
    )
    ok &= _shape_and_finite(
        step, f"teacher_forces in the store at {name}", forces, (batch.num_nodes, 3), "eV/Å", problems
    )
    if not ok or problems:
        return
    gap, mean = _force_gap(ns["finetuned_model"], batch, batch["teacher_forces"])
    if gap > FORCE_TOLERANCE:
        problems.append(
            f"{step}: teacher_forces in the store at {name} are not finetuned_model's forces on the same "
            f"structures: they differ by {gap:.3g} eV/Å on average (mean force {mean:.3g} eV/Å); the teacher "
            "is finetuned_model"
        )
        return
    copies = _helpers().copies_in_store(batch)
    kinds = _nearest_start(batch, copies, ns["starts"], ns["test_starts"])
    if kinds.count("sample") < len(kinds):
        problems.append(
            f"{step}: the run in the store at {name} does not start from starts; the teacher's MD starts from the "
            "cells in starts"
        )
        return
    _liquid_note(step, name, _helpers().still_liquid(ns[name], steps=int(ns["md_steps"])), notes)


def _student_is_trained(step, ns, problems) -> bool:
    """The student is the helper's small model, and it has been trained."""

    if not _is_model(step, "student_model", ns, problems):
        return False
    student = ns["student_model"]
    fresh = _helpers().make_student(ns["labeled_path"])
    count, wanted = (sum(p.numel() for p in m.parameters()) for m in (student, fresh))
    if count != wanted:
        problems.append(
            f"{step}: student_model has {count:,} weights; the student from helpers.make_student(labeled_path) "
            f"has {wanted:,}"
        )
        return False
    moved = sum(
        float((a.detach().cpu().double() - b.detach().cpu().double()).abs().sum())
        for a, b in zip(student.parameters(), fresh.parameters())
    )
    if moved == 0.0:
        problems.append(f"{step}: student_model still has its starting weights; the training did not update it")
        return False
    return True


def _check_distill(ns, problems, notes, ctx) -> None:
    step = "distill"
    if not _student_is_trained(step, ns, problems):
        return
    if _weight_digest(ns["finetuned_model"]) != ctx["teacher_weights"]:
        problems.append(f"{step}: finetuned_model's weights changed during the training; only the student trains")
        return
    _loss_lines(step, "loss_history", ns, problems, notes)
    if problems:
        return
    _, batch = _helpers().read_store(ns["labeled_path"])
    gap, mean = _force_gap(ns["student_model"], batch, batch["teacher_forces"])
    if gap > 0.5 * mean:
        notes.append(
            f"{step}: unexpected: student_model's forces are far from teacher_forces on the store: they differ "
            f"by {gap:.3g} eV/Å on average, where the teacher's force averages {mean:.3g} eV/Å"
        )


def _is_stability_of_run(step, ns, problems) -> bool:
    """``stability`` is the metrics of a ``StabilityMonitor`` that watched the student's run to its end."""

    value = ns["stability"]
    if type(value).__name__ != "StabilityMetrics" or not hasattr(value, "first_divergence_step"):
        problems.append(
            f"{step}: stability is {type(value).__name__}; it should be the metrics of the StabilityMonitor "
            "registered on the student's run (monitor.metrics() after the run)"
        )
        return False
    wanted = int(ns["test_steps"])
    if not 0.5 * wanted < value.last_step <= wanted:  # the monitor's last firing falls within its last interval
        problems.append(
            f"{step}: stability ends at step {value.last_step}; the student's run has test_steps = {wanted} steps "
            "(register the StabilityMonitor on that run)"
        )
        return False
    return True


def _check_run_student(ns, problems, notes, ctx) -> None:
    step, name = "run_student", "student_frames_path"
    for other in ("labeled_path", "train_path", "test_path"):
        if _same_path(ns[name], ns[other]):
            problems.append(f"{step}: {name} is the store of {other}; the student's frames go to a new store")
            return
    if not _is_stability_of_run(step, ns, problems):
        return
    opened = _open_store(step, name, ns, problems)
    if opened is None:
        return
    reader, batch = opened
    wanted = ("positions", "atomic_numbers", "cell", "pbc", "forces", "velocities")
    if not _has_fields(step, name, reader, wanted, problems):
        return
    _enough(step, name, reader, problems, notes)
    if not _are_liquid_cells(step, name, batch, ns, problems):
        return
    _frames_differ(step, name, batch, problems)
    if not _shape_and_finite(
        step, f"forces in the store at {name}", batch.forces, (batch.num_nodes, 3), "eV/Å", problems
    ):
        return
    if problems:
        return
    gap, mean = _force_gap(ns["student_model"], batch, batch.forces)
    if gap > FORCE_TOLERANCE:
        problems.append(
            f"{step}: the forces in the store at {name} are not student_model's forces on the same frames: "
            f"they differ by {gap:.3g} eV/Å on average (mean force {mean:.3g} eV/Å); run the MD with student_model"
        )
        return
    helpers = _helpers()
    copies = helpers.copies_in_store(batch)
    if copies != len(ns["test_starts"]):
        problems.append(
            f"{step}: the store at {name} holds {copies} copies per saved frame; the student's run has one copy "
            f"of each cell in test_starts ({len(ns['test_starts'])})"
        )
        return
    kinds = _nearest_start(batch, copies, ns["starts"], ns["test_starts"])
    if kinds.count("test") < len(kinds):
        problems.append(
            f"{step}: the student's run does not start from test_starts; start it from those cells, so that the "
            "frames are new to the student"
        )
        return
    _liquid_note(step, name, helpers.still_liquid(ns[name], ns["stability"]), notes)
    result = helpers.compare_on_frames(ns[name], ns["finetuned_model"])
    if result["mean_error"] > 0.5 * result["mean_teacher_force"]:
        notes.append(
            f"{step}: unexpected: on the student's own frames its forces differ from finetuned_model's by "
            f"{result['mean_error']:.3g} eV/Å on average, where the teacher's force averages "
            f"{result['mean_teacher_force']:.3g} eV/Å"
        )


def _check_structure(ns, problems, notes, ctx) -> None:
    import torch
    from nvalchemi.training.distillation.evaluation import RadialDistribution, radial_distribution

    step = "structure"
    helpers = _helpers()
    for name in ("teacher_rdf", "student_rdf"):
        if not isinstance(ns[name], RadialDistribution):
            problems.append(
                f"{step}: {name} is {type(ns[name]).__name__}; it should be the RadialDistribution that Toolkit's "
                "radial_distribution returns"
            )
    if problems:
        return
    teacher, student = ns["teacher_rdf"], ns["student_rdf"]
    if teacher.pair is not None or student.pair is not None:
        problems.append(f"{step}: teacher_rdf and student_rdf should count all pairs (pair=None)")
        return
    if teacher.edges.shape != student.edges.shape or not bool(torch.allclose(teacher.edges, student.edges)):
        problems.append(
            f"{step}: teacher_rdf and student_rdf have different bins (r_max {teacher.r_max} and {student.r_max}, "
            f"{teacher.counts.numel()} and {student.counts.numel()} bins); give both the same r_max and num_bins"
        )
        return
    for name, store in (("teacher_rdf", "labeled_path"), ("student_rdf", "student_frames_path")):
        rdf = ns[name]
        frames = len(helpers.read_store(ns[store])[0])
        if not frames // 2 <= rdf.num_frames <= frames:
            problems.append(
                f"{step}: {name} counts {rdf.num_frames} frames; the store at {store} holds {frames} "
                f"(use the frames of {store}, or most of them)"
            )
            continue
        if rdf.num_frames == frames:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batches = [b.to(device) for b in helpers.store_batches(ns[store], 64)]
            fresh = radial_distribution(batches, r_max=rdf.r_max, num_bins=rdf.counts.numel())
            if not bool(torch.allclose(fresh.g_r, rdf.g_r, rtol=1e-3, atol=1e-6)):
                problems.append(f"{step}: {name} is not the radial distribution of the frames in {store}")
    if problems:
        return
    partials = helpers.partial_rdfs(
        ns["labeled_path"], ns["student_frames_path"], teacher.r_max, teacher.counts.numel()
    )
    verdict = helpers.follows_teacher(teacher, student, partials)
    if verdict["judged"] and not verdict["follows"]:
        notes.append(
            f"{step}: unexpected: the student's structure departs from the teacher's ({'; '.join(verdict['reasons'])})"
        )


_CONTENT = {
    "finetune": _check_finetune,
    "sample_label": _check_sample_label,
    "distill": _check_distill,
    "run_student": _check_run_student,
    "structure": _check_structure,
}


# ---------------------------------------------------------------------------
# Results of steps that passed, for the next step's check
# ---------------------------------------------------------------------------


def _budgets(ns: dict) -> list[int]:
    return [int(ns[k]) for k in _helpers().BUDGET_NAMES]


def _save_results(step: str, ns: dict, work: Path) -> None:
    import torch

    cache = work / _CACHE_FOLDER
    cache.mkdir(exist_ok=True)
    values: dict = {}
    for name in STEPS[step].names:
        value = ns[name]
        if isinstance(value, torch.nn.Module):
            torch.save(value.state_dict(), cache / f"{name}.pt")
            values[name] = {"weights": f"{name}.pt"}
        elif _is_path(value):
            values[name] = {"path": str(value)}
        elif name.endswith("loss_history"):
            steps, losses = _helpers().loss_points(value)
            values[name] = {"loss": [[float(s), float(v)] for s, v in zip(steps, losses)]}
        else:
            torch.save(value, cache / f"{name}.obj")
            values[name] = {"object": f"{name}.obj"}
    record = {"file_sha": _sha(work / STEPS[step].file), "budgets": _budgets(ns), "values": values}
    (cache / f"{step}.json").write_text(json.dumps(record))


def _load_model(name: str, weights: Path, ns: dict, labeled_path):
    """A saved model: the teacher as a frozen copy of ``base_model``, the student from ``make_student``."""

    import torch

    teacher = name == "finetuned_model"
    model = copy.deepcopy(ns["base_model"]) if teacher else _helpers().make_student(labeled_path, ns["device"])
    model.load_state_dict(torch.load(weights, map_location=ns["device"], weights_only=True))
    if teacher:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model.eval()


def _load_results(step: str, ns: dict, work: Path) -> bool:
    """Put a passed step's results into ``ns``; ``False`` when there are none for the file as it is now."""

    import torch

    cache = work / _CACHE_FOLDER
    record_file, step_file = cache / f"{step}.json", work / STEPS[step].file
    if not record_file.exists() or not step_file.exists():
        return False
    try:
        record = json.loads(record_file.read_text())
        if record["file_sha"] != _sha(step_file) or record["budgets"] != _budgets(ns):
            return False
        loaded: dict = {}
        for name, value in record["values"].items():
            if "path" in value:
                if not Path(value["path"]).exists():
                    return False
                loaded[name] = Path(value["path"])
            elif "weights" in value:
                labeled_path = loaded.get("labeled_path", ns.get("labeled_path"))
                loaded[name] = _load_model(name, cache / value["weights"], ns, labeled_path)
            elif "loss" in value:
                loaded[name] = [(s, v) for s, v in value["loss"]]
            elif "object" in value:
                loaded[name] = torch.load(cache / value["object"], weights_only=False)
    except Exception:
        return False
    ns.update(loaded)
    return True


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _run_file(ns: dict, path: Path):
    """Run one step file in ``ns``; its own prints go to stderr. Returns the exception, or ``None``."""

    real_out, sys.stdout = sys.stdout, sys.stderr
    try:
        exec(compile(path.read_text(), str(path), "exec"), ns)
    except Exception as exc:  # reported to the learner as one line
        return exc
    finally:
        sys.stdout = real_out
    return None


def _setup_namespace() -> dict:
    ns: dict = {"__name__": "__main__"}
    real_out, sys.stdout = sys.stdout, sys.stderr
    try:
        exec(_helpers().SETUP_CELL, ns)
        override = os.environ.get(_BUDGETS_VARIABLE)
        if override:
            for name, value in zip(_helpers().BUDGET_NAMES, override.split(","), strict=True):
                ns[name] = int(value)
            ns["starts"], ns["test_starts"] = _helpers().load_starts(ns["n_replicas"])
    finally:
        sys.stdout = real_out
    return ns


def _earlier_steps(step: str, ns: dict, work: Path, use_saved: bool) -> tuple[str | None, bool]:
    """Bring the earlier steps into ``ns``. Returns (a problem line or None, whether saved results were used)."""

    used_saved = False
    for earlier in STEPS[step].after:
        if use_saved and _load_results(earlier, ns, work):
            used_saved = True
            continue
        path = work / STEPS[earlier].file
        if not path.exists():
            return f"{step}: the earlier step file {STEPS[earlier].file} is missing in {work}", used_saved
        print(f"  (running the earlier file {path.name})", file=sys.stderr, flush=True)
        error = _run_file(ns, path)
        if error is not None:
            return f"{step}: an earlier file failed: {path.name}: {_first_line(error)}", used_saved
    return None, used_saved


def _error_line(step: str, error: Exception, attempt: int) -> str:
    file = STEPS[step].file
    if isinstance(error, FileExistsError):
        return (
            f"{step}: a store already exists at the path the file writes to; use a fresh path or "
            f"remove the old store first"
        )
    if isinstance(error, NameError) and getattr(error, "name", None) in _SCRIPT_NAMES:
        return f"{step}: {file} uses {error.name}, which does not exist in a notebook cell; remove that line"
    if isinstance(error, NameError) and getattr(error, "name", None):
        return f"{step}: {file} uses {error.name}, which is not defined in an earlier file; define it in this file"
    if attempt == 1:
        return f"{step}: running {file} failed: {_first_line(error)}"
    return f"{step}: running {file} a second time failed: {_first_line(error)}"


def _run_chain(step: str, work: Path, use_saved: bool):
    """Run the setup cell, bring in the earlier steps, then run the step's file once.

    Returns ``(ns, ctx, problem, error, used_saved)``: the namespace, the weight fingerprints taken before the
    step's file ran, a problem line when an earlier step is missing or failed, the exception of the step's file
    or ``None``, and whether saved results stood in for an earlier file.
    """

    ns = _setup_namespace()
    ctx = {"base_weights": _weight_digest(ns["base_model"])}
    problem, used_saved = _earlier_steps(step, ns, work, use_saved)
    if problem:
        return ns, ctx, problem, None, used_saved
    if "finetuned_model" in ns:
        ctx["teacher_weights"] = _weight_digest(ns["finetuned_model"])
    return ns, ctx, None, _run_file(ns, work / STEPS[step].file), used_saved


def check_step(step: str) -> tuple[list[str], list[str]]:
    """Run the chain up to ``step`` and check its contract. Returns (problem lines, unexpected lines)."""

    work = _helpers().work_dir_path()
    path = work / STEPS[step].file
    if not path.exists():
        return [f"{step}: the step file {STEPS[step].file} is missing in {work}"], []
    ns, ctx, problem, error, used_saved = _run_chain(step, work, use_saved=True)
    if isinstance(error, NameError) and used_saved:
        # the name may come from an earlier file's imports: run the earlier files for real
        ns, ctx, problem, error, _ = _run_chain(step, work, use_saved=False)
    if problem:
        return [problem], []
    if error is not None:
        return [_error_line(step, error, 1)], []
    error = _run_file(ns, path)  # a learner runs the cell again
    if error is not None:
        return [_error_line(step, error, 2)], []
    problems = [f"{step}: {n} is missing; the step asks for it" for n in STEPS[step].names if n not in ns]
    if problems:
        return problems, []
    notes: list[str] = []
    try:
        _CONTENT[step](ns, problems, notes, ctx)
    except Exception as exc:
        problems.append(f"{step}: the results could not be checked: {_first_line(exc)}")
    if not problems:
        _save_results(step, ns, work)
    return problems, notes


def resolve_step(name: str) -> str:
    """Return the step a learner's spelling names: a stem, a file name or a path."""

    return Path(name).name.removesuffix(".py")


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "check":
        print("usage: python -m helpers.agent09 check <" + "|".join(STEPS) + ">")
        return 2
    step = resolve_step(argv[2])
    if step not in STEPS:
        print(f"unknown step {step!r}; steps: {', '.join(STEPS)}")
        return 2
    start = time.time()
    problems, notes = check_step(step)
    seen: set[str] = set()
    for line in [*problems, *notes]:
        if line not in seen:
            seen.add(line)
            print(line)
    if not problems:
        count = len(STEPS[step].names)
        print(f"{step}: PASS ({count} name{'s' if count != 1 else ''})")
    print(f"  (check took {time.time() - start:.0f} s)", file=sys.stderr)
    return 1 if problems else 0
