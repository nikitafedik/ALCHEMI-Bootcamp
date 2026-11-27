# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step contracts and the step check of Agent 08, Paths of least resistance, batched.

Run from ``notebooks/``::

    python -m helpers.agent08 check <step>

The check runs the setup cell, the files of the earlier steps, then the file
of the step twice (a learner runs a cell again), all in one namespace as the
notebook does after ``%load``. Then it checks the names the step must create.
It prints one plain line per problem, or ``<step>: OK``. Lines marked
``note`` report what the check measured; they never fail a step. A result
that departs from the expected relation (a path that is not converged, a top
image that is not a saddle point) is reported in a note as "unexpected" and
the step passes. The contracts hold what the step tables leave out: shapes,
units and order. Every line the check can print is in ``LINES``.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import agent08_views as views

# The setup cell of the notebook, run before every step file.
NEB_SETTINGS_LINE = (
    "neb_settings = dict(num_images=8, fmax=0.003, regular_fmax=0.03, n_steps=600)"
    "  # images per path, both ends; eV/Å; eV/Å; steps"
)
SETUP_CELL = (
    "# Load the course helpers, the model, four reactions and the NEB settings\n"
    "import torch\n"
    "from helpers import agent08 as helpers\n"
    "helpers.configure_tutorial()             # course display defaults\n"
    "work_dir = helpers.make_work_dir()       # notebooks/agent08_work\n"
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    "model = helpers.load_model(device)\n"
    "reactants, products, reaction_names = helpers.load_reactions(model)\n"
    f"{NEB_SETTINGS_LINE}\n"
    "helpers.show_session()                   # prints the session lines"
)

END_TOLERANCE = 1.0e-3  # Å, an end of a path against the structure the loader gave
ENERGY_TOLERANCE = 5.0e-3  # eV, a learner's energy against a fresh model call
# eV/Å, the largest force on an atom of a structure the table calls a saddle point. The number is
# fixed here and does not follow the force at which the NEB stopped. On the softest path, the
# ethane rotation, a top image with this force is 2.4 degrees from the eclipsed saddle and
# 0.006 kcal/mol below it; a top that stopped 5 degrees short carries 0.0098 eV/Å and is refused.
SADDLE_FORCE_TOLERANCE = 5.0e-3
# cm^-1. The check recounts the imaginary modes of every image from the model's projected Hessian. A mode below
# -50 counts as imaginary; one between -50 and -10 may be counted either way, so a table's own cutoff in that band
# passes. With the notebook's model every vibrational mode of the 32 images is at least 105 cm^-1 from zero,
# so the count does not depend on the cutoff.
IMAGINARY_SURE, IMAGINARY_MAYBE = 50.0, 10.0
KCAL_PER_EV = views.KCAL_PER_EV


@dataclass(frozen=True)
class Step:
    """One step of the notebook: its file, the names the file leaves, and the steps whose files run first."""

    file: str
    names: tuple[str, ...]
    after: tuple[str, ...] = ()


# The contract of each step is in its `_check_*` function below and, in words, in solutions/agent-08/solutions.md.
STEPS: dict[str, Step] = {
    "paths": Step("paths.py", ("paths",)),
    "neb": Step(
        "neb.py", ("relaxed_paths", "neb_log", "path_energy", "barriers", "converged"), after=("paths",)
    ),
    "saddle": Step("saddle.py", ("saddle_table",), after=("paths", "neb")),
}
ALIASES = {"barriers": "neb"}  # the barriers are read in the step neb; the check takes this name for it
# Names a script has and a notebook cell does not; a file that uses one is told to drop that line.
NOT_IN_A_CELL = frozenset({"__file__"})

# Every line the check prints to the learner.
LINES = {
    # common
    "setup_failed": "the notebook's setup cell failed: {error}",
    "step_file_missing": "the step file {file} is missing in {folder}",
    "earlier_file_missing": "the earlier step file {file} is missing in {folder}",
    "earlier_file_failed": "an earlier file failed: {file}: {error}",
    "undefined_name": "{file} uses {name}, which no earlier cell or file defines; define it in this file",
    "not_in_a_cell": "{file} uses {name}, which does not exist in a notebook cell; remove that line",
    "run_failed": "running {file}{again} failed: {error}",
    "name_missing": "{name} is missing; the step asks for it",
    "alias": "the barriers are read in the step neb now; checking neb",
    # paths
    "paths_type": "paths is {type}; it should be one Batch",
    "paths_no_groups": "paths has no groups; interpolate_paths returns one Batch with one group of images per path",
    "paths_count": "paths holds {got} paths; the {want} reactions give {want}, all in one Batch",
    "paths_images": "{name} has {got} images; neb_settings asks for {want}, both ends included",
    "paths_atoms": "an image of {name} has {got} atoms; the reactant has {want}",
    "paths_end": (
        "the {which} image of {name} is not the {word} the loader gave (largest difference {off:.3f} Å); "
        "the ends are already laid over each other, so build the path without aligning them again"
    ),
    "paths_no_field": (
        "paths has no {field}; interpolate_paths drops energy, forces and velocities, "
        "and the NEB run reads all three. Attach them to paths"
    ),
    "paths_field_shape": "paths.{field} has shape {got}; one row per atom gives {want}",
    "paths_energy_size": "paths.energy has {got} values; one per image gives {want}",
    # neb
    "relaxed_type": "relaxed_paths is {type}; it should be one Batch, as NEB.run returns",
    "relaxed_groups": (
        "relaxed_paths has paths of {got} images; paths has {want}. One NEB call takes the whole Batch and returns it"
    ),
    "no_neb_call": "neb.py made no NEB.run call with model; the step asks for one NEB call on the Batch of paths",
    "many_neb_calls": "neb.py called NEB.run with model {count} times; one call relaxes every path in the Batch",
    "no_climbing": "the NEB ran without a climbing image; pass climbing=ClimbingImageConfig(...) to NEB",
    "log_type": "neb_log is {type}; it should be the path of the diagnostics file the NEB run wrote",
    "log_paths": "neb_log holds {got} paths; the run should log all {want}",
    "second_run": (
        "running neb.py a second time left a log with a single step: the first run changed paths "
        "itself, so the second run started from finished paths. Give the run a copy, "
        "paths.clone(), so that the cell can be run again"
    ),
    "not_moved": "relaxed_paths equals the straight-line paths; the NEB did not move any image",
    "no_threshold": (
        "the check cannot read the force at which this NEB stops a path, so it cannot test converged; "
        "pass fmax=neb_settings['fmax'] to NEB"
    ),
    "energy_type": "path_energy is {type}; it should hold one energy per image, in eV",
    "energy_size": "path_energy has {got} values; one per image of relaxed_paths gives {want}",
    "energy_off": (
        "path_energy differs from the model's energy of the images by up to {off:.3g} eV; "
        "the contract is the energy of every image in eV, in the order of relaxed_paths"
    ),
    "per_path_type": "{name} is {type}; it should hold one value per path",
    "per_path_size": "{name} has {got} values; one per path gives {want}",
    "barrier_off": (
        "the barrier of {name} is {got:.4g}; the forward barrier, the highest image "
        "between the ends minus the first image, is {want:.4f} eV"
    ),
    "converged_off": (
        "converged is {flag} for {name}; its last force in the log is "
        "{force:.4f} eV/Å and the run stops a path at {fmax:g} eV/Å"
    ),
    "note_settings": "the run stops a path at {got:g} eV/Å; neb_settings has fmax {want:g}",
    "note_converged": (
        "{name}: barrier {barrier:.2f} kcal/mol, converged at step {step} with force {force:.4f} eV/Å"
    ),
    "note_not_converged": (
        "{name}: barrier {barrier:.2f} kcal/mol, not converged: unexpected. The path ended at step {step} "
        "with force {force:.4f} eV/Å, above {fmax:g}; its barrier is an estimate"
    ),
    "note_stopped": "{name} stopped at step {step} with force {force:.4f} eV/Å",
    # saddle
    "table_type": "saddle_table is {type}; it should be a table with one row per image",
    "table_rows": "saddle_table has {got} rows; one per image of relaxed_paths gives {want}",
    "table_no_answer": "saddle_table has no column of yes or no; Done when asks for a yes or no for every image",
    "table_no_numbers": "saddle_table holds no numbers behind the answers; Done when asks for them",
    "table_no_path": (
        "saddle_table has no column that names the path of each row; add one that holds the names in reaction_names"
    ),
    "table_no_image": (
        "saddle_table has no column with the image number of each row; add one that counts the images of each path from 0"
    ),
    "table_no_count": (
        "saddle_table has no column with the count of imaginary modes of each image; add one of whole numbers "
        "with \"imaginary\" in its name: a saddle point has exactly one imaginary frequency, and the check reads that count"
    ),
    "table_images": "saddle_table does not list every image of {name} once; the path has images 0 to {last}",
    "end_is_yes": (
        "{name}: the table says yes for the {which} image, an end of the path; "
        "the ends are relaxed minima, so the answer for an end is no"
    ),
    "yes_above_tolerance": (
        "{name}, image {image}: the table says yes, and the largest force on an atom there is {force:.4f} eV/Å; "
        "a saddle point has forces near zero, here at or below {tol:g} eV/Å whatever force the NEB stopped at"
    ),
    "yes_not_top": (
        "{name}, image {image}: the table says yes, and the image is not higher in energy than both its "
        "neighbours on the path; a saddle point on a path is a top: the images on both sides are lower"
    ),
    "yes_imaginary_count": (
        "{name}, image {image}: the table says yes, and its column {column} gives {count} imaginary modes; "
        "a saddle point has exactly one imaginary frequency"
    ),
    "note_top": (
        "{name}: highest image {image}, your answer {said}, largest force {force:.4f} eV/Å; the top of the "
        "fitted path is {distance:.2f} Å along the path from it and {below:.3f} kcal/mol higher"
    ),
    "note_top_unexpected": ". Unexpected: the top image of a converged path is usually a saddle point.",
    "imaginary_count_off": (
        "the column {column} differs from the check's count of imaginary modes at {count} of {total} images, first at "
        "{name}, image {image}: {said} in the table, {want} in the check, which projects the rigid translations and "
        "rotations out of the model's Hessian before it counts"
    ),
    "note_no_hessian": "the check could not take the model's Hessian ({error}), so it did not recount the imaginary modes",
    "ok": "{step}: OK ({count} names)",
    "usage": "usage: python -m helpers.agent08 check <{steps}>",
}


# ---------------------------------------------------------------------------
# Small tools
# ---------------------------------------------------------------------------


_array = views._array


def _type(value) -> str:
    return type(value).__name__


def _is_batch(value) -> bool:
    from nvalchemi.data import Batch

    return isinstance(value, Batch)


def _groups(batch) -> list[int] | None:
    layout = getattr(batch, "group_layout", None)
    return None if layout is None else [int(n) for n in layout.num_graphs_per_group.tolist()]


def _field(batch, name):
    try:
        return getattr(batch, name)
    except (AttributeError, KeyError):
        return None


def _split(values: np.ndarray, counts: list[int]) -> list[np.ndarray]:
    return np.split(values, np.cumsum(counts)[:-1])


def _images(batch) -> list[np.ndarray]:
    """Positions of every image, in Å, one array per image."""

    counts = [int(n) for n in batch.num_nodes_per_graph.tolist()]
    return _split(_array(batch.positions).astype(float).reshape(-1, 3), counts)


def _fresh_call(model, batch) -> tuple[np.ndarray, list[np.ndarray]]:
    """Energy of every image (eV) and the model's forces on it (eV/Å), from a fresh call."""

    work = batch.clone()
    if hasattr(model, "make_neighbor_hooks"):  # a fresh neighbor list for the model at hand
        from .model_support import prepare_model_batch

        work = prepare_model_batch(work, model)
    out = model(work)
    counts = [int(n) for n in batch.num_nodes_per_graph.tolist()]
    forces = _array(out["forces"]).astype(float).reshape(-1, 3)
    return _array(out["energy"]).astype(float).reshape(-1), _split(forces, counts)


def path_top(positions: list[np.ndarray], energy: np.ndarray, forces: list[np.ndarray]) -> dict:
    """Where the highest image of one path sits against the top of the path.

    The energy between two neighbouring images is fitted with a cubic that
    takes the two energies and the model's forces along the line between
    them. Returns the index of the highest image between the ends, the
    largest force on one of its atoms (norm of the force vector, eV/Å), the
    distance along the path from that image to the highest point of the
    fitted profile (Å), and how much higher that point is (kcal/mol).
    """

    count = len(positions)
    top = 1 + int(np.argmax(energy[1:-1])) if count > 2 else int(np.argmax(energy))
    along = np.concatenate(
        [[0.0], np.cumsum([np.linalg.norm(positions[i + 1] - positions[i]) for i in range(count - 1)])]
    )
    best_energy, best_place = float(energy[top]), float(along[top])
    x = np.linspace(0.0, 1.0, 201)
    h00, h10 = 2 * x**3 - 3 * x**2 + 1, x**3 - 2 * x**2 + x
    h01, h11 = -2 * x**3 + 3 * x**2, x**3 - x**2
    for i in range(count - 1):
        step = positions[i + 1] - positions[i]
        length = float(np.linalg.norm(step))
        if length < 1.0e-9:
            continue
        tangent = step / length
        slope0 = -float((forces[i] * tangent).sum())
        slope1 = -float((forces[i + 1] * tangent).sum())
        fitted = h00 * energy[i] + h10 * length * slope0 + h01 * energy[i + 1] + h11 * length * slope1
        k = int(np.argmax(fitted))
        if fitted[k] > best_energy:
            best_energy, best_place = float(fitted[k]), float(along[i] + x[k] * length)
    return {
        "image": top,
        "force": float(np.linalg.norm(forces[top], axis=1).max()),
        "distance": abs(best_place - float(along[top])),
        "below": (best_energy - float(energy[top])) * KCAL_PER_EV,
    }


CM_PER_SQRT_EV_A2_AMU = 521.47  # sqrt(eV / (Å^2 amu)) in cm^-1


def projected_frequencies(hessian: np.ndarray, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Vibrational frequencies of one structure, in cm^-1, sorted; an imaginary one is negative.

    ``hessian`` is the (3n, 3n) Cartesian Hessian in eV/Å^2. It is mass-weighted, the rigid
    translations and rotations are projected out (by rank, so a linear molecule loses five), and
    the as many eigenvalues closest to zero are dropped.
    """

    n = len(positions)
    h = 0.5 * (hessian + hessian.T)
    w = np.repeat(np.sqrt(np.asarray(masses, dtype=float)), 3)
    hm = h / np.outer(w, w)
    r = positions - (masses[:, None] * positions).sum(0) / masses.sum()
    vecs = []
    for a in range(3):
        t = np.zeros((n, 3))
        t[:, a] = 1.0
        vecs.append(t)
        e = np.zeros(3)
        e[a] = 1.0
        vecs.append(np.cross(np.broadcast_to(e, (n, 3)), r))
    rigid = np.stack([v.reshape(-1) * w for v in vecs], axis=1)
    u, s, _ = np.linalg.svd(rigid, full_matrices=False)
    u = u[:, s > 1.0e-6 * s.max()]
    p = np.eye(3 * n) - u @ u.T
    values = np.linalg.eigvalsh(p @ hm @ p)
    keep = np.sort(np.argsort(np.abs(values))[u.shape[1]:])
    vib = values[keep]
    return np.sign(vib) * np.sqrt(np.abs(vib)) * CM_PER_SQRT_EV_A2_AMU


def image_frequencies(model, batch) -> list[np.ndarray] | str:
    """The projected frequencies of every image of ``batch`` from ``model.compute_hessian``.

    Returns one array per image, or the first line of the error when the
    model gives no Hessian.
    """

    try:
        from .model_support import prepare_model_batch

        work = prepare_model_batch(batch.clone(), model)
        hessian = _array(model.compute_hessian(work)["hessian"]).astype(float)
    except Exception as exc:  # noqa: BLE001 - reported to the learner as a note
        return _first_line(exc)
    counts = [int(n) for n in batch.num_nodes_per_graph.tolist()]
    masses = _array(batch.atomic_masses).astype(float).reshape(-1)
    out, atom, row = [], 0, 0
    for n, positions in zip(counts, _images(batch)):
        h = hessian[row : row + n * n].reshape(n, n, 3, 3).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
        out.append(projected_frequencies(h, positions, masses[atom : atom + n]))
        atom, row = atom + n, row + n * n
    return out


class _WatchNEB:
    """Record every ``NEB.run`` call while step files run."""

    def __init__(self) -> None:
        self.runs: list = []

    def on(self, model) -> list:
        """The runs on one model; an IDPP pre-pass runs on another and is not counted."""

        return [neb for neb in self.runs if getattr(neb, "model", None) is model]

    def __enter__(self):
        from nvalchemi.dynamics.mep import NEB

        self._cls, self._own = NEB, NEB.__dict__.get("run")
        original, runs = NEB.run, self.runs

        def run(neb, *args, **kwargs):
            runs.append(neb)
            return original(neb, *args, **kwargs)

        NEB.run = run
        return self

    def __exit__(self, *exc) -> None:
        if self._own is None:
            del self._cls.run
        else:
            self._cls.run = self._own


# ---------------------------------------------------------------------------
# The contracts
# ---------------------------------------------------------------------------


def _check_paths(ns: dict, ctx: dict) -> tuple[list[str], list[str]]:
    paths, names = ns["paths"], list(ns["reaction_names"])
    if not _is_batch(paths):
        return [LINES["paths_type"].format(type=_type(paths))], []
    groups = _groups(paths)
    if groups is None:
        return [LINES["paths_no_groups"]], []
    if len(groups) != len(names):
        return [LINES["paths_count"].format(got=len(groups), want=len(names))], []
    problems = []
    want_images = int(ns["neb_settings"]["num_images"])
    images, first = _images(paths), 0
    for name, count, reactant, product in zip(names, groups, ns["reactants"], ns["products"]):
        mine = images[first : first + count]
        first += count
        if count != want_images:
            problems.append(LINES["paths_images"].format(name=name, got=count, want=want_images))
        if any(len(image) != len(reactant) for image in mine):
            problems.append(LINES["paths_atoms"].format(name=name, got=len(mine[0]), want=len(reactant)))
            continue
        for image, end, word in ((mine[0], reactant, "reactant"), (mine[-1], product, "product")):
            off = float(np.abs(image - end.positions).max())
            if off > END_TOLERANCE:
                which = "first" if word == "reactant" else "last"
                problems.append(LINES["paths_end"].format(which=which, name=name, word=word, off=off))
    for field, shape in (("energy", None), ("forces", (paths.num_nodes, 3)), ("velocities", (paths.num_nodes, 3))):
        value = _field(paths, field)
        if value is None:
            problems.append(LINES["paths_no_field"].format(field=field))
        elif shape is not None and tuple(value.shape) != shape:
            problems.append(LINES["paths_field_shape"].format(field=field, got=tuple(value.shape), want=shape))
        elif shape is None and int(value.numel()) != paths.num_graphs:
            problems.append(LINES["paths_energy_size"].format(got=int(value.numel()), want=paths.num_graphs))
    return problems, []


def _log_rows(ns: dict) -> list[dict] | str:
    """The stop of every path in the learner's log, or one line saying what is wrong with it."""

    log = ns["neb_log"]
    if not isinstance(log, (str, os.PathLike)):
        return LINES["log_type"].format(type=_type(log))
    try:
        data = views.read_neb_log(log)
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)
    if len(data) != len(ns["reaction_names"]):
        return LINES["log_paths"].format(got=len(data), want=len(ns["reaction_names"]))
    return views.neb_log_data(log, ns["reaction_names"])


def _run_threshold(neb) -> float | None:
    """The force threshold, in eV/Å, at which the final stage of ``neb`` stops a path.

    ``NEB(fmax=...)`` leaves ``neb.convergence_hook`` at None and the run builds its hook from
    ``fmax``. A ``convergence_hook`` given to NEB takes precedence over ``fmax``; its threshold
    is read when it is a single force-norm criterion, the metric of the log's ``fmax`` column.
    Any other hook gives None: the check cannot compare the log with it, and says so.
    """

    if neb is None:
        return None
    hook = neb.convergence_hook
    if hook is None:
        return float(neb.fmax)
    criteria = list(getattr(hook, "criteria", []))
    if len(criteria) != 1:
        return None
    c = criteria[0]
    if c.key != "forces" or c.reduce_op != "norm" or c.custom_op is not None:
        return None
    return float(c.threshold)


def _per_path(ns: dict, name: str, count: int):
    """``barriers`` as floats or ``converged`` as booleans, or one line saying what is wrong."""

    try:
        if name == "barriers":
            values = views._flat(ns[name])
        elif isinstance(ns[name], (list, tuple)):
            values = np.asarray([bool(_array(v).reshape(-1)[0]) for v in ns[name]])
        else:
            values = np.asarray([bool(v) for v in _array(ns[name]).reshape(-1)])
    except (TypeError, ValueError, IndexError):
        return LINES["per_path_type"].format(name=name, type=_type(ns[name]))
    if values.size != count:
        return LINES["per_path_size"].format(name=name, got=values.size, want=count)
    return values


def _check_neb(ns: dict, ctx: dict) -> tuple[list[str], list[str]]:
    """The merged step: the NEB run, then the barrier and the converged flag of every path."""

    relaxed, names = ns["relaxed_paths"], list(ns["reaction_names"])
    problems, notes = [], []
    if not _is_batch(relaxed):
        return [LINES["relaxed_type"].format(type=_type(relaxed))], []
    groups = _groups(relaxed)
    if groups != ctx["groups"]:
        return [LINES["relaxed_groups"].format(got=groups, want=ctx["groups"])], []
    calls = ctx["neb_calls"]
    if calls == 0:
        problems.append(LINES["no_neb_call"])
    elif calls > 1:
        problems.append(LINES["many_neb_calls"].format(count=calls))
    neb = ctx["neb"]
    if neb is not None and neb.climbing is None:
        problems.append(LINES["no_climbing"])
    rows = _log_rows(ns)
    if isinstance(rows, str):
        return problems + [rows], notes
    first = ctx.get("first_log")
    rerun_broken = False
    if isinstance(first, list) and max(r["stop_step"] for r in first) > 0 and max(r["stop_step"] for r in rows) == 0:
        problems.append(LINES["second_run"])
        rows = first
        # converged now comes from the second run, which started from finished paths, so it is not
        # compared with the log of the first run.
        rerun_broken = True
    if not problems and ctx.get("start") is not None:
        moved = max(float(np.abs(a - b).max()) for a, b in zip(_images(relaxed), ctx["start"]))
        if moved < 1.0e-6:
            problems.append(LINES["not_moved"])
    fmax = _run_threshold(neb)
    if neb is not None and fmax is None:
        problems.append(LINES["no_threshold"])
    want_fmax = float(ns["neb_settings"]["fmax"])
    if fmax is not None and abs(fmax - want_fmax) > 1.0e-12:
        notes.append(LINES["note_settings"].format(got=fmax, want=want_fmax))

    # the energies, the barriers and the flags
    try:
        energy = views._flat(ns["path_energy"])
    except (TypeError, ValueError):
        return problems + [LINES["energy_type"].format(type=_type(ns["path_energy"]))], notes
    if energy.size != sum(groups):
        return problems + [LINES["energy_size"].format(got=energy.size, want=sum(groups))], notes
    fresh, _ = _fresh_call(ns["model"], relaxed)
    worst = float(np.abs(energy - fresh).max())
    energy_ok = bool(np.isfinite(worst) and worst <= ENERGY_TOLERANCE)
    if not energy_ok:
        problems.append(LINES["energy_off"].format(off=worst))
    source = energy if energy_ok else fresh  # the learner's own energies once they are right
    expected, start = [], 0
    for count in groups:
        mine = source[start : start + count]
        expected.append(float(mine[1:-1].max() - mine[0]) if count > 2 else float(mine.max() - mine[0]))
        start += count
    barriers, flags = _per_path(ns, "barriers", len(names)), _per_path(ns, "converged", len(names))
    problems += [value for value in (barriers, flags) if isinstance(value, str)]
    if not isinstance(barriers, str):
        for name, got, want in zip(names, barriers, expected):
            if not np.isfinite(got) or abs(got - want) > ENERGY_TOLERANCE:
                problems.append(LINES["barrier_off"].format(name=name, got=got, want=want))
    for k, (name, row) in enumerate(zip(names, rows)):
        if fmax is None:
            notes.append(LINES["note_stopped"].format(name=name, step=row["stop_step"], force=row["stop_force"]))
            continue
        below = row["stop_force"] <= fmax
        if not isinstance(flags, str) and not rerun_broken and bool(flags[k]) != below:
            problems.append(LINES["converged_off"].format(flag=bool(flags[k]), name=name, force=row["stop_force"], fmax=fmax))
        line = LINES["note_converged"] if below else LINES["note_not_converged"]
        notes.append(
            line.format(name=name, barrier=expected[k] * KCAL_PER_EV, step=row["stop_step"], force=row["stop_force"], fmax=fmax)
        )
    return problems, notes


saddle_answers = views.saddle_answers


def _image_rows(frame, names: list[str], groups: list[int]) -> tuple[list[int] | None, str | None, str | None, str | None]:
    """The row of the table that belongs to each image of the batch.

    Returns (rows, problem line, path column, image column). The table must
    name the path and the image of every row; the image column counts each
    path from 0 or from 1.
    """

    path_col = views.path_column(frame, names)
    if path_col is None:
        return None, LINES["table_no_path"], None, None
    image_col = views.image_column(frame, path_col)
    if image_col is None:
        return None, LINES["table_no_image"], path_col, None
    shift = 1 if int(frame[image_col].min()) == 1 else 0
    where: dict = {}
    for row, (p, i) in enumerate(zip(frame[path_col], frame[image_col])):
        where.setdefault((p, int(i) - shift), []).append(row)
    for name, count in zip(names, groups):
        if any(len(where.get((name, k), [])) != 1 for k in range(count)):
            return None, LINES["table_images"].format(name=name, last=count - 1), path_col, image_col
    return [where[(name, k)][0] for name, count in zip(names, groups) for k in range(count)], None, path_col, image_col


def imaginary_count_column(frame) -> str | None:
    """The table's column that counts imaginary modes per image, if it has one.

    A column of whole numbers whose name says "imag" and does not name a
    frequency or wavenumber (``n_imaginary``, ``imaginary modes``, ``n_imag``).
    """

    return next((column for column in frame.columns if views._is_count(frame, column)), None)


def _check_saddle(ns: dict, ctx: dict) -> tuple[list[str], list[str]]:
    relaxed, names = ns["relaxed_paths"], list(ns["reaction_names"])
    groups = _groups(relaxed)
    try:
        frame = views.saddle_frame(ns["saddle_table"])
    except Exception:
        return [LINES["table_type"].format(type=_type(ns["saddle_table"]))], []
    if len(frame) != sum(groups):
        return [LINES["table_rows"].format(got=len(frame), want=sum(groups))], []
    problems = []
    found = saddle_answers(frame)
    if found is None:
        return [LINES["table_no_answer"]], []
    answers = found[1]
    rows, problem, _, image_col = _image_rows(frame, names, groups)
    numbers = [c for c in frame.columns if frame[c].dtype.kind in "fiu" and c != image_col]
    if not numbers:
        problems.append(LINES["table_no_numbers"])
    if rows is None:
        return problems + [problem], []
    energy, forces = _fresh_call(ns["model"], relaxed)
    largest = [float(np.linalg.norm(f, axis=1).max()) for f in forces]
    count_col = imaginary_count_column(frame)
    if count_col is None:  # without the count a table would skip the curvature test
        problems.append(LINES["table_no_count"])
    images, notes, first = _images(relaxed), [], 0
    for name, count in zip(names, groups):
        mine = slice(first, first + count)
        for end, word in ((first, "first"), (first + count - 1, "last")):
            if answers[rows[end]]:
                problems.append(LINES["end_is_yes"].format(name=name, which=word))
        for k in range(1, count - 1):
            i = first + k
            if not answers[rows[i]]:
                continue
            # A saddle point has forces near zero and exactly one imaginary frequency. The force is measured
            # here; the curvature is held by the path (a saddle on a path is higher than both neighbours) and by
            # the table's own count of imaginary modes.
            if largest[i] > SADDLE_FORCE_TOLERANCE:
                problems.append(
                    LINES["yes_above_tolerance"].format(name=name, image=k, force=largest[i], tol=SADDLE_FORCE_TOLERANCE)
                )
            if not (energy[i] > energy[i - 1] and energy[i] > energy[i + 1]):
                problems.append(LINES["yes_not_top"].format(name=name, image=k))
            if count_col is not None:
                said = frame[count_col].iloc[rows[i]]
                if said != 1:
                    problems.append(LINES["yes_imaginary_count"].format(name=name, image=k, column=count_col, count=said))
        top = path_top(images[mine], energy[mine], forces[mine])
        said = answers[rows[first + top["image"]]]
        note = LINES["note_top"].format(
            name=name, image=top["image"], said="yes" if said else "no", force=top["force"],
            distance=top["distance"], below=top["below"],
        )
        notes.append(note + ("" if said else LINES["note_top_unexpected"]))
        first += count
    if count_col is not None:
        problems += _recount(ns["model"], relaxed, frame, rows, count_col, names, groups, notes)
    return problems, notes


def _recount(model, relaxed, frame, rows, count_col, names, groups, notes) -> list[str]:
    """One line when the table's count of imaginary modes differs from the model's projected Hessian."""

    if not hasattr(model, "compute_hessian"):
        return []
    freqs = image_frequencies(model, relaxed)
    if isinstance(freqs, str):
        notes.append(LINES["note_no_hessian"].format(error=freqs))
        return []
    labels = [(name, k) for name, count in zip(names, groups) for k in range(count)]
    off = []
    for i, ((name, k), f) in enumerate(zip(labels, freqs)):
        said = frame[count_col].iloc[rows[i]]
        sure, maybe = int((f < -IMAGINARY_SURE).sum()), int((f < -IMAGINARY_MAYBE).sum())
        if not sure <= said <= maybe:
            off.append((name, k, said, sure if said < sure else maybe))
    if not off:
        return []
    name, k, said, want = off[0]
    return [LINES["imaginary_count_off"].format(
        column=count_col, count=len(off), total=len(labels), name=name, image=k, said=said, want=want)]


CHECKS = {"paths": _check_paths, "neb": _check_neb, "saddle": _check_saddle}


# ---------------------------------------------------------------------------
# Running the files
# ---------------------------------------------------------------------------


def _run_file(path: Path, ns: dict) -> Exception | None:
    """Run one step file in the notebook's namespace; its prints go to stderr."""

    started = time.time()
    try:
        with contextlib.redirect_stdout(sys.stderr):  # the file's prints must not mix with the verdict
            exec(compile(path.read_text(), str(path), "exec"), ns)
    except Exception as exc:  # reported to the learner as one line
        return exc
    print(f"  ({path.name} ran in {time.time() - started:.1f} s)", file=sys.stderr, flush=True)
    return None


def _first_line(exc: Exception) -> str:
    """The type of an exception and the first line of its message."""

    text = str(exc).splitlines()[0][:200] if str(exc) else ""
    return f"{type(exc).__name__}: {text}".rstrip(": ")


def resolve_step(name: str) -> str:
    """Return the step a learner's spelling names: a stem, a file name, a path, or its number."""

    step = Path(name).name.removesuffix(".py")
    if step.isdigit() and 1 <= int(step) <= len(STEPS):
        step = list(STEPS)[int(step) - 1]
    return ALIASES.get(step, step)


def check_step(step: str) -> tuple[list[str], list[str]]:
    """Run the chain up to ``step`` and check its contract. Returns (problems, notes)."""

    spec = STEPS[step]
    ns: dict = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            exec(SETUP_CELL, ns)
    except Exception as exc:  # for example a model checkpoint that fails to load: one line, not a traceback
        return [f"{step}: " + LINES["setup_failed"].format(error=_first_line(exc))], []
    work = Path(ns["work_dir"])
    ctx: dict = {}
    with _WatchNEB() as watch:
        for earlier in spec.after:
            path = work / STEPS[earlier].file
            if not path.exists():
                return [f"{step}: " + LINES["earlier_file_missing"].format(file=path.name, folder=work)], []
            error = _run_file(path, ns)
            if error is not None:
                return [f"{step}: " + LINES["earlier_file_failed"].format(file=path.name, error=_first_line(error))], []
        path = work / spec.file
        if not path.exists():
            return [f"{step}: " + LINES["step_file_missing"].format(file=spec.file, folder=work)], []
        if step == "neb" and _is_batch(ns.get("paths")):
            ctx["groups"], ctx["start"] = _groups(ns["paths"]), _images(ns["paths"])
        before = len(watch.on(ns.get("model")))
        for attempt in (1, 2):  # the second run is a learner running the cell again
            error = _run_file(path, ns)
            if error is not None:
                if isinstance(error, NameError) and getattr(error, "name", None):
                    key = "not_in_a_cell" if error.name in NOT_IN_A_CELL else "undefined_name"
                    return [f"{step}: " + LINES[key].format(file=spec.file, name=error.name)], []
                again = " a second time" if attempt == 2 else ""
                return [f"{step}: " + LINES["run_failed"].format(file=spec.file, again=again, error=_first_line(error))], []
            if attempt == 1:
                ctx["neb_calls"] = len(watch.on(ns.get("model"))) - before
                if step == "neb" and "neb_log" in ns:
                    ctx["first_log"] = _log_rows(ns)
    runs = watch.on(ns.get("model"))
    ctx["neb"] = runs[-1] if runs else None
    missing = [name for name in spec.names if name not in ns]
    if missing:
        return [f"{step}: " + LINES["name_missing"].format(name=name) for name in missing], []
    problems, notes = CHECKS[step](ns, ctx)
    return [f"{step}: {line}" for line in problems], [f"{step}: note: {line}" for line in notes]


def main(argv: list[str]) -> int:
    """Entry point of ``python -m helpers.agent08 check <step>``; ``argv`` starts with the program name."""

    step = resolve_step(argv[2]) if len(argv) == 3 and argv[1] == "check" else None
    if step not in STEPS:
        print(LINES["usage"].format(steps="|".join(STEPS)), file=sys.stderr)
        return 2
    asked = Path(argv[2]).name.removesuffix(".py")
    if asked in ALIASES:
        print(f"{asked}: note: " + LINES["alias"])
    started = time.time()
    problems, notes = check_step(step)
    for line in notes + list(dict.fromkeys(problems)):
        print(line)
    if not problems:
        print(LINES["ok"].format(step=step, count=len(STEPS[step].names)))
    print(f"  (check took {time.time() - started:.1f} s)", file=sys.stderr)
    return 1 if problems else 0
