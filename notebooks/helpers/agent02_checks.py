# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step checks for Agent 02, Many molecules, one batch.

Run from ``notebooks/``::

    python -m helpers.agent02 check <step>

The check runs the setup cell, the files of the earlier steps, then the file of
the step (twice, as a learner re-runs a cell), in one namespace, and prints one
plain line per problem. A result that departs from the usual relation is a note;
the step still passes. The words are in ``agent02_text.py``.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .agent02_text import (
    PROVISIONAL_BUILD_ROUTES_REPLACES_OWN_BATCH,
    PROVISIONAL_N_STRUCTURES,
    PROVISIONAL_UNEXPECTED_ORDER_PASSES,
    line,
)

COUNTS = [24, 21, 33, 20, 49]  # caffeine, aspirin, ibuprofen, paracetamol, capsaicin; any order is accepted
SAVED_FIELDS = ("atomic_numbers", "positions", "energy", "forces")  # compared between what was saved and what was read back
SYSTEM_FIELDS = ("energy",)  # one row per structure; the others have one row per atom


@dataclass(frozen=True)
class Step:
    """One step of the notebook: its file, the names the file leaves, and the steps whose files run first."""

    file: str
    names: tuple[str, ...]
    after: tuple[str, ...] = field(default=())


# Names that exist when a file runs as a script but not in a notebook cell (%load): the check says to remove the line
NOT_IN_A_CELL = ("__file__",)

# The setup cell of the notebook, as the check runs it (the call to show_session only prints, so it is left out)
SETUP = (
    "from helpers import agent02 as helpers\n"
    "helpers.configure_tutorial()\n"
    "work_dir = helpers.make_work_dir()\n"
    f"n_structures = {PROVISIONAL_N_STRUCTURES}\n"  # the input row of build_routes names it
)

OPTIONAL_STEPS = ("my_batch", "build_routes")  # the "Try it" steps: a learner may skip them

# `after` lists every earlier step of the notebook except the optional "Try it" steps,
# in notebook order: a file may use names and imports that an earlier cell made.
STEPS: dict[str, Step] = {
    "atomic_data": Step("atomic_data.py", ("caffeine_data",)),
    "batch": Step("batch.py", ("molecule_data", "batch"), after=("atomic_data",)),
    "my_batch": Step("my_batch.py", ("my_batch",), after=("atomic_data", "batch")),
    "build_routes": Step("build_routes.py", ("build_table",), after=("atomic_data", "batch")),
    "model_call": Step("model_call.py", ("aimnet_model", "energy", "forces"), after=("atomic_data", "batch")),
    "zarr_roundtrip": Step("zarr_roundtrip.py", ("store_path", "loaded_batch"), after=("atomic_data", "batch", "model_call")),
    "own_batch": Step("own_batch.py", ("own_batch", "own_energy", "own_forces", "own_store"), after=("atomic_data", "batch", "model_call", "zarr_roundtrip")),
}

# The steps of the notebook, in file order: a learner may name a step by its number, and the line for an
# unknown step lists them. While build_routes takes the place of own_batch (the switch in agent02_text.py),
# the six numbered steps end with zarr_roundtrip; otherwise all seven are numbered.
# own_batch keeps its check either way and can be run by name.
NUMBERED_STEPS = tuple(name for name in STEPS if not (PROVISIONAL_BUILD_ROUTES_REPLACES_OWN_BATCH and name == "own_batch"))


def _counts(batch) -> list[int]:
    return [int(n) for n in batch.num_nodes_per_graph.cpu().tolist()]


def _run_file(ns: dict, work: Path, file: str):
    """Run one file of the work folder in the namespace; return the error, or None."""

    path = work / file
    if not path.exists():
        return FileNotFoundError(line("file_missing", file=file))
    try:
        exec(compile(path.read_text(), str(path), "exec"), ns)
    except Exception as exc:  # reported to the learner as one line
        return exc
    return None


def _first_line(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}".strip()


# ---------------------------------------------------------------------------
# What was saved against what was read back: one function for the view and the check
# ---------------------------------------------------------------------------
def compare_saved_and_read(batch, loaded) -> dict:
    """Compare ``batch`` with ``loaded_batch``.

    A store that was appended to holds more structures than ``batch``: then the last
    ``batch.num_graphs`` structures are compared. The comparison is to float32 precision,
    because the reader may hand back float32 for a float64 field.

    Returns ``{"enough": bool, "tail": bool, "missing": [...], "no_saved": [...], "differs": [...], "same": bool}``.
    """

    import torch

    enough = loaded.num_graphs >= batch.num_graphs and loaded.num_nodes >= batch.num_nodes
    result = {"enough": enough, "tail": loaded.num_graphs > batch.num_graphs, "missing": [], "no_saved": [], "differs": [], "same": False}
    if not enough:
        return result
    for name in SAVED_FIELDS:
        saved, read = getattr(batch, name, None), getattr(loaded, name, None)
        if read is None:
            result["missing"].append(name)
        elif saved is None:
            result["no_saved"].append(name)
        else:
            rows = batch.num_graphs if name in SYSTEM_FIELDS else batch.num_nodes
            saved = saved.detach().cpu().double()
            read = read.detach().cpu().double()[-rows:]
            if saved.shape != read.shape or not bool(torch.allclose(saved, read, rtol=1e-6, atol=1e-6)):
                result["differs"].append(name)
    result["same"] = not (result["missing"] or result["no_saved"] or result["differs"])
    return result


# ---------------------------------------------------------------------------
# The build-routes table
# ---------------------------------------------------------------------------
BUILD_ORDER = ("tensors", "cpu_then_moved", "direct_gpu")  # fastest expected first
TIE_FRACTION = 0.10  # best times closer than this fraction of the slower route count as a tie, shown as unexpected


def _route_of(label: str) -> str | None:
    """The route a row label names, by its words; None when it names none."""

    if "tensor" in label or "raw" in label:
        return "tensors"
    if "cpu" in label:
        return "cpu_then_moved"
    if "gpu" in label or "direct" in label:
        return "direct_gpu"
    return None


def build_route_rows(frame) -> dict[str, int] | None:
    """Map each route to its row by the words in the route's name; None when a route cannot be told apart."""

    for column in [c for c in frame.columns if frame[c].dtype == object]:
        rows: dict[str, int] = {}
        for index, label in enumerate(frame[column].astype(str).str.lower()):
            role = _route_of(label)
            if role is not None:
                rows.setdefault(role, index)
        if len(rows) == 3 and len(set(rows.values())) == 3:
            return rows
    return None


_NOT_A_TIME = ("struct", "atom", "run", "repeat", "count", "graph", "node", "size", "field")  # size columns of the table
_TIME_WORDS = ("best", "min", "median", "mean", "avg", "sec", "time", "_s", "ms", "wall", "duration")


def _time_column(numbers):
    """The column of the numeric part of a build table that holds the times; None when there is none."""

    candidates = [
        c for c in numbers.columns
        if bool((numbers[c] > 0).all()) and bool(numbers[c].map(math.isfinite).all())
        and not any(word in str(c).lower() for word in _NOT_A_TIME)
    ]
    for word in _TIME_WORDS:
        named = next((c for c in candidates if word in str(c).lower()), None)
        if named is not None:
            return named
    decimals = [c for c in candidates if numbers[c].dtype.kind == "f"]
    return decimals[0] if decimals else None


def build_route_times(frame) -> list[float]:
    """The time of each row, in seconds.

    Columns that count something (structures, atoms, runs) are never times. Among the others the column named
    best or min is taken first, then median, mean and any other name that says time; else the first column of
    positive decimals. A column whose name ends in ``ms`` is read as milliseconds.
    """

    numbers = frame.select_dtypes("number")
    chosen = _time_column(numbers)
    if chosen is None:
        return []
    name = str(chosen).lower().rstrip(")] ")
    scale = 1e-3 if name.endswith("ms") else 1.0
    return [float(v) * scale for v in numbers[chosen]]


def build_route_order(table) -> dict:
    """Judge a build_table: the routes, their times, and whether they follow tensors < CPU then moved < directly on the GPU.

    status: "usual", "unexpected" or "unnamed" (the rows cannot be told apart, or have no times).
    ``marked`` lists the routes whose time is out of order against the route expected to be slower or faster.
    """

    import pandas as pd

    frame = pd.DataFrame(table)
    rows = build_route_rows(frame)
    times = build_route_times(frame)
    if rows is None or len(times) != len(frame):
        return {"status": "unnamed", "rows": rows, "times": {}, "marked": []}
    by_role = {role: times[index] for role, index in rows.items()}
    marked: set[str] = set()
    for faster, slower in zip(BUILD_ORDER, BUILD_ORDER[1:]):
        if not by_role[faster] < by_role[slower] * (1 - TIE_FRACTION):
            marked.update((faster, slower))
    return {"status": "unexpected" if marked else "usual", "rows": rows, "times": by_role, "marked": sorted(marked)}


# ---------------------------------------------------------------------------
# The content check of each step. ``add(key, **fields)`` reports one line of agent02_text.CHECK_LINES.
# ---------------------------------------------------------------------------
def _check_tensors(add, expected, why: str) -> None:
    """Each ``(name, value, shape)`` of ``expected`` is a finite tensor of that shape."""

    import torch

    for name, tensor, shape in expected:
        if not isinstance(tensor, torch.Tensor):
            add("not_tensor", name=name, type=type(tensor).__name__)
        elif tuple(tensor.shape) != shape:
            add("tensor_shape", name=name, got=tuple(tensor.shape), why=why, expected=shape)
        elif not bool(torch.isfinite(tensor).all()):
            add("not_finite", name=name)


def _structures_in_store(path) -> int | None:
    """The number of structures in a Zarr store; None when nothing exists at the path."""

    from nvalchemi.data.datapipes import AtomicDataZarrReader

    if not os.path.exists(str(path)):
        return None
    return len(AtomicDataZarrReader(str(path)))


def _check_atomic_data(ns: dict, add) -> None:
    from nvalchemi.data import AtomicData

    data = ns["caffeine_data"]
    if not isinstance(data, AtomicData):
        add("caffeine_type", type=type(data).__name__)
    elif data.num_nodes != COUNTS[0]:
        add("caffeine_atoms", n=data.num_nodes, expected=COUNTS[0])


def _check_batch(ns: dict, add) -> None:
    from nvalchemi.data import AtomicData, Batch

    data_list, batch = ns["molecule_data"], ns["batch"]
    if not (isinstance(data_list, list) and all(isinstance(d, AtomicData) for d in data_list)):
        add("molecule_data_type")
    else:
        got = [d.num_nodes for d in data_list]
        if sorted(got) != sorted(COUNTS):
            add("molecule_data_counts", got=got, expected=COUNTS)
    if not isinstance(batch, Batch):
        add("batch_type", type=type(batch).__name__)
    elif sorted(_counts(batch)) != sorted(COUNTS):
        add("batch_counts", g=batch.num_graphs, got=_counts(batch), expected=COUNTS)


def _check_my_batch(ns: dict, add) -> None:
    from nvalchemi.data import Batch

    mine, five = ns["my_batch"], ns["molecule_data"]
    if not isinstance(mine, Batch):
        add("my_batch_type", type=type(mine).__name__)
        return
    counts = _counts(mine)
    if len(counts) != len(five) + 1:
        add("my_batch_size", g=len(counts), expected=len(five) + 1)
    elif not any(counts[:i] + counts[i + 1:] == [d.num_nodes for d in five] for i in range(len(counts))):
        add("my_batch_order", got=counts)


def _check_build_table(table, add) -> None:
    """build_table: one row per route with a time; the names of the columns are the learner's agent's choice."""

    import pandas as pd

    try:
        frame = pd.DataFrame(table)
    except Exception:
        add("table_type", type=type(table).__name__)
        return
    if len(frame) < 3:
        add("table_rows", n=len(frame))
        return
    times = build_route_times(frame)  # the same pick as the view: size columns are not times
    if not times:
        add("table_no_times")
    names = [c for c in frame.columns if frame[c].dtype == object and frame[c].nunique() >= 3]
    if not names:
        add("table_no_routes")
    elif times and build_route_order(frame)["status"] == "unnamed":
        add("table_unnamed")


def _check_build_routes(ns: dict, add) -> None:
    _check_build_table(ns["build_table"], add)


def _check_model_call(ns: dict, add) -> None:
    import torch

    batch, energy, forces = ns["batch"], ns["energy"], ns["forces"]
    n_graphs, n_atoms = batch.num_graphs, batch.num_nodes
    _check_tensors(add, (("energy", energy, (n_graphs, 1)), ("forces", forces, (n_atoms, 3))), "one call on the batch")
    if isinstance(forces, torch.Tensor) and tuple(forces.shape) == (n_atoms, 3):
        # the forces of an isolated molecule sum to zero; a larger sum says forces and atoms are out of step
        owner = batch.batch_idx.to(forces.device)
        sums = torch.zeros(n_graphs, 3, device=forces.device).index_add_(0, owner, forces.detach().float())
        worst = float(sums.abs().max())
        if worst > 0.05:
            add("forces_sum", worst=worst)


def _check_zarr_roundtrip(ns: dict, add) -> None:
    from nvalchemi.data import Batch

    path, loaded, batch = ns["store_path"], ns["loaded_batch"], ns["batch"]
    stored = _structures_in_store(path)
    if stored is None:
        add("store_path_missing", name="store_path", path=path)
    elif stored < batch.num_graphs:  # an appended store holds more
        add("store_too_small", n=stored, name="batch", m=batch.num_graphs)
    if not isinstance(loaded, Batch):
        add("loaded_type", type=type(loaded).__name__)
        return
    verdict = compare_saved_and_read(batch, loaded)
    if not verdict["enough"]:
        add("loaded_size", g=loaded.num_graphs, a=loaded.num_nodes, g2=batch.num_graphs, a2=batch.num_nodes)
        return
    for name in verdict["missing"]:
        add("loaded_no_field", name=name)
    for name in verdict["no_saved"]:
        add("saved_no_field", name=name)
    for name in verdict["differs"]:
        add("loaded_differs", name=name)


def _check_own_batch(ns: dict, add) -> None:
    from nvalchemi.data import Batch

    own, energy, forces, store = ns["own_batch"], ns["own_energy"], ns["own_forces"], ns["own_store"]
    if not isinstance(own, Batch):
        add("own_batch_type", type=type(own).__name__)
        return
    _check_tensors(add, (("own_energy", energy, (own.num_graphs, 1)), ("own_forces", forces, (own.num_nodes, 3))), "own_batch")
    stored = _structures_in_store(store)
    if stored is None:
        add("store_path_missing", name="own_store", path=store)
    elif stored != own.num_graphs:
        add("store_count_own", n=stored, m=own.num_graphs)


_CONTENT_CHECKS = {
    "atomic_data": _check_atomic_data,
    "batch": _check_batch,
    "my_batch": _check_my_batch,
    "build_routes": _check_build_routes,
    "model_call": _check_model_call,
    "zarr_roundtrip": _check_zarr_roundtrip,
    "own_batch": _check_own_batch,
}


def _check_names(step: str, ns: dict, problems: list[str]) -> bool:
    missing = [n for n in STEPS[step].names if n not in ns]
    for name in missing:
        problems.append(line("missing_name", step=step, name=name))
    return not missing


def _check_content(step: str, ns: dict, problems: list[str]) -> None:
    def add(key: str, **fields) -> None:
        problems.append(line(key, step=step, **fields))

    check = _CONTENT_CHECKS.get(step)
    if check is not None:
        check(ns, add)


# ---------------------------------------------------------------------------
# Running the files of a check
# ---------------------------------------------------------------------------
def _names_left_by(steps, ns: dict) -> dict:
    """The names that the given steps left in the namespace, with their values."""

    return {name: ns[name] for earlier in steps for name in STEPS[earlier].names if name in ns}


def _changed_names(step: str, file: str, before: dict, ns: dict) -> list[str]:
    """One line for each name of ``before`` that ``file`` bound to another object."""

    return [line("changed_name", step=step, file=file, name=name) for name, value in before.items() if ns.get(name) is not value]


def _run_earlier_files(step: str, ns: dict, work: Path, problems: list[str]) -> str | None:
    """Run the files of the earlier steps in notebook order; return the line of a required file that failed.

    A "Try it" file runs only if it exists, and its errors are ignored here. It runs between the numbered
    steps, so it must leave their names as they were: a later step would otherwise run on what it built.
    """

    order = list(STEPS)
    for earlier in order[: order.index(step)]:
        file = STEPS[earlier].file
        if earlier in STEPS[step].after:
            error = _run_file(ns, work, file)
            if error is not None:
                return line("earlier_failed", step=step, file=file, error=_first_line(error))
        elif earlier in OPTIONAL_STEPS and (work / file).exists():
            before = _names_left_by(STEPS[earlier].after, ns)
            _run_file(ns, work, file)
            problems.extend(_changed_names(step, file, before, ns))
    return None


def _run_step_file(step: str, ns: dict, work: Path) -> str | None:
    """Run the file of the step twice, as a learner who runs the cell again; return the line of a failure."""

    file = STEPS[step].file
    for attempt in (1, 2):
        error = _run_file(ns, work, file)
        if error is None:
            continue
        name = getattr(error, "name", None) if isinstance(error, NameError) else None
        if isinstance(error, FileExistsError):
            return line("store_exists", step=step)
        if name in NOT_IN_A_CELL:
            return line("cell_only_name", step=step, file=file, name=name)
        if name:
            return line("name_error", step=step, file=file, name=name)
        if attempt == 1:
            return line("run_failed", step=step, file=file, error=_first_line(error))
        return line("run_failed_second", step=step, file=file, error=_first_line(error))
    return None


def resolve_step(name: str) -> str:
    """Return the step a learner's spelling names: a stem, a file name, a path, or the step's number in file order."""

    step = Path(name).name.removesuffix(".py")
    if step.isdigit() and 1 <= int(step) <= len(NUMBERED_STEPS):
        step = NUMBERED_STEPS[int(step) - 1]
    return step


def run_check(step: str, root: Path | None = None) -> int:
    """Run one step check; print the problems; return 0 when there are none (a note does not fail the step)."""

    step = resolve_step(step)
    if step not in STEPS:
        print(line("unknown_step", name=repr(step), steps=", ".join(NUMBERED_STEPS)))
        return 2
    root = root or Path.cwd()
    work = root / "agent02_work"
    sys.path.insert(0, str(root))
    ns: dict = {"__name__": "__main__"}
    exec(SETUP, ns)
    problems: list[str] = []
    failure = _run_earlier_files(step, ns, work, problems)
    if failure is not None:
        print(failure)
        return 1
    kept = _names_left_by(STEPS[step].after, ns) if step in OPTIONAL_STEPS else {}
    failure = _run_step_file(step, ns, work)
    if failure is not None:
        print(failure)
        return 1
    problems.extend(_changed_names(step, STEPS[step].file, kept, ns))
    if _check_names(step, ns, problems):
        _check_content(step, ns, problems)
    for text in problems:
        print(text)
    if problems:
        return 1
    if step == "build_routes" and build_route_order(ns["build_table"])["status"] == "unexpected":
        print(line("order_note", step=step))  # a note; the switch in agent02_text.py decides whether the step still passes
        if not PROVISIONAL_UNEXPECTED_ORDER_PASSES:
            return 1
    print(line("ok", step=step, n=len(STEPS[step].names)))
    return 0
