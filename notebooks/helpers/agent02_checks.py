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

import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .agent02_text import (
    BUILD_ORDER_LINE,  # noqa: F401  (re-exported for the view and the tests)
    BUILD_ROUTE_TEXT,  # noqa: F401
    PROVISIONAL_BUILD_ROUTES_REPLACES_OWN_BATCH,
    PROVISIONAL_UNEXPECTED_ORDER_PASSES,
    line,
)

COUNTS = [24, 21, 33, 20, 49]  # caffeine, aspirin, ibuprofen, paracetamol, capsaicin; any order is accepted
SAVED_FIELDS = ("atomic_numbers", "positions", "energy", "forces")  # compared between what was saved and what was read back
SYSTEM_FIELDS = ("energy",)  # one row per structure; the others have one row per atom


@dataclass(frozen=True)
class Step:
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
    "n_structures = 1000\n"  # the input row of build_routes names it (revision 26 Nov; equals PROVISIONAL_N_STRUCTURES)
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
# unknown step lists them. PROVISIONAL (a02q-build-routes, switch in agent02_text.py): build_routes replaces
# own_batch, so the six steps end with zarr_roundtrip; as a third Try it all seven are listed.
# own_batch keeps its check either way and can be run by name.
NUMBERED_STEPS = tuple(name for name in STEPS if not (PROVISIONAL_BUILD_ROUTES_REPLACES_OWN_BATCH and name == "own_batch"))


def _counts(batch) -> list[int]:
    return [int(n) for n in batch.num_nodes_per_graph.cpu().tolist()]


def _run_file(ns: dict, work: Path, file: str):
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
    result = {"enough": enough, "tail": loaded.num_graphs > batch.num_graphs, "missing": [], "no_saved": [], "differs": []}
    if enough:
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
    result["same"] = enough and not (result["missing"] or result["no_saved"] or result["differs"])
    return result


# ---------------------------------------------------------------------------
# The build-routes table
# ---------------------------------------------------------------------------
BUILD_ORDER = ("tensors", "cpu_then_moved", "direct_gpu")  # fastest expected first
TIE_FRACTION = 0.10  # best times closer than this fraction of the slower route count as a tie, shown as unexpected


def build_route_rows(frame) -> dict[str, int] | None:
    """Map each route to its row by the words in the route's name; None when a route cannot be told apart."""

    rows: dict[str, int] = {}
    for column in [c for c in frame.columns if frame[c].dtype == object]:
        for index, label in enumerate(frame[column].astype(str).str.lower()):
            if "tensor" in label or "raw" in label:
                role = "tensors"
            elif "cpu" in label:
                role = "cpu_then_moved"
            elif "gpu" in label or "direct" in label:
                role = "direct_gpu"
            else:
                continue
            rows.setdefault(role, index)
        if len(rows) == 3 and len(set(rows.values())) == 3:
            return rows
        rows = {}
    return None


_NOT_A_TIME = ("struct", "atom", "run", "repeat", "count", "graph", "node", "size", "field")  # size columns of the table
_TIME_WORDS = ("best", "min", "median", "mean", "avg", "sec", "time", "_s", "ms", "wall", "duration")


def build_route_times(frame) -> list[float]:
    """The time of each row, in seconds.

    Columns that count something (structures, atoms, runs) are never times. Among the others the column named
    best or min is taken first, then median, mean and any other name that says time; else the first column of
    positive decimals. A column whose name ends in ``ms`` is read as milliseconds.
    """

    numbers = frame.select_dtypes("number")
    candidates = [
        c for c in numbers.columns
        if bool((numbers[c] > 0).all()) and bool(numbers[c].map(lambda v: v == v and v != float("inf")).all())
        and not any(word in str(c).lower() for word in _NOT_A_TIME)
    ]
    chosen = None
    for word in _TIME_WORDS:
        chosen = next((c for c in candidates if word in str(c).lower()), None)
        if chosen is not None:
            break
    if chosen is None:
        decimals = [c for c in candidates if numbers[c].dtype.kind == "f"]
        chosen = decimals[0] if decimals else None
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


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------
def _check_names(step: str, ns: dict, problems: list[str]) -> bool:
    missing = [n for n in STEPS[step].names if n not in ns]
    for name in missing:
        problems.append(line("missing_name", step=step, name=name))
    return not missing


def _check_content(step: str, ns: dict, problems: list[str]) -> None:
    import torch
    from nvalchemi.data import AtomicData, Batch

    def add(key: str, **fields) -> None:
        problems.append(line(key, step=step, **fields))

    def tensor_checks(pairs, why: str) -> None:
        for name, tensor, shape in pairs:
            if not isinstance(tensor, torch.Tensor):
                add("not_tensor", name=name, type=type(tensor).__name__)
            elif tuple(tensor.shape) != shape:
                add("tensor_shape", name=name, got=tuple(tensor.shape), why=why, expected=shape)
            elif not bool(torch.isfinite(tensor).all()):
                add("not_finite", name=name)

    if step == "atomic_data":
        data = ns["caffeine_data"]
        if not isinstance(data, AtomicData):
            add("caffeine_type", type=type(data).__name__)
        elif data.num_nodes != COUNTS[0]:
            add("caffeine_atoms", n=data.num_nodes, expected=COUNTS[0])
    elif step == "batch":
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
    elif step == "my_batch":
        mine, five = ns["my_batch"], ns["molecule_data"]
        if not isinstance(mine, Batch):
            add("my_batch_type", type=type(mine).__name__)
            return
        counts = _counts(mine)
        if len(counts) != len(five) + 1:
            add("my_batch_size", g=len(counts), expected=len(five) + 1)
        elif not any(counts[:i] + counts[i + 1:] == [d.num_nodes for d in five] for i in range(len(counts))):
            add("my_batch_order", got=counts)
    elif step == "build_routes":
        _check_build_table(ns["build_table"], lambda key, **fields: add(key, **fields))
    elif step == "model_call":
        batch, energy, forces = ns["batch"], ns["energy"], ns["forces"]
        n_graphs, n_atoms = batch.num_graphs, batch.num_nodes
        tensor_checks((("energy", energy, (n_graphs, 1)), ("forces", forces, (n_atoms, 3))), "one call on the batch")
        if isinstance(forces, torch.Tensor) and tuple(forces.shape) == (n_atoms, 3):
            owner = batch.batch_idx.to(forces.device)
            sums = torch.zeros(n_graphs, 3, device=forces.device).index_add_(0, owner, forces.detach().float())
            worst = float(sums.abs().max())
            if worst > 0.05:
                add("forces_sum", worst=worst)
    elif step == "zarr_roundtrip":
        from nvalchemi.data.datapipes import AtomicDataZarrReader

        path, loaded, batch = ns["store_path"], ns["loaded_batch"], ns["batch"]
        if not os.path.exists(str(path)):
            add("store_path_missing", name="store_path", path=path)
        else:
            n = len(AtomicDataZarrReader(str(path)))
            if n < batch.num_graphs:  # an appended store holds more
                add("store_too_small", n=n, name="batch", m=batch.num_graphs)
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
    elif step == "own_batch":
        from nvalchemi.data.datapipes import AtomicDataZarrReader

        ob, oe, of, store = ns["own_batch"], ns["own_energy"], ns["own_forces"], ns["own_store"]
        if not isinstance(ob, Batch):
            add("own_batch_type", type=type(ob).__name__)
            return
        tensor_checks((("own_energy", oe, (ob.num_graphs, 1)), ("own_forces", of, (ob.num_nodes, 3))), "own_batch")
        if not os.path.exists(str(store)):
            add("store_path_missing", name="own_store", path=store)
        else:
            n = len(AtomicDataZarrReader(str(store)))
            if n != ob.num_graphs:
                add("store_count_own", n=n, m=ob.num_graphs)


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
    # earlier files in notebook order; the optional Try it file runs only if it exists, and its errors are ignored here
    position = list(STEPS).index(step)
    for earlier in list(STEPS)[:position]:
        if earlier in STEPS[step].after:
            error = _run_file(ns, work, STEPS[earlier].file)
            if error is not None:
                print(line("earlier_failed", step=step, file=STEPS[earlier].file, error=_first_line(error)))
                return 1
        elif earlier in OPTIONAL_STEPS and (work / STEPS[earlier].file).exists():
            before = {name: ns[name] for e in STEPS[earlier].after for name in STEPS[e].names if name in ns}
            _run_file(ns, work, STEPS[earlier].file)
            # a Try it file that replaced an earlier name fails the later step too, which would otherwise run on it
            # (validation, 26 Nov: with the first build_routes sample in place, `check model_call` passed on 50 structures)
            for name, value in before.items():
                if ns.get(name) is not value:
                    problems.append(line("changed_name", step=step, file=STEPS[earlier].file, name=name))
    # a "Try it" file runs between the numbered steps: it must leave their names as they were
    # (the first sample of build_routes named its large batch `batch`, and the model call then ran on it)
    kept = {name: ns[name] for earlier in STEPS[step].after for name in STEPS[earlier].names if name in ns} if step in OPTIONAL_STEPS else {}
    for attempt in (1, 2):  # the second run is a learner running the cell again
        error = _run_file(ns, work, STEPS[step].file)
        if error is None:
            continue
        if isinstance(error, FileExistsError):
            print(line("store_exists", step=step))
        elif isinstance(error, NameError) and getattr(error, "name", None) in NOT_IN_A_CELL:
            print(line("cell_only_name", step=step, file=STEPS[step].file, name=error.name))
        elif isinstance(error, NameError) and getattr(error, "name", None):
            print(line("name_error", step=step, file=STEPS[step].file, name=error.name))
        elif attempt == 1:
            print(line("run_failed", step=step, file=STEPS[step].file, error=_first_line(error)))
            if os.environ.get("A02_TRACE"):
                traceback.print_exception(error)
        else:
            print(line("run_failed_second", step=step, file=STEPS[step].file, error=_first_line(error)))
        return 1
    for name, value in kept.items():
        if ns.get(name) is not value:
            problems.append(line("changed_name", step=step, file=STEPS[step].file, name=name))
    if _check_names(step, ns, problems):
        _check_content(step, ns, problems)
    for text in problems:
        print(text)
    if not problems:
        if step == "build_routes" and build_route_order(ns["build_table"])["status"] == "unexpected":
            print(line("order_note", step=step))  # a note, not a failure: the step still passes (PROVISIONAL, a02q-order-line)
            if not PROVISIONAL_UNEXPECTED_ORDER_PASSES:
                return 1
        print(line("ok", step=step, n=len(STEPS[step].names)))
    return 1 if problems else 0
