# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step contracts for Agent 07, One out, one in.

``python -m helpers.agent07 check <step>`` runs the setup cell, then the files of
the earlier steps and the step's own file in one namespace, and prints one plain
line per problem, or ``<step>: OK``. A result off the expected relation (no
staggered switches, trajectories that leave NVT on its step cap, a refill run
that admitted nothing) is reported in a note as "unexpected" and passes.

Shapes and units are the step's, never the learner's:

- ``fused_batches``, ``staged_batches``, ``relax_md_batches``: the finished
  batches of the route, a list of ``Batch`` (or a store with ``read()``); every
  trajectory once, each batch within ``max_batch_size`` and ``max_atoms``.
- ``completed``: the store (``read()``) or a list of ``Batch``; every trajectory once.
- ``fused_seconds``, ``inflight_seconds``, ``staged_seconds``: seconds, a number.
- ``stage_trace``, ``relax_md_trace``: one row per trajectory in
  ``simulation_id`` order, two integer columns: the MD step of the move to the
  second stage and the MD step of the finish, counted from the trajectory's own
  first step. No trajectory may leave NVT in fewer than ``window`` MD steps (its
  stopping rule averages over the last ``window`` steps), and every trajectory
  runs NVE for its fixed length of ``nve_steps`` MD steps (one step of slack
  either way).
- A store must have a ``read()`` that works; the check says so when it stops.
- ``build_stages()``: no arguments, returns a fresh ``(nvt, nve)`` pair.

"Finished both stages" is counted, not read from ``status``: for the files that
run after step 1 the check hands them a ``build_stages`` that counts the MD
steps every trajectory takes in each stage. A staged run written with plain
engines therefore passes without a ``status`` column.

The study size follows the names of the setup cell. ``ALCHEMI_AGENT07_SMALL=1``
runs the check on a small study (for a CPU). Every line the check prints is in
``LINES``.
"""

from __future__ import annotations

import functools
import math
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .agent07 import BATCH_MULTIPLIER, MAX_ATOMS, MAX_BATCH_SIZE, STAGE_SETTINGS

# The setup cell, in two sizes. "full" is the notebook's; "small" is for a CPU: the first
# complexes of the set (17 trajectories) and a capacity they do not fit in, so that every route
# runs several groups or refills, with shorter stages.
SETUP_VALUES = {
    "full": dict(
        batch_multiplier=BATCH_MULTIPLIER,
        max_batch_size=MAX_BATCH_SIZE,
        max_atoms=MAX_ATOMS,
        stage_settings=dict(STAGE_SETTINGS),
    ),
    "small": dict(
        batch_multiplier=0.1,
        max_batch_size=6,
        max_atoms=110,
        stage_settings=dict(STAGE_SETTINGS, window=50, nvt_steps=200, nve_steps=50),
    ),
}

_SETUP = (
    "import torch\n"
    "from helpers import agent07 as helpers\n"
    "helpers.configure_tutorial()\n"
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "batch_multiplier = {batch_multiplier!r}\n"
    "max_batch_size = {max_batch_size!r}\n"
    "max_atoms = {max_atoms!r}\n"
    "# NVT ends once the mean temperature over the last `window` MD steps is within `tolerance` K\n"
    "# of `temperature`, or at `nvt_steps`; NVE then runs `nve_steps` MD steps. `dt` is the timestep\n"
    "# in fs, `friction` the Langevin friction per fs.\n"
    "stage_settings = dict({settings})\n"
    "trajectories, composed_model = helpers.prepare_study(\n"
    "    device, batch_multiplier=batch_multiplier)\n"
    "work_dir = helpers.make_work_dir()\n"
)

# Names a script has but a notebook cell does not (a %load cell runs without them).
NOT_IN_A_CELL = ("__file__",)

# Every line the check prints.
LINES = {
    "usage": "usage: python -m helpers.agent07 check <step>\nsteps: {steps}",
    "ok": "{step}: OK",
    "missing_file": "{step}: {file} is missing in {folder}",
    "stops": "{step}: {file} stops with {error}",
    "notebook_only": "{step}: {file} uses {name}, which does not exist in a notebook cell; remove that line",
    "missing_name": "{step}: {name} is missing",
    "check_stops": "{step}: check stops with {error}",
    "not_batches": (
        "{name} must be the finished batches of the route, a list of Batch (or a store with "
        "read()), each trajectory with its simulation_id; got {got}"
    ),
    "read_stops": (
        "{name} is a store ({got}), but its read() stops with {error}; keep the finished "
        "trajectories in a store whose read() returns them as one Batch, or as a list of Batch"
    ),
    "not_once": "{name} holds {count} trajectories, {distinct} distinct; expected {total}, each once",
    "over_capacity": (
        "{name}: one batch holds {graphs} trajectories and {atoms} atoms; the capacity is "
        "max_batch_size = {max_batch_size} and max_atoms = {max_atoms}, and every route runs at it"
    ),
    "seconds": "{name} must be the run time in seconds, a number above 0; got {got}",
    "trace_shape": (
        "{name} has shape {shape}, expected ({total}, 2): one row per trajectory in simulation_id "
        "order, column 0 the MD step of the move to the second stage, column 1 the MD step of the "
        "finish, both counted from the trajectory's own first step"
    ),
    "trace_order": (
        "{name}: {low} trajectories have a switch step of 0 or less and {late} a finish step before "
        "the switch step (row {row}: switch {switch}, finish {finish}); both count MD steps from "
        "the trajectory's own first step"
    ),
    "early": (
        "{name}: {count} of {total} trajectories left {stage} after fewer than `window` = {window} MD "
        "steps in it (row {row}: {steps}); its stopping rule averages over the last `window` MD "
        "steps of the stage, so none can pass sooner"
    ),
    "nve_length": (
        "{name}: {count} of {total} trajectories ran NVE for other than `nve_steps` = {length} MD "
        "steps (row {row}: {steps}); NVE has a fixed length: every trajectory runs `nve_steps` MD "
        "steps in it"
    ),
    "pair": "build_stages() must take no arguments and return a fresh (nvt, nve) pair of stages; {got}",
    "changed_input": (
        "{file} changed `trajectories` (positions, velocities or status); start every route from "
        "a copy, for example [t.clone() for t in trajectories]"
    ),
    "no_build_stages": (
        "{file} did not call build_stages() from step 1; every route runs the same stages"
    ),
    "not_both": (
        "{count} of {total} trajectories did not take a step in both stages "
        "({first} without NVT, {second} without NVE)"
    ),
    "active_over": (
        "the active batch held up to {graphs} trajectories and {atoms} atoms; the capacity is "
        "max_batch_size = {max_batch_size} and max_atoms = {max_atoms}"
    ),
    "no_refill": (
        "no trajectory entered a batch that others were still running in, although {total} do "
        "not fit in one batch: the run did not refill"
    ),
    "note_same": (
        "{name}: every trajectory switched at step {step}: unexpected. Does the first stage have "
        "its own stopping rule?"
    ),
    "note_cap": (
        "{count} of {total} trajectories left {stage} at its step cap of {cap} without passing "
        "the rule: unexpected"
    ),
    "note_cap_differs": (
        "build_stages() gives {stage} a step cap of {got}; stage_settings has {want}"
    ),
    "note_length_differs": (
        "build_stages() gives NVE a length of {got} steps; stage_settings has nve_steps = {want}"
    ),
    "note_fits": "all {total} trajectories fit in the first batch, so none was admitted later: unexpected",
    "note_store": "{name} is not the course store, so the late entries are not recorded in the table",
}


@dataclass
class _Watch:
    """What the stages of one file did: MD steps per trajectory and stage, and the batch sizes."""

    total: int
    calls: int = 0
    steps: list = field(default_factory=list)
    max_graphs: int = 0
    max_atoms: int = 0
    seen: torch.Tensor | None = None  # per trajectory: has been in a batch of the first stage
    joined: torch.Tensor | None = None  # trajectories that entered a batch others were already in

    def count(self, which: int) -> torch.Tensor:
        while len(self.steps) <= which:
            self.steps.append(torch.zeros(self.total, dtype=torch.long))
        return self.steps[which]


class _WatchHook:
    """Counts the MD steps of every trajectory in one stage (no host transfer during the run)."""

    frequency = 1

    def __init__(self, watch: _Watch, which: int) -> None:
        from nvalchemi.dynamics import DynamicsStage

        self.stage = DynamicsStage.AFTER_STEP
        self.watch, self.which = watch, which

    def __call__(self, ctx: Any, stage: Any) -> None:
        batch, watch = ctx.batch, self.watch
        count = batch.num_graphs
        ids = batch.simulation_id.reshape(-1)[:count].long()
        steps = watch.count(self.which)
        if steps.device != ids.device:
            steps = watch.steps[self.which] = steps.to(ids.device)
        active = ctx.active_graph_mask
        steps[ids] += 1 if active is None else active.long()
        watch.max_graphs = max(watch.max_graphs, int(count))
        watch.max_atoms = max(watch.max_atoms, int(batch.num_nodes))
        if self.which == 0:
            if watch.seen is None:
                watch.seen = torch.zeros(watch.total, dtype=torch.bool, device=ids.device)
                watch.joined = torch.zeros((), dtype=torch.long, device=ids.device)
            new = ~watch.seen[ids]
            # a refill: newcomers arrive in a batch that still holds earlier trajectories
            watch.joined += new.sum() * (~new).any()
            watch.seen[ids] = True


def _watched(build: Callable, watch: _Watch) -> Callable:
    """``build_stages`` that also counts what its stages do."""

    @functools.wraps(build)
    def build_stages(*args, **kwargs):
        pair = build(*args, **kwargs)
        watch.calls += 1
        for which, stage in enumerate(pair):
            stage.register_hook(_WatchHook(watch, which))
        return pair

    build_stages._agent07_watch = watch
    return build_stages


def _fingerprint(trajectories) -> list[torch.Tensor]:
    """Copies of the positions, velocities and status of every trajectory."""

    out = []
    for item in trajectories:
        for name in ("positions", "velocities", "status"):
            value = getattr(item, name, None)
            if value is not None:
                out.append(value.detach().cpu().clone())
    return out


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


def _finished(ns: dict, name: str, *, capacity: bool) -> list[str]:
    """Every trajectory once in ``name``; each batch within the capacity when ``capacity``."""

    from . import agent07 as helpers

    total = len(ns["trajectories"])
    if hasattr(ns[name], "read"):
        try:
            ns[name].read()
        except Exception:
            return [LINES["read_stops"].format(name=name, got=type(ns[name]).__name__, error=_last_line())]
    try:
        batches = helpers._batches(ns[name])
        ids = helpers._ids(ns[name])
    except Exception:
        return [LINES["not_batches"].format(name=name, got=type(ns[name]).__name__)]
    distinct = int(torch.unique(ids).numel())
    if ids.numel() != total or distinct != total:
        return [LINES["not_once"].format(name=name, count=ids.numel(), distinct=distinct, total=total)]
    if capacity and not hasattr(ns[name], "read"):
        for batch in batches:
            graphs, atoms = int(batch.num_graphs), int(batch.num_nodes)
            if graphs > ns["max_batch_size"] or atoms > ns["max_atoms"]:
                return [
                    LINES["over_capacity"].format(
                        name=name,
                        graphs=graphs,
                        atoms=atoms,
                        max_batch_size=ns["max_batch_size"],
                        max_atoms=ns["max_atoms"],
                    )
                ]
    return []


def _seconds(ns: dict, name: str) -> list[str]:
    """``name`` is a run time: a finite number above 0."""

    value = ns[name]
    try:
        number = float(value)
    except Exception:
        number = float("nan")
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        return [LINES["seconds"].format(name=name, got=repr(value)[:60])]
    return []


def _trace(ns: dict, name: str) -> tuple[list[str], list[str]]:
    """The shape and the order of a trace as problems; switches all at one step as a note."""

    total = len(ns["trajectories"])
    try:
        trace = torch.as_tensor(ns[name]).detach().cpu()
    except Exception:
        return [LINES["trace_shape"].format(name=name, shape=type(ns[name]).__name__, total=total)], []
    if tuple(trace.shape) != (total, 2) or trace.is_floating_point() and bool((trace != trace.round()).any()):
        return [LINES["trace_shape"].format(name=name, shape=tuple(trace.shape), total=total)], []
    switch, finish = trace[:, 0].long(), trace[:, 1].long()
    bad = (switch <= 0) | (finish < switch)
    if bool(bad.any()):
        row = int(bad.float().argmax())
        return [
            LINES["trace_order"].format(
                name=name,
                low=int((switch <= 0).sum()),
                late=int((finish < switch).sum()),
                row=row,
                switch=int(switch[row]),
                finish=int(finish[row]),
            )
        ], []
    notes = []
    if total > 1 and int(switch.min()) == int(switch.max()):
        notes.append(LINES["note_same"].format(name=name, step=int(switch.min())))
    return [], notes


def _cap_notes(ns: dict, first_steps: torch.Tensor) -> list[str]:
    """Trajectories that left NVT at its step cap, when the setup cell has stage_settings.

    NVE has a fixed length (``nve_steps``), so every trajectory ends it there: no note.
    """

    settings = ns.get("stage_settings")
    if not isinstance(settings, dict) or settings.get("nvt_steps") is None:
        return []
    cap, total = int(settings["nvt_steps"]), int(first_steps.numel())
    count = int((first_steps.cpu() >= cap).sum())
    if count:
        return [LINES["note_cap"].format(count=count, total=total, stage="NVT", cap=cap)]
    return []


def _stage_pair(ns: dict) -> tuple[list[str], list[str]]:
    """``build_stages()`` returns a pair of stages; a step cap or a length off ``stage_settings`` is a note."""

    notes: list[str] = []
    try:
        pair = ns["build_stages"]()
        if len(pair) != 2 or not all(callable(getattr(s, "run", None)) for s in pair):
            raise TypeError(f"it returned {type(pair).__name__} of {len(pair)}")
        settings = ns.get("stage_settings")
        if isinstance(settings, dict):
            got = getattr(pair[0], "n_steps", None)
            if "nvt_steps" in settings and got != settings["nvt_steps"]:
                notes.append(
                    LINES["note_cap_differs"].format(stage="NVT", got=got, want=settings["nvt_steps"])
                )
            got = getattr(pair[1], "n_steps", None)
            if "nve_steps" in settings and got != settings["nve_steps"]:
                notes.append(LINES["note_length_differs"].format(got=got, want=settings["nve_steps"]))
    except Exception as error:
        return [LINES["pair"].format(got=str(error).splitlines()[0][:120])], notes
    return [], notes


def _early(ns: dict, name: str, first_steps: torch.Tensor) -> list[str]:
    """NVT left before its rule had `window` MD steps to average over.

    A rule over the last `window` MD steps cannot pass sooner. One step of slack
    allows for traces that count from 0. A rule on the temperature of a single
    step lets a trajectory leave NVT this early, and is caught here.
    """

    settings = ns.get("stage_settings")
    if not isinstance(settings, dict) or not settings.get("window"):
        return []
    window, cap = int(settings["window"]), settings.get("nvt_steps")
    early = first_steps < window - 1
    if cap is not None:
        early &= first_steps < int(cap)  # a step cap below the window is the cap, not the rule
    if bool(early.any()):
        row = int(early.float().argmax())
        return [
            LINES["early"].format(
                name=name, count=int(early.sum()), total=int(first_steps.numel()), stage="NVT",
                window=window, row=row, steps=int(first_steps[row]),
            )
        ]
    return []


def _nve_length(ns: dict, name: str, second_steps: torch.Tensor) -> list[str]:
    """NVE ran for other than its fixed length; one step of slack either way for the counting."""

    settings = ns.get("stage_settings")
    if not isinstance(settings, dict) or settings.get("nve_steps") is None:
        return []
    length = int(settings["nve_steps"])
    off = (second_steps - length).abs() > 1
    if bool(off.any()):
        row = int(off.float().argmax())
        return [
            LINES["nve_length"].format(
                name=name, count=int(off.sum()), total=int(second_steps.numel()), length=length,
                row=row, steps=int(second_steps[row]),
            )
        ]
    return []


def _check_fused(ns: dict, watch: _Watch | None) -> tuple[list[str], list[str]]:
    del watch  # the stages of step 1 are read from the trace of the file itself
    problems = _finished(ns, "fused_batches", capacity=True) + _seconds(ns, "fused_seconds")
    trace_problems, notes = _trace(ns, "stage_trace")
    pair_problems, pair_notes = _stage_pair(ns)
    problems += trace_problems + pair_problems
    notes += pair_notes
    if not trace_problems:
        trace = torch.as_tensor(ns["stage_trace"]).long()
        problems += _early(ns, "stage_trace", trace[:, 0])
        problems += _nve_length(ns, "stage_trace", trace[:, 1] - trace[:, 0])
        notes += _cap_notes(ns, trace[:, 0])
    return problems, notes


def _both_stages(ns: dict, watch: _Watch, file: str) -> tuple[list[str], list[str]]:
    """Every trajectory took a step in both stages of the watched ``build_stages``."""

    if watch.calls == 0:
        return [LINES["no_build_stages"].format(file=file)], []
    first, second = watch.count(0).cpu(), watch.count(1).cpu()
    no_first, no_second = int((first == 0).sum()), int((second == 0).sum())
    if no_first or no_second:
        missing = int(((first == 0) | (second == 0)).sum())
        return [
            LINES["not_both"].format(count=missing, total=watch.total, first=no_first, second=no_second)
        ], []
    return [], _cap_notes(ns, first)


def _check_refill(ns: dict, watch: _Watch) -> tuple[list[str], list[str]]:
    problems = _finished(ns, "completed", capacity=False) + _seconds(ns, "inflight_seconds")
    both, notes = _both_stages(ns, watch, STEPS["refill"].file)
    problems += both
    if both:
        return problems, notes
    if watch.max_graphs > ns["max_batch_size"] or watch.max_atoms > ns["max_atoms"]:
        problems.append(
            LINES["active_over"].format(
                graphs=watch.max_graphs,
                atoms=watch.max_atoms,
                max_batch_size=ns["max_batch_size"],
                max_atoms=ns["max_atoms"],
            )
        )
    total = watch.total
    late = int(watch.joined) if watch.joined is not None else 0
    sizes = [int(item.num_nodes) for item in ns["trajectories"]]
    fits = total <= ns["max_batch_size"] and sum(sizes) <= ns["max_atoms"]
    if late == 0:
        if fits:
            notes.append(LINES["note_fits"].format(total=total))
        else:
            problems.append(LINES["no_refill"].format(total=total))
    if getattr(ns["completed"], "admitted_inflight", None) is None:
        notes.append(LINES["note_store"].format(name="completed"))
    return problems, notes


def _check_staged(ns: dict, watch: _Watch) -> tuple[list[str], list[str]]:
    problems = _finished(ns, "staged_batches", capacity=True) + _seconds(ns, "staged_seconds")
    # No cap notes here: in a staged run a trajectory keeps stepping while it waits for its group,
    # so its step count does not say whether its own rule passed.
    both, _ = _both_stages(ns, watch, STEPS["staged"].file)
    return problems + both, []


def _check_relax_md(ns: dict, watch: _Watch | None) -> tuple[list[str], list[str]]:
    del watch
    problems = _finished(ns, "relax_md_batches", capacity=True)
    trace_problems, notes = _trace(ns, "relax_md_trace")
    return problems + trace_problems, notes


@dataclass(frozen=True)
class Step:
    """One step of the notebook: its file, the names the file leaves, its contract, and the steps whose files run first."""

    file: str
    names: tuple[str, ...]
    check: Callable[[dict, Any], tuple[list[str], list[str]]]
    after: tuple[str, ...] = ()


STEPS: dict[str, Step] = {
    "fused": Step(
        "fused.py", ("build_stages", "fused_batches", "fused_seconds", "stage_trace"), _check_fused
    ),
    "refill": Step("refill.py", ("completed", "inflight_seconds"), _check_refill, after=("fused",)),
    "staged": Step("staged.py", ("staged_batches", "staged_seconds"), _check_staged, after=("fused",)),
    "relax_md": Step("relax_md.py", ("relax_md_batches", "relax_md_trace"), _check_relax_md),
}


def setup_source(size: str | None = None) -> str:
    """The text of the setup cell the check runs, for ``size`` "full" or "small"."""

    if size is None:
        size = "small" if os.environ.get("ALCHEMI_AGENT07_SMALL") else "full"
    values = SETUP_VALUES[size]
    settings = ", ".join(f"{k}={v!r}" for k, v in values["stage_settings"].items())
    return _SETUP.format(settings=settings, **{k: v for k, v in values.items() if k != "stage_settings"})


def _order(step: str) -> list[str]:
    """``step`` and the steps whose files run before it, in the order they run."""

    seen: list[str] = []

    def visit(name: str) -> None:
        for earlier in STEPS[name].after:
            visit(earlier)
        if name not in seen:
            seen.append(name)

    visit(step)
    return seen


def _last_line() -> str:
    """The last line of the traceback of the exception being handled."""

    return traceback.format_exc().strip().splitlines()[-1][:300]


def _run_file(name: str, ns: dict, work_dir: Path) -> tuple[str | None, _Watch | None]:
    """Run the file of step ``name`` in ``ns``; return (problem line or None, what its stages did).

    When ``ns`` holds ``build_stages`` from step 1, the file gets one that counts the MD steps of
    its stages. A file that replaces it with its own leaves an empty count.
    """

    spec = STEPS[name]
    path = work_dir / spec.file
    if not path.exists():
        return LINES["missing_file"].format(step=name, file=spec.file, folder=work_dir), None
    build = ns.get("build_stages")
    watch = None
    if callable(build):
        watch = _Watch(total=len(ns["trajectories"]))
        ns["build_stages"] = _watched(getattr(build, "__wrapped__", build), watch)
    before = _fingerprint(ns["trajectories"])
    try:
        exec(compile(path.read_text(), str(path), "exec"), ns)
    except Exception as error:
        if isinstance(error, NameError) and getattr(error, "name", None) in NOT_IN_A_CELL:
            return LINES["notebook_only"].format(step=name, file=spec.file, name=error.name), None
        return LINES["stops"].format(step=name, file=spec.file, error=_last_line()), None
    if not _same(before, _fingerprint(ns["trajectories"])):
        return f"{name}: " + LINES["changed_input"].format(file=spec.file), None
    if getattr(ns.get("build_stages"), "_agent07_watch", None) is not watch:
        watch = _Watch(total=len(ns["trajectories"])) if name != "fused" else None
    return None, watch


def check_namespace(step: str, ns: dict, work_dir: Path) -> tuple[list[str], list[str]]:
    """Run the files of ``step`` in ``ns`` (the setup cell has run); return (problems, notes)."""

    watch: _Watch | None = None
    for name in _order(step):
        problem, watch = _run_file(name, ns, Path(work_dir))
        if problem is not None:
            return [problem], []
    spec = STEPS[step]
    problems = [LINES["missing_name"].format(step=step, name=n) for n in spec.names if n not in ns]
    notes: list[str] = []
    if not problems:
        try:
            found, notes = spec.check(ns, watch)
            problems = [f"{step}: {line}" for line in found]
            notes = [f"{step}: {line}" for line in notes]
        except Exception:
            problems = [LINES["check_stops"].format(step=step, error=_last_line())]
    return problems, notes


def check_step(step: str, work_dir: Path | None = None, size: str | None = None) -> tuple[list[str], list[str]]:
    """Run the setup cell and the files of ``step``; return (problems, notes)."""

    from . import agent07 as helpers

    work_dir = Path(work_dir or helpers.WORK_DIR)
    ns: dict[str, Any] = {"__name__": "__main__"}
    exec(setup_source(size), ns)
    return check_namespace(step, ns, work_dir)


def main(argv: list[str]) -> int:
    """Entry point of ``python -m helpers.agent07 check <step>``."""

    step = Path(argv[1]).name.removesuffix(".py") if len(argv) == 2 else None
    if len(argv) != 2 or argv[0] != "check" or step not in STEPS:
        print(LINES["usage"].format(steps=", ".join(STEPS)))
        return 2
    problems, notes = check_step(step)
    print("\n".join(problems or [LINES["ok"].format(step=step)]))
    if notes:
        print("\n".join(notes))
    return 1 if problems else 0
