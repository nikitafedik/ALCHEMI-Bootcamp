# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step check of Agent 03, Ready-made.

Run from ``notebooks/``::

    python -m helpers.agent03 check <step>

The check runs the setup cell, the files of the earlier steps in contract order, then the
step's file, all in one namespace as the notebook does after ``%load``. Then it checks the
names the step must create and prints one plain line per problem: what was expected, what
was found. A line that starts with "unexpected:" is a note; the step still passes. Exit
code 0 passes, 1 fails.

In order below: the settings and the contracts (``STEPS``), the check of one name, the checks
that look across names (one group per step, listed in ``STEP_CHECKS``), the three watches that
record what a step file does while it runs, and the runner (``check_step``, ``main``). The
words of every line are in ``agent03.MESSAGES``.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agent03 import (
    LADDER,
    SCAN_BLOWUP_EV_PER_ATOM,
    SCAN_BLOWUP_EV_PER_COPY,
    SPEED_CASES,
    SPEED_FLAGS,
    SPEED_ROLES,
    SPEED_TIMES,
    _column,
    _flag_value,
    _say,
    _speed_case_words,
    ladder_data,
    prepare_model_batch,
    scan_blowup_rows,
    speed_notes,
)

# The settings of the setup cell, in one place: the numbers the notebook's setup cell sets.
SETTINGS = {"n_crystals": 200, "n_temperatures": 12}
# The check reads these environment variables in place of the numbers above (short runs on a CPU).
SETTINGS_ENV = {"n_crystals": "ALCHEMI_AGENT03_N_CRYSTALS", "n_temperatures": "ALCHEMI_AGENT03_N_TEMPERATURES"}


def setup_cell() -> str:
    """The notebook's setup cell as the check runs it, with the settings in force."""

    values = {k: int(os.environ.get(SETTINGS_ENV[k]) or v) for k, v in SETTINGS.items()}
    return (
        "from helpers import agent03 as helpers\n"
        "helpers.configure_tutorial()\n"
        "work_dir = helpers.make_work_dir()\n"
        f"n_crystals = {values['n_crystals']}\n"
        f"n_temperatures = {values['n_temperatures']}\n"
        "helpers.show_session()"
    )


SETUP_CELL = setup_cell()


@dataclass
class Name:
    """The contract of one name a step file leaves."""

    kind: str  # "Batch", "model", "tensor", "number" or "any"
    shape: tuple | None = None  # for tensors: ints or the symbols below
    unit: str = ""
    rule: str = ""  # the rest of the contract in words, for the reader; STEP_CHECKS holds its checks
    finite: bool = False
    positive: bool = False
    increasing: bool = False
    integer: bool = False


@dataclass
class Step:
    """One step: its file, the steps whose files run first, the names it leaves, what the check records."""

    file: str
    after: list
    names: dict
    watch_md_batch: bool = False  # record md_batch before the file runs, to see which atoms moved
    watch_models: bool = False  # record every MACE model the file calls
    watch_dynamics: bool = False  # record when each user hook fires, in simulated time
    watch_runs: bool = False  # record every dynamics run: its timesteps and its starting batch


# Shape symbols: n_structures = n_crystals; n_saved = len(time_fs).
STEPS = {
    "relax_build": Step(
        "relax_build.py",
        after=[],
        names={
            "crystal_batch": Name("Batch", rule="n_crystals graphs, every graph periodic with a cell"),
            "mace_model": Name("model", rule="a Toolkit model"),
            "start_max_force": Name("tensor", shape=("n_structures",), unit="eV/Å", finite=True),
        },
    ),
    "relax_run": Step(
        "relax_run.py",
        after=["relax_build"],
        names={
            "relaxed_batch": Name("Batch", rule="the same graphs as crystal_batch"),
            "relaxed_max_force": Name(
                "tensor",
                shape=("n_structures",),
                unit="eV/Å",
                finite=True,
                rule="every value at or below force_threshold",
            ),
            "force_threshold": Name("number", unit="eV/Å"),
            "batch_seconds": Name("number", unit="s", positive=True),
            "single_seconds": Name("tensor", shape=(3,), unit="s", finite=True, positive=True),
        },
        watch_models=True,
        watch_runs=True,
    ),
    "md_build": Step(
        "md_build.py",
        after=["relax_build", "relax_run"],
        names={
            "md_batch": Name(
                "Batch",
                rule="the same graphs as crystal_batch, velocities present, no graph marked finished",
            ),
            "start_temperature": Name(
                "tensor", shape=("n_structures",), unit="K", finite=True, positive=True
            ),
        },
    ),
    "md_run": Step(
        "md_run.py",
        after=["relax_build", "relax_run", "md_build"],
        names={
            "time_fs": Name("tensor", shape=("n_saved",), unit="fs", finite=True, increasing=True),
            "crystal_energy": Name(
                "tensor", shape=("n_saved", "n_structures"), unit="eV", finite=True
            ),
        },
        watch_md_batch=True,
        watch_dynamics=True,
    ),
    "timestep_scan": Step(
        # No cells in the notebook. It needs only mace_model (and a crystal, which the learner may
        # bring or take from crystal_batch), so it runs after relax_build alone: the relaxation
        # and the md files would add minutes to the check and nothing it uses.
        "timestep_scan.py",
        after=["relax_build"],
        names={
            "replica_dt": Name(
                "tensor", shape=("n_replicas",), unit="fs", finite=True, positive=True,
                rule="one timestep for each copy; the values differ",
            ),
            "saved_step": Name(
                "tensor", shape=("n_saved",), unit="steps", finite=True, increasing=True, integer=True,
            ),
            "total_energy": Name(
                "tensor", shape=("n_saved", "n_replicas"), unit="eV",
                rule="total energy of each copy; the first row is finite; a copy that blew up may turn non-finite",
            ),
        },
        watch_runs=True,
    ),
    "temperature_ladder": Step(
        # Like the timestep scan it needs only mace_model and a crystal, so it runs after
        # relax_build alone. The rules and numbers are in LADDER.
        "temperature_ladder.py",
        after=["relax_build"],
        names={
            "time_fs": Name("tensor", shape=("n_saved",), unit="fs", finite=True, increasing=True),
            "replica_temperature": Name(
                "tensor", shape=("n_temperatures",), unit="K", finite=True, positive=True,
                rule="one temperature for each copy; the values differ",
            ),
            "replica_energy": Name(
                "tensor", shape=("n_saved", "n_temperatures"), unit="eV/atom",
                rule="the model's potential energy of each copy, per atom; the first row is finite; "
                "a copy that blew up may turn non-finite",
            ),
        },
        watch_runs=True,
        watch_dynamics=True,
    ),
    "speed_settings": Step(
        "speed_settings.py",
        after=["relax_build"],
        names={
            "speed_table": Name(
                "any",
                rule="a pandas DataFrame, one row for each case (cueq and compile off or on); "
                "columns cueq and compile (True or False), first_call_s and repeat_s (s, finite, above 0)",
            ),
            "setting_roles": Name(
                "any", rule="a dict with a non-empty text for each of the keys cueq and compile"
            ),
        },
        watch_models=True,
    ),
}


def _tensor(value):
    import torch

    try:
        return torch.as_tensor(value).detach().cpu()
    except Exception:
        return None


def _is_numeric(t) -> bool:
    """True for a tensor of floats or of 32 or 64 bit integers; a tensor of True and False is not numbers."""
    import torch

    return t is not None and t.dtype != torch.bool and (t.is_floating_point() or t.dtype in (torch.int32, torch.int64))


def _max_per_graph(values, graph_of_atom, n_graphs):
    """The largest of ``values`` (one per atom, float64 on the host) in each graph."""
    import torch

    out = torch.zeros(n_graphs, dtype=torch.float64)
    return out.scatter_reduce_(0, graph_of_atom, values, reduce="amax", include_self=False)


def _describe(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        return _say("describe_shape", kind=type(value).__name__, shape=tuple(shape))
    if hasattr(value, "__len__") and not isinstance(value, str):
        return _say("describe_length", kind=type(value).__name__, n=len(value))
    return type(value).__name__


def _symbols(ns, step=""):
    syms = {"n_structures": int(ns["n_crystals"])}
    if "n_temperatures" in ns:
        try:
            syms["n_temperatures"] = int(ns["n_temperatures"])
        except (TypeError, ValueError):
            pass
    # n_saved is the length of time_fs (md_run, temperature_ladder) or of saved_step
    # (timestep_scan); n_replicas is the length of replica_dt, so total_energy is judged
    # against both.
    own = {"timestep_scan": "saved_step"}.get(step, "time_fs")
    t = _tensor(ns[own]) if own in ns else None
    if t is not None and t.dim() >= 1:
        syms["n_saved"] = int(t.shape[0])
    if step == "timestep_scan":
        r = _tensor(ns["replica_dt"]) if "replica_dt" in ns else None
        if r is not None and r.dim() >= 1:
            syms["n_replicas"] = int(r.shape[0])
    return syms


def _shape_words(shape, syms):
    return (
        "("
        + ", ".join(str(s) for s in shape)
        + (",)" if len(shape) == 1 else ")")
        + " = ("
        + ", ".join(str(syms.get(s, s)) for s in shape)
        + (",)" if len(shape) == 1 else ")")
    )


def _check_name(step, name, spec, ns, syms):
    """Problems for one name, as a list of plain lines."""
    import torch
    from nvalchemi.data import Batch
    from nvalchemi.models.base import BaseModelMixin

    if name not in ns:
        return [_say("missing", step=step, name=name)]
    value = ns[name]
    unit = _say("unit", unit=spec.unit) if spec.unit else ""
    if spec.kind == "Batch":
        if not isinstance(value, Batch):
            return [_say("not_batch", step=step, name=name, found=_describe(value))]
        return []
    if spec.kind == "model":
        if not isinstance(value, BaseModelMixin):
            return [_say("not_model", step=step, name=name, found=_describe(value))]
        return []
    if spec.kind == "any":
        return []
    t = _tensor(value)
    if not _is_numeric(t):
        return [_say("not_numbers", step=step, name=name, unit=unit, found=_describe(value))]
    t = t.double()
    if spec.kind == "number":
        if t.numel() != 1:
            return [_say("not_one_number", step=step, name=name, unit=unit, found=_describe(value))]
        t = t.reshape(())
    else:
        want = tuple(syms.get(s, s) for s in spec.shape)
        if tuple(t.shape) != want:
            return [
                _say(
                    "wrong_shape", step=step, name=name, unit=unit,
                    shape=_shape_words(spec.shape, syms), found=tuple(t.shape),
                )
            ]
    problems = []
    if (spec.finite or spec.positive or spec.kind == "number") and not torch.isfinite(t).all():
        bad = int((~torch.isfinite(t)).sum())
        problems.append(_say("not_finite", step=step, name=name, bad=bad, n=t.numel()))
    elif spec.positive and not (t > 0).all():
        bad = int((~(t > 0)).sum())
        problems.append(
            _say("not_positive", step=step, name=name, bad=bad, n=t.numel(), smallest=float(t.min()))
        )
    if spec.integer and torch.isfinite(t).all() and not (t == t.round()).all():
        problems.append(_say("not_integer", step=step, name=name))
    if spec.increasing and t.numel() > 1 and not (t[1:] > t[:-1]).all():
        problems.append(_say("not_increasing", step=step, name=name))
    return problems


# --- the checks that look across names ---------------------------------------


def _graph_sizes(batch):
    return batch.num_nodes_per_graph.detach().cpu().long()


def _same_graphs(step, name, ns):
    import torch
    from nvalchemi.data import Batch

    cb, b = ns.get("crystal_batch"), ns.get(name)
    if not isinstance(cb, Batch) or not isinstance(b, Batch):
        return []
    if b.num_graphs != cb.num_graphs or not torch.equal(_graph_sizes(b), _graph_sizes(cb)):
        return [_say("same_graphs", step=step, name=name, n=cb.num_graphs, m=b.num_graphs)]
    return []


def _finished_graphs(batch):
    status = getattr(batch, "status", None)
    if status is None:
        return 0
    status = status.reshape(-1)[: batch.num_graphs]
    return int((status >= 1).sum())


def _same_precision(step, ns):
    """mace_model's weights and crystal_batch.positions must share one number precision.

    MACEWrapper.forward casts the positions to the model's dtype, so one force call works
    with a mismatch. The forces then come back in the model's dtype, and the optimizer and
    integrators, which update float32 positions with those forces, fail later.
    """
    from nvalchemi.data import Batch
    from nvalchemi.models.base import BaseModelMixin

    model, cb = ns.get("mace_model"), ns.get("crystal_batch")
    if not isinstance(model, BaseModelMixin) or not isinstance(cb, Batch):
        return []
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        return []
    pos_dtype = getattr(getattr(cb, "positions", None), "dtype", None)
    if pos_dtype is None or model_dtype == pos_dtype:
        return []
    return [
        _say(
            "same_precision", step=step,
            model=str(model_dtype).replace("torch.", ""), positions=str(pos_dtype).replace("torch.", ""),
        )
    ]


def _largest_force_per_graph(batch, forces):
    """Largest force on any atom of each graph: the maximum over atoms of the vector length."""

    lengths = forces.detach().cpu().double().norm(dim=-1)
    return _max_per_graph(lengths, batch.batch_idx.detach().cpu().long(), batch.num_graphs)


def _stored_forces(batch):
    try:
        forces = getattr(batch, "forces", None)
    except Exception:
        return None
    if forces is None or tuple(getattr(forces, "shape", ())) != (batch.num_nodes, 3):
        return None
    return forces


def _fresh_forces(batch, model):
    """One forward call on a copy of the batch, so the learner's batch is left as it is."""
    work = batch.clone()
    prepare_model_batch(work, model)
    return model(work)["forces"]


def _force_is_vector_length(step, name, batch_name, ns):
    """The learner's per-crystal maximum force must be the largest vector length, in eV/Å.

    The reference comes from the forces stored on the batch if they agree; otherwise from
    one fresh forward call of mace_model on a copy (stored forces can be one step old). The
    check says nothing when the reference cannot be computed, so a correct file never fails
    here. The largest Cartesian component, a common slip, is smaller than the vector length.
    """
    import torch
    from nvalchemi.data import Batch

    batch, value, model = ns.get(batch_name), ns.get(name), ns.get("mace_model")
    t = _tensor(value) if value is not None else None
    if (
        not isinstance(batch, Batch)
        or not callable(model)
        or not _is_numeric(t)
        or tuple(t.shape) != (batch.num_graphs,)
        or not torch.isfinite(t.double()).all()
    ):
        return []
    t = t.double()
    try:
        expected, bad = None, None
        stored = _stored_forces(batch)
        sources = ([lambda: stored] if stored is not None else []) + [
            lambda: _fresh_forces(batch, model)
        ]
        for source in sources:
            expected = _largest_force_per_graph(batch, source())
            if not torch.isfinite(expected).all():
                return []
            bad = (t - expected).abs() > 0.01 * expected.abs() + 1e-4
            if not bad.any():
                return []
    except Exception:
        return []
    k = int(bad.sum())
    worst = int(((t - expected).abs() * bad).argmax())
    return [
        _say(
            "force_vector", step=step, name=name, k=k, n=batch.num_graphs,
            found=float(t[worst]), expected=float(expected[worst]),
        )
    ]


# The crystals timed one by one stand for the batch when their mean atom count is within this
# fraction of the batch's mean; outside it the check adds a note.
SINGLE_SIZE_TOLERANCE = 0.25


def _single_sizes_notes(ns, watch):
    """A note when the crystals relaxed one at a time are not of about the batch's average size.

    The one-by-one bar is the mean of single_seconds times the number of crystals, so it stands
    for the batch only when the timed crystals are of about average atom count. The timed runs
    are the last runs of one graph the file made, as many as single_seconds has entries. Silent
    when the watch saw fewer such runs.
    """
    from nvalchemi.data import Batch

    cb, single = ns.get("crystal_batch"), ns.get("single_seconds")
    t = _tensor(single) if single is not None else None
    if watch is None or not isinstance(cb, Batch) or t is None or t.dim() != 1 or t.numel() == 0:
        return []
    ones = [int(state["sizes"][0]) for _engine, state in watch.runs if state["n"] == 1]
    if len(ones) < t.numel():
        return []
    sizes = ones[-t.numel():]
    mean = float(_graph_sizes(cb).double().mean())
    timed = sum(sizes) / len(sizes)
    if abs(timed - mean) <= SINGLE_SIZE_TOLERANCE * mean:
        return []
    return [_say("single_sizes", sizes=", ".join(str(n) for n in sizes), mean=mean)]


def _relax_build_contract(step, ns, ctx):
    """One precision for model and batch, the largest force as a vector length, n_crystals periodic crystals."""
    from nvalchemi.data import Batch

    problems = _same_precision(step, ns)
    problems += _force_is_vector_length(step, "start_max_force", "crystal_batch", ns)
    batch = ns.get("crystal_batch")
    if isinstance(batch, Batch):
        if batch.num_graphs != ns["n_crystals"]:
            problems.append(_say("crystal_count", want=ns["n_crystals"], found=batch.num_graphs))
        problems += _periodic_crystals(batch)
    return problems


def _periodic_crystals(batch):
    """Every graph of crystal_batch has periodic boundaries on all three axes and a cell with a volume."""
    import torch

    pbc, cell = getattr(batch, "pbc", None), getattr(batch, "cell", None)
    if pbc is None or cell is None:
        return [_say("periodic_none")]
    pbc = pbc.reshape(batch.num_graphs, -1).cpu()
    cell = cell.reshape(batch.num_graphs, 3, 3).detach().cpu().double()
    nopbc = int((~pbc.all(dim=1)).sum())
    nocell = int((torch.linalg.det(cell).abs() < 1e-9).sum() + (~torch.isfinite(cell).all()).sum())
    if nopbc or nocell:
        return [_say("periodic_count", nopbc=nopbc, n=batch.num_graphs, nocell=nocell)]
    return []


def _relax_run_contract(step, ns, ctx):
    """The same graphs as crystal_batch, every largest force at or below the threshold, and a
    one-by-one timing on the model settings of the batched run."""

    problems = _same_graphs(step, "relaxed_batch", ns)
    problems += _force_is_vector_length(step, "relaxed_max_force", "relaxed_batch", ns)
    problems += _below_threshold(ns)
    problems += _same_model_settings(step, ns, ctx.get("watch"))
    ctx.setdefault("notes", []).extend(_single_sizes_notes(ns, ctx.get("run_watch")))
    return problems


def _below_threshold(ns):
    """Every relaxed_max_force is at or below force_threshold.

    A force recomputed after the run can sit a hair above the threshold the optimizer stopped at
    (float32 noise on the GPU), so 1 percent is allowed.
    """
    import torch

    force, threshold = ns.get("relaxed_max_force"), ns.get("force_threshold")
    if force is None or threshold is None:
        return []
    force, threshold = _tensor(force), _tensor(threshold)
    if force is None or threshold is None or threshold.numel() != 1 or force.numel() == 0:
        return []
    if not torch.isfinite(force).all():
        return []
    threshold = float(threshold.reshape(()))
    over = int((force.double() > threshold * 1.01 + 1e-6).sum())
    if not over:
        return []
    return [_say("threshold", threshold=threshold, over=over, n=force.numel(), largest=float(force.max()))]


def _md_build_contract(step, ns, ctx):
    """The same graphs as crystal_batch, with velocities, and no graph marked finished."""
    import torch
    from nvalchemi.data import Batch

    problems = _same_graphs(step, "md_batch", ns)
    batch = ns.get("md_batch")
    if not isinstance(batch, Batch):
        return problems
    velocities = getattr(batch, "velocities", None)
    if velocities is None:
        problems.append(_say("no_velocities"))
    elif tuple(velocities.shape) != (batch.num_nodes, 3) or not torch.isfinite(velocities).all():
        problems.append(_say("bad_velocities", atoms=batch.num_nodes, found=tuple(velocities.shape)))
    finished = _finished_graphs(batch)
    if finished:
        problems.append(_say("md_finished", k=finished, n=batch.num_graphs))
    return problems


# The name that holds the saved energy rows of a step that records energies along a run.
ENERGY_ROWS = {"md_run": "crystal_energy", "temperature_ladder": "replica_energy"}


def _saved_rows_differ(step, ns, ctx):
    return _rows_repeat(step, ENERGY_ROWS[step], ns)


def _rows_repeat(step, name, ns):
    """Every saved row after the first holds exactly the same numbers: the rows were not recorded
    along the run. Seen with SnapshotHook and a HostMemory sink in Toolkit 0.3.0 on a CPU device:
    HostMemory.write keeps ``data.to("cpu")``, which is no copy when the batch is already on the CPU,
    so every saved frame holds the batch's last energy. On a GPU ``.to("cpu")`` copies."""
    import torch

    t = _tensor(ns.get(name)) if name in ns else None
    if t is None or t.dim() != 2 or t.shape[0] < 3 or not t.is_floating_point():
        return []
    rest = t[1:].double()
    if not torch.isfinite(rest).all():
        return []
    if bool((rest == rest[0]).all()) and not bool((t[0].double() == rest[0]).all()):
        return [_say("rows_repeat", step=step, name=name, n=int(rest.shape[0]))]
    return []


# An atom moved when it ends further than this from where it started, in Å.
MOVED_ANGSTROM = 1e-6


def _atoms_moved(step, ns, ctx):
    """Every crystal's atoms moved during the MD, judged against md_batch before the file ran.

    The MD may run on md_batch itself or on a copy. So the positions are compared for
    whatever md_batch is bound to now, and for any Batch the step file created or rebound
    (a copy the MD ran on). Batches the step file did not touch (crystal_batch,
    relaxed_batch) are left out: they differ from md_batch for other reasons.
    """
    import torch
    from nvalchemi.data import Batch

    snap = ctx.get("md_positions")
    if snap is None:
        return []
    pos0, bidx, n_graphs, finished = snap
    before = ctx.get("batch_ids", {})
    moved = torch.zeros(n_graphs, dtype=torch.float64)
    for key, value in ns.items():
        if key.startswith("_") or not isinstance(value, Batch):
            continue
        if key != "md_batch" and before.get(key) == id(value):
            continue
        if value.num_nodes != pos0.shape[0] or value.num_graphs != n_graphs:
            continue
        shift = (value.positions.detach().cpu().double() - pos0).norm(dim=-1)
        moved = torch.maximum(moved, _max_per_graph(shift, bidx, n_graphs))
    still = int((moved <= MOVED_ANGSTROM).sum())
    if still:
        hint = _say("md_not_moved_hint") if finished else ""
        return [_say("md_not_moved", k=still, n=n_graphs, hint=hint)]
    return []


def _model_settings(wrapper):
    """(cuEquivariance on, compiled, dtype) of a loaded MACEWrapper."""
    import torch
    from nvalchemi.models.mace import _mace_uses_cueq

    inner = wrapper.model
    compiled = isinstance(inner, torch._dynamo.eval_frame.OptimizedModule)
    return (bool(_mace_uses_cueq(inner)), compiled, next(wrapper.parameters()).dtype)


def _settings_words(settings):
    cueq, compiled, dtype = settings
    return _say(
        "settings_words",
        cueq=_say("on" if cueq else "off"),
        compiled=_say("compiled" if compiled else "not_compiled"),
        dtype=str(dtype).replace("torch.", ""),
    )


class _WatchMaceCalls:
    """For the duration of a step file, record the settings of every MACEWrapper that is called."""

    def __init__(self):
        self.models = {}  # id -> (wrapper, settings at its first call)

    def __enter__(self):
        from nvalchemi.models.mace import MACEWrapper

        self._cls, self._orig = MACEWrapper, MACEWrapper.forward
        recorder = self

        def forward(model, *args, **kwargs):
            if id(model) not in recorder.models:
                recorder.models[id(model)] = (model, _model_settings(model))
            return recorder._orig(model, *args, **kwargs)

        MACEWrapper.forward = forward
        return self

    def __exit__(self, *exc):
        self._cls.forward = self._orig


def _same_model_settings(step, ns, watch):
    """The one-by-one timing must run on the same model settings as the batched run."""
    ref = ns.get("mace_model")
    if watch is None or ref is None:
        return []
    try:
        ref_settings = _model_settings(ref)
    except Exception:
        return []
    found = []
    for model, settings in watch.models.values():
        if model is not ref and settings != ref_settings and settings not in found:
            found.append(settings)
    return [
        _say("same_settings", step=step, ref=_settings_words(ref_settings), found=_settings_words(s))
        for s in found
    ]


class _WatchDynamicsHooks:
    """For the duration of a step file, record the simulated time of every call of a hook
    the file wrote itself (a hook whose class is not part of nvalchemi).

    A hook dispatched at AFTER_STEP while step_count is k sees the state after k + 1 steps,
    because the engine adds 1 to step_count after the hooks of that step have run. A hook at
    BEFORE_STEP sees the state after k steps. The timestep is read from the engine (kernel
    time units converted back to fs). Other stages are not recorded.
    """

    def __init__(self):
        self.hooks = {}  # id(hook) -> [hook, dt_fs, [times in fs, one per call]]

    def __enter__(self):
        from nvalchemi.dynamics._units import FS_PER_INTERNAL_TIME
        from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage

        self._cls, self._orig = BaseDynamics, BaseDynamics._call_hooks
        recorder, orig = self, BaseDynamics._call_hooks

        def call_hooks(engine, stage, *args, **kwargs):
            try:
                offset = {DynamicsStage.AFTER_STEP: 1, DynamicsStage.BEFORE_STEP: 0}.get(stage)
                dt_init = getattr(engine, "_dt_init", None)
                if offset is not None and dt_init is not None:
                    dt = float(dt_init) * FS_PER_INTERNAL_TIME
                    k = int(engine.step_count)
                    for hook in engine.hooks:
                        if type(hook).__module__.startswith("nvalchemi"):
                            continue
                        runs = getattr(hook, "_runs_on_stage", None)
                        if (runs(stage) if runs is not None else stage == hook.stage) and (
                            kwargs.get("ignore_frequency") or k % hook.frequency == 0
                        ):
                            entry = recorder.hooks.setdefault(id(hook), [hook, dt, []])
                            entry[2].append((k + offset) * dt)
            except Exception:
                pass  # the watch must never break the file it watches
            return orig(engine, stage, *args, **kwargs)

        BaseDynamics._call_hooks = call_hooks
        return self

    def __exit__(self, *exc):
        self._cls._call_hooks = self._orig


def _time_axis(step, ns, ctx):
    """md_run: a time_fs that does not match the saved rows fails (see _time_axis_miss)."""

    miss = _time_axis_miss(ns, ctx, ENERGY_ROWS["md_run"])
    return [_say("time_axis", **miss)] if miss else []


def _ladder_time_axis_notes(ns, ctx):
    """temperature_ladder: the same test as md_run, printed as an "unexpected" note (the step passes)."""

    miss = _time_axis_miss(ns, ctx, ENERGY_ROWS["temperature_ladder"])
    return [_say("ladder_time_axis", **miss)] if miss else []


def _time_axis_miss(ns, ctx, energy_name):
    """time_fs must give the simulated time of each saved energy row; None when it does.

    The rows come from a hook the file wrote. The watch knows the simulated time of every
    call of that hook. The hook whose calls match the rows (one call per row, or one row
    for time 0 plus one call per row) is the reference; if none matches, the check stays
    silent, so a file that records its rows another way is never failed by this check.
    A difference of up to one timestep passes: step_count counts the step in progress, and
    files that label a row with step_count * dt are one step early.
    """
    import torch

    watch = ctx.get("dyn_watch")
    if watch is None or "time_fs" not in ns or energy_name not in ns:
        return None
    tf, ce = _tensor(ns["time_fs"]), _tensor(ns[energy_name])
    if tf is None or ce is None or tf.dim() != 1 or ce.dim() < 1 or ce.shape[0] != tf.shape[0]:
        return None
    tf = tf.double()
    best = None  # (largest miss in steps, expected times, dt)
    for _hook, dt, times in watch.hooks.values():
        for lead in ([], [0.0]):
            if len(lead) + len(times) != tf.shape[0]:
                continue
            expected = torch.tensor(lead + times, dtype=torch.float64)
            miss = float((tf - expected).abs().max()) / dt
            if best is None or miss < best[0]:
                best = (miss, expected, dt)
    if best is None or best[0] <= 1.0 + 1e-3:
        return None
    _, expected, dt = best
    off = (tf - expected).abs() / dt
    row = tf.shape[0] - 1 if off[-1] > 1.0 + 1e-3 else int(off.argmax())
    which = _say("time_axis_last") if row == tf.shape[0] - 1 else _say("time_axis_row", row=row + 1)
    return {"which": which, "expected": float(expected[row]), "found": float(tf[row])}


# --- the timestep scan ------------------------------------------------------------


def _start_state(batch):
    """What a dynamics run started from: sizes, atoms, positions, cells and pbc of each graph."""

    idx = batch.batch_idx.detach().cpu().long()
    n = int(batch.num_graphs)
    pbc = getattr(batch, "pbc", None)
    cell = getattr(batch, "cell", None)
    return {
        "n": n,
        "sizes": _graph_sizes(batch),
        "idx": idx,
        "finished": _finished_graphs(batch),
        "z": batch.atomic_numbers.detach().cpu().long().clone(),
        "pos": batch.positions.detach().cpu().double().clone(),
        "pbc": None if pbc is None else pbc.detach().cpu().reshape(n, -1).clone(),
        "cell": None if cell is None else cell.detach().cpu().double().reshape(n, 3, 3).clone(),
    }


class _WatchEngineRuns:
    """For the duration of a step file, record every dynamics run: the engine, its starting batch and its end positions."""

    def __init__(self):
        self.runs = []  # (engine, start state)

    def __enter__(self):
        from nvalchemi.dynamics.base import BaseDynamics

        self._cls, self._orig = BaseDynamics, BaseDynamics.run
        recorder, orig = self, BaseDynamics.run

        def run(engine, batch, *args, **kwargs):
            state = None
            try:
                state = _start_state(batch)
                state["optimizer"] = _is_optimizer(engine)
                recorder.runs.append((engine, state))
            except Exception:
                pass  # the watch must never break the file it watches
            result = orig(engine, batch, *args, **kwargs)
            try:
                if state is not None:
                    done = result if hasattr(result, "positions") else batch
                    state["end_pos"] = done.positions.detach().cpu().double().clone()
                    _record_run_end(engine, done, state)
            except Exception:
                pass
            return result

        BaseDynamics.run = run
        return self

    def __exit__(self, *exc):
        self._cls.run = self._orig


def _is_optimizer(engine) -> bool:
    """True for a Toolkit optimizer, and for a fused stage made of optimizers only."""

    stages = getattr(engine, "sub_stages", None)
    if stages:
        return all(_is_optimizer(dynamics) for _code, dynamics in stages)
    return type(engine).__module__.startswith("nvalchemi.dynamics.optimizers")


# Thermostats (classes of nvalchemi) whose per-copy target the check reads. Both keep the
# target in engine._state.temperature as kT in eV, one value per graph (Toolkit 0.3.0).
_THERMOSTATS = {"NVTLangevin", "NVTNoseHoover"}


def _record_run_end(engine, done, state):
    """Add what the ladder check needs to a run's record: end velocities, masses, the targets.

    ``target_init_K`` is the temperature the engine was built with (K, one per graph);
    ``target_end_K`` the target held per graph when the run ended. They differ when something
    rebound the targets during the run (replica exchange does). ``exchange`` is True when the
    batch carries replica-exchange labels.
    """
    import torch

    n = state["n"]
    for key, source in (("end_vel", "velocities"), ("masses", "atomic_masses")):
        value = getattr(done, source, None)
        if value is not None:
            state[key] = value.detach().cpu().double().reshape(-1, 3 if key == "end_vel" else 1).clone()
    state["exchange"] = getattr(done, "thermodynamic_state_id", None) is not None
    cls = type(engine)
    if cls.__name__ not in _THERMOSTATS or not cls.__module__.startswith("nvalchemi"):
        return
    from nvalchemi.dynamics import KB_EV

    init = getattr(engine, "_temperature_init", None)
    if init is not None:
        init = torch.as_tensor(init).detach().cpu().double().reshape(-1)
        state["target_init_K"] = init.expand(n).clone() if init.numel() == 1 else init.clone()
    st = getattr(engine, "_state", None)
    kt = getattr(st, "temperature", None) if st is not None else None
    if kt is not None:
        state["target_end_K"] = kt.detach().cpu().double().reshape(-1).clone() / KB_EV


def _engine_dt_fs(engine):
    """The timesteps an engine was built with, in fs (a scalar is one value), or None."""
    import torch
    from nvalchemi.dynamics._units import FS_PER_INTERNAL_TIME

    value = getattr(engine, "_dt_init", None)
    if value is None:
        return None
    return torch.as_tensor(value).detach().cpu().double().reshape(-1) * FS_PER_INTERNAL_TIME


def _fs_list(values, limit=8):
    shown = ", ".join(f"{v:.4g}" for v in values[:limit])
    return shown + (", ..." if len(values) > limit else "")


def _scan_copies(step, ns, ctx):
    """The copies must run together as one crystal each, periodic, each with its own timestep.

    Judged on the runs the file made (the watch), using the runs that start from as many
    graphs as replica_dt has timesteps. Silent when the watch saw no run.
    """

    watch = ctx.get("run_watch")
    dt = _tensor(ns["replica_dt"]) if "replica_dt" in ns else None
    if watch is None or not watch.runs or dt is None or dt.dim() != 1:
        return []
    dt = dt.double()
    k = int(dt.numel())
    runs = [(engine, state) for engine, state in watch.runs if state["n"] == k]
    if not runs:
        sizes = sorted({state["n"] for _, state in watch.runs})
        return [_say("scan_one_batch", step=step, k=k, sizes=", ".join(str(n) for n in sizes))]
    state = runs[-1][1]
    # every copy is the same structure: same atoms in the same order, same positions
    problems = _copies_same_atoms(step, state, k)
    if not problems:
        positions = state["pos"].reshape(k, -1, 3)
        if float((positions - positions[0]).abs().max()) > 1e-4:
            problems.append(_say("scan_positions", step=step))
    # every copy is a periodic crystal
    problems += _copies_periodic(step, state, k)
    # no copy is marked finished (a leftover of the relaxation: it would not move)
    if state["finished"]:
        problems.append(_say("scan_finished", step=step, k=state["finished"], n=k))
    # every copy moved during the run (a copy that did not cannot drift, at any timestep)
    still = _copies_that_stayed(state["pos"], state.get("end_pos"), state["idx"], k)
    if still:
        hint = _say("copies_finished_hint") if state["finished"] else ""
        problems.append(_say("scan_not_moved", step=step, still=still, k=k, hint=hint))
    # every copy really ran with its own timestep, equal to replica_dt
    if not any(_ran_with_timesteps(engine, dt) for engine, _ in runs):
        used = _engine_dt_fs(runs[-1][0])
        if used is not None:
            problems.append(
                _say("scan_dt", step=step, used=_fs_list(used.tolist()), want=_fs_list(dt.tolist()))
            )
    return problems


def _ran_with_timesteps(engine, dt) -> bool:
    """True when the engine was built with the timesteps ``dt`` (fs, one per copy)."""
    import torch

    used = _engine_dt_fs(engine)
    if used is None:
        return False
    used = used.expand(dt.numel()) if used.numel() == 1 else used
    return used.numel() == dt.numel() and bool(torch.allclose(used, dt, rtol=1e-3, atol=1e-6))


def _copies_same_atoms(step, state, k):
    """The copies of a run hold the same atoms in the same order."""

    sizes, numbers = state["sizes"], state["z"]
    if not bool((sizes == sizes[0]).all()):
        return [_say("copies_sizes", step=step)]
    numbers = numbers.reshape(k, -1)
    if not bool((numbers == numbers[0]).all()):
        return [_say("copies_atoms", step=step)]
    return []


def _copies_that_stayed(start, end, graph_of_atom, k) -> int:
    """How many of the k copies have no atom that moved between ``start`` and ``end``; 0 when the
    end positions are not known."""

    if end is None or end.shape != start.shape:
        return 0
    moved = _max_per_graph((end - start).norm(dim=-1), graph_of_atom, k)
    return int((moved <= MOVED_ANGSTROM).sum())


def _copies_periodic(step, state, k):
    """The copies of a run must be a periodic crystal: a cell and pbc on all three axes."""
    import torch

    pbc, cell = state["pbc"], state["cell"]
    if pbc is None or cell is None:
        found = _say("copies_periodic_none")
    else:
        nopbc = int((~pbc.bool().all(dim=1)).sum())
        nocell = int(((torch.linalg.det(cell).abs() < 1e-9) | ~torch.isfinite(cell).all(dim=(1, 2))).sum())
        found = None if not (nopbc or nocell) else _say("copies_periodic_count", nopbc=nopbc, k=k, nocell=nocell)
    return [_say("copies_periodic", step=step, found=found)] if found else []


def _scan_blowup_limit(ctx, n_copies):
    """(limit in eV, words) for a blown-up copy: 1 eV per atom when the run's copies are known
    to hold the same number of atoms, else 10 eV for the copy."""

    watch = ctx.get("run_watch")
    for _engine, state in reversed(getattr(watch, "runs", [])):
        if state["n"] == n_copies and bool((state["sizes"] == state["sizes"][0]).all()):
            atoms = int(state["sizes"][0])
            return SCAN_BLOWUP_EV_PER_ATOM * atoms, _say("scan_limit_atom", limit=SCAN_BLOWUP_EV_PER_ATOM)
    return SCAN_BLOWUP_EV_PER_COPY, _say("scan_limit_copy", limit=SCAN_BLOWUP_EV_PER_COPY)


def _scan_values(step, ns, ctx):
    """Different timesteps, a finite first row, and at least one copy that does not blow up."""
    import torch

    problems = []
    dt, saved, energy = (_tensor(ns[n]) if n in ns else None for n in ("replica_dt", "saved_step", "total_energy"))
    if dt is not None and dt.dim() == 1 and torch.isfinite(dt.double()).all():
        distinct = len(set(dt.double().tolist()))
        if dt.numel() < 2 or distinct != dt.numel():
            problems.append(
                _say("scan_distinct", step=step, distinct=distinct, s="s" if distinct != 1 else "", n=dt.numel())
            )
    if saved is not None and saved.dim() == 1 and saved.numel() < 2:
        problems.append(_say("scan_rows", step=step, n=saved.numel()))
    if _is_numeric(energy) and energy.dim() == 2 and energy.shape[0] >= 1:
        e = energy.double()
        first = ~torch.isfinite(e[0])
        if first.any():
            problems.append(_say("scan_first_row", step=step, k=int(first.sum()), n=e.shape[1]))
        else:
            limit, words = _scan_blowup_limit(ctx, e.shape[1])
            rows = scan_blowup_rows((e - e[0]).numpy(), limit)
            if not bool((rows == e.shape[0]).any()):
                problems.append(_say("scan_none_stable", step=step, words=words))
    return problems


# --- speed_settings -------------------------------------------------------------


def _speed_contract(step, ns, ctx):
    """speed_table holds each of the four cases once with two good times; setting_roles names both.

    Fails only on: a missing case or a case twice, a missing or bad time, missing roles (and,
    through the generic check, a missing name). "Both slower than neither" and the model watch
    add notes; they never fail the step.
    """
    problems = []
    if "speed_table" in ns:
        problems += _speed_table_problems(step, ns["speed_table"])
        if not problems:
            ctx["notes"].extend(speed_notes(ns["speed_table"]))
            ctx["notes"].extend(_speed_model_notes(ns["speed_table"], ctx.get("watch")))
    if "setting_roles" in ns:
        problems += _speed_roles_problems(step, ns["setting_roles"])
    return problems


def _speed_table_problems(step, table):
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas ships with the runtime
        return []
    if not isinstance(table, pd.DataFrame):
        return [_say("speed_not_table", step=step, found=_describe(table))]
    columns = {c: _column(table, c) for c in SPEED_FLAGS + SPEED_TIMES}
    missing = [c for c, found in columns.items() if found is None]
    if missing:
        return [_say("speed_columns", step=step, missing=", ".join(missing))]
    problems = []
    cases = []
    for c in SPEED_FLAGS:
        values = [_flag_value(v) for v in table[columns[c]].tolist()]
        bad = [v for v, f in zip(table[columns[c]].tolist(), values) if f is None]
        if bad:
            problems.append(_say("speed_flags", step=step, column=c, found=repr(bad[0])))
        cases.append(values)
    if problems:
        return problems
    rows = list(zip(*cases))  # (cueq, compile) of each row
    for case in SPEED_CASES:
        count = rows.count(case)
        if count == 0:
            problems.append(_say("speed_case_missing", step=step, case=_speed_case_words(case)))
        elif count > 1:
            problems.append(_say("speed_case_twice", step=step, case=_speed_case_words(case), count=count))
    for c in SPEED_TIMES:
        for case, value in zip(rows, table[columns[c]].tolist()):
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                seconds = None
            if isinstance(value, (bool, np.bool_)) or seconds is None or not np.isfinite(seconds) or seconds <= 0:
                problems.append(
                    _say("speed_time", step=step, column=c, found=repr(value), case=_speed_case_words(case))
                )
                break  # one line per column
    return problems


def _speed_roles_problems(step, roles):
    if not isinstance(roles, dict):
        return [_say("speed_roles", step=step, found=_describe(roles))]
    lower = {str(k).lower(): v for k, v in roles.items()}
    absent = [k for k in SPEED_ROLES if k not in lower]
    empty = [k for k in SPEED_ROLES if k in lower and not (isinstance(lower[k], str) and lower[k].strip())]
    if not absent and not empty:
        return []
    if absent:
        found = _say("speed_roles_keys", keys=", ".join(str(k) for k in roles)) if roles else _say("speed_roles_none")
    else:
        found = _say("speed_roles_empty", keys=", ".join(empty))
    return [_say("speed_roles", step=step, found=found)]


def _speed_model_notes(table, watch):
    """One note per case in speed_table that no MACE model called during the file had.

    The watch holds every MACEWrapper called while the file ran, with its settings at its first
    call. Silent when the watch saw no model (a file that times another way is not judged).
    """
    if watch is None or not watch.models:
        return []
    seen = {(cueq, compiled) for _model, (cueq, compiled, _dtype) in watch.models.values()}
    cueq, comp = _column(table, "cueq"), _column(table, "compile")
    rows = {(_flag_value(a), _flag_value(b)) for a, b in zip(table[cueq].tolist(), table[comp].tolist())}
    found = "; ".join(sorted({_settings_words(s) for _m, s in watch.models.values()}))
    return [
        _say("speed_model", case=_speed_case_words(case), found=found)
        for case in SPEED_CASES
        if case in rows and case not in seen
    ]


# --- temperature_ladder ---------------------------------------------------------


def _ladder_runs(ctx, k):
    """The runs the file made with one graph for each temperature, in order; None when no run was seen."""

    watch = ctx.get("run_watch")
    if watch is None or not watch.runs:
        return None
    return [state for _engine, state in watch.runs if state["n"] == k]


def _ladder_contract(step, ns, ctx):
    """Hard fails: temperatures not all different, a first row that is not finite, copies not run
    together as one periodic crystal, a copy that did not move, a thermostat target that differs
    from replica_temperature (only where the target can be read). The rest adds notes."""
    import torch

    problems = []
    temps, energy = (_tensor(ns[n]) if n in ns else None for n in ("replica_temperature", "replica_energy"))
    if temps is None or temps.dim() != 1 or temps.numel() == 0 or not torch.isfinite(temps.double()).all():
        return problems  # the generic check reports it
    temps = temps.double()
    k = int(temps.numel())
    distinct = len(set(temps.tolist()))
    if distinct != k:
        problems.append(_say("ladder_distinct", step=step, distinct=distinct, s="s" if distinct != 1 else "", n=k))
    usable = _is_numeric(energy) and energy.dim() == 2 and energy.shape[1] == k and energy.shape[0] >= 1
    if usable:
        first = ~torch.isfinite(energy.double()[0])
        if first.any():
            problems.append(_say("ladder_first_row", step=step, k=int(first.sum()), n=k))
            usable = False
    runs = _ladder_runs(ctx, k)
    n_atoms = None
    if runs is not None:
        if not runs:
            sizes = sorted({s["n"] for _e, s in ctx["run_watch"].runs})
            problems.append(_say("ladder_one_batch", step=step, k=k, sizes=", ".join(str(n) for n in sizes)))
        else:
            problems += _ladder_copies(step, runs, k, ctx)
            sizes = runs[-1]["sizes"]
            n_atoms = int(sizes[0]) if bool((sizes == sizes[0]).all()) else None
            problems += _ladder_targets(step, runs, temps, ctx, energy if usable else None)
    # The data alone: an energy that never changes means the copy did not move (this also holds
    # when the watch saw no run).
    if usable and energy.shape[0] >= 2 and not ctx.get("ladder_still"):
        e = energy.double()
        column = e[:, torch.isfinite(e).all(dim=0)]
        scale = float(n_atoms) if n_atoms and float(e[0].abs().median()) > LADDER["whole_copy_ev"] else 1.0
        flat = int(((column.max(dim=0).values - column.min(dim=0).values) / scale <= LADDER["flat_ev_per_atom"]).sum())
        if flat:
            finished = bool(runs) and any(r["finished"] for r in runs)
            hint = _say("copies_finished_hint") if finished else ""
            problems.append(_say("ladder_flat", step=step, still=flat, k=k, hint=hint))
    if usable and "time_fs" in ns and not problems:  # the notes read a run that passed
        ctx["notes"].extend(_ladder_time_axis_notes(ns, ctx))
        try:
            ctx["notes"].extend(ladder_data(ns["time_fs"], temps, energy, n_atoms)["notes"])
        except ValueError:
            pass  # a shape the generic check reports
    return problems


def _ladder_copies(step, runs, k, ctx):
    """Same periodic crystal in every copy (atoms, cell, pbc), and every copy moved.

    ``ctx["ladder_still"]`` is set to the number of copies that did not move."""

    state = runs[-1]
    problems = _copies_same_atoms(step, state, k)
    periodic = _copies_periodic(step, state, k)
    problems += periodic
    cell = state["cell"]
    if not periodic and cell is not None and float((cell - cell[0]).abs().max()) > LADDER["cell_angstrom"]:
        problems.append(_say("ladder_cells", step=step))
    # moved: from the start of the first dynamics run of the copies to the end of the last one.
    # A relaxation of the copies before the dynamics does not count as the run.
    dynamics = [r for r in runs if not r.get("optimizer")] or runs
    still = _copies_that_stayed(dynamics[0]["pos"], dynamics[-1].get("end_pos"), dynamics[0]["idx"], k)
    if still:
        ctx["ladder_still"] = still
        finished = any(r["finished"] for r in dynamics)
        hint = _say("copies_finished_hint") if finished else ""
        problems.append(_say("ladder_not_moved", step=step, still=still, k=k, hint=hint))
    return problems


def _ladder_targets(step, runs, temps, ctx, energy):
    """The thermostat target of each copy must be replica_temperature, where the target can be read.

    Readable: a run of NVTLangevin or NVTNoseHoover whose targets at the end equal the ones it was
    built with (nothing rebound them) and whose batch carries no replica-exchange labels. Where no
    run is readable, the copies' kinetic temperatures at the end of the last run are compared in
    order instead, and a mismatch is a note, not a failure.
    """
    import torch

    rtol = LADDER["target_rtol"]
    readable = [
        r for r in runs
        if "target_end_K" in r and "target_init_K" in r and not r.get("exchange")
        and r["target_end_K"].numel() == temps.numel() and r["target_init_K"].numel() == temps.numel()
        and bool(torch.allclose(r["target_end_K"], r["target_init_K"], rtol=rtol))
    ]
    if readable:
        if any(bool(torch.allclose(r["target_end_K"], temps, rtol=rtol)) for r in readable):
            return []
        used = readable[-1]["target_end_K"]
        return [_say("ladder_target", step=step, used=_fs_list(used.tolist()), want=_fs_list(temps.tolist()))]
    ctx["notes"].extend(_ladder_kinetic_notes(runs[-1], temps, energy))
    return []


def _ladder_kinetic_notes(state, temps, energy):
    """Copies whose kinetic temperature at the end of the run is out of order with replica_temperature."""
    import torch
    from nvalchemi.dynamics import KB_EV

    vel, masses = state.get("end_vel"), state.get("masses")
    k = int(temps.numel())
    if vel is None or masses is None or vel.shape[0] != masses.shape[0] or vel.shape[0] != state["idx"].numel():
        return []
    ke = torch.zeros(k, dtype=torch.float64).index_add_(
        0, state["idx"], 0.5 * masses.reshape(-1) * vel.pow(2).sum(-1)
    )
    atoms = state["sizes"].double()
    kinetic = 2.0 * ke / (3.0 * atoms * KB_EV)
    keep = torch.isfinite(kinetic)
    if energy is not None and float(energy.double()[0].abs().median()) <= LADDER["whole_copy_ev"]:
        blown = scan_blowup_rows((energy.double() - energy.double()[0]).numpy(), LADDER["blowup_ev_per_atom"])
        keep &= torch.as_tensor(blown == energy.shape[0])
    ranked = [int(i) for i in torch.argsort(temps) if bool(keep[int(i)])]
    notes = []
    for cold, hot in zip(ranked[:-1], ranked[1:]):
        if not temps[hot] > temps[cold]:
            continue
        sigma = float(torch.sqrt(2.0 / (3.0 * atoms[hot]))) * float(torch.hypot(temps[hot], temps[cold]))
        if float(kinetic[hot] - kinetic[cold]) < -LADDER["kinetic_sigmas"] * sigma:
            notes.append(
                _say(
                    "ladder_kinetic", hot=float(temps[hot]), cold=float(temps[cold]),
                    hot_k=float(kinetic[hot]), cold_k=float(kinetic[cold]),
                )
            )
    return notes


# The checks of each step that look across names, in the order their lines print. Each takes
# (step, ns, ctx) and returns the problems; a check adds its notes to ctx["notes"].
STEP_CHECKS = {
    "relax_build": (_relax_build_contract,),
    "relax_run": (_relax_run_contract,),
    "md_build": (_md_build_contract,),
    "md_run": (_saved_rows_differ, _atoms_moved, _time_axis),
    "timestep_scan": (_scan_copies, _scan_values),
    "temperature_ladder": (_saved_rows_differ, _ladder_contract),
    "speed_settings": (_speed_contract,),
}


def _contract_extras(step, ns, ctx):
    """The problems of a step that the name checks cannot see (see STEP_CHECKS)."""

    return [line for check in STEP_CHECKS[step] for line in check(step, ns, ctx)]


def _md_batch_before(ns):
    """md_batch as it is before the step file runs: positions, graph of each atom, number of
    graphs, number of graphs marked finished. None when there is no md_batch."""

    batch = ns.get("md_batch")
    if batch is None or not hasattr(batch, "positions"):
        return None
    return (
        batch.positions.detach().cpu().double().clone(),
        batch.batch_idx.detach().cpu().long().clone(),
        batch.num_graphs,
        _finished_graphs(batch),
    )


_SCRIPT_NAMES = {"__file__"}  # names a script has and a notebook cell does not


def _run_file(path, ns, label):
    """Run one step file in the namespace, as a notebook cell runs it after %load.

    Returns None, or the one line that says how the file stopped. The file's own prints go to
    stderr, so they do not mix with the verdict lines on stdout.
    """

    t0 = time.time()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            exec(compile(Path(path).read_text(), path, "exec"), ns)
    except Exception as exc:
        traceback.print_exc()
        if isinstance(exc, NameError) and getattr(exc, "name", None) in _SCRIPT_NAMES:
            return _say("script_name", step=label, file=os.path.basename(path), name=exc.name)
        return _say(
            "stopped", label=label, file=os.path.basename(path), error=type(exc).__name__,
            first_line=str(exc).splitlines()[0][:200] if str(exc) else "",
        )
    print(_say("ran_in", file=os.path.basename(path), seconds=time.time() - t0), file=sys.stderr, flush=True)
    return None


def check_step(step, notes=None):
    """Run the chain up to `step`, check its contract. Returns (problems, n_names).

    Lines that do not fail the step (the "unexpected" notes) are appended to ``notes``
    when a list is given.
    """
    spec = STEPS[step]
    ns = {"__name__": "__main__"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(setup_cell(), ns)
    work = ns["work_dir"]
    for earlier in spec.after:
        path = os.path.join(work, STEPS[earlier].file)
        if not os.path.exists(path):
            return [_say("earlier_missing", step=step, file=STEPS[earlier].file, work=work)], 0
        err = _run_file(path, ns, earlier)
        if err:
            return [_say("earlier_failed", step=step, error=err)], 0
    path = os.path.join(work, spec.file)
    if not os.path.exists(path):
        return [_say("step_missing", step=step, file=spec.file, work=work)], len(spec.names)
    ctx = {
        "notes": [] if notes is None else notes,
        "watch": _WatchMaceCalls() if spec.watch_models else None,
        "dyn_watch": _WatchDynamicsHooks() if spec.watch_dynamics else None,
        "run_watch": _WatchEngineRuns() if spec.watch_runs else None,
    }
    if spec.watch_md_batch:
        from nvalchemi.data import Batch

        ctx["md_positions"] = _md_batch_before(ns)
        ctx["batch_ids"] = {k: id(v) for k, v in ns.items() if isinstance(v, Batch)}
    with contextlib.ExitStack() as stack:
        for watch in (ctx["watch"], ctx["dyn_watch"], ctx["run_watch"]):
            if watch is not None:
                stack.enter_context(watch)
        err = _run_file(path, ns, step)
    problems = [err] if err else []
    syms = _symbols(ns, step)
    for name, name_spec in spec.names.items():
        problems += _check_name(step, name, name_spec, ns, syms)
    problems += _contract_extras(step, ns, ctx)
    return problems, len(spec.names)


def main(argv):
    if len(argv) != 3 or argv[1] != "check" or argv[2] not in STEPS:
        print(_say("usage", steps="|".join(STEPS)), file=sys.stderr)
        return 2
    step = argv[2]
    t0 = time.time()
    notes = []
    problems, n = check_step(step, notes)
    seen = set()
    for line in problems + [_say("note", step=step, note=note) for note in notes]:
        if line not in seen:
            seen.add(line)
            print(line)
    if not problems:
        print(_say("ok", step=step, n=n))
    print(_say("check_took", seconds=time.time() - t0), file=sys.stderr)
    return 1 if problems else 0

