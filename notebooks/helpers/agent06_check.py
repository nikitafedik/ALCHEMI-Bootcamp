# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step contracts and the agent's own check for Agent 06, Hooked in at every step.

Run from ``notebooks/``::

    python -m helpers.agent06 check <step>

The check runs the setup cell, the files of the earlier steps, then the step's
file twice (a learner runs a cell again), in one namespace, as the notebook
does after ``%load``. It then checks the names the step must leave behind and
prints one plain line per problem. The contracts hold the shapes and units, so
the learner's prompt only needs the names:

``interaction``     ``saved_step`` (n,), the MD step of each record;
                    ``interaction_energy`` (n, 3), one column per complex, and
                    ``start_interaction_energy`` (3,), a single point at the
                    starting structure, both in one unit of the learner's
                    choice: kcal/mol, eV, kJ/mol or Hartree, read from the ratio to the
                    check's own value; ``plain_store``: the three complexes at
                    every saved step; ``plain_log``: a DataFrame or the path of
                    a CSV file, one row per complex at each logged step.
``separation_hook`` ``separation`` and ``nearest_contact`` (n, 3) in Å, one
                    column per complex, on the ``saved_step`` of the first step.
``freeze``          ``frozen_mask`` boolean (atoms,), True for fragment A;
                    ``frozen_store``: frames on the schedule of the plain run.
``retire``          ``stop_step`` (3,) integer, the MD step at which a complex
                    was retired and -1 for a complex that ran to the end;
                    ``final_status`` (3,); ``retire_store``: the three
                    complexes at every saved step.

A line that starts with "unexpected:" reports a result off the expected
relation. It does not fail the step: the check still prints OK and exits 0.

A file that assigns a name of an earlier step again (it re-runs that step
inside itself) gets one "unexpected:" line and the step still passes.

A short run for a CPU: set ``AGENT06_CHECK_STEPS`` to a number of MD steps and
the check's setup cell uses it in place of ``run_settings["n_steps"]``.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agent06_views import N_COMPLEXES, UNITS, frames_all_equal, read_unit, store_frames

__all__ = ["OPTIONAL_STEPS", "SETUP_CELL", "STEPS", "check_step", "main", "resolve_step"]

SETUP_CELL = (
    "from helpers import agent06 as helpers\n"
    "helpers.configure_tutorial()\n"
    "work_dir = helpers.make_work_dir()\n"
    "complexes, composed_model, fragment_a_sizes = helpers.prepare_complexes()\n"
    "run_settings = dict(temperature=1000.0, dt=1.0, friction=0.01, n_steps=1000)\n"
    "helpers.show_session()\n"
)

STILL = 1.0e-6  # Å: a complex that moves less than this between saved frames stands still
FROZEN = 1.0e-4  # Å: a frozen atom stays within this of its start
SAME_LIMIT = 0.2  # Å: slack when the saved frames are compared with one limit for every complex
UNEXPECTED = "unexpected: "  # a line with this prefix is reported and does not fail the step
KCAL = UNITS["kcal/mol"]  # kcal/mol per eV
FIRST_RECORD = 0.5  # kcal/mol: from rest the first record lies this close to the starting value
BOUND = 5.0  # kcal/mol: a bound pair stays between its starting value minus this and plus this
RECOMPUTED = 0.2  # kcal/mol: a record and the same saved frame recomputed agree this well
RECORD_EVERY = 10  # MD steps between records, as the first step asks
SAME_RUN = 0.05  # Å: two runs that start the same way give the same nearest contacts this well
_SCRIPT_NAMES = {"__file__"}  # names a script has and a notebook cell does not
# a file that assigns names of an earlier step again: one note, the step passes (advisor, 26 Nov)
CHANGED_LINE = "{step}: " + UNEXPECTED + "{file} assigned {name} again; the cells above now show this run"
DIGITS = {"Hartree": 4}  # decimals of an energy in the check's lines; other units: 2
SAME_LINE = (
    "every saved frame is the same: the store kept the last state of the run, "
    "not one state per saved step"
)


@dataclass(frozen=True)
class Step:
    """One ``%load`` step: its file, the names it creates, the earlier steps, its checks."""

    file: str
    names: tuple[str, ...]
    after: tuple[str, ...]
    checks: tuple[Callable[[dict, dict], list[str]], ...]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else np.asarray(value).shape


def _atom_counts(ns: dict) -> list[int]:
    return [int(c.num_nodes) for c in ns["complexes"]]


def _fragment_a_mask(ns: dict) -> np.ndarray:
    """True for the atoms of fragment A, over the atoms of the batch in order."""

    parts = [np.arange(n) < a for n, a in zip(_atom_counts(ns), ns["fragment_a_sizes"])]
    return np.concatenate(parts)


def _start_distances(ns: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fragment separation and nearest contact of each complex at its start, in Å."""

    separation, nearest = [], []
    for data, split in zip(ns["complexes"], ns["fragment_a_sizes"]):
        positions = _array(data.positions).astype(float)
        masses = _array(data.atomic_masses).astype(float).reshape(-1, 1)
        s, n = _distances(positions[None], masses, int(split))
        separation.append(s[0])
        nearest.append(n[0])
    return np.array(separation), np.array(nearest)


def _distances(positions: np.ndarray, masses: np.ndarray, split: int) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: distance between the centers of mass of A and B, and the shortest A to B distance."""

    a, b = positions[:, :split], positions[:, split:]
    com_a = (a * masses[:split]).sum(1) / masses[:split].sum()
    com_b = (b * masses[split:]).sum(1) / masses[split:].sum()
    pair = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)
    return np.linalg.norm(com_a - com_b, axis=1), pair.reshape(len(positions), -1).min(axis=1)


def _store_positions(ns: dict, name: str, problems: list[str]) -> list[np.ndarray] | None:
    """Positions of a store, one array (frames, atoms, 3) per complex; None with a line on a problem."""

    try:
        positions = store_frames(ns[name])["positions"]
    except Exception as exc:
        problems.append(f"{name}: {str(exc).splitlines()[0]}")
        return None
    counts = [p.shape[1] for p in positions]
    if counts != _atom_counts(ns):
        problems.append(
            f"{name} holds complexes with {counts} atoms; the three complexes have {_atom_counts(ns)}, in this order"
        )
        return None
    if len(positions[0]) < 2:
        problems.append(f"{name} holds one saved frame; save the frames while the run goes")
        return None
    if frames_all_equal(positions):
        problems.append(f"{name}: {SAME_LINE}")
        return None
    return positions


def _own_hooks(ctx: dict) -> list[Any]:
    return [h for h in ctx.get("hooks", []) if not type(h).__module__.startswith("nvalchemi")]


def _built_in(ctx: dict, word: str) -> bool:
    return any(word in type(h).__name__ and type(h).__module__.startswith("nvalchemi") for h in ctx.get("hooks", []))


def _interaction_ev(ns: dict, positions: list[np.ndarray]) -> np.ndarray:
    """Interaction energy of each complex at each given frame, in eV, with the composed model.

    ``positions`` holds one array (frames, atoms, 3) per complex, in Å. Returns
    (frames, complexes): E(AB) - E(A) - E(B), every fragment neutral, at the
    geometry of the frame.
    """

    import torch
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import DynamicsStage
    from nvalchemi.hooks import DynamicsContext

    model = ns["composed_model"]
    frames = len(positions[0])

    def energies(parts: list) -> np.ndarray:
        batch = Batch.from_data_list(parts)
        context = DynamicsContext(batch=batch, model=model)
        for hook in model.make_neighbor_hooks():
            hook(context, DynamicsStage.BEFORE_COMPUTE)
        return model(batch)["energy"].detach().reshape(-1).double().cpu().numpy()

    def part(data, xyz: np.ndarray, atoms: slice):
        pos = torch.as_tensor(xyz[atoms], dtype=data.positions.dtype, device=data.positions.device)
        return AtomicData(
            positions=pos,
            atomic_numbers=data.atomic_numbers[atoms].clone(),
            charge=torch.zeros((1, 1), dtype=pos.dtype, device=pos.device),
            forces=torch.zeros_like(pos),
            energy=torch.zeros((1, 1), dtype=pos.dtype, device=pos.device),
        )

    whole, fragments = [], []
    for f in range(frames):
        for data, split, xyz in zip(ns["complexes"], ns["fragment_a_sizes"], positions):
            whole.append(part(data, xyz[f], slice(None)))
            fragments.append(part(data, xyz[f], slice(0, int(split))))
            fragments.append(part(data, xyz[f], slice(int(split), None)))
    e_whole, e_parts = energies(whole), energies(fragments)
    return (e_whole - e_parts[0::2] - e_parts[1::2]).reshape(frames, len(positions))


def _start_interaction_ev(ns: dict) -> np.ndarray:
    """Interaction energy of each complex at its starting structure, in eV."""

    start = [_array(data.positions).astype(float)[None] for data in ns["complexes"]]
    return _interaction_ev(ns, start)[0]


def _steps(ns: dict, out: list[str]) -> np.ndarray | None:
    """``saved_step`` as an array when it is a usable list of MD steps; a line otherwise."""

    step = _array(ns["saved_step"])
    if step.ndim != 1 or step.size < 2:
        out.append(f"saved_step has shape {_shape(ns['saved_step'])}; one MD step per record gives (n,)")
        return None
    if not np.all(np.diff(step) > 0):
        out.append("saved_step does not increase from one record to the next")
        return None
    gaps = np.unique(np.diff(step))
    if gaps.size != 1:
        out.append(f"saved_step advances by {gaps.tolist()} steps; one interval for the whole run")
    n_steps = int(ns["run_settings"]["n_steps"])
    if step[-1] > n_steps or step[-1] < n_steps - 2 * int(gaps.max()):
        out.append(f"saved_step ends at step {int(step[-1])}; run_settings asks for {n_steps} steps")
    return step


def _numbers(values, digits: int = 2) -> str:
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in np.asarray(values).reshape(-1)) + "]"


# --------------------------------------------------------------------------
# Checks per step
# --------------------------------------------------------------------------


def check_plain_store_and_log(ns: dict) -> list[str]:
    """The saved frames and the log of the first run, which the later steps use."""

    import pandas as pd

    out: list[str] = []
    positions = _store_positions(ns, "plain_store", out)
    if positions is not None and max(float(np.abs(p[-1] - p[0]).max()) for p in positions) < 1e-3:
        out.append("plain_store: no atom moved between the first and the last saved frame")
    log = ns["plain_log"]
    if isinstance(log, (str, Path)):
        if not Path(log).exists():
            return out + [f"plain_log names {log}, and no such file exists"]
        log = pd.read_csv(log)
    if not isinstance(log, pd.DataFrame):
        return out + [f"plain_log is {type(log).__name__}; it should be a DataFrame or the path of the log file"]
    for column in ("energy", "temperature"):
        if column not in log.columns:
            out.append(f"plain_log has no {column} column; found {list(log.columns)}")
    if "step" in log.columns:
        per_step = log.groupby("step").size()
        if not (per_step == N_COMPLEXES).all():
            twice = bool((per_step % N_COMPLEXES == 0).all()) and len(per_step) > 0
            out.append(
                f"plain_log has {int(per_step.max())} rows for one step; one row per complex gives {N_COMPLEXES}"
                + ("; a second run added to the old log file, start from an empty file" if twice else "")
            )
        steps = np.sort(log["step"].unique())
        n_steps = int(ns["run_settings"]["n_steps"])
        spacing = int(np.diff(steps).max()) if len(steps) > 1 else n_steps
        if len(steps) < 2:
            out.append("plain_log holds one logged step; log while the run goes")
        elif steps[-1] > n_steps or steps[-1] < n_steps - spacing - 1:
            out.append(f"plain_log ends at step {int(steps[-1])}; run_settings asks for {n_steps} steps")
    elif len(log) % N_COMPLEXES:
        out.append(f"plain_log has {len(log)} rows; one row per complex at each logged step gives a multiple of {N_COMPLEXES}")
    return out


def check_interaction(ns: dict, ctx: dict) -> list[str]:
    out = check_plain_store_and_log(ns)
    if "hooks" in ctx:
        for word, job in (("Snapshot", "save the frames"), ("Logging", "write the log")):
            if not _built_in(ctx, word):
                out.append(f"interaction.py registers no built-in {word}Hook to {job}")
    step = _steps(ns, out)
    energy = _array(ns["interaction_energy"]).astype(float)
    start = _array(ns["start_interaction_energy"]).astype(float)
    if start.shape != (N_COMPLEXES,):
        out.append(
            f"start_interaction_energy has shape {start.shape}; one value per complex gives ({N_COMPLEXES},)"
        )
    elif not np.isfinite(start).all():
        out.append("start_interaction_energy holds values that are not finite")
    if step is not None and energy.shape != (step.size, N_COMPLEXES):
        out.append(
            f"interaction_energy has shape {energy.shape}; one row per record and one column per complex "
            f"give ({step.size}, {N_COMPLEXES})"
        )
    elif not np.isfinite(energy).all():
        out.append("interaction_energy holds values that are not finite")
    if out:
        return out
    reference = _start_interaction_ev(ns)  # eV, the check's own single point
    unit = read_unit(start, reference)
    if unit is None:
        return [
            f"start_interaction_energy is {_numbers(start)}; at the starting structure the complexes have "
            f"{_numbers(reference * KCAL)} kcal/mol, {_numbers(reference, 3)} eV, "
            f"{_numbers(reference * UNITS['kJ/mol'], 1)} kJ/mol or "
            f"{_numbers(reference * UNITS['Hartree'], DIGITS['Hartree'])} Hartree, in the order of complexes: "
            "energy of the complex minus the energies of its two fragments"
        ]
    if float(np.ptp(energy, axis=0).max()) == 0.0:
        return ["interaction_energy does not change during the run; record it while the atoms move"]
    gap = int(np.diff(step).max())
    if gap > RECORD_EVERY:
        out.append(
            f"{UNEXPECTED}saved_step advances by {gap} steps; the step records every {RECORD_EVERY} steps. "
            "The plot of section 2 sits on these steps, and a limit read off a coarser plot can retire a second complex"
        )
    to_kcal = KCAL / UNITS[unit]
    digits = DIGITS.get(unit, 2)
    if np.abs(energy[0] - start).max() * to_kcal > FIRST_RECORD:
        out.append(
            f"{UNEXPECTED}the first record is {_numbers(energy[0], digits)} {unit} and the starting value "
            f"{_numbers(start, digits)} {unit}; a run that starts at rest begins at the starting value"
        )
    low, high = energy.min(axis=0), energy.max(axis=0)
    if (high * to_kcal > BOUND).any() or ((low - start) * to_kcal < -BOUND).any():
        out.append(
            f"{UNEXPECTED}interaction_energy runs from {_numbers(low, digits)} to {_numbers(high, digits)} {unit}; "
            f"these pairs start at {_numbers(start, digits)} {unit}. Both fragments are computed at each recorded geometry"
        )
    positions = store_frames(ns["plain_store"])["positions"]
    if len(positions[0]) == step.size:  # frames on the steps of the records: recompute three of them
        rows = sorted({0, step.size // 2, step.size - 1})
        again = _interaction_ev(ns, [p[rows] for p in positions]) * UNITS[unit]
        worst = int(np.abs(again - energy[rows]).max(axis=1).argmax())
        if np.abs(again - energy[rows]).max() * to_kcal > RECOMPUTED:
            out.append(
                f"{UNEXPECTED}at step {int(step[rows[worst]])} interaction_energy is "
                f"{_numbers(energy[rows[worst]], digits)} {unit}; from the frame saved at that record it is "
                f"{_numbers(again[worst], digits)} {unit}"
            )
    return out


def check_separation_hook(ns: dict, ctx: dict) -> list[str]:
    out: list[str] = []
    step = _steps(ns, out)
    if step is None:
        return out
    n = step.size
    start = dict(zip(("separation", "nearest_contact"), _start_distances(ns)))
    slack = {"separation": 0.3, "nearest_contact": 0.6}
    for name in ("separation", "nearest_contact"):
        values = _array(ns[name]).astype(float)
        if values.shape != (n, N_COMPLEXES):
            out.append(
                f"{name} has shape {values.shape}; one row per record of saved_step and one column per complex "
                f"give ({n}, {N_COMPLEXES}) in Å"
            )
            continue
        if not np.isfinite(values).all():
            out.append(f"{name} holds values that are not finite")
            continue
        first = values[0]
        if np.abs(first - start[name]).max() > slack[name]:
            out.append(
                f"{name} starts at {np.round(first, 2).tolist()} Å; the complexes start at "
                f"{np.round(start[name], 2).tolist()} Å, in the order of complexes"
            )
    records = _shape(ns["interaction_energy"])[0] if "interaction_energy" in ns else n
    if not out and records != n:
        out.append(
            f"separation has {n} records and interaction_energy {records}; "
            "record the distances on the steps of the interaction energy"
        )
    if out:
        return out
    if float(np.ptp(_array(ns["separation"]).astype(float), axis=0).max()) < 1e-3:
        return ["separation does not change during the run; record it while the atoms move"]
    if "hooks" in ctx:
        own = [h for h in _own_hooks(ctx) if hasattr(h, "stage") and hasattr(h, "frequency")]
        if not own:
            out.append(
                "separation_hook.py registers no hook of its own (a class with a stage and a frequency); "
                "the step records the distances with a hook while the run goes"
            )
    with contextlib.suppress(Exception):  # the first run's frames, when they sit on the same records
        positions = store_frames(ns["plain_store"])["positions"]
        if len(positions[0]) == n:
            nearest = np.stack(
                [
                    _distances(p, _array(data.atomic_masses).astype(float).reshape(-1, 1), int(split))[1]
                    for p, data, split in zip(positions, ns["complexes"], ns["fragment_a_sizes"])
                ],
                axis=1,
            )
            gap = float(np.abs(nearest - _array(ns["nearest_contact"]).astype(float)).max())
            if gap > SAME_RUN:
                out.append(
                    f"{UNEXPECTED}the nearest contacts differ by up to {gap:.2f} Å from the saved frames of the "
                    "first run: this run did not follow it, so the plot sets two different runs side by side. "
                    "A repeated run follows the first only if it starts the same way"
                )
    return out


def check_freeze(ns: dict, ctx: dict) -> list[str]:
    out: list[str] = []
    mask = ns["frozen_mask"]
    expected = _fragment_a_mask(ns)
    if isinstance(mask, (list, tuple)) and mask and all(np.ndim(_array(m)) >= 1 for m in mask):
        lengths = [_array(m).shape[0] for m in mask]  # one mask per complex, not one over the batch
        return [
            f"frozen_mask holds {len(lengths)} separate masks of {', '.join(map(str, lengths))} entries; "
            f"one entry per atom of the batch gives one mask of shape {expected.shape}"
        ]
    values = _array(mask)
    if values.dtype != np.bool_:
        out.append(f"frozen_mask has type {values.dtype}; it should be boolean, True for a frozen atom")
    if values.shape != expected.shape:
        return out + [
            f"frozen_mask has shape {values.shape}; one entry per atom of the batch gives {expected.shape}"
        ]
    values = values.astype(bool)
    if not np.array_equal(values, expected):
        out.append(
            f"frozen_mask marks {int(values.sum())} atoms; fragment A is the first "
            f"{list(ns['fragment_a_sizes'])} atoms of the complexes, {int(expected.sum())} atoms"
        )
        return out
    frozen = _store_positions(ns, "frozen_store", out)
    if frozen is None:
        return out
    if "plain_store" in ns:
        with contextlib.suppress(Exception):
            plain_frames = len(store_frames(ns["plain_store"])["positions"][0])
            if len(frozen[0]) != plain_frames:
                out.append(
                    f"frozen_store holds {len(frozen[0])} saved frames; the plain run saved {plain_frames}, "
                    "and the frozen run repeats it"
                )
    starts = [_array(c.positions).astype(float) for c in ns["complexes"]]
    moved = np.concatenate([np.linalg.norm(p - s, axis=2) for p, s in zip(frozen, starts)], axis=1)  # (frames, atoms)
    worst = float(moved[:, values].max())
    if worst > FROZEN:
        out.append(f"a frozen atom moved {worst:.3f} Å from its start; every frozen atom stays at 0 Å")
    if float(moved[:, ~values].max()) < 1e-3:
        out.append("the free atoms did not move; only fragment A is frozen")
    if "hooks" in ctx and not _built_in(ctx, "FreezeAtoms"):
        out.append("freeze.py registers no FreezeAtomsHook; the step uses Toolkit's FreezeAtomsHook")
    return out


def still_from(positions: np.ndarray) -> int | None:
    """Index of the saved frame from which a complex no longer moves; None if it moves to the end."""

    moves = np.abs(np.diff(positions, axis=0)).reshape(len(positions) - 1, -1).max(axis=1) > STILL
    if moves[-1]:
        return None
    moving = np.flatnonzero(moves)
    return int(moving[-1]) + 1 if moving.size else 0


def check_retire(ns: dict, ctx: dict) -> list[str]:
    out: list[str] = []
    stop = _array(ns["stop_step"])
    if stop.shape != (N_COMPLEXES,):
        out.append(f"stop_step has shape {stop.shape}; one MD step per complex gives ({N_COMPLEXES},)")
    elif not np.issubdtype(stop.dtype, np.integer):
        out.append(f"stop_step has type {stop.dtype}; it holds whole MD steps, -1 for a complex that ran to the end")
    status = _array(ns["final_status"])
    if status.shape != (N_COMPLEXES,):
        out.append(f"final_status has shape {status.shape}; one status per complex gives ({N_COMPLEXES},)")
    positions = _store_positions(ns, "retire_store", out)
    if out or positions is None:
        return out
    stop = stop.astype(int)
    n_steps = int(ns["run_settings"]["n_steps"])
    frames = len(positions[0])
    per_frame = n_steps / frames  # MD steps between saved frames
    still = [still_from(p) for p in positions]
    for k in range(N_COMPLEXES):
        label = f"complex {k + 1}"
        if stop[k] >= n_steps:
            out.append(
                f"{label}: stop_step is {stop[k]}, the end of the run; a complex that ran out of steps "
                "was not retired, use -1"
            )
        elif stop[k] >= 0 and still[k] is None:
            out.append(f"{label} is marked retired at step {stop[k]} and still moves in the saved frames after it")
        elif stop[k] < 0 and still[k] is not None:
            out.append(
                f"{label} stands still from saved frame {still[k]} on (about step {round(still[k] * per_frame)}) "
                "and stop_step is -1"
            )
        elif stop[k] >= 0 and abs(still[k] - stop[k] / per_frame) > 2:
            out.append(
                f"{label}: stop_step is {stop[k]}; in the saved frames it stands still from about step "
                f"{round(still[k] * per_frame)}"
            )
    if out:
        return out
    retired = [k for k in range(N_COMPLEXES) if stop[k] >= 0]
    if not retired:
        return ["no complex was retired; lower your limit"]
    if len(retired) > 1:
        names = ", ".join(f"complex {k + 1} at step {stop[k]}" for k in retired)
        out.append(
            f"{UNEXPECTED}{len(retired)} complexes were retired ({names}); the step retires one. "
            "Raise your limit so that one complex passes it and the others run on"
        )
    ran_on = [k for k in range(N_COMPLEXES) if k not in retired]
    if ran_on and {int(status[k]) for k in retired} & {int(status[k]) for k in ran_on}:
        out.append(
            f"{UNEXPECTED}final_status is {status.astype(int).tolist()}, the same for a retired complex and for "
            "one that ran to the end; the saved frames tell which complex was stopped"
        )
    # one limit for every complex: a complex that ran on never reached the contact at which another was retired
    nearest = []
    for k, (data, split) in enumerate(zip(ns["complexes"], ns["fragment_a_sizes"])):
        masses = _array(data.atomic_masses).astype(float).reshape(-1, 1)
        nearest.append(_distances(positions[k], masses, int(split))[1])
    at_stop = min(float(nearest[k][-1]) for k in retired)
    for k in range(N_COMPLEXES):
        if k not in retired and float(nearest[k].max()) > at_stop + SAME_LIMIT:
            out.append(
                f"complex {k + 1} ran to the end with a nearest contact of up to {nearest[k].max():.2f} Å; "
                f"another complex was retired at {at_stop:.2f} Å, and one limit holds for every complex"
            )
    for k in retired:
        if still[k] and float(nearest[k][: still[k]].max()) > float(nearest[k][-1]) + SAME_LIMIT:
            out.append(
                f"complex {k + 1} was retired at a nearest contact of {nearest[k][-1]:.2f} Å after it had "
                f"reached {nearest[k][: still[k]].max():.2f} Å; it is retired once it is above the limit"
            )
    with contextlib.suppress(Exception):  # does this run follow the plot the limit was read from?
        plotted = _array(ns["nearest_contact"]).astype(float)
        if plotted.shape == (frames, N_COMPLEXES):
            before = min(still[k] for k in retired)  # saved frames before the first stop
            gap = float(np.abs(np.stack(nearest, axis=1)[:before] - plotted[:before]).max()) if before else 0.0
            if gap > SAME_RUN:
                out.append(
                    f"{UNEXPECTED}before the first stop the nearest contacts differ by up to {gap:.2f} Å from "
                    "your plot of section 2: this run did not follow it. A repeated run follows the plot only "
                    "if it starts the same way"
                )
    return out


OPTIONAL_STEPS = ("freeze",)  # no later step needs its file: a learner may skip it

STEPS: dict[str, Step] = {
    "interaction": Step(
        "interaction.py",
        ("saved_step", "interaction_energy", "start_interaction_energy", "plain_store", "plain_log"),
        (),
        (check_interaction,),
    ),
    "separation_hook": Step(
        "separation_hook.py",
        ("separation", "nearest_contact"),
        ("interaction",),
        (check_separation_hook,),
    ),
    "freeze": Step(
        "freeze.py",
        ("frozen_mask", "frozen_store"),
        ("interaction",),
        (check_freeze,),
    ),
    "retire": Step(
        "retire.py",
        ("stop_step", "final_status", "retire_store"),
        ("interaction", "separation_hook"),
        (check_retire,),
    ),
}


# --------------------------------------------------------------------------
# Running a check
# --------------------------------------------------------------------------


class _WatchHooks:
    """For the duration of a step file, record every hook it registers on an engine.

    The registry's ``register_hook`` is watched: an engine's constructor (``hooks=[...]``)
    and a fused stage both register through it.
    """

    def __init__(self) -> None:
        self.hooks: list[Any] = []

    def __enter__(self):
        from nvalchemi.hooks._registry import HookRegistryMixin

        self._cls, self._orig = HookRegistryMixin, HookRegistryMixin.register_hook
        recorder, orig = self, HookRegistryMixin.register_hook

        def register_hook(engine, hook, *args, **kwargs):
            recorder.hooks.append(hook)
            return orig(engine, hook, *args, **kwargs)

        HookRegistryMixin.register_hook = register_hook
        return self

    def __exit__(self, *exc) -> None:
        self._cls.register_hook = self._orig


@contextlib.contextmanager
def _in_folder(folder: Path):
    """Run with ``folder`` as the working directory, as the notebook runs in ``notebooks/``."""

    before = Path.cwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def _run_file(path: Path, ns: dict) -> Exception | None:
    """Run one step file in the notebook's namespace; its own prints go to stderr."""

    started, real_out = time.time(), sys.stdout
    sys.stdout = sys.stderr  # the file's own prints must not mix with the verdict lines
    try:
        exec(compile(path.read_text(), str(path), "exec"), ns)
    except Exception as exc:
        sys.stdout = real_out
        if os.environ.get("AGENT06_TRACE"):
            traceback.print_exc()
        return exc
    finally:
        sys.stdout = real_out
    print(f"  ({path.name} ran in {time.time() - started:.1f} s)", file=sys.stderr, flush=True)
    return None


def _first_line(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}".strip()


def resolve_step(name: str) -> str:
    """Return the step a learner's spelling names: a stem, a file name or a path."""

    return Path(name).name.removesuffix(".py")


def setup_namespace(n_steps: int | None = None) -> dict:
    """Run the setup cell and return its namespace; ``n_steps`` shortens the run."""

    ns: dict[str, Any] = {"__name__": "__main__"}
    real_out, sys.stdout = sys.stdout, sys.stderr
    try:
        exec(SETUP_CELL, ns)
    finally:
        sys.stdout = real_out
    if n_steps is not None:
        ns["run_settings"]["n_steps"] = int(n_steps)
    return ns


def _earlier_names(ns: dict, step: str) -> dict[str, Any]:
    """The names the steps before ``step`` have left in ``ns``, each with the object it holds now."""

    earlier = list(STEPS)[: list(STEPS).index(step)]
    return {name: ns[name] for e in earlier for name in STEPS[e].names if name in ns}


def _changed(step: str, file: str, before: dict[str, Any], ns: dict) -> list[str]:
    """One "unexpected" line naming the earlier names that ``file`` assigned again; none if it left them."""

    names = [name for name, value in before.items() if ns.get(name) is not value]
    return [CHANGED_LINE.format(step=step, file=file, name=", ".join(names))] if names else []


def check_step(step: str, work_dir: Path | None = None, ns: dict | None = None) -> list[str]:
    """Run ``step`` after the earlier steps in one namespace; return one line per problem.

    ``work_dir`` is the folder with the step files (default ``notebooks/agent06_work``).
    ``ns`` is a namespace in which the setup cell has run (default: run it now).
    """

    from . import agent06 as helpers

    spec = STEPS[step]
    elsewhere = work_dir is not None
    work_dir = Path(work_dir or helpers.WORK_DIR).resolve()
    if ns is None:
        short = os.environ.get("AGENT06_CHECK_STEPS")
        ns = setup_namespace(int(short) if short else None)
    if elsewhere:
        ns["work_dir"] = work_dir  # the files write beside themselves, not into notebooks/agent06_work
    changed: list[str] = []  # earlier names the step file assigned again: a note, not a failure
    with _in_folder(work_dir.parent):
        for earlier in list(STEPS)[: list(STEPS).index(step)]:
            path = work_dir / STEPS[earlier].file
            if earlier in spec.after:
                if not path.exists():
                    return [f"{step}: the earlier step file {path.name} is missing in {work_dir}"]
                error = _run_file(path, ns)
                if error is not None:
                    return [f"{step}: an earlier file failed: {path.name}: {_first_line(error)}"]
            elif earlier in OPTIONAL_STEPS and path.exists():
                _run_file(path, ns)  # the "Try it" file runs if it is there; its errors are its own
        path = work_dir / spec.file
        if not path.exists():
            return [f"{step}: the step file {spec.file} is missing in {work_dir}"]
        kept = _earlier_names(ns, step)  # as the earlier steps left them
        ctx: dict[str, Any] = {}
        for attempt in (1, 2):  # the second run is a learner running the cell again
            with _WatchHooks() as watch:
                error = _run_file(path, ns)
            ctx["hooks"] = watch.hooks
            if error is None:
                continue
            if isinstance(error, NameError) and getattr(error, "name", None) in _SCRIPT_NAMES:
                return [f"{step}: {spec.file} uses {error.name}, which does not exist in a notebook cell; remove that line"]
            if isinstance(error, NameError) and getattr(error, "name", None):
                return [f"{step}: {spec.file} uses {error.name}, which no earlier cell defines; define it in this file"]
            again = " a second time" if attempt == 2 else ""
            return [f"{step}: running {spec.file}{again} failed: {_first_line(error)}"]
        changed += _changed(step, spec.file, kept, ns)
        problems = [f"{step}: {name} is missing; the step asks for it" for name in spec.names if name not in ns]
        if not problems:
            for check in spec.checks:
                try:
                    problems += [f"{step}: {line}" for line in check(ns, ctx)]
                except Exception as exc:
                    problems.append(f"{step}: the check could not read the results: {_first_line(exc)}")
    problems += changed
    notes = [line for line in problems if line.startswith(f"{step}: {UNEXPECTED}")]
    failures = [line for line in problems if line not in notes]
    return failures + notes if failures else [f"{step}: OK ({len(spec.names)} names)"] + notes


def main(argv: list[str]) -> int:
    """Entry point of ``python -m helpers.agent06 check <step>``."""

    step = resolve_step(argv[1]) if len(argv) in (2, 3) else ""
    if len(argv) not in (2, 3) or argv[0] != "check" or step not in STEPS:
        print("usage: python -m helpers.agent06 check <step>\nsteps: " + ", ".join(STEPS))
        return 2
    started = time.time()
    lines = check_step(step, Path(argv[2]) if len(argv) == 3 else None)
    print("\n".join(dict.fromkeys(lines)))
    print(f"  (check took {time.time() - started:.1f} s)", file=sys.stderr)
    return 0 if ": OK (" in lines[0] else 1
