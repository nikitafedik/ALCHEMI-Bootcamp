# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 03, Ready-made.

The loader for the crystal set, the plotters and the table of the notebook, and
the step check (``python -m helpers.agent03 check <step>``, run from
``notebooks/``). The step contracts are in ``STEPS``; every line the check or a
view prints is in ``MESSAGES``.

Every plotter has two layers: a ``build_*`` function that returns a Matplotlib
figure (used by the tests and for saving PNGs) and the public function that
shows the figure as one 920 px picture.
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._agent03_loaders import build_crystals as _build_crystals
from .common import configure_tutorial as _configure_tutorial

__all__ = [
    "configure_tutorial",
    "load_crystals",
    "load_ladder_crystal",
    "make_work_dir",
    "plot_batch_speedup",
    "plot_energy_drift",
    "plot_force_histograms",
    "plot_start_temperature",
    "plot_temperature_ladder",
    "plot_timestep_scan",
    "prepare_model_batch",
    "show_crystal_grid",
    "show_session",
    "show_speed_table",
]

# MACE prints this line to stdout each time a model loads from the cache (fix pass 27 Nov, R4).
_MACE_DEFAULT_LINE = "Using medium MPA-0 model as default MACE-MP model"


def _quiet_mace_default_line() -> None:
    """Drop MACE's "Using medium MPA-0 model as default" line; every other line passes through.

    ``MACEWrapper.from_checkpoint`` calls ``mace.calculators.foundations_models.download_mace_mp_checkpoint``
    at each load, and that function prints the line whenever the checkpoint is not the copy inside the
    mace package, so the setup cell and the speed step printed it once per model. It is a print, not a
    warning, so no warnings filter reaches it: the function is wrapped once, by message, in its own module.
    """

    try:
        # Importing mace makes e3nn warn about TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD; that
        # notice does not describe the learner's run, so import inside the course's filter.
        from .model_support import suppress_known_mace_load_noise

        with suppress_known_mace_load_noise():
            from mace.calculators import foundations_models
    except Exception:  # without MACE there is no line to drop
        return
    original = foundations_models.download_mace_mp_checkpoint
    if getattr(original, "_alchemi_quiet", False):
        return

    @functools.wraps(original)
    def quiet(*args, **kwargs):
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                return original(*args, **kwargs)
        finally:
            kept = [line for line in buffer.getvalue().splitlines(keepends=True) if not line.startswith(_MACE_DEFAULT_LINE)]
            sys.stdout.write("".join(kept))

    quiet._alchemi_quiet = True
    foundations_models.download_mace_mp_checkpoint = quiet


def configure_tutorial() -> None:
    """Apply the course style (light only) and drop MACE's repeated default-model line."""

    _configure_tutorial()
    _quiet_mace_default_line()


# The work folder. ALCHEMI_AGENT03_WORK names another one (the tests and the CPU proof use it).
WORK_DIR = Path(os.environ.get("ALCHEMI_AGENT03_WORK") or Path(__file__).resolve().parents[1] / "agent03_work")
_DATA = Path(__file__).resolve().parents[1] / "data" / "agent03"
# The saved default crystal of the temperature ladder (data/agent03/README.md).
LADDER_CRYSTAL = _DATA / "ladder-default-Cu-fcc-4x4x4.extxyz"
NOTEBOOK = "notebooks/alchemi-agent-03-ready-made.ipynb"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STYLE = _PROJECT_ROOT / "shared" / "alchemi-figure.mplstyle"
DISPLAY_PX = 920

# ---------------------------------------------------------------------------
# Every line the check or a view prints for the learner (or raises at them), in one
# place. The code fills the {placeholders}. Keys marked DRAFT are new wording for the
# educator; the others are the wording that was already in use. The list with each
# key and its text is notes/builder-03/eng/printed_lines.md.
# ---------------------------------------------------------------------------
MESSAGES = {
    # --- the session and the loaders ---------------------------------------------------
    "session": (
        "Give these to your agent when you start the session:\n"
        "  notebook    {notebook}\n"
        "  helpers     from helpers import agent03 as helpers\n"
        "  work in     notebooks/agent03_work/\n"
        "  check with  python -m helpers.agent03 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
    ),
    "load_crystals_n": "load_crystals: n should be at least 1; found {n}",
    # --- the plotters (raised as ValueError) ----------------------------------------
    "grid_mismatch": "show_crystal_grid: start_max_force has {n} values but crystal_batch holds {m} crystals",
    "drift_shape": (
        "plot_energy_drift: crystal_energy should have shape (n_saved, n_structures) = "
        "({n_saved}, {n_structures}); found {found}"
    ),
    "scan_shape": (
        "plot_timestep_scan: total_energy should have shape (n_saved, n_replicas) = "
        "({n_saved}, {n_replicas}); found {found}"
    ),
    "speed_view_columns": (  # DRAFT
        "show_speed_table: speed_table should have the columns cueq, compile, first_call_s and "
        "repeat_s; missing {missing}"
    ),
    "ladder_shape": (  # DRAFT
        "plot_temperature_ladder: replica_energy should have shape (n_saved, n_temperatures) = "
        "({n_saved}, {n_temperatures}); found {found}"
    ),
    # --- every step: the names and their contract -------------------------------------
    "missing": "{step}: {name} is missing; the step file should create it",
    "not_batch": "{step}: {name} should be a Batch; found {found}",
    "not_model": "{step}: {name} should be a Toolkit model; found {found}",
    "unit": " in {unit}",
    "describe_shape": "{kind} of shape {shape}",
    "describe_length": "{kind} of length {n}",
    "not_numbers": "{step}: {name} should be numbers{unit}; found {found}",
    "not_one_number": "{step}: {name} should be one number{unit}; found {found}",
    "wrong_shape": "{step}: {name} should have shape {shape}{unit}; found shape {found}",
    "not_finite": (
        "{step}: {name} should be finite everywhere; found {bad} of {n} values that are not (nan or inf)"
    ),
    "not_positive": (
        "{step}: {name} should be above 0 everywhere; found {bad} of {n} values at or below 0 "
        "(smallest {smallest:.4g})"
    ),
    "not_integer": "{step}: {name} should hold whole numbers (step counts); found values with a fraction",
    "not_increasing": "{step}: {name} should increase from one saved row to the next; found rows that do not",
    "same_graphs": (
        "{step}: {name} should hold the same {n} graphs as crystal_batch (same atom counts, same order); "
        "found {m} graphs"
    ),
    # --- relax_build, relax_run ---------------------------------------------------------
    "same_precision": (
        "{step}: mace_model and crystal_batch should use the same number precision; "
        "found model {model} and positions {positions}"
    ),
    "force_vector": (
        "{step}: {name} should be the largest force on any atom, measured as the length of the "
        "force vector, in eV/Å; found values that differ for {k} of {n} crystals "
        "(for example {found:.4g} instead of {expected:.4g})"
    ),
    "crystal_count": "relax_build: crystal_batch should have {want} graphs (n_crystals); found {found}",
    "periodic_none": (
        "relax_build: every crystal should be periodic with a cell; found a batch with no pbc or no cell"
    ),
    "periodic_count": (
        "relax_build: every crystal should be periodic with a cell; found {nopbc} of {n} without full "
        "pbc and {nocell} with an empty cell"
    ),
    "threshold": (
        "relax_run: every relaxed_max_force should be at or below force_threshold ({threshold:g} eV/Å); "
        "found {over} of {n} above it (largest {largest:.4g} eV/Å)"
    ),
    "same_settings": (
        "{step}: the one-by-one timing should use the same model settings as the batched run "
        "(mace_model: {ref}); found a model with {found}"
    ),
    "single_sizes": (  # DRAFT (a note: the check still passes)
        "unexpected: the crystals timed one by one hold {sizes} atoms, while the batch averages "
        "{mean:.0f} atoms per crystal; the projected one-by-one time depends on that choice"
    ),
    "relaxed_all_below": (  # DRAFT: printed under the force histograms
        "The largest force of every relaxed crystal is below the threshold (dashed line)."
    ),
    "relaxed_above": (  # DRAFT: printed under the force histograms
        "unexpected: {k} relaxed crystals are above the threshold"
    ),
    "settings_words": "cuEquivariance {cueq}, {compiled}, {dtype}",
    "on": "on",
    "off": "off",
    "compiled": "compiled",
    "not_compiled": "not compiled",
    # --- md_build, md_run ---------------------------------------------------------------
    "no_velocities": "md_build: md_batch should carry velocities; found none",
    "bad_velocities": "md_build: md_batch.velocities should be finite with shape ({atoms}, 3); found shape {found}",
    "md_finished": (
        "md_build: no graph of md_batch should be marked finished before the MD starts; found {k} of "
        "{n} marked finished (status 1 or more, left over from the relaxation)"
    ),
    "md_not_moved": "md_run: no atom moved in {k} of {n} crystals during the MD{hint}",
    "md_not_moved_hint": "; md_batch may still carry the relaxation's finished marks",
    "time_axis": (
        "md_run: time_fs should give the time of each saved energy row; {which} was recorded at "
        "{expected:g} fs but time_fs says {found:g} fs"
    ),
    "ladder_time_axis": (  # DRAFT (a note: the check still passes)
        "unexpected: time_fs should give the time of each saved energy row; {which} was recorded "
        "at {expected:g} fs but time_fs says {found:g} fs"
    ),
    "time_axis_last": "the last row",
    "time_axis_row": "row {row}",
    # --- timestep_scan (draft exercise); some lines serve the ladder as well ------------
    "scan_one_batch": (
        "{step}: the copies should run together in one batch, one graph for each entry of replica_dt "
        "({k}); found runs of {sizes} graphs"
    ),
    "copies_sizes": "{step}: every copy should be the same crystal; found copies with different atom counts",
    "copies_atoms": "{step}: every copy should be the same crystal; found copies with different atoms",
    "scan_positions": "{step}: every copy should start from the same positions; found copies that differ",
    "copies_periodic": (
        "{step}: the copies should be a periodic crystal (a cell and periodic boundaries on all three "
        "axes); found {found}; MACE-MPA-0 is a model for materials"
    ),
    "copies_periodic_none": "copies that carry no cell or no periodic boundaries",
    "copies_periodic_count": (
        "{nopbc} of {k} copies without periodic boundaries on all three axes and {nocell} with an empty cell"
    ),
    "scan_finished": (
        "{step}: no copy should be marked finished before the run starts; found {k} of {n} marked "
        "finished (status 1 or more, left over from the relaxation)"
    ),
    "scan_not_moved": (
        "{step}: the atoms of {still} of {k} copies did not move during the run, so their energy "
        "cannot drift{hint}"
    ),
    "copies_finished_hint": "; the copies may still carry the relaxation's finished marks",
    "scan_dt": (
        "{step}: replica_dt should be the timestep each copy ran with (fs); the run used {used} but "
        "replica_dt says {want}"
    ),
    "scan_distinct": (
        "{step}: replica_dt should hold two or more different timesteps, one for each copy, in fs; "
        "found {distinct} different timestep{s} among {n}"
    ),
    "scan_rows": "{step}: saved_step should hold two or more saved rows; found {n}",
    "scan_first_row": (
        "{step}: the first row of total_energy should be finite for every copy; found {k} of {n} "
        "copies that are not"
    ),
    "scan_none_stable": (
        "{step}: total_energy should stay finite and within {words} of the first row for at least "
        "one copy; found none"
    ),
    "scan_limit_atom": "{limit:g} eV per atom",
    "scan_limit_copy": "{limit:g} eV per copy",
    # --- temperature_ladder (draft exercise): every line is DRAFT -----------------------
    "ladder_one_batch": (  # DRAFT
        "{step}: the copies should run together in one batch, one graph for each temperature "
        "(n_temperatures = {k}); found runs of {sizes} graphs"
    ),
    "ladder_cells": (  # DRAFT
        "{step}: every copy should be the same crystal; found copies with different cells"
    ),
    "ladder_not_moved": (  # DRAFT
        "{step}: the atoms of {still} of {k} copies did not move during the run{hint}"
    ),
    "ladder_flat": (  # DRAFT
        "{step}: replica_energy of {still} of {k} copies does not change along the run, so their "
        "atoms did not move{hint}"
    ),
    "ladder_distinct": (  # DRAFT
        "{step}: replica_temperature should hold a different temperature for each copy, in K; "
        "found {distinct} different temperature{s} among {n}"
    ),
    "ladder_first_row": (  # DRAFT
        "{step}: the first row of replica_energy should be finite for every copy; found {k} of {n} "
        "copies that are not"
    ),
    "ladder_target": (  # DRAFT
        "{step}: replica_temperature should be the temperature each copy's thermostat held (K); "
        "the run used {used} but replica_temperature says {want}"
    ),
    "ladder_whole_copy": (  # DRAFT (a note: the check still passes)
        "replica_energy should be in eV per atom; its first row is about {value:.4g} eV, the size of "
        "the energy of a whole copy{atoms}; divide each copy's energy by its number of atoms"
    ),
    "ladder_whole_copy_atoms": " of {n} atoms",  # DRAFT
    "ladder_order": (  # DRAFT (a note: the check still passes)
        "unexpected: over the last third of the run, the copy at {hot:.0f} K sits below the copy at "
        "{cold:.0f} K ({hot_e:.0f} against {cold_e:.0f} meV per atom)"
    ),
    "ladder_below_start": (  # DRAFT (a note: the check still passes)
        "unexpected: the energy of {k} of {n} copies falls below the first row, so the copies did "
        "not start from a relaxed crystal; the zero of the plot is the first row"
    ),
    "ladder_start_differs": (  # DRAFT (a note: the check still passes)
        "unexpected: the first row of replica_energy differs between the copies by {spread:.3g} meV "
        "per atom; it should be the energy of the crystal before the first step, the same for every copy"
    ),
    "ladder_all_blown": (  # DRAFT (a note: the check still passes)
        "unexpected: every copy blew up, so the order of the energies cannot be judged"
    ),
    "ladder_kinetic": (  # DRAFT (a note: the check still passes)
        "unexpected: at the end of the run the copy at {hot:.0f} K had a lower kinetic temperature "
        "({hot_k:.0f} K) than the copy at {cold:.0f} K ({cold_k:.0f} K)"
    ),
    # --- speed_settings: every line is DRAFT --------------------------------------------
    "speed_not_table": (  # DRAFT
        "{step}: speed_table should be a table (pandas DataFrame); found {found}"
    ),
    "speed_columns": (  # DRAFT
        "{step}: speed_table should have the columns cueq, compile, first_call_s and repeat_s; "
        "missing {missing}"
    ),
    "speed_flags": (  # DRAFT
        "{step}: the {column} column of speed_table should hold True or False; found {found}"
    ),
    "speed_case_missing": "{step}: speed_table is missing the case {case}",  # DRAFT
    "speed_case_twice": (  # DRAFT
        "{step}: speed_table should hold each case once; found {case} {count} times"
    ),
    "speed_time": (  # DRAFT
        "{step}: {column} should be a time in s, finite and above 0, for every case; found {found} "
        "for {case}"
    ),
    "speed_roles": (  # DRAFT
        "{step}: setting_roles should be a dict with a short text for each of the keys cueq and "
        "compile; found {found}"
    ),
    "speed_roles_keys": "the keys {keys}",  # DRAFT
    "speed_roles_none": "an empty dict",  # DRAFT
    "speed_roles_empty": "no text for {keys}",  # DRAFT
    "speed_case": "cuEquivariance {cueq}, compile {compile}",  # DRAFT
    "speed_slower": (  # DRAFT (a note: the check still passes, the view prints it too)
        "unexpected: with cuEquivariance and compile both on, a repeat took {both:.3g} s, longer "
        "than the {neither:.3g} s with both off"
    ),
    "speed_model": (  # DRAFT (a note: the check still passes)
        "unexpected: no model with {case} was called while the file ran, so that row may come from "
        "other settings (models called: {found})"
    ),
    "rows_repeat": (  # DRAFT (validation of 26 Nov 2026)
        "{step}: the {n} saved rows of {name} after the first hold exactly the same numbers, so they "
        "were not recorded along the run; keep a copy of the energies each time a row is saved"
    ),
    # --- running the files --------------------------------------------------------------
    "note": "{step}: {note}",  # DRAFT: how a note is printed by the check
    "earlier_missing": "{step}: the earlier step file {file} is missing in {work}",
    "earlier_failed": "{step}: could not run the earlier steps; {error}",
    "step_missing": "{step}: the step file {file} is missing in {work}",
    "stopped": "{label}: {file} stopped with {error}: {first_line}",
    # DRAFT (track rule B, 26 Nov): a name a script has and a notebook cell does not.
    "script_name": "{step}: {file} uses {name}, which does not exist in a notebook cell; remove that line",
    "ran_in": "  ({file} ran in {seconds:.1f} s)",
    "ok": "{step}: OK ({n} names)",
    "check_took": "  (check took {seconds:.1f} s)",
    "usage": "usage: python -m helpers.agent03 check <{steps}>",
}


def _say(key: str, **values) -> str:
    """One learner-facing line from MESSAGES, filled in."""

    return MESSAGES[key].format(**values)


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    WORK_DIR.mkdir(exist_ok=True)
    return WORK_DIR


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(_say("session", notebook=NOTEBOOK))


def prepare_model_batch(batch, model):
    """Apply a model's neighbor hooks before one standalone forward call."""

    from .model_support import prepare_model_batch as implementation

    return implementation(batch, model)


# ---------------------------------------------------------------------------
# The crystal set
# ---------------------------------------------------------------------------


def load_crystals(n: int = 200):
    """Return ``n`` common crystals as ASE ``Atoms`` (ASE bulk supercells, atoms rattled 0.1 Å).

    The set is built once from 285 crystals and ``n`` of them are picked evenly,
    so every ``n`` gives a mix of metals, semiconductors and salts of varied size.
    The result is the same on every call. ``n`` above 285 returns all 285.
    """

    if n < 1:
        raise ValueError(_say("load_crystals_n", n=n))
    frames = _build_crystals()
    if n >= len(frames):
        return frames
    pick = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [frames[i] for i in pick]


def load_ladder_crystal():
    """Return the default crystal of the temperature ladder as ASE ``Atoms``.

    Copper, fcc, 4 x 4 x 4 primitive cells: 64 atoms, periodic on all three axes, every atom
    on its lattice site (no rattle), so the forces are zero and no relaxation is needed. The
    file is ``data/agent03/ladder-default-Cu-fcc-4x4x4.extxyz``; the result is the same on
    every call.
    """

    from ase.io import read

    return read(LADDER_CRYSTAL)


# ---------------------------------------------------------------------------
# Wording on the figures: plain axis labels "Quantity [unit]" and the names
# the plan or the board give. Alt texts describe what the picture shows.
# ---------------------------------------------------------------------------
LABELS = {
    "grid": {
        "colorbar": "Largest force [eV/Å]",
        "alt": "One box for each crystal, coloured by the largest force on any of its atoms, from grey for small forces to red for large ones.",
    },
    "histograms": {
        "xlabel_relaxed": "Largest force after relaxation [eV/Å]",
        "xlabel_start": "Largest force at the start [eV/Å]",
        "ylabel": "Number of crystals",
        "threshold": "threshold",
        "alt": "Two histograms of the largest force on an atom in each crystal: after the relaxation on the left, at the start on the right. A dashed line marks the force threshold.",
    },
    "speedup": {
        "ylabel": "Time [s]",
        "batched": "batched",
        "one_by_one": "one by one",
        "projected": "projected from {n} structures",
        "alt": "Two bars: the time of the batched relaxation and the time of relaxing the same structures one by one, projected from the structures that were timed.",
    },
    "temperature": {
        "xlabel": "Crystal",
        "ylabel": "Starting temperature [K]",
        "alt": "The starting temperature of each crystal, one dot per crystal.",
    },
    "drift": {
        "xlabel": "Time [fs]",
        "ylabel": "Change in total energy [meV/atom]",
        "alt": "The change in total energy per atom since the start of the run, one thin line for each crystal.",
    },
    # DRAFT exercise (timestep scan), not in the notebook yet. The axis labels follow the
    # "drift" labels; the alt text is a PLACEHOLDER for the educator to word.
    "timestep_scan": {
        "xlabel": "Time [fs]",
        "ylabel": "Change in total energy [meV]",
        "timestep": "{dt:g} fs",
        "alt": "PLACEHOLDER: change in total energy over time for copies of one crystal, one line for each timestep.",
    },
    # DRAFT exercise (temperature ladder), not in the notebook yet. Every entry is DRAFT.
    "temperature_ladder": {
        "xlabel": "Time [fs]",
        "ylabel": "Potential energy above the relaxed crystal [meV/atom]",
        "temperature": "{t:.0f} K",
        "alt": (
            "Potential energy per atom above the relaxed crystal over time for copies of one "
            "crystal, one line for each temperature, coloured from cold (blue) to hot (red); "
            "a cross marks a copy that blew up."
        ),
    },
    "speed_table": {
        "cueq": "cuEquivariance",
        "compile": "Compile",
        "seconds": "Time [s]",
        "first_call_seconds": "First call [s]",
        "first_run_seconds": "First run [s]",
        "first_call_s": "First call [s]",  # DRAFT: the first_call_s column
        "repeat_s": "Repeat [s]",  # DRAFT: the repeat_s column
        "setting": "Setting",
        "role": "Role",
        "on": "on",
        "off": "off",
    },
}

# Colours of the course figure style: green main result, grey baseline, ink reference.
GREEN, GREY, LIGHT_GREY, INK = "#76B900", "#6F6F6F", "#B8B8B8", "#1A1A1A"
# One constant for the force colours of the crystal grid: neutral grey to red (mock-red is open on the board).
GRID_COLORS = ("#E4E4E4", "#F0B49C", "#D1352B")
# The energy-drift axis spans at least this much on each side of zero, in meV per atom, and grows
# when a crystal drifts further. Below this, noise is not magnified into drama.
DRIFT_AXIS_FLOOR_MEV = 1.0


def _array(values: Any) -> np.ndarray:
    """Return a tensor or array-like as a float NumPy array on the host."""

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    return np.asarray(values, dtype=float)


@contextlib.contextmanager
def _course_style():
    """The course figure style (Open Sans, 920 px wide) for everything drawn inside."""

    import matplotlib.pyplot as plt

    from .common import register_course_fonts

    register_course_fonts()  # before the style names the course font
    with plt.style.context(str(_STYLE)):
        yield


def _in_course_style(build):
    """Run a ``build_*`` function inside the course figure style."""

    @functools.wraps(build)
    def wrapper(*args, **kwargs):
        with _course_style():
            return build(*args, **kwargs)

    return wrapper


def _new_figure(*args, **kwargs):
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


def _show(figure, alt: str) -> None:
    """Show a figure as one picture, at most 920 px wide, with alt text."""

    from base64 import b64encode
    from html import escape
    from io import BytesIO

    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    buffer = BytesIO()
    with _course_style():
        figure.savefig(buffer, format="png", dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)
    payload = b64encode(buffer.getvalue()).decode("ascii")
    display(
        HTML(
            f'<img src="data:image/png;base64,{payload}" alt="{escape(alt, quote=True)}" '
            f'style="display:block;box-sizing:border-box;width:100%;max-width:{DISPLAY_PX}px;'
            'height:auto;margin:0">'
        )
    )


def _seconds_text(value: float) -> str:
    if value >= 100:
        return f"{value:.0f} s"
    return f"{value:.1f} s" if value >= 10 else f"{value:.2f} s"


# ---------------------------------------------------------------------------
# Cell 11: show_crystal_grid
# ---------------------------------------------------------------------------


@_in_course_style
def build_crystal_grid_figure(crystal_batch, start_max_force):
    """One box per crystal, coloured by its largest force on an atom; red = far from the minimum."""

    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    force = _array(start_max_force).reshape(-1)
    n = force.size
    if hasattr(crystal_batch, "num_graphs") and int(crystal_batch.num_graphs) != n:
        raise ValueError(_say("grid_mismatch", n=n, m=int(crystal_batch.num_graphs)))
    finite = force[np.isfinite(force) & (force > 0)]
    vmin = float(finite.min()) if finite.size else 1.0
    vmax = float(finite.max()) if finite.size else 10.0
    if vmax <= vmin:
        vmax = vmin * 10.0
    cmap = LinearSegmentedColormap.from_list("force", GRID_COLORS).with_extremes(bad="#FFFFFF")
    cols = min(n, 25)
    rows = -(-n // cols)
    values = np.full(rows * cols, np.nan)
    values[:n] = force
    values = np.ma.masked_invalid(values.reshape(rows, cols))
    figure, axis = _new_figure(figsize=(9.2, max(1.8, 0.34 * rows + 1.0)))
    mesh = axis.pcolormesh(
        values, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), edgecolors="white", linewidth=2.0
    )
    axis.set_xlim(0, cols)
    axis.set_ylim(rows, 0)
    axis.set_aspect("equal")
    axis.grid(False)
    axis.axis("off")
    colorbar = figure.colorbar(mesh, ax=axis, orientation="horizontal", shrink=0.5, aspect=40, pad=0.04)
    colorbar.set_label(LABELS["grid"]["colorbar"])
    colorbar.outline.set_visible(False)
    return figure


def show_crystal_grid(crystal_batch, start_max_force) -> None:
    """Show one box per crystal, coloured by its largest force on an atom."""

    _show(build_crystal_grid_figure(crystal_batch, start_max_force), LABELS["grid"]["alt"])


# ---------------------------------------------------------------------------
# Cell 14: plot_force_histograms
# ---------------------------------------------------------------------------


def _log_edges(low: float, threshold: float, high: float, total: int = 30) -> np.ndarray:
    """Log-spaced bin edges with the threshold as one edge, so no bin straddles it."""

    below = np.log10(threshold / low)
    above = np.log10(high / threshold)
    n_below = max(3, round(total * below / (below + above)))
    n_above = max(3, total - n_below)
    return np.unique(
        np.concatenate(
            [np.geomspace(low, threshold, n_below + 1), np.geomspace(threshold, high, n_above + 1)]
        )
    )


def histogram_data(start_max_force, relaxed_max_force, force_threshold) -> dict[str, Any]:
    """Bin edges and counts of both histograms, and how many relaxed crystals sit above the threshold."""

    start = _array(start_max_force).reshape(-1)
    relaxed = _array(relaxed_max_force).reshape(-1)
    threshold = float(_array(force_threshold).reshape(-1)[0])
    floor = 1e-6
    start = np.clip(start[np.isfinite(start)], floor, None)
    relaxed = np.clip(relaxed[np.isfinite(relaxed)], floor, None)
    both = np.concatenate([start, relaxed])
    low = min(float(both.min()) * 0.8, threshold * 0.5)
    high = max(float(both.max()) * 1.25, threshold * 2.0)
    edges = _log_edges(low, threshold, high)
    return {
        "edges": edges,
        "start": np.histogram(start, bins=edges)[0],
        "relaxed": np.histogram(relaxed, bins=edges)[0],
        "threshold": threshold,
        "relaxed_above": int((relaxed > threshold).sum()),
    }


@_in_course_style
def build_force_histograms_figure(start_max_force, relaxed_max_force, force_threshold):
    """Two histograms on one log axis: after the relaxation on the left, at the start on the right."""

    data = histogram_data(start_max_force, relaxed_max_force, force_threshold)
    edges = data["edges"]
    labels = LABELS["histograms"]
    figure, (left, right) = _new_figure(1, 2, figsize=(9.2, 4.0), sharex=True, sharey=True)
    for axis, key, color, xlabel in (
        (left, "relaxed", GREEN, labels["xlabel_relaxed"]),
        (right, "start", GREY, labels["xlabel_start"]),
    ):
        axis.stairs(data[key], edges, fill=True, color=color, alpha=0.9, linewidth=0)
        axis.axvline(data["threshold"], color=INK, linestyle="--", linewidth=1.4, zorder=5)
        axis.set_xscale("log")
        axis.set_xlabel(xlabel)
    left.set_ylabel(labels["ylabel"])
    left.annotate(
        labels["threshold"],
        xy=(data["threshold"], 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=7.92,
        color=INK,
    )
    left.set_xlim(edges[0], edges[-1])
    return figure


def plot_force_histograms(start_max_force, relaxed_max_force, force_threshold) -> None:
    """Show the largest force of each crystal after the relaxation (left) and at the start (right);
    then print how the relaxed crystals sit against the threshold.

    start_max_force, relaxed_max_force: (n_structures,) eV/Å, the largest force on any atom of each
    crystal (the length of the force vector). force_threshold: one number, eV/Å.
    """

    _show(
        build_force_histograms_figure(start_max_force, relaxed_max_force, force_threshold),
        LABELS["histograms"]["alt"],
    )
    above = histogram_data(start_max_force, relaxed_max_force, force_threshold)["relaxed_above"]
    print(_say("relaxed_above", k=above) if above else _say("relaxed_all_below"))


# ---------------------------------------------------------------------------
# Cell 16: plot_batch_speedup
# ---------------------------------------------------------------------------


def speedup_data(batch_seconds, single_seconds, n_structures) -> dict[str, Any]:
    """The batched time, and the one-by-one time projected from the timed structures."""

    singles = _array(single_seconds).reshape(-1)
    batched = float(_array(batch_seconds).reshape(-1)[0])
    return {
        "batched": batched,
        "one_by_one": float(singles.mean()) * int(n_structures),
        "n_timed": int(singles.size),
    }


@_in_course_style
def build_batch_speedup_figure(batch_seconds, single_seconds, n_structures):
    """Two bars: the batched run and the one-by-one time projected from the timed structures."""

    data = speedup_data(batch_seconds, single_seconds, n_structures)
    labels = LABELS["speedup"]
    figure, axis = _new_figure(figsize=(9.2, 4.0))
    names = [
        labels["batched"],
        f'{labels["one_by_one"]}\n({labels["projected"].format(n=data["n_timed"])})',
    ]
    values = [data["batched"], data["one_by_one"]]
    bars = axis.bar(names, values, width=0.5, color=[GREEN, GREY])
    top = max(values)
    for bar, value in zip(bars, values):
        axis.annotate(
            _seconds_text(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.36,
            color=INK,
        )
    axis.set_ylim(0, top * 1.15)
    axis.set_ylabel(labels["ylabel"])
    axis.grid(False)
    axis.tick_params(axis="x", length=0)
    return figure


def plot_batch_speedup(batch_seconds, single_seconds, n_structures) -> None:
    """Show the time of the batched run beside the one-by-one time projected from the timed structures.

    batch_seconds: one number, s, the time to relax the whole batch in one run.
    single_seconds: (3,) s, the time to relax three crystals one at a time, each of about the
    average atom count of the batch; the helper projects their mean to n_structures crystals.
    n_structures: the number of crystals in the batch.
    """

    _show(
        build_batch_speedup_figure(batch_seconds, single_seconds, n_structures),
        LABELS["speedup"]["alt"],
    )


# ---------------------------------------------------------------------------
# Cell 21: plot_start_temperature
# ---------------------------------------------------------------------------


@_in_course_style
def build_start_temperature_figure(start_temperature):
    """One dot per crystal: its starting temperature, on an axis that starts at 0 K."""

    temperature = _array(start_temperature).reshape(-1)
    labels = LABELS["temperature"]
    figure, axis = _new_figure(figsize=(9.2, 4.0))
    axis.plot(np.arange(1, temperature.size + 1), temperature, "o", color=GREEN, markersize=3.5)
    top = float(np.nanmax(temperature)) if temperature.size else 1.0
    axis.set_ylim(0, max(top * 1.25, 1.0))
    axis.set_xlim(0, temperature.size + 1)
    axis.set_xlabel(labels["xlabel"])
    axis.set_ylabel(labels["ylabel"])
    return figure


def plot_start_temperature(start_temperature) -> None:
    """Show the starting temperature of each crystal."""

    _show(build_start_temperature_figure(start_temperature), LABELS["temperature"]["alt"])


# ---------------------------------------------------------------------------
# Cell 24: plot_energy_drift
# ---------------------------------------------------------------------------


def drift_data(time_fs, crystal_energy, md_batch) -> dict[str, np.ndarray]:
    """Change of the total energy per atom since the first saved row, in meV, one column per crystal."""

    time = _array(time_fs).reshape(-1)
    energy = _array(crystal_energy)
    atoms = _array(md_batch.num_nodes_per_graph).reshape(-1)
    if energy.ndim != 2 or energy.shape != (time.size, atoms.size):
        raise ValueError(
            _say("drift_shape", n_saved=time.size, n_structures=atoms.size, found=tuple(energy.shape))
        )
    return {"time": time, "drift": (energy - energy[0]) / atoms * 1000.0}


@_in_course_style
def build_energy_drift_figure(time_fs, crystal_energy, md_batch):
    """One thin line per crystal: total energy per atom minus its start value."""

    data = drift_data(time_fs, crystal_energy, md_batch)
    labels = LABELS["drift"]
    figure, axis = _new_figure(figsize=(9.2, 4.0))
    axis.plot(data["time"], data["drift"], color=GREEN, linewidth=0.6, alpha=0.5)
    axis.axhline(0.0, color=INK, linewidth=0.8, zorder=1)
    largest = float(np.nanmax(np.abs(data["drift"]))) if data["drift"].size else 0.0
    half = max(largest * 1.15, DRIFT_AXIS_FLOOR_MEV)
    axis.set_ylim(-half, half)
    axis.set_xlim(float(data["time"].min()), float(data["time"].max()))
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.set_xlabel(labels["xlabel"])
    axis.set_ylabel(labels["ylabel"])
    return figure


def plot_energy_drift(time_fs, crystal_energy, md_batch) -> None:
    """Show the change in total energy per atom of each crystal during the run."""

    _show(build_energy_drift_figure(time_fs, crystal_energy, md_batch), LABELS["drift"]["alt"])


# ---------------------------------------------------------------------------
# Draft exercise (timestep scan): plot_timestep_scan
# ---------------------------------------------------------------------------


# A copy counts as blown up when its total energy is not finite or leaves its first row by more
# than this (a stable copy moves by meV; the blow-ups seen on crystals were 1e5 eV and more).
SCAN_BLOWUP_EV_PER_ATOM = 1.0
SCAN_BLOWUP_EV_PER_COPY = 10.0  # used where the atom count of the copies is not known (the plot)


def scan_blowup_rows(change_ev, limit_ev) -> np.ndarray:
    """For each copy, the first saved row whose change of total energy is not finite or larger
    than limit_ev (eV, one number or one per copy); the number of rows when it never is."""

    change = np.asarray(change_ev, dtype=float)
    bad = ~np.isfinite(change) | (np.abs(change) > limit_ev)
    return np.where(bad.any(axis=0), bad.argmax(axis=0), change.shape[0])


def timestep_scan_data(replica_dt, saved_step, total_energy) -> dict[str, np.ndarray]:
    """Time of each saved row for each copy (step x that copy's dt, fs) and the change of total energy.

    The change is the total energy of each copy minus its first saved value, in meV for the whole
    copy (not per atom), the convention of the Agent 05 plot.
    """

    dt = _array(replica_dt).reshape(-1)
    step = _array(saved_step).reshape(-1)
    energy = _array(total_energy)
    if energy.shape != (step.size, dt.size):
        raise ValueError(
            _say("scan_shape", n_saved=step.size, n_replicas=dt.size, found=tuple(energy.shape))
        )
    return {"dt": dt, "time": step[:, None] * dt[None, :], "change": (energy - energy[0]) * 1000.0}


def _label_line_ends(axis, ends, gap_pt=4.0, steps_pt=(0, 8, -8, 14, -14, 20, -20)):
    """Name each line at its end, nudged up or down until no other line runs under the name.

    Ported from helpers/agent05.py (the artist's label-placement fix).
    """

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
                    continue  # the label's own line
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


@_in_course_style
def build_timestep_scan_figure(replica_dt, saved_step, total_energy):
    """One grey line per timestep (darker for larger), named at its end; a copy that blows up
    (its total energy turns non-finite or leaves its first row by more than SCAN_BLOWUP_EV_PER_COPY)
    is drawn to its last point under that limit and marked with a cross."""

    import matplotlib.pyplot as plt

    data = timestep_scan_data(replica_dt, saved_step, total_energy)
    dt, time, change = data["dt"], data["time"], data["change"]
    labels = LABELS["timestep_scan"]
    figure, axis = _new_figure(figsize=(9.2, 4.0))
    order = np.argsort(dt)
    blown = scan_blowup_rows(change / 1000.0, SCAN_BLOWUP_EV_PER_COPY)  # first blown-up row of each copy
    ends = []  # (label, x, y, colour) at each line's last point under the limit
    for rank, c in enumerate(order):
        last = max(int(blown[c]) - 1, 0)
        shade = plt.cm.Greys(0.5 + 0.45 * rank / max(len(order) - 1, 1))
        axis.plot(time[: last + 1, c], change[: last + 1, c], color=shade, linewidth=1.8)
        ends.append((labels["timestep"].format(dt=dt[c]), time[last, c], change[last, c], shade))
        if blown[c] < change.shape[0]:
            axis.scatter([time[last, c]], [change[last, c]], marker="x", color=shade, zorder=5)
    axis.set_yscale("symlog", linthresh=1.0)
    axis.axhline(0.0, color=LIGHT_GREY, linewidth=0.8)
    axis.set(xlabel=labels["xlabel"], ylabel=labels["ylabel"])
    axis.set_xlim(right=axis.get_xlim()[1] * 1.08)
    _label_line_ends(axis, ends)
    return figure


def plot_timestep_scan(replica_dt, saved_step, total_energy) -> None:
    """Show the change of total energy over time, one line per timestep.

    replica_dt: (n_replicas,) fs. saved_step: (n_saved,) steps. total_energy: (n_saved, n_replicas) eV.
    """

    _show(build_timestep_scan_figure(replica_dt, saved_step, total_energy), LABELS["timestep_scan"]["alt"])


# ---------------------------------------------------------------------------
# Draft exercise (temperature ladder): plot_temperature_ladder
# ---------------------------------------------------------------------------

# The ladder's rules and numbers, in one place (PROVISIONAL: the contract may change).
LADDER = {
    # A copy counts as blown up when its energy per atom is not finite or leaves the first row by
    # more than this, in eV per atom. Intact copies reached 3.5 (copper) and 4.7 (diamond) eV per
    # atom at 100,000 K in the advisor's trial; the blow-ups seen on crystals were 1e5 eV per atom
    # and more (notes/evidence/lesson-02/l02-temperature-ladder--20261126-0215--advisor-opus.md).
    "blowup_ev_per_atom": 50.0,
    # The zero of the plot is the first row (PROVISIONAL): the copies start from one relaxed
    # crystal, so the first row is its energy. A copy that falls below it by more than this, or a
    # first row that differs between copies by more than this, gets a note. In meV per atom.
    "start_mev": 1.0,
    # A copy whose energy changes by less than this along the run did not move, in eV per atom.
    "flat_ev_per_atom": 1e-9,
    # "Hotter above cooler" is judged on the mean of the last part of the run; the words in
    # MESSAGES["ladder_order"] say "the last third".
    "tail_fraction": 1.0 / 3.0,
    # A first row whose typical size is above this, in eV, reads as the energy of a whole copy,
    # not per atom (MACE-MPA-0 energies per atom stay well inside it).
    "whole_copy_ev": 20.0,
    # The thermostat's target and replica_temperature agree within this relative tolerance.
    "target_rtol": 1e-3,
    # Kinetic temperatures out of order by more than this many standard deviations of the
    # difference are noted; one copy's instantaneous spread is sqrt(2 / (3 N)) of its temperature.
    "kinetic_sigmas": 3.0,
    # A copy moved when one of its atoms moved by more than this, in Å.
    "moved_angstrom": 1e-6,
    # The copies' cells agree within this, in Å.
    "cell_angstrom": 1e-4,
    # Line colours from the coldest to the hottest copy (kit blue, grey, kit red).
    "colors": ("#00A3E0", "#B8B8B8", "#E05252"),
}


def ladder_data(time_fs, replica_temperature, replica_energy, n_atoms=None) -> dict[str, Any]:
    """The ladder's arrays, which copies blew up, and the notes the data alone can show.

    ``above``: the energy of each copy above the first row, in meV per atom; the first row is
    the crystal before the first step (the lowest value of the row is taken as the zero).
    ``blown``: for each copy, the first saved row that is not finite or leaves the first row by
    more than LADDER["blowup_ev_per_atom"] (the number of rows when it never does). When the energy
    looks like a whole copy's energy, it is divided by ``n_atoms`` if known; otherwise only
    non-finite rows count as blown up. ``notes``: the size note and the "unexpected" lines.
    """

    time = _array(time_fs).reshape(-1)
    temps = _array(replica_temperature).reshape(-1)
    energy = _array(replica_energy)
    if energy.ndim != 2 or energy.shape != (time.size, temps.size):
        raise ValueError(
            _say("ladder_shape", n_saved=time.size, n_temperatures=temps.size, found=tuple(energy.shape))
        )
    notes = []
    rows = energy.shape[0]
    out = {"time": time, "temps": temps, "energy": energy, "whole": False, "notes": notes}
    first = energy[0][np.isfinite(energy[0])] if rows else np.empty(0)
    if first.size == 0:
        return out | {"above": np.full(energy.shape, np.nan), "blown": np.zeros(temps.size, dtype=int)}
    whole = float(np.median(np.abs(first))) > LADDER["whole_copy_ev"]
    per_atom = energy
    if whole:
        atoms = _say("ladder_whole_copy_atoms", n=int(n_atoms)) if n_atoms else ""
        notes.append(_say("ladder_whole_copy", value=float(np.median(first)), atoms=atoms))
        per_atom = energy / float(n_atoms) if n_atoms else energy
    above = (per_atom - per_atom[0][np.isfinite(per_atom[0])].min()) * 1000.0  # meV per atom
    if whole and not n_atoms:
        blown = np.where((~np.isfinite(energy)).any(axis=0), (~np.isfinite(energy)).argmax(axis=0), rows)
    else:
        blown = scan_blowup_rows(above / 1000.0, LADDER["blowup_ev_per_atom"])
    out |= {"above": above, "blown": blown, "whole": whole}
    survive = np.flatnonzero(blown == rows)
    if survive.size == 0:
        notes.append(_say("ladder_all_blown"))
        return out
    if whole and not n_atoms:
        return out  # the scale is not known, so nothing more is judged
    spread = float(np.ptp(above[0][np.isfinite(above[0])]))
    if spread > LADDER["start_mev"]:
        notes.append(_say("ladder_start_differs", spread=spread))
    else:
        below = int((above[:, survive].min(axis=0) < -LADDER["start_mev"]).sum())
        if below:
            notes.append(_say("ladder_below_start", k=below, n=temps.size))
    if survive.size > 1:
        start = min(int(np.floor(rows * (1.0 - LADDER["tail_fraction"]))), rows - 1)
        mean = above[start:].mean(axis=0)
        ranked = survive[np.argsort(temps[survive], kind="stable")]
        for cold, hot in zip(ranked[:-1], ranked[1:]):
            if temps[hot] > temps[cold] and mean[hot] < mean[cold]:  # equal means (no motion) are not "below"
                notes.append(
                    _say("ladder_order", hot=temps[hot], cold=temps[cold], hot_e=mean[hot], cold_e=mean[cold])
                )
    return out


@_in_course_style
def build_temperature_ladder_figure(time_fs, replica_temperature, replica_energy):
    """One line per copy: its energy per atom above the first row against time, coloured from cold
    to hot and named with its temperature at its end; a copy that blew up is drawn to its last
    point under the limit and marked with a cross."""

    from matplotlib.colors import LinearSegmentedColormap

    data = ladder_data(time_fs, replica_temperature, replica_energy)
    time, temps, energy, blown = data["time"], data["temps"], data["above"], data["blown"]
    labels = LABELS["temperature_ladder"]
    cmap = LinearSegmentedColormap.from_list("ladder", LADDER["colors"])
    figure, axis = _new_figure(figsize=(9.2, 4.0))
    order = np.argsort(temps, kind="stable")
    ends = []
    for rank, c in enumerate(order):
        last = max(int(blown[c]) - 1, 0)
        colour = cmap(rank / max(len(order) - 1, 1))
        axis.plot(time[: last + 1], energy[: last + 1, c], color=colour, linewidth=1.8)
        ends.append((labels["temperature"].format(t=temps[c]), time[last], energy[last, c], colour))
        if blown[c] < energy.shape[0]:
            axis.scatter([time[last]], [energy[last, c]], marker="x", color=colour, zorder=5)
    axis.set(xlabel=labels["xlabel"], ylabel=labels["ylabel"])
    if time.size > 1:
        axis.set_xlim(float(time[0]), float(time[-1]) + 0.08 * float(time[-1] - time[0]))
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    _label_line_ends(axis, ends)
    return figure


def plot_temperature_ladder(time_fs, replica_temperature, replica_energy) -> None:
    """Show the potential energy of each copy above the relaxed crystal over time, one line per
    temperature, coloured from cold to hot; then print the "unexpected" notes, if any.

    time_fs: (n_saved,) fs, the time of each saved row; the first row is time 0, the crystal
    before the first step.
    replica_temperature: (n_temperatures,) K, the temperature each copy is held at.
    replica_energy: (n_saved, n_temperatures) eV per atom, the model's potential energy of each
    copy divided by its number of atoms. The plot shows it above the first row, in meV per atom.
    """

    _show(
        build_temperature_ladder_figure(time_fs, replica_temperature, replica_energy),
        LABELS["temperature_ladder"]["alt"],
    )
    for note in ladder_data(time_fs, replica_temperature, replica_energy)["notes"]:
        print(note)


# ---------------------------------------------------------------------------
# Cell 28: show_speed_table
# ---------------------------------------------------------------------------


def _column(table, *names):
    """The first of ``names`` that is a column of ``table``, matched without regard to case."""

    lookup = {str(c).lower(): c for c in table.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


# The speed_settings contract, in one place. speed_table: a pandas DataFrame, one row per case;
# the flag columns hold True or False, the time columns seconds (finite, above 0). setting_roles:
# a dict with a non-empty text for each role key. Column and key names match without regard to case.
SPEED_FLAGS = ("cueq", "compile")
SPEED_TIMES = ("first_call_s", "repeat_s")
SPEED_ROLES = ("cueq", "compile")
# The four cases as (cueq, compile): neither, cuEquivariance only, compile only, both.
SPEED_CASES = ((False, False), (True, False), (False, True), (True, True))


def _flag_value(value):
    """True or False for a bool-like table cell (bool, numpy bool, 0 or 1); None otherwise."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    return None


def _speed_case_words(case) -> str:
    cueq, comp = case
    return _say("speed_case", cueq=_say("on" if cueq else "off"), compile=_say("on" if comp else "off"))


def speed_notes(speed_table) -> list[str]:
    """The "unexpected" notes the data alone can show: "both" repeats slower than "neither".

    Silent when the table does not hold those two cases with usable times.
    """

    try:
        cueq, comp, rep = (_column(speed_table, c) for c in ("cueq", "compile", "repeat_s"))
        if cueq is None or comp is None or rep is None:
            return []
        times = {}
        for _, row in speed_table.iterrows():
            case = (_flag_value(row[cueq]), _flag_value(row[comp]))
            times.setdefault(case, float(row[rep]))
        both, neither = times.get((True, True)), times.get((False, False))
    except Exception:
        return []
    if both is None or neither is None or not (np.isfinite(both) and np.isfinite(neither)):
        return []
    return [_say("speed_slower", both=both, neither=neither)] if both > neither else []


def speed_table_rows(speed_table) -> tuple[list[str], list[list[str]]]:
    """Headers and text rows of the speed table: the four cases, their first-call and repeat times."""

    labels = LABELS["speed_table"]
    columns = [_column(speed_table, c) for c in SPEED_FLAGS + SPEED_TIMES]
    missing = [c for c, found in zip(SPEED_FLAGS + SPEED_TIMES, columns) if found is None]
    if missing:
        raise ValueError(_say("speed_view_columns", missing=", ".join(missing)))
    cueq, comp, first, repeat = columns
    headers = [labels["cueq"], labels["compile"], labels["first_call_s"], labels["repeat_s"]]
    rows = []
    for _, row in speed_table.iterrows():
        def flag(column):
            return labels["on"] if bool(row[column]) else labels["off"]

        def number(column):
            value = float(row[column])
            return "" if value != value else f"{value:.3g}"

        rows.append([flag(cueq), flag(comp), number(first), number(repeat)])
    return headers, rows


def _role_rows(setting_roles) -> list[list[str]]:
    names = {"cueq": LABELS["speed_table"]["cueq"], "compile": LABELS["speed_table"]["compile"]}
    roles = {str(k).lower(): v for k, v in dict(setting_roles).items()}
    return [[names[key], str(roles[key])] for key in SPEED_ROLES if key in roles]


def speed_table_html(speed_table, setting_roles) -> str:
    """The speed table and the role of each setting, as two small HTML tables."""

    from html import escape

    def table(headers, rows, numeric=()):
        head = "".join(f"<th style='text-align:left'>{escape(h)}</th>" for h in headers)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td style='text-align:{'right' if i in numeric else 'left'}'>{escape(c)}</td>"
                for i, c in enumerate(row)
            )
            + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    headers, rows = speed_table_rows(speed_table)
    labels = LABELS["speed_table"]
    return table(headers, rows, numeric=range(2, len(headers))) + table(
        [labels["setting"], labels["role"]], _role_rows(setting_roles)
    )


def show_speed_table(speed_table, setting_roles) -> None:
    """Show the first-call and repeat time of each of the four cases and the role of each setting;
    then print the "unexpected" notes, if any.

    speed_table: a pandas DataFrame with one row for each of the four cases and the columns
    ``cueq`` and ``compile`` (True or False: the setting is on or off), ``first_call_s`` (seconds,
    the first call of a freshly loaded model) and ``repeat_s`` (seconds, a repeat of the same call).
    setting_roles: a dict with the keys ``cueq`` and ``compile``; each value is one or two
    sentences on what the setting is for.
    """

    from IPython.display import HTML, display

    display(HTML(speed_table_html(speed_table, setting_roles)))
    for note in speed_notes(speed_table):
        print(note)


# ---------------------------------------------------------------------------
# Step contracts and the step check
#
#   python -m helpers.agent03 check <step>        (run from notebooks/)
#
# The check runs the setup cell, the files of the earlier steps in contract
# order, then the step's file, all in one namespace as the notebook does after
# %load. Then it checks the names the step must create. One line per problem,
# plain words: what was expected, what was found. Exit code 0 passes, 1 fails.
# ---------------------------------------------------------------------------

# The settings of the setup cell, in one place. n_temperatures is PROVISIONAL: its course value
# is to be measured on the course GPU (what fits in two minutes; the command is in
# notes/evidence/lesson-02/l02-temperature-ladder--20261126-0300--owner03.md).
SETTINGS = {"n_crystals": 200, "n_temperatures": 12}
# The check reads these environment variables in place of the numbers above (small CPU runs).
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
    kind: str  # "Batch", "model", "tensor", "number" or "any"
    shape: tuple | None = None  # for tensors: ints or the symbols below
    unit: str = ""
    rule: str = ""  # extra rule, in words, for the contract table
    finite: bool = False
    positive: bool = False
    increasing: bool = False
    integer: bool = False


@dataclass
class Step:
    file: str
    after: list
    names: dict
    extra: list = field(default_factory=list)  # names of extra checks
    snapshot: str = ""  # what to record before the file runs
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
        extra=["atoms_moved", "time_axis"],
        snapshot="md_positions",
        watch_dynamics=True,
    ),
    "timestep_scan": Step(
        # DRAFT exercise, not in the notebook yet. It needs only mace_model (and a crystal, which
        # the learner may bring or take from crystal_batch), so it runs after relax_build alone:
        # the relaxation and the md files would add minutes to the check and nothing it uses.
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
        extra=["scan_copies", "scan_values"],
        watch_runs=True,
    ),
    "temperature_ladder": Step(
        # DRAFT exercise, not in the notebook yet. Like the timestep scan it needs only mace_model
        # and a crystal, so it runs after relax_build alone. The rules and numbers are in LADDER.
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
    if (
        t is None
        or t.dtype == torch.bool
        or not (t.is_floating_point() or t.dtype in (torch.int32, torch.int64))
    ):
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
    import torch

    f = forces.detach().cpu().double()
    out = torch.zeros(batch.num_graphs, dtype=torch.float64)
    return out.scatter_reduce_(
        0, batch.batch_idx.detach().cpu().long(), f.norm(dim=-1), reduce="amax", include_self=False
    )


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
        or t is None
        or not callable(model)
        or tuple(t.shape) != (batch.num_graphs,)
        or not (t.is_floating_point() or t.dtype in (torch.int32, torch.int64))
        or t.dtype == torch.bool
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


def _contract_extras(step, ns, ctx):
    import torch
    from nvalchemi.data import Batch

    problems = []
    if step == "relax_build":
        cb = ns.get("crystal_batch")
        problems += _same_precision(step, ns)
        problems += _force_is_vector_length(step, "start_max_force", "crystal_batch", ns)
        if isinstance(cb, Batch):
            if cb.num_graphs != ns["n_crystals"]:
                problems.append(_say("crystal_count", want=ns["n_crystals"], found=cb.num_graphs))
            pbc = getattr(cb, "pbc", None)
            cell = getattr(cb, "cell", None)
            if pbc is None or cell is None:
                problems.append(_say("periodic_none"))
            else:
                pbc = pbc.reshape(cb.num_graphs, -1).cpu()
                cell = cell.reshape(cb.num_graphs, 3, 3).detach().cpu().double()
                nopbc = int((~pbc.all(dim=1)).sum())
                nocell = int(
                    (torch.linalg.det(cell).abs() < 1e-9).sum() + (~torch.isfinite(cell).all()).sum()
                )
                if nopbc or nocell:
                    problems.append(_say("periodic_count", nopbc=nopbc, n=cb.num_graphs, nocell=nocell))
    elif step == "relax_run":
        problems += _same_graphs(step, "relaxed_batch", ns)
        problems += _force_is_vector_length(step, "relaxed_max_force", "relaxed_batch", ns)
        rmf, thr = ns.get("relaxed_max_force"), ns.get("force_threshold")
        if rmf is not None and thr is not None:
            t, th = _tensor(rmf), _tensor(thr)
            if (
                t is not None
                and th is not None
                and th.numel() == 1
                and t.numel() > 0
                and torch.isfinite(t).all()
            ):
                # A force recomputed after the run can sit a hair above the threshold the
                # optimizer stopped at (float32 noise on the GPU); allow 1 percent.
                limit = float(th.reshape(())) * 1.01 + 1e-6
                over = int((t.double() > limit).sum())
                if over:
                    problems.append(
                        _say(
                            "threshold", threshold=float(th.reshape(())), over=over, n=t.numel(),
                            largest=float(t.max()),
                        )
                    )
        problems += _same_model_settings(step, ns, ctx.get("watch"))
        ctx.setdefault("notes", []).extend(_single_sizes_notes(ns, ctx.get("run_watch")))
    elif step == "md_build":
        problems += _same_graphs(step, "md_batch", ns)
        mb = ns.get("md_batch")
        if isinstance(mb, Batch):
            vel = getattr(mb, "velocities", None)
            if vel is None:
                problems.append(_say("no_velocities"))
            elif tuple(vel.shape) != (mb.num_nodes, 3) or not torch.isfinite(vel).all():
                problems.append(_say("bad_velocities", atoms=mb.num_nodes, found=tuple(vel.shape)))
            done = _finished_graphs(mb)
            if done:
                problems.append(_say("md_finished", k=done, n=mb.num_graphs))
    elif step == "md_run":
        problems += _rows_repeat(step, "crystal_energy", ns)
    elif step == "speed_settings":
        problems += _speed_contract(step, ns, ctx)
    elif step == "temperature_ladder":
        problems += _rows_repeat(step, "replica_energy", ns)
        problems += _ladder_contract(step, ns, ctx)
    return problems


def _rows_repeat(step, name, ns):
    """Every saved row after the first holds exactly the same numbers: the rows were not recorded
    along the run. Seen with SnapshotHook and a HostMemory sink in Toolkit 0.3.0 on a CPU device:
    HostMemory.write keeps ``data.to("cpu")``, which is no copy when the batch is already on the CPU,
    so every saved frame holds the batch's last energy (validation of 26 Nov 2026,
    notes/evidence/lesson-02/validation/direct/snapshot_sink). On a GPU ``.to("cpu")`` copies."""
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
        d = (value.positions.detach().cpu().double() - pos0).norm(dim=-1)
        per = torch.zeros(n_graphs, dtype=torch.float64).scatter_reduce_(
            0, bidx, d, reduce="amax", include_self=False
        )
        moved = torch.maximum(moved, per)
    still = int((moved <= 1e-6).sum())
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

    miss = _time_axis_miss(ns, ctx, "crystal_energy")
    return [_say("time_axis", **miss)] if miss else []


def _ladder_time_axis_notes(ns, ctx):
    """temperature_ladder: the same test as md_run, printed as an "unexpected" note (the step passes)."""

    miss = _time_axis_miss(ns, ctx, "replica_energy")
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


# --- the timestep scan (draft exercise) -----------------------------------------


def _start_state(batch):
    """What a dynamics run started from: sizes, atoms, positions, cells and pbc of each graph."""
    import torch

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
    import torch

    watch = ctx.get("run_watch")
    dt = _tensor(ns["replica_dt"]) if "replica_dt" in ns else None
    if watch is None or not watch.runs or dt is None or dt.dim() != 1:
        return []
    dt = dt.double()
    k = int(dt.numel())
    runs = [(e, s) for e, s in watch.runs if s["n"] == k]
    if not runs:
        sizes = sorted({s["n"] for _, s in watch.runs})
        return [_say("scan_one_batch", step=step, k=k, sizes=", ".join(str(n) for n in sizes))]
    problems = []
    engine, state = runs[-1]
    # every copy is the same structure: same atoms in the same order, same positions
    sizes, z, pos, idx = state["sizes"], state["z"], state["pos"], state["idx"]
    if not bool((sizes == sizes[0]).all()):
        problems.append(_say("copies_sizes", step=step))
    else:
        zz, pp = z.reshape(k, -1), pos.reshape(k, -1, 3)
        if not bool((zz == zz[0]).all()):
            problems.append(_say("copies_atoms", step=step))
        elif float((pp - pp[0]).abs().max()) > 1e-4:
            problems.append(_say("scan_positions", step=step))
    # every copy is a periodic crystal
    problems += _copies_periodic(step, state, k)
    # no copy is marked finished (a leftover of the relaxation: it would not move)
    if state["finished"]:
        problems.append(_say("scan_finished", step=step, k=state["finished"], n=k))
    # every copy moved during the run (a copy that did not cannot drift, at any timestep)
    end = state.get("end_pos")
    if end is not None and end.shape == pos.shape:
        shift = (end - pos).norm(dim=-1)
        per_copy = torch.zeros(k, dtype=torch.float64).scatter_reduce_(0, idx, shift, reduce="amax", include_self=False)
        still = int((per_copy <= 1e-6).sum())
        if still:
            hint = _say("copies_finished_hint") if state["finished"] else ""
            problems.append(_say("scan_not_moved", step=step, still=still, k=k, hint=hint))
    # every copy really ran with its own timestep, equal to replica_dt
    for engine, _ in runs:
        used = _engine_dt_fs(engine)
        if used is not None:
            used = used.expand(k) if used.numel() == 1 else used
            if used.numel() == k and bool(torch.allclose(used, dt, rtol=1e-3, atol=1e-6)):
                break
    else:
        used = _engine_dt_fs(engine)
        if used is not None:
            problems.append(
                _say("scan_dt", step=step, used=_fs_list(used.tolist()), want=_fs_list(dt.tolist()))
            )
    return problems


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
    if (
        energy is not None
        and energy.dim() == 2
        and energy.shape[0] >= 1
        and (energy.is_floating_point() or energy.dtype in (torch.int32, torch.int64))
        and energy.dtype != torch.bool
    ):
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
    usable = (
        energy is not None
        and energy.dim() == 2
        and energy.shape[1] == k
        and energy.shape[0] >= 1
        and (energy.is_floating_point() or energy.dtype in (torch.int32, torch.int64))
        and energy.dtype != torch.bool
    )
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
    import torch

    problems = []
    state = runs[-1]
    sizes, z = state["sizes"], state["z"]
    if not bool((sizes == sizes[0]).all()):
        problems.append(_say("copies_sizes", step=step))
    elif not bool((z.reshape(k, -1) == z.reshape(k, -1)[0]).all()):
        problems.append(_say("copies_atoms", step=step))
    periodic = _copies_periodic(step, state, k)
    problems += periodic
    cell = state["cell"]
    if not periodic and cell is not None and float((cell - cell[0]).abs().max()) > LADDER["cell_angstrom"]:
        problems.append(_say("ladder_cells", step=step))
    # moved: from the start of the first dynamics run of the copies to the end of the last one.
    # A relaxation of the copies before the dynamics does not count as the run.
    dynamics = [r for r in runs if not r.get("optimizer")] or runs
    start, end = dynamics[0]["pos"], dynamics[-1].get("end_pos")
    if end is not None and end.shape == start.shape:
        shift = (end - start).norm(dim=-1)
        per_copy = torch.zeros(k, dtype=torch.float64).scatter_reduce_(
            0, dynamics[0]["idx"], shift, reduce="amax", include_self=False
        )
        still = int((per_copy <= LADDER["moved_angstrom"]).sum())
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


EXTRA_CHECKS = {
    "atoms_moved": _atoms_moved,
    "time_axis": _time_axis,
    "scan_copies": _scan_copies,
    "scan_values": _scan_values,
}


def _take_snapshot(kind, ns):
    if kind == "md_positions":
        mb = ns.get("md_batch")
        if mb is None or not hasattr(mb, "positions"):
            return None
        return (
            mb.positions.detach().cpu().double().clone(),
            mb.batch_idx.detach().cpu().long().clone(),
            mb.num_graphs,
            _finished_graphs(mb),
        )
    return None


_SCRIPT_NAMES = {"__file__"}  # names a script has and a notebook cell does not


def _run_file(path, ns, label):
    t0 = time.time()
    real_out = sys.stdout
    sys.stdout = sys.stderr  # the file's own prints must not mix with the verdict lines
    try:
        exec(compile(open(path).read(), path, "exec"), ns)
    except Exception as exc:
        sys.stdout = real_out
        traceback.print_exc()
        if isinstance(exc, NameError) and getattr(exc, "name", None) in _SCRIPT_NAMES:
            return _say("script_name", step=label, file=os.path.basename(path), name=exc.name)
        return _say(
            "stopped", label=label, file=os.path.basename(path), error=type(exc).__name__,
            first_line=str(exc).splitlines()[0][:200] if str(exc) else "",
        )
    finally:
        sys.stdout = real_out
    print(_say("ran_in", file=os.path.basename(path), seconds=time.time() - t0), file=sys.stderr, flush=True)
    return None


def check_step(step, notes=None):
    """Run the chain up to `step`, check its contract. Returns (problems, n_names).

    Lines that do not fail the step (the "unexpected" notes) are appended to ``notes``
    when a list is given.
    """
    spec = STEPS[step]
    notes = [] if notes is None else notes
    ns = {"__name__": "__main__"}
    real_out, sys.stdout = sys.stdout, sys.stderr
    try:
        exec(setup_cell(), ns)
    finally:
        sys.stdout = real_out
    work = ns["work_dir"]
    for earlier in spec.after:
        path = os.path.join(work, STEPS[earlier].file)
        if not os.path.exists(path):
            return [_say("earlier_missing", step=step, file=STEPS[earlier].file, work=work)], 0
        err = _run_file(path, ns, earlier)
        if err:
            return [_say("earlier_failed", step=step, error=err)], 0
    ctx = {"notes": notes}
    if spec.snapshot:
        ctx[spec.snapshot] = _take_snapshot(spec.snapshot, ns)
        from nvalchemi.data import Batch

        ctx["batch_ids"] = {k: id(v) for k, v in ns.items() if isinstance(v, Batch)}
    path = os.path.join(work, spec.file)
    if not os.path.exists(path):
        return [_say("step_missing", step=step, file=spec.file, work=work)], len(spec.names)
    watch = _WatchMaceCalls() if spec.watch_models else None
    dyn_watch = _WatchDynamicsHooks() if spec.watch_dynamics else None
    run_watch = _WatchEngineRuns() if spec.watch_runs else None
    with contextlib.ExitStack() as stack:
        for w in (watch, dyn_watch, run_watch):
            if w is not None:
                stack.enter_context(w)
        err = _run_file(path, ns, step)
    problems = [err] if err else []
    syms = _symbols(ns, step)
    for name, nspec in spec.names.items():
        problems += _check_name(step, name, nspec, ns, syms)
    ctx["watch"] = watch
    ctx["dyn_watch"] = dyn_watch
    ctx["run_watch"] = run_watch
    problems += _contract_extras(step, ns, ctx)
    for extra in spec.extra:
        problems += EXTRA_CHECKS[extra](step, ns, ctx)
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


if __name__ == "__main__":
    sys.exit(main(sys.argv))
