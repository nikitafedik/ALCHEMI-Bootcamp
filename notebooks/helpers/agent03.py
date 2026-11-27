# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 03, Ready-made.

The module reads top to bottom in three parts:

1. the setup of the notebook: display defaults, the work folder, the session lines;
2. the loaders: the course crystal set and the default crystal of the temperature ladder;
3. the views: the plotters and the speed table. Every plotter has a ``build_*`` function
   that returns a Matplotlib figure and a public function that shows it as one 920 px picture.

The step check, ``python -m helpers.agent03 check <step>`` (run from ``notebooks/``), is in
``agent03_check``, with the step contracts (``STEPS``).

Every line a view or the check prints or raises is in ``MESSAGES``; the words on the
figures are in ``LABELS``.
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
import sys
import warnings
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

# MACE prints this line to stdout each time a model loads from the cache.
_MACE_DEFAULT_LINE = "Using medium MPA-0 model as default MACE-MP model"


def _quiet_mace_default_line() -> None:
    """Drop MACE's "Using medium MPA-0 model as default" line; every other line passes through.

    ``MACEWrapper.from_checkpoint`` calls ``mace.calculators.foundations_models.download_mace_mp_checkpoint``
    at each load, and that function prints the line whenever the checkpoint is not the copy inside the
    mace package: once per model in the setup cell and in the speed step. It is a print, not a warning,
    so no warnings filter reaches it. The function is wrapped once, in its own module, and the wrapper
    drops the line by its text.
    """

    try:
        # The first import of mace raises library notices that do not describe the learner's run
        # (e3nn's TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD notice, torch's "`torch.jit.script` is deprecated").
        # The course guard drops those two and shows every other warning of the import.
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


def _ignore_speed_step_notices() -> None:
    """Ignore two library notices of the speed step, each by message and module.

    MACE warns "To copy construct from a tensor ..." each time a model is converted to cuEquivariance,
    and torch.compile warns that TensorFloat32 is available but not enabled. Neither describes the
    learner's run. Call this after the mace import, so the rules land in front of the "once" rules
    that Toolkit and cuEquivariance set when they are imported.
    """

    warnings.filterwarnings(
        "ignore",
        message=r"To copy construct from a tensor.*",
        category=UserWarning,
        module=r"mace\.modules\.models",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*",
        category=UserWarning,
        module=r"torch\._inductor\.compile_fx",
    )


def configure_tutorial() -> None:
    """Apply the course style (light only), drop MACE's repeated default-model line and three library notices."""

    _configure_tutorial()
    _quiet_mace_default_line()
    _ignore_speed_step_notices()


# The work folder. The environment variable ALCHEMI_AGENT03_WORK names another one (for short runs
# that must leave notebooks/agent03_work alone).
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
# place. The code fills the {placeholders}. A line that starts with "unexpected:" is
# a note: the check prints it and the step still passes.
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
    "speed_view_columns": (
        "show_speed_table: speed_table should have the columns cueq, compile, first_call_s and "
        "repeat_s; missing {missing}"
    ),
    "ladder_shape": (
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
    "single_sizes": (  # a note
        "unexpected: the crystals timed one by one hold {sizes} atoms, while the batch averages "
        "{mean:.0f} atoms per crystal; the projected one-by-one time depends on that choice"
    ),
    "relaxed_all_below": (  # printed under the force histograms
        "The largest force of every relaxed crystal is below the threshold (dashed line)."
    ),
    "relaxed_above": (  # printed under the force histograms
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
    "ladder_time_axis": (  # a note
        "unexpected: time_fs should give the time of each saved energy row; {which} was recorded "
        "at {expected:g} fs but time_fs says {found:g} fs"
    ),
    "time_axis_last": "the last row",
    "time_axis_row": "row {row}",
    # --- timestep_scan; some lines serve the ladder as well -----------------------------
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
    # --- temperature_ladder ---------------------------------------------------------------
    "ladder_one_batch": (
        "{step}: the copies should run together in one batch, one graph for each temperature "
        "(n_temperatures = {k}); found runs of {sizes} graphs"
    ),
    "ladder_cells": (
        "{step}: every copy should be the same crystal; found copies with different cells"
    ),
    "ladder_not_moved": (
        "{step}: the atoms of {still} of {k} copies did not move during the run{hint}"
    ),
    "ladder_flat": (
        "{step}: replica_energy of {still} of {k} copies does not change along the run, so their "
        "atoms did not move{hint}"
    ),
    "ladder_distinct": (
        "{step}: replica_temperature should hold a different temperature for each copy, in K; "
        "found {distinct} different temperature{s} among {n}"
    ),
    "ladder_first_row": (
        "{step}: the first row of replica_energy should be finite for every copy; found {k} of {n} "
        "copies that are not"
    ),
    "ladder_target": (
        "{step}: replica_temperature should be the temperature each copy's thermostat held (K); "
        "the run used {used} but replica_temperature says {want}"
    ),
    "ladder_whole_copy": (  # a note
        "replica_energy should be in eV per atom; its first row is about {value:.4g} eV, the size of "
        "the energy of a whole copy{atoms}; divide each copy's energy by its number of atoms"
    ),
    "ladder_whole_copy_atoms": " of {n} atoms",
    "ladder_order": (  # a note
        "unexpected: over the last third of the run, the copy at {hot:.0f} K sits below the copy at "
        "{cold:.0f} K ({hot_e:.0f} against {cold_e:.0f} meV per atom)"
    ),
    "ladder_below_start": (  # a note
        "unexpected: the energy of {k} of {n} copies falls below the first row, so the copies did "
        "not start from a relaxed crystal; the zero of the plot is the first row"
    ),
    "ladder_start_differs": (  # a note
        "unexpected: the first row of replica_energy differs between the copies by {spread:.3g} meV "
        "per atom; it should be the energy of the crystal before the first step, the same for every copy"
    ),
    "ladder_all_blown": (  # a note
        "unexpected: every copy blew up, so the order of the energies cannot be judged"
    ),
    "ladder_kinetic": (  # a note
        "unexpected: at the end of the run the copy at {hot:.0f} K had a lower kinetic temperature "
        "({hot_k:.0f} K) than the copy at {cold:.0f} K ({cold_k:.0f} K)"
    ),
    # --- speed_settings -------------------------------------------------------------------
    "speed_not_table": (
        "{step}: speed_table should be a table (pandas DataFrame); found {found}"
    ),
    "speed_columns": (
        "{step}: speed_table should have the columns cueq, compile, first_call_s and repeat_s; "
        "missing {missing}"
    ),
    "speed_flags": (
        "{step}: the {column} column of speed_table should hold True or False; found {found}"
    ),
    "speed_case_missing": "{step}: speed_table is missing the case {case}",
    "speed_case_twice": (
        "{step}: speed_table should hold each case once; found {case} {count} times"
    ),
    "speed_time": (
        "{step}: {column} should be a time in s, finite and above 0, for every case; found {found} "
        "for {case}"
    ),
    "speed_roles": (
        "{step}: setting_roles should be a dict with a short text for each of the keys cueq and "
        "compile; found {found}"
    ),
    "speed_roles_keys": "the keys {keys}",
    "speed_roles_none": "an empty dict",
    "speed_roles_empty": "no text for {keys}",
    "speed_case": "cuEquivariance {cueq}, compile {compile}",
    "speed_slower": (  # a note; the view prints it too
        "unexpected: with cuEquivariance and compile both on, a repeat took {both:.3g} s, longer "
        "than the {neither:.3g} s with both off"
    ),
    "speed_model": (  # a note
        "unexpected: no model with {case} was called while the file ran, so that row may come from "
        "other settings (models called: {found})"
    ),
    "rows_repeat": (
        "{step}: the {n} saved rows of {name} after the first hold exactly the same numbers, so they "
        "were not recorded along the run; keep a copy of the energies each time a row is saved"
    ),
    # --- running the files --------------------------------------------------------------
    "note": "{step}: {note}",  # how the check prints a note
    "earlier_missing": "{step}: the earlier step file {file} is missing in {work}",
    "earlier_failed": "{step}: could not run the earlier steps; {error}",
    "step_missing": "{step}: the step file {file} is missing in {work}",
    "stopped": "{label}: {file} stopped with {error}: {first_line}",
    # a name a script has and a notebook cell does not
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
# Wording on the figures: plain axis labels "Quantity [unit]". Alt texts
# describe what the picture shows.
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
    # The timestep scan has a plotter and a check and no cells in the notebook. Its alt text is a
    # placeholder, and the text says so.
    "timestep_scan": {
        "xlabel": "Time [fs]",
        "ylabel": "Change in total energy [meV]",
        "timestep": "{dt:g} fs",
        "alt": "PLACEHOLDER: change in total energy over time for copies of one crystal, one line for each timestep.",
    },
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
        "first_call_s": "First call [s]",  # the first_call_s column
        "repeat_s": "Repeat [s]",  # the repeat_s column
        "setting": "Setting",
        "role": "Role",
        "on": "on",
        "off": "off",
    },
}

# Colours of the course figure style: green main result, grey baseline, ink reference.
GREEN, GREY, LIGHT_GREY, INK = "#76B900", "#6F6F6F", "#B8B8B8", "#1A1A1A"
# The force colours of the crystal grid, from small to large forces: neutral grey to red.
GRID_COLORS = ("#E4E4E4", "#F0B49C", "#D1352B")
# The energy-drift axis spans at least this much on each side of zero, in meV per atom, and grows
# when a crystal drifts further, so that numerical noise does not fill the plot.
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
    """``plt.subplots``, with Matplotlib imported only when a figure is drawn."""

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
    """A time for a bar label: three figures up to 100 s, whole seconds above."""

    if value >= 100:
        return f"{value:.0f} s"
    return f"{value:.1f} s" if value >= 10 else f"{value:.2f} s"


# ---------------------------------------------------------------------------
# The crystal grid: show_crystal_grid
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
# Forces before and after the relaxation: plot_force_histograms
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
# Batched against one by one: plot_batch_speedup
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
# Starting temperatures: plot_start_temperature
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
# Energy conservation: plot_energy_drift
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
# The timestep scan (no cells in the notebook): plot_timestep_scan
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
    copy (not per atom).
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
    """Name each line at its end, nudged up or down until no other line runs under the name."""

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
# The temperature ladder: plot_temperature_ladder
# ---------------------------------------------------------------------------

# The ladder's rules and numbers, in one place.
LADDER = {
    # A copy counts as blown up when its energy per atom is not finite or leaves the first row by
    # more than this, in eV per atom. Intact copies reached 3.5 (copper) and 4.7 (diamond) eV per
    # atom at 100,000 K in a trial run; the blow-ups seen on crystals were 1e5 eV per atom and more.
    "blowup_ev_per_atom": 50.0,
    # The zero of the plot is the first row: the copies start from one relaxed
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
    out = {"time": time, "temps": temps, "notes": notes}
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
    out |= {"above": above, "blown": blown}
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
# The speed settings: show_speed_table
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

    def flag(value) -> str:
        return labels["on"] if bool(value) else labels["off"]

    def seconds(value) -> str:
        value = float(value)
        return "" if value != value else f"{value:.3g}"  # an empty cell for a time that is not a number

    rows = [
        [flag(row[cueq]), flag(row[comp]), seconds(row[first]), seconds(row[repeat])]
        for _, row in speed_table.iterrows()
    ]
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


def main(argv) -> int:
    """Run ``python -m helpers.agent03 check <step>``; the check itself is in ``agent03_check``."""

    from .agent03_check import main as check_main

    return check_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
