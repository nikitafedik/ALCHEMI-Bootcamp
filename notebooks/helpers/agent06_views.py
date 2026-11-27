# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result views of Agent 06, Hooked in at every step.

``show_run`` and ``show_retired`` show a player of the saved frames and a
table; ``plot_interaction``, ``plot_separation`` and ``plot_displacement`` draw
to the course figure standard (light only, 920 px wide, Open Sans). A view
that has something to say about its result prints one line under it. Each ``build_*`` and ``*_data``
function is free of display calls, so the tests can read what a view shows.

Every word on a figure or a table comes from the notebook's text: the wording
lives in ``LABELS`` and is the educator's to change.
"""

from __future__ import annotations

import contextlib
from base64 import b64encode
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STYLE = _PROJECT_ROOT / "shared" / "alchemi-figure.mplstyle"
DISPLAY_PX = 920
GREEN, GREY, LIGHT_GREY, INK = "#76B900", "#6F6F6F", "#B8B8B8", "#1A1A1A"
N_COMPLEXES = 3

# Energy units a learner may choose for the interaction energy: unit -> value of 1 eV in it.
UNITS = {"kcal/mol": 23.060547830619, "eV": 1.0, "kJ/mol": 96.48533212331, "Hartree": 1.0 / 27.211386245988}
LINE_DIGITS = {"Hartree": 4}  # decimals of an energy in the line under the plot; other units: 1
UNIT_SLACK = 0.01  # relative; a start value within 1 % of the reference reads as that unit
UNIT_FLOOR_EV = 0.2 / 23.060547830619  # or within 0.2 kcal/mol, whichever is larger

LABELS = {
    "interaction": {
        "xlabel": "MD step",
        "ylabel": "Interaction energy [{unit}]",
        "ylabel_no_unit": "Interaction energy",
        "run": "along the run",
        "start": "starting structure",
        "alt": "Each panel follows one complex: its interaction energy against the MD step, with a line for the starting value.",
        "line": "Mean along the run: {means} {unit}. At the starting structure: {starts} {unit}.",
    },
    "separation": {
        "xlabel": "MD step",
        "ylabel": "Distance [Å]",
        "separation": "fragment separation",
        "nearest": "nearest contact",
        "alt": "Each panel follows one complex: its fragment separation and its nearest contact against the MD step.",
        "alt_energy": "Each column follows one complex: its interaction energy above, its fragment separation and its nearest contact below, against the MD step.",
        "line": "Largest nearest contact: {peaks} Å.",
    },
    "displacement": {
        "plain": "plain run",
        "frozen": "frozen run",
        "xlabel": "Saved frame",
        "ylabel": "Distance from start [Å]",
        "frozen_atoms": "frozen atoms",
        "free_atoms": "free atoms",
        "alt": "How far each atom moved from its start, plain run beside frozen run.",
    },
    "retired": {
        "complex": "Complex",
        "step": "Retired at step",
        "to_the_end": "ran to the end",
        "one": "{name} was retired at step {step}. The others ran to the end.",
        "none": "No complex was retired: every complex ran to the end.",
        "several": "Unexpected: {count} complexes were retired ({names}). The step retires one and lets the others run to the end.",
    },
    "log": {"complex": "complex"},
    # shown in place of a result when the saved frames cannot show one
    "same_frames": "Every saved frame is the same: the store kept the last state of the run, not one state per saved step.",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _array(values: Any) -> np.ndarray:
    """Return a tensor or array-like as a float NumPy array on the host."""

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    return np.asarray(values, dtype=float)


def complex_names(n: int = N_COMPLEXES) -> list[str]:
    """Names of the complexes of the setup cell, or "complex 1" ... before it has run."""

    from . import agent06

    names = list(agent06.COMPLEX_NAMES)
    return names if len(names) == n else [f"complex {i + 1}" for i in range(n)]


def store_frames(store: Any, n_complexes: int = N_COMPLEXES) -> dict[str, Any]:
    """Read a store of saved frames.

    Returns ``positions``, a list (one entry per complex) of arrays of shape
    (frames, atoms, 3) in Å, and ``numbers``, the atomic numbers of each complex.
    The store holds every complex at every saved step, in batch order.
    """

    if not callable(getattr(store, "read", None)):
        raise TypeError(f"expected a store of saved frames with read(); found {type(store).__name__}")
    records = store.read().to_data_list()
    if not records or len(records) % n_complexes:
        raise ValueError(
            f"the store holds {len(records)} saved structures; "
            f"{n_complexes} complexes at every saved step give a multiple of {n_complexes}"
        )
    positions, numbers = [], []
    for k in range(n_complexes):
        mine = records[k::n_complexes]
        counts = {int(r.positions.shape[0]) for r in mine}
        if len(counts) != 1:
            raise ValueError("the saved frames are not in batch order: a complex changes its atom count")
        positions.append(np.stack([_array(r.positions) for r in mine]))
        numbers.append(np.asarray(_array(mine[0].atomic_numbers), dtype=int).reshape(-1))
    return {"positions": positions, "numbers": numbers}


def frames_all_equal(positions: list[np.ndarray]) -> bool:
    """True when a store has several frames and every one equals the first."""

    several = any(len(p) > 1 for p in positions)
    return several and all(float(np.abs(p - p[0]).max()) == 0.0 for p in positions)


@contextlib.contextmanager
def _course_style():
    """The course figure style (Open Sans, 920 px wide) for everything drawn inside."""

    import matplotlib.pyplot as plt

    from .common import register_course_fonts

    register_course_fonts()  # before the style names the course font
    with plt.style.context(str(_STYLE)):
        yield


def _show(figure, alt: str) -> None:
    """Show a figure as one picture, at most 920 px wide, with alt text."""

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


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th style='text-align:left'>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td style='text-align:left'>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_PLAYER_HEIGHT = 300
_PLAYER_SCRIPT = """
const items = __ITEMS__;
const colors = __COLORS__;
const viewers = items.map(function (item, index) {
  const viewer = $3Dmol.createViewer(document.getElementById('pane' + index), {backgroundColor: '#FFFFFF', antialias: true});
  viewer.addModelsAsFrames(item.xyz, 'xyz');
  viewer.setStyle({}, {sphere: {scale: 0.25, color: '#76B900'}, stick: {radius: 0.09, color: '#76B900'}});
  Object.keys(colors).forEach(function (symbol) {
    viewer.setStyle({elem: symbol}, {sphere: {scale: 0.25, color: colors[symbol]}, stick: {radius: 0.09, color: colors[symbol]}});
  });
  viewer.zoomTo();
  viewer.zoom(0.8);
  viewer.setFrame(0);
  viewer.render();
  return viewer;
});
const frames = __FRAMES__;
const slider = document.getElementById('frame');
const label = document.getElementById('label');
const button = document.getElementById('play');
let timer = null;
function show(frame) {
  slider.value = frame;
  label.textContent = (frame + 1) + ' / ' + frames;
  viewers.forEach(function (viewer) { viewer.setFrame(frame); viewer.render(); });
}
function stop() { clearInterval(timer); timer = null; button.textContent = '\u25B6'; }
slider.addEventListener('input', function () { stop(); show(Number(slider.value)); });
button.addEventListener('click', function () {
  if (timer) { stop(); return; }
  button.textContent = '\u23F8';
  timer = setInterval(function () { show((Number(slider.value) + 1) % frames); }, __INTERVAL__);
});
show(0);
"""


def player_html(frames: dict[str, Any], names: list[str], *, frame_interval_ms: int = 80) -> str:
    """One document with a pane per complex, a play button and a slider over the saved frames.

    The panes share one frame index, so the complexes are shown at the same
    saved step. The only words in the player are the names of the complexes.
    """

    import json

    from ase.data import chemical_symbols

    from . import viewers

    items = []
    for positions, numbers in zip(frames["positions"], frames["numbers"], strict=True):
        symbols = [chemical_symbols[int(z)] for z in numbers]
        blocks = []
        for frame in positions:
            rows = [f"{s} {x:.4f} {y:.4f} {z:.4f}" for s, (x, y, z) in zip(symbols, frame)]
            blocks.append(f"{len(symbols)}\n\n" + "\n".join(rows) + "\n")
        items.append({"xyz": "".join(blocks)})
    count = len(frames["positions"][0])
    script = (
        _PLAYER_SCRIPT.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
        .replace("__COLORS__", json.dumps(viewers._ELEMENT_COLORS))
        .replace("__FRAMES__", str(count))
        .replace("__INTERVAL__", str(int(frame_interval_ms)))
    )
    panes = "".join(
        f'<div style="flex:1 1 0;min-width:0"><div style="font:13px Open Sans,Helvetica,Arial,sans-serif;'
        f'color:#1A1A1A;padding:0 0 4px 2px">{escape(name)}</div>'
        f'<div id="pane{k}" style="position:relative;width:100%;height:{_PLAYER_HEIGHT - 70}px;'
        f'border:1px solid #D8D8D8;border-radius:4px;box-sizing:border-box"></div></div>'
        for k, name in enumerate(names)
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>html,body{margin:0;background:#FFFFFF;overflow:hidden}*{box-sizing:border-box}"
        "</style></head><body>"
        f"{viewers._bundle_preamble()}"
        f'<div style="display:flex;gap:10px">{panes}</div>'
        '<div style="display:flex;align-items:center;gap:10px;padding:8px 2px 0;'
        'font:13px Open Sans,Helvetica,Arial,sans-serif;color:#1A1A1A">'
        '<button id="play" style="width:34px;height:26px;border:1px solid #B8B8B8;border-radius:4px;'
        'background:#FFFFFF;cursor:pointer">&#9654;</button>'
        f'<input id="frame" type="range" min="0" max="{count - 1}" value="0" style="flex:1 1 auto;accent-color:#76B900">'
        '<span id="label" style="min-width:70px;text-align:right"></span></div>'
        f"<script>{script}</script></body></html>"
    )


def _player(frames: dict[str, Any], names: list[str]):
    """The player of the saved frames as a notebook output, at most 920 px wide."""

    from IPython.display import HTML

    import warnings

    document = player_html(frames, names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # IPython suggests IFrame for the player's srcdoc
        return HTML(
            f'<iframe title="{escape(", ".join(names), quote=True)}" srcdoc="{escape(document, quote=True)}" '
            f'style="display:block;width:100%;max-width:{DISPLAY_PX}px;height:{_PLAYER_HEIGHT}px;border:0;'
            'background:#FFFFFF;" loading="eager"></iframe>'
        )


def _show_player(store: Any) -> None:
    from IPython.display import Markdown, display

    frames = store_frames(store)
    if frames_all_equal(frames["positions"]):
        display(Markdown(LABELS["same_frames"]))
        return
    display(_player(frames, complex_names()))


# --------------------------------------------------------------------------
# Cell 9: show_run
# --------------------------------------------------------------------------


def log_table(plain_log: Any):
    """The log as a DataFrame; takes a DataFrame or the path of the CSV file."""

    import pandas as pd

    if isinstance(plain_log, pd.DataFrame):
        table = plain_log.copy()
    elif isinstance(plain_log, (str, Path)):
        table = pd.read_csv(plain_log)
    else:
        raise TypeError(
            f"plain_log should be a DataFrame or the path of the log file; found {type(plain_log).__name__}"
        )
    # The log writes counters as floats (step 10.0, status 0.0): show them as whole numbers.
    for column in ("step", "graph_idx", "status"):
        if column in table.columns:
            values = pd.to_numeric(table[column], errors="coerce")
            if np.isfinite(values).all() and (values == values.round()).all():
                table[column] = values.astype("int64")
    if "graph_idx" in table.columns:
        names = complex_names()
        index = table["graph_idx"].astype(int)
        if index.between(0, len(names) - 1).all():
            table.insert(
                list(table.columns).index("graph_idx") + 1,
                LABELS["log"]["complex"],
                [names[i] for i in index],
            )
    return table


def show_run(plain_store, plain_log) -> None:
    """Play the saved frames of the plain run and show its log as a table.

    ``plain_store`` holds the three complexes at every saved step.
    ``plain_log`` is the log, as a DataFrame or as the path of the CSV file:
    one row per complex at each logged step.
    """

    from IPython.display import HTML, display

    _show_player(plain_store)
    display(HTML(log_table(plain_log).to_html(index=False, max_rows=12, border=0)))


# --------------------------------------------------------------------------
# plot_interaction
# --------------------------------------------------------------------------


def _listed(values, digits: int = 1) -> str:
    """Numbers as "a, b and c"."""

    words = [f"{float(v):.{digits}f}" for v in values]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def read_unit(start_values, reference_ev) -> str | None:
    """The energy unit of ``start_values``, read from their ratio to the same values in eV.

    ``reference_ev`` holds the interaction energy of each complex at its
    starting structure in eV. Returns "kcal/mol", "eV", "kJ/mol" or "Hartree" when every
    complex agrees with the reference in that unit, and None otherwise.
    """

    values = _array(start_values).reshape(-1)
    reference = _array(reference_ev).reshape(-1)
    if values.shape != reference.shape or not np.isfinite(values).all():
        return None
    for unit, factor in UNITS.items():
        slack = np.maximum(UNIT_SLACK * np.abs(reference), UNIT_FLOOR_EV) * factor
        if np.all(np.abs(values - reference * factor) <= slack):
            return unit
    return None


def interaction_data(saved_step, interaction_energy, start_interaction_energy) -> dict[str, Any]:
    """Check and return the plotted arrays, the unit and the mean of each complex along the run."""

    from . import agent06

    step = _array(saved_step).reshape(-1)
    energy, start = _array(interaction_energy), _array(start_interaction_energy).reshape(-1)
    if energy.ndim != 2 or energy.shape[0] != step.shape[0] or energy.shape[1] != start.shape[0]:
        raise ValueError(
            "expected saved_step (saved steps,), interaction_energy (saved steps, complexes) and "
            f"start_interaction_energy (complexes,); found {step.shape}, {energy.shape} and {start.shape}"
        )
    if not (np.isfinite(energy).all() and np.isfinite(start).all()):
        raise ValueError("interaction_energy or start_interaction_energy holds values that are not finite")
    reference = list(agent06.START_INTERACTION_EV)
    return {
        "step": step,
        "energy": energy,
        "start": start,
        "names": complex_names(energy.shape[1]),
        "unit": read_unit(start, reference) if len(reference) == start.shape[0] else None,
        "mean": energy.mean(axis=0),
    }


def interaction_line(data: dict[str, Any]) -> str:
    """The line under the plot: the mean of each complex along the run and its starting value."""

    digits = LINE_DIGITS.get(data["unit"], 1)
    return LABELS["interaction"]["line"].format(
        means=_listed(data["mean"], digits), starts=_listed(data["start"], digits), unit=data["unit"] or ""
    ).replace(" .", ".")


def build_interaction_figure(saved_step, interaction_energy, start_interaction_energy):
    """One panel per complex: interaction energy against the MD step, a dashed line at the starting value."""

    import matplotlib.pyplot as plt

    data = interaction_data(saved_step, interaction_energy, start_interaction_energy)
    labels = LABELS["interaction"]
    count = len(data["names"])
    with _course_style():
        figure, axes = plt.subplots(1, count, figsize=(9.2, 3.6), sharey=True, squeeze=False)
        for k, axis in enumerate(axes[0]):
            axis.plot(data["step"], data["energy"][:, k], color=GREEN, label=labels["run"])
            axis.axhline(data["start"][k], color=INK, linestyle="--", linewidth=1.0, label=labels["start"])
            axis.set_title(data["names"][k])
            axis.set_xlabel(labels["xlabel"])
        axes[0][0].set_ylabel(
            labels["ylabel"].format(unit=data["unit"]) if data["unit"] else labels["ylabel_no_unit"]
        )
        axes[0][0].legend(loc="upper left")
    return figure


def plot_interaction(saved_step, interaction_energy, start_interaction_energy) -> None:
    """Plot the interaction energy of each complex against the MD step, one panel per complex.

    ``saved_step`` has shape (saved steps,); ``interaction_energy`` has shape
    (saved steps, complexes); ``start_interaction_energy`` has shape
    (complexes,), the value at the starting structure, drawn as a dashed line.
    Any of kcal/mol, eV, kJ/mol or Hartree: the axis names the unit the values are in.
    """

    from IPython.display import Markdown, display

    data = interaction_data(saved_step, interaction_energy, start_interaction_energy)
    _show(build_interaction_figure(saved_step, interaction_energy, start_interaction_energy), LABELS["interaction"]["alt"])
    display(Markdown(interaction_line(data)))


# --------------------------------------------------------------------------
# plot_separation
# --------------------------------------------------------------------------


def separation_data(saved_step, separation, nearest_contact, interaction_energy=None) -> dict[str, Any]:
    """Check and return the plotted arrays and the largest nearest contact of each complex."""

    step = _array(saved_step).reshape(-1)
    sep, near = _array(separation), _array(nearest_contact)
    if sep.ndim != 2 or sep.shape != near.shape or sep.shape[0] != step.shape[0]:
        raise ValueError(
            "expected saved_step (saved steps,) and separation and nearest_contact "
            f"(saved steps, complexes); found {step.shape}, {sep.shape} and {near.shape}"
        )
    if not (np.isfinite(sep).all() and np.isfinite(near).all()):
        raise ValueError("separation or nearest_contact holds values that are not finite")
    energy = None
    if interaction_energy is not None:
        energy = _array(interaction_energy)
        if energy.shape != sep.shape:
            raise ValueError(
                f"interaction_energy has shape {energy.shape}; on the same saved steps it has shape {sep.shape}"
            )
    peak_index = near.argmax(axis=0)
    return {
        "step": step,
        "separation": sep,
        "nearest_contact": near,
        "interaction_energy": energy,
        "names": complex_names(sep.shape[1]),
        "peak_step": step[peak_index],
        "peak": near.max(axis=0),
    }


def separation_line(data: dict[str, Any]) -> str:
    """The line under the plot: the largest nearest contact of each complex."""

    return LABELS["separation"]["line"].format(peaks=_listed(data["peak"], 2))


def _distance_panel(axis, data: dict[str, Any], k: int) -> None:
    labels = LABELS["separation"]
    axis.plot(data["step"], data["separation"][:, k], color=GREY, label=labels["separation"])
    axis.plot(data["step"], data["nearest_contact"][:, k], color=GREEN, label=labels["nearest"])
    axis.plot([data["peak_step"][k]], [data["peak"][k]], "o", color=GREEN, markeredgecolor=INK)
    axis.annotate(
        f"{data['peak'][k]:.2f} Å",
        xy=(data["peak_step"][k], data["peak"][k]),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.92,
        color=INK,
    )
    axis.set_xlabel(labels["xlabel"])


def build_separation_figure(saved_step, separation, nearest_contact, interaction_energy=None):
    """One column per complex: fragment separation and nearest contact against the MD step.

    A marker and a value in Å show the largest nearest contact of each complex.
    With ``interaction_energy`` a row above shows it on the same steps.
    """

    import matplotlib.pyplot as plt

    from . import agent06

    data = separation_data(saved_step, separation, nearest_contact, interaction_energy)
    labels = LABELS["separation"]
    count = len(data["names"])
    with _course_style():
        if data["interaction_energy"] is None:
            figure, axes = plt.subplots(1, count, figsize=(9.2, 3.6), sharey=True, squeeze=False)
            distance_axes = axes[0]
        else:
            figure, axes = plt.subplots(
                2, count, figsize=(9.2, 5.6), sharex=True, sharey="row", squeeze=False
            )
            distance_axes = axes[1]
            for k, axis in enumerate(axes[0]):
                axis.plot(data["step"], data["interaction_energy"][:, k], color=INK)
            axes[0][0].set_ylabel(LABELS["interaction"]["ylabel_no_unit"])
        for k, axis in enumerate(distance_axes):
            _distance_panel(axis, data, k)
        for k, axis in enumerate(axes[0]):
            axis.set_title(data["names"][k])
        distance_axes[0].set_ylabel(labels["ylabel"])
        distance_axes[0].legend(loc="upper left")
    return figure


def plot_separation(saved_step, interaction_energy, separation, nearest_contact=None) -> None:
    """Plot the interaction energy and both distances of each complex against the MD step.

    ``saved_step`` has shape (saved steps,); ``interaction_energy``,
    ``separation`` and ``nearest_contact`` have shape (saved steps, complexes),
    the distances in Å. One column per complex: the interaction energy above,
    the two distances below. The marked value is the largest nearest contact of
    that complex; the line under the plot lists the three.

    Called with three arguments, ``plot_separation(saved_step, separation,
    nearest_contact)``, it draws the two distances alone.
    """

    from IPython.display import Markdown, display

    if nearest_contact is None:  # the three-argument form: no interaction energy
        interaction_energy, separation, nearest_contact = None, interaction_energy, separation
    data = separation_data(saved_step, separation, nearest_contact, interaction_energy)
    alt = LABELS["separation"]["alt" if interaction_energy is None else "alt_energy"]
    _show(build_separation_figure(saved_step, separation, nearest_contact, interaction_energy), alt)
    display(Markdown(separation_line(data)))


# --------------------------------------------------------------------------
# Cell 19: plot_displacement
# --------------------------------------------------------------------------


def displacement_data(plain_store, frozen_store, frozen_mask) -> dict[str, Any]:
    """Distance of every atom from its first saved frame, for both runs.

    Returns ``plain`` and ``frozen``, arrays of shape (frames, atoms) in Å with
    the atoms in batch order, ``mask`` (atoms,), True for a frozen atom, and
    ``same_frames``, True when a store kept one state for every frame.
    """

    mask = np.asarray(_array(frozen_mask)).astype(bool).reshape(-1)
    out: dict[str, Any] = {"mask": mask, "same_frames": False}
    for key, store in (("plain", plain_store), ("frozen", frozen_store)):
        positions = store_frames(store)["positions"]
        atoms = sum(p.shape[1] for p in positions)
        if atoms != mask.shape[0]:
            raise ValueError(
                f"frozen_mask has {mask.shape[0]} entries; the complexes have {atoms} atoms, one entry per atom"
            )
        out["same_frames"] |= frames_all_equal(positions)
        out[key] = np.concatenate([np.linalg.norm(p - p[0], axis=2) for p in positions], axis=1)
    return out


def build_displacement_figure(plain_store, frozen_store, frozen_mask):
    """Two panels, plain run and frozen run: distance of each atom from its start, by saved frame."""

    import matplotlib.pyplot as plt

    data = displacement_data(plain_store, frozen_store, frozen_mask)
    labels = LABELS["displacement"]
    mask = data["mask"]
    with _course_style():
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
        for axis, key in zip(axes, ("plain", "frozen")):
            moved = data[key]
            frame = np.arange(moved.shape[0])
            if (~mask).any():
                axis.plot(frame, moved[:, ~mask], color=LIGHT_GREY, linewidth=0.8)
                axis.plot([], [], color=LIGHT_GREY, label=labels["free_atoms"])
            if mask.any():
                axis.plot(frame, moved[:, mask], color=GREEN, linewidth=1.2)
                axis.plot([], [], color=GREEN, label=labels["frozen_atoms"])
            axis.set_title(labels[key])
            axis.set_xlabel(labels["xlabel"])
        axes[0].set_ylabel(labels["ylabel"])
        axes[0].legend(loc="upper left")
    return figure


def plot_displacement(plain_store, frozen_store, frozen_mask) -> None:
    """Plot how far each atom moved from its start, plain run beside frozen run.

    ``frozen_mask`` is boolean with one entry per atom of the batch, True for a
    frozen atom. Both stores hold the three complexes at every saved step.
    """

    from IPython.display import Markdown, display

    figure = build_displacement_figure(plain_store, frozen_store, frozen_mask)
    if displacement_data(plain_store, frozen_store, frozen_mask)["same_frames"]:
        import matplotlib.pyplot as plt

        plt.close(figure)
        display(Markdown(LABELS["same_frames"]))
        return
    _show(figure, LABELS["displacement"]["alt"])


# --------------------------------------------------------------------------
# Cell 23: show_retired
# --------------------------------------------------------------------------


def retired_rows(stop_step, final_status) -> list[list[Any]]:
    """One row per complex: its name and the step it was retired at, or "ran to the end"."""

    step = np.asarray(_array(stop_step)).reshape(-1)
    status = np.asarray(_array(final_status)).reshape(-1)
    if step.shape[0] != status.shape[0]:
        raise ValueError(
            f"stop_step has {step.shape[0]} entries and final_status {status.shape[0]}; both hold one per complex"
        )
    labels = LABELS["retired"]
    names = complex_names(step.shape[0])
    return [[names[k], int(step[k]) if step[k] >= 0 else labels["to_the_end"]] for k in range(step.shape[0])]


def retired_line(stop_step) -> str:
    """What happened, in plain words; it starts with "Unexpected" when several complexes were retired."""

    step = np.asarray(_array(stop_step)).reshape(-1)
    labels = LABELS["retired"]
    names = complex_names(step.shape[0])
    retired = [k for k in range(step.shape[0]) if step[k] >= 0]
    if not retired:
        return labels["none"]
    if len(retired) == 1:
        return labels["one"].format(name=names[retired[0]], step=int(step[retired[0]]))
    return labels["several"].format(count=len(retired), names=", ".join(names[k] for k in retired))


def show_retired(stop_step, final_status, retire_store) -> None:
    """Show which complex was retired and when, and play the saved frames.

    ``stop_step`` holds one MD step per complex, ``-1`` for a complex that ran
    to the end. ``final_status`` holds the status of each complex after the
    run. ``retire_store`` holds the three complexes at every saved step. A line
    under the table says what happened; it starts with "Unexpected" when the
    number of retired complexes is not one.
    """

    from IPython.display import HTML, Markdown, display

    labels = LABELS["retired"]
    display(HTML(_html_table([labels["complex"], labels["step"]], retired_rows(stop_step, final_status))))
    display(Markdown(retired_line(stop_step)))
    _show_player(retire_store)
