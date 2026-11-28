# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result views of Agent 08, Paths of least resistance, batched.

Every plot has two layers: a ``build_*`` function that returns a Matplotlib
figure (used by the tests) and the public function that shows it as one
920 px picture. Tables are shown as tables. The 3D views use the course
viewer (``shared/alchemi-viewer``), light only.

The views accept what a learner's agent returns: tensors or arrays, energies
of shape (n,) or (n, 1), a ``Path`` or a string for the log, a DataFrame or a
list of rows for the table.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import re
import textwrap
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "plot_energy_profiles",
    "show_neb_log",
    "show_paths",
    "show_reactions",
    "show_saddle_table",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STYLE = _PROJECT_ROOT / "shared" / "alchemi-figure.mplstyle"
DISPLAY_PX = 920
KCAL_PER_EV = 23.0605  # kcal/mol in one eV
GREEN, GREY = "#76B900", "#6F6F6F"

# ---------------------------------------------------------------------------
# Wording on the views: plain labels "Quantity [unit]".
# ---------------------------------------------------------------------------
LABELS = {
    "reactions": {"reactant": "reactant", "product": "product", "title": "The two ends of each reaction"},
    "paths": {
        "path": "Path",
        "images": "Images",
        "atoms": "Atoms",
        "image": "Image",
        "title": "The images of every path",
    },
    "neb_log": {
        "path": "Path",
        "step": "Stopped at step",
        "force": "Force there [eV/Å]",
        "xlabel": "Step",
        "ylabel": "Force [eV/Å]",
        "alt": "The force of each path against the step, one panel for each path. A dot marks the step at which the path stopped.",
    },
    "profiles": {
        "xlabel": "Image",
        "ylabel": "Energy [kcal/mol]",
        "barrier": "{barrier:.2f} kcal/mol",
        "converged": "converged",
        "not_converged": "not converged",
        "alt": "The energy of each image along each path, relative to the reactant, in kcal/mol, one panel for each path.",
        # the table under the plot: one row per path
        "path": "Path",
        "barrier_column": "Barrier [kcal/mol]",
        "converged_column": "Converged",
        "yes": "yes",
        "no": "no",
        "note_column": "",
        "unexpected": "unexpected: the path ran out of steps, so this barrier is an estimate",
    },
    "saddle": {
        # headers of the columns the view recognises; any other column gets its own name, made readable
        "verdict": "Saddle point",
        "image": "Image",
        "count": "Imaginary modes",
        "force": "Largest force",
        "energy": "Energy",
        "frequency": "frequency",
        "eigenvalue": "eigenvalue",
        "number": "{value:.4f}",  # one format for every number that is not a whole number
        "no_path": "saddle_table has no column that names the path of each row, so the rows are shown in one list.",
        "no_image": "saddle_table has no column with the image number of each row.",
        "none": "unexpected: no image of {path} is a saddle point in this table",
        "several": "unexpected: {count} images of {path} are saddle points in this table",
    },
}


def _array(values: Any) -> np.ndarray:
    """Return a tensor or array-like as a NumPy array on the host."""

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    return np.asarray(values)


def _flat(values: Any) -> np.ndarray:
    """A float vector from a tensor, an array or a list of 0-d tensors."""

    if isinstance(values, (list, tuple)):
        values = [float(_array(v).reshape(-1)[0]) for v in values]
    return _array(values).astype(float).reshape(-1)


def images_per_path(batch) -> list[int]:
    """Number of images in each path of a batch built by ``interpolate_paths``."""

    layout = getattr(batch, "group_layout", None)
    if layout is None:
        raise ValueError(
            "this batch has no groups; a batch of paths comes from interpolate_paths, "
            "one group of images per path"
        )
    return [int(n) for n in layout.num_graphs_per_group.tolist()]


def path_images(batch) -> list[list]:
    """The images of every path as ASE ``Atoms``: one list per path."""

    from ase import Atoms

    counts = [int(n) for n in batch.num_nodes_per_graph.tolist()]
    numbers = _array(batch.atomic_numbers).reshape(-1)
    positions = _array(batch.positions).astype(float).reshape(-1, 3)
    images, start = [], 0
    for count in counts:
        images.append(Atoms(numbers=numbers[start : start + count], positions=positions[start : start + count]))
        start += count
    out, first = [], 0
    for n in images_per_path(batch):
        out.append(images[first : first + n])
        first += n
    return out


def _names(reaction_names, count: int) -> list[str]:
    names = [str(name) for name in reaction_names]
    if len(names) != count:
        raise ValueError(f"reaction_names has {len(names)} names; there are {count} paths")
    return names


# ---------------------------------------------------------------------------
# Tables and pictures
# ---------------------------------------------------------------------------


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th style='text-align:left'>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td style='text-align:left'>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _display_html(markup: str) -> None:
    import warnings

    from IPython.display import HTML, display

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # IPython suggests IFrame for a srcdoc iframe
        display(HTML(markup))


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


def _show(figure, alt: str) -> None:
    """Show a figure as one picture, at most 920 px wide, with alt text."""

    from base64 import b64encode
    from io import BytesIO

    import matplotlib.pyplot as plt

    buffer = BytesIO()
    with _course_style():
        figure.savefig(buffer, format="png", dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)
    payload = b64encode(buffer.getvalue()).decode("ascii")
    _display_html(
        f'<img src="data:image/png;base64,{payload}" alt="{escape(alt, quote=True)}" '
        f'style="display:block;box-sizing:border-box;width:100%;max-width:{DISPLAY_PX}px;'
        'height:auto;margin:0">'
    )


def _panels(count: int, height: float = 2.9):
    """One row of ``count`` panels across the 920 px figure."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, count, figsize=(9.2, height), squeeze=False)
    return figure, list(axes[0])


def _title(name: str, extra: str = "") -> str:
    """A panel title that fits a quarter of the figure: the name wrapped, then one more line."""

    lines = textwrap.wrap(name, 22) or [name]
    return "\n".join(lines + ([extra] if extra else []))


# ---------------------------------------------------------------------------
# 3D views (course viewer)
# ---------------------------------------------------------------------------

_VIEW_CSS = (
    "#root{font-family:'NVIDIA Sans','Helvetica Neue',Arial,sans-serif;font-size:13px;color:#1A1A1A}"
    ".a08-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}"
    ".a08-card{border:1px solid #D4D4D4;border-radius:10px;padding:8px}"
    ".a08-card>strong{display:block;margin:0 2px 6px;font-weight:700}"
    ".a08-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}"
    ".a08-end>span{display:block;margin:4px 2px 0;font-size:12px;color:#6F6F6F}"
    ".av-readout{display:none}"
)


def _structure_json(name: str, atoms, positions: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "formula": atoms.get_chemical_formula(),
        "atom_count": len(atoms),
        "symbols": atoms.get_chemical_symbols(),
        "positions": np.round(positions, 4).tolist(),
        "periodic": False,
        "cell": None,
        "cell_edges": [],
    }


def _view_frame(reference: np.ndarray):
    """A turn that faces a structure to the camera; the same turn serves both ends of a path."""

    from .viewers import _to_view_frame, _viewing_rotation

    origin = reference.mean(axis=0)
    rotation = _viewing_rotation(reference)
    return lambda points: _to_view_frame(np.asarray(points, dtype=float), origin, rotation)


def _iframe(body: str, mount: str, *, title: str, height: int) -> str:
    from . import viewers_three as three

    document = three._viewer_document(
        f"<style>{_VIEW_CSS}</style><div id='root'>{body}</div>",
        mount + three._THEME_TO_BODY + three._AUTO_HEIGHT,
    )
    return (
        f'<iframe title="{escape(title, quote=True)}" srcdoc="{escape(document, quote=True)}" '
        f'style="display:block;width:100%;max-width:{DISPLAY_PX}px;height:{height}px;border:0;'
        'background:transparent;" loading="eager"></iframe>'
    )


def reactions_markup(reactants, products, reaction_names) -> str:
    """The iframe that shows the reactant and the product of each reaction side by side."""

    from .viewers import _json_for_script

    names = _names(reaction_names, len(reactants))
    if len(products) != len(reactants):
        raise ValueError(f"{len(reactants)} reactants and {len(products)} products; one of each per reaction")
    words = LABELS["reactions"]
    cards, mounts = [], []
    for k, (name, a, b) in enumerate(zip(names, reactants, products)):
        turn = _view_frame(a.positions)
        cards.append(
            f"<div class='a08-card'><strong>{escape(name)}</strong><div class='a08-pair'>"
            f"<div class='a08-end'><div id='r{k}'></div><span>{escape(words['reactant'])}</span></div>"
            f"<div class='a08-end'><div id='p{k}'></div><span>{escape(words['product'])}</span></div>"
            "</div></div>"
        )
        for slot, atoms, word in ((f"r{k}", a, words["reactant"]), (f"p{k}", b, words["product"])):
            data = _structure_json(f"{name}, {word}", atoms, turn(atoms.positions))
            mounts.append(
                f"AlchemiViewer.mountStructure(document.getElementById('{slot}'),"
                f"{_json_for_script(data)},{{height:150,theme:\"light\"}});"
            )
    rows = (len(names) + 1) // 2
    return _iframe(
        f"<div class='a08-grid'>{''.join(cards)}</div>",
        "".join(mounts),
        title=words["title"],
        height=rows * 220,
    )


def show_reactions(reactants, products, reaction_names) -> None:
    """Show the reactant and the product of each reaction side by side in 3D."""

    _display_html(reactions_markup(reactants, products, reaction_names))


def paths_rows(paths, reaction_names) -> tuple[list[str], list[list[Any]]]:
    """Header and rows of the paths table: one row per path, its images and atoms."""

    images = path_images(paths)
    names = _names(reaction_names, len(images))
    words = LABELS["paths"]
    rows = [[name, len(frames), len(frames[0])] for name, frames in zip(names, images)]
    return [words["path"], words["images"], words["atoms"]], rows


def paths_player_markup(paths, reaction_names) -> str:
    """The iframe that plays through the images of every path, one card per path."""

    from .viewers import _json_for_script

    images = path_images(paths)
    names = _names(reaction_names, len(images))
    longest = max(len(frames) for frames in images)
    items = []
    for name, frames in zip(names, images):
        turn = _view_frame(frames[0].positions)
        shown = [np.round(turn(frame.positions), 4).tolist() for frame in frames]
        shown += [shown[-1]] * (longest - len(shown))  # a shorter path rests on its last image
        items.append(
            {
                "name": name,
                "atom_count": len(frames[0]),
                "symbols": frames[0].get_chemical_symbols(),
                "cell": None,
                "cell_edges": [],
                "frames": shown,
                "atom_forces": [[0.0] * len(frames[0]) for _ in shown],  # element colours
                "fmax": [1.0e9] * len(shown),  # no image is marked as finished
            }
        )
    # The course player counts relaxation steps; here every frame is one image of a path.
    relabel = (
        "(function(){var s=document.querySelector('.av-step');if(!s)return;"
        f"var fix=function(){{var t=s.textContent.replace(/^Step/,{_json_for_script(LABELS['paths']['image'])});"
        "if(t!==s.textContent)s.textContent=t;};"
        "new MutationObserver(fix).observe(s,{childList:true,characterData:true,subtree:true});fix();})();"
    )
    mount = (
        "AlchemiViewer.mountPlayer(document.getElementById('root'),"
        f"{_json_for_script(items)},{{height:200,frame_interval_ms:400,theme:\"light\"}});" + relabel
    )
    return _iframe("", mount, title=LABELS["paths"]["title"], height=330)


def show_paths(paths, reaction_names) -> None:
    """Show one row per path (images, atoms) and the images of each path in a 3D player."""

    headers, rows = paths_rows(paths, reaction_names)
    _display_html(_html_table(headers, rows))
    _display_html(paths_player_markup(paths, reaction_names))


# ---------------------------------------------------------------------------
# The NEB log
# ---------------------------------------------------------------------------

FINISHED = 2  # status of a path that has left the climbing stage


def read_neb_log(neb_log) -> dict[int, dict[str, np.ndarray]]:
    """Read the diagnostics CSV of an NEB run: per path, its steps, forces and statuses.

    The file holds one row per path per logged step. A path that has finished
    keeps its row in every later step, so the rows after its stop repeat it.
    """

    path = Path(str(neb_log))
    if not path.is_file():
        raise FileNotFoundError(f"neb_log {path} does not exist; it is the diagnostics file of the NEB run")
    table: dict[int, dict[str, list]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"step", "group_idx", "status", "fmax"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"neb_log {path.name} has no column {min(missing)}; it is not an NEB diagnostics file")
        for row in reader:
            entry = table.setdefault(int(float(row["group_idx"])), {"step": [], "fmax": [], "status": []})
            entry["step"].append(int(float(row["step"])))
            entry["fmax"].append(float(row["fmax"]))
            entry["status"].append(int(float(row["status"])))
    return {k: {name: np.asarray(v) for name, v in entry.items()} for k, entry in sorted(table.items())}


def neb_log_data(neb_log, reaction_names) -> list[dict[str, Any]]:
    """Per path: its name, steps and forces up to its stop, the stop step and the force there.

    The stop is the first row at which the path is finished; a path that the
    run's step budget cut off stops at its last row.
    """

    log = read_neb_log(neb_log)
    names = _names(reaction_names, len(log))
    out = []
    for name, entry in zip(names, log.values()):
        finished = np.flatnonzero(entry["status"] >= FINISHED)
        last = int(finished[0]) if finished.size else len(entry["step"]) - 1
        out.append(
            {
                "name": name,
                "steps": entry["step"][: last + 1],
                "force": entry["fmax"][: last + 1],
                "stop_step": int(entry["step"][last]),
                "stop_force": float(entry["fmax"][last]),
            }
        )
    return out


def neb_log_rows(neb_log, reaction_names) -> tuple[list[str], list[list[Any]]]:
    """Header and rows of the table under the plot: path, stop step, force there."""

    words = LABELS["neb_log"]
    rows = [[d["name"], d["stop_step"], f"{d['stop_force']:.4f}"] for d in neb_log_data(neb_log, reaction_names)]
    return [words["path"], words["step"], words["force"]], rows


@_in_course_style
def build_neb_log_figure(neb_log, reaction_names):
    """One panel per path: its force against the step, with a dot where the path stopped."""

    data = neb_log_data(neb_log, reaction_names)
    words = LABELS["neb_log"]
    figure, axes = _panels(len(data))
    low = min(float(np.min(d["force"][d["force"] > 0])) for d in data if np.any(d["force"] > 0))
    high = max(float(np.max(d["force"])) for d in data)
    for axis, d in zip(axes, data):
        axis.plot(d["steps"], d["force"], color=GREY, linewidth=1.4)
        axis.plot([d["stop_step"]], [d["stop_force"]], "o", color=GREEN, markersize=6, zorder=3)
        axis.set_yscale("log")
        axis.set_ylim(low * 0.6, high * 1.6)
        axis.set_xlim(left=0)
        axis.set_title(_title(d["name"]))
        axis.set_xlabel(words["xlabel"])
    axes[0].set_ylabel(words["ylabel"])
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    return figure


def show_neb_log(neb_log, reaction_names) -> None:
    """Plot each path's force against the step and list where each path stopped."""

    _show(build_neb_log_figure(neb_log, reaction_names), LABELS["neb_log"]["alt"])
    headers, rows = neb_log_rows(neb_log, reaction_names)
    _display_html(_html_table(headers, rows))


# ---------------------------------------------------------------------------
# Energy profiles
# ---------------------------------------------------------------------------


def profile_data(relaxed_paths, path_energy, reaction_names, barriers, converged) -> list[dict[str, Any]]:
    """Per path: energies of its images relative to the first, kcal/mol, its barrier and flag.

    ``path_energy`` holds one value per image in eV, (n,) or (n, 1), in the
    order of the batch; ``barriers`` one value per path in eV; ``converged``
    one true or false per path.
    """

    counts = images_per_path(relaxed_paths)
    names = _names(reaction_names, len(counts))
    energy = _flat(path_energy)
    if energy.size != sum(counts):
        raise ValueError(f"path_energy has {energy.size} values; relaxed_paths has {sum(counts)} images")
    barrier = _flat(barriers)
    flags = _array(converged).reshape(-1) if not isinstance(converged, (list, tuple)) else np.asarray(
        [bool(_array(v).reshape(-1)[0]) for v in converged]
    )
    for label, values in (("barriers", barrier), ("converged", flags)):
        if values.size != len(counts):
            raise ValueError(f"{label} has {values.size} values; there are {len(counts)} paths")
    out, start = [], 0
    for k, (name, count) in enumerate(zip(names, counts)):
        relative = (energy[start : start + count] - energy[start]) * KCAL_PER_EV
        top = 1 + int(np.argmax(relative[1:-1])) if count > 2 else int(np.argmax(relative))
        out.append(
            {
                "name": name,
                "energy": relative,
                "top": top,
                "barrier": float(barrier[k]) * KCAL_PER_EV,
                "converged": bool(flags[k]),
            }
        )
        start += count
    return out


@_in_course_style
def build_energy_profiles_figure(relaxed_paths, path_energy, reaction_names, barriers, converged):
    """One panel per path: the energy of each image relative to the reactant, in kcal/mol."""

    data = profile_data(relaxed_paths, path_energy, reaction_names, barriers, converged)
    words = LABELS["profiles"]
    figure, axes = _panels(len(data), height=3.2)
    for axis, d in zip(axes, data):
        images = np.arange(len(d["energy"]))
        axis.plot(images, d["energy"], "-o", color=GREY, linewidth=1.4, markersize=4)
        axis.plot([d["top"]], [d["energy"][d["top"]]], "o", color=GREEN, markersize=7, zorder=3)
        state = words["converged"] if d["converged"] else words["not_converged"]
        axis.set_title(_title(d["name"], words["barrier"].format(barrier=d["barrier"]) + ", " + state))
        axis.set_xlabel(words["xlabel"])
        axis.set_xticks(images if len(images) <= 12 else images[:: max(1, len(images) // 6)])
    axes[0].set_ylabel(words["ylabel"])
    return figure


def profile_rows(relaxed_paths, path_energy, reaction_names, barriers, converged) -> tuple[list[str], list[list[Any]]]:
    """Header and rows of the table under the plot: path, barrier, converged yes or no, and a note.

    A path that is not converged is marked "unexpected"; its barrier is shown all the same.
    """

    words = LABELS["profiles"]
    rows = [
        [
            d["name"],
            f"{d['barrier']:.2f}",
            words["yes"] if d["converged"] else words["no"],
            "" if d["converged"] else words["unexpected"],
        ]
        for d in profile_data(relaxed_paths, path_energy, reaction_names, barriers, converged)
    ]
    return [words["path"], words["barrier_column"], words["converged_column"], words["note_column"]], rows


def plot_energy_profiles(relaxed_paths, path_energy, reaction_names, barriers, converged) -> None:
    """Plot the energy along every path, relative to its first image, in kcal/mol, and list the barriers.

    ``relaxed_paths`` is the Batch the NEB returned; ``path_energy`` holds one
    energy per image in eV, in the order of the batch; ``barriers`` one forward
    barrier per path in eV; ``converged`` one true or false per path. Tensors,
    arrays and lists are taken. The plot converts to kcal/mol. Each
    panel title gives the barrier and whether the path converged. The table
    under the plot has one row per path: its barrier and a converged yes or
    no; a path that is not converged is marked "unexpected".
    """

    _show(
        build_energy_profiles_figure(relaxed_paths, path_energy, reaction_names, barriers, converged),
        LABELS["profiles"]["alt"],
    )
    headers, rows = profile_rows(relaxed_paths, path_energy, reaction_names, barriers, converged)
    _display_html(_html_table(headers, rows))


# ---------------------------------------------------------------------------
# The learner's saddle table
# ---------------------------------------------------------------------------

_PATH_WORDS = ("path", "reaction", "system", "group")
_YES = {"yes", "y", "true"}
_NO = {"no", "n", "false"}


def saddle_frame(saddle_table):
    """The learner's table as a DataFrame, unchanged in content."""

    import pandas as pd

    if isinstance(saddle_table, pd.DataFrame):
        return saddle_table
    return pd.DataFrame(saddle_table)


def _answers(column) -> list[bool] | None:
    """A column as yes-or-no answers, or None when it holds anything else."""

    out = []
    for value in column:
        if isinstance(value, (bool, np.bool_)):
            out.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in _YES | _NO:
            out.append(value.strip().lower() in _YES)
        else:
            return None
    return out


def saddle_answers(frame) -> tuple[str, list[bool]] | None:
    """The column that holds the verdict, and its answers.

    A column whose name says "saddle" wins; otherwise the last column that
    holds only yes or no.
    """

    found = {str(c): a for c in frame.columns if (a := _answers(frame[c].tolist())) is not None}
    if not found:
        return None
    named = [c for c in found if "saddle" in c.lower()]
    column = named[-1] if named else list(found)[-1]
    return column, found[column]


def path_column(frame, names=None) -> str | None:
    """The column that names the path of each row, if the table has one.

    With ``names`` (the reaction names) it is the first column whose values
    are all among them. Without, it is the first text column whose name says
    path, reaction, system or group.
    """

    if names is not None:
        known = {str(n) for n in names}
        for column in frame.columns:
            values = frame[column].tolist()
            if values and all(isinstance(v, str) and v in known for v in values):
                return column
        return None
    for column in frame.columns:
        values = frame[column].tolist()
        named = any(word in str(column).lower() for word in _PATH_WORDS)
        if named and _answers(values) is None and all(isinstance(v, str) for v in values):
            return column
    # Otherwise the first text column, not yes or no, whose values each fill one block of rows, as a path name
    # does: a column called "name" is read as the path. A verdict in words ("slope", "saddle point", "slope")
    # breaks into several blocks and is not taken.
    for column in frame.columns:
        values = frame[column].tolist()
        if values and all(isinstance(v, str) for v in values) and _answers(values) is None:
            starts = [v for i, v in enumerate(values) if i == 0 or v != values[i - 1]]
            if len(starts) == len(set(starts)) and len(starts) < len(values):
                return column
    return None


def image_column(frame, path_col=None) -> str | None:
    """The column with the image number of each row: whole numbers that count each path from 0 or 1.

    A column whose name says "image" wins; otherwise the first column of
    whole numbers that counts up by one within every path.
    """

    def counts_up(column) -> bool:
        if frame[column].dtype.kind not in "iu":
            return False
        blocks = [frame[column].tolist()] if path_col is None else [
            frame.loc[frame[path_col] == value, column].tolist() for value in dict.fromkeys(frame[path_col].tolist())
        ]
        return all(sorted(b) == list(range(min(b), min(b) + len(b))) and min(b) in (0, 1) for b in blocks if b)

    candidates = [c for c in frame.columns if c != path_col and counts_up(c)]
    named = [c for c in candidates if "image" in str(c).lower()]
    return (named or candidates or [None])[0]


# Unit spellings in a column name, at its end, and the unit they mean.
_UNITS = (
    (r"ev_?per_?(a|ang|angstrom|å)|ev/(a|å)|ev_a", "eV/Å"),
    (r"cm\^?-?1|cm_?inv|inv_?cm|wavenumbers?", "cm⁻¹"),
    (r"kcal_?(per_?)?mol|kcal/mol", "kcal/mol"),
    (r"ev_?per_?(a|å)2|ev/(a|å)\^?2", "eV/Å²"),
    (r"ev", "eV"),
)


def _is_count(frame, column) -> bool:
    """True for a column of whole numbers whose name says "imag" and names no frequency or wavenumber."""

    label = str(column).lower()
    return (
        "imag" in label
        and not any(w in label for w in ("image", "freq", "cm", "wave", "lowest", "value"))
        and frame[column].dtype.kind in "iu"
    )


def saddle_header(column, frame=None, verdict=None, image=None) -> str:
    """A readable header with its unit for a column of the learner's table.

    The verdict, the image number and the count of imaginary modes get fixed
    headers. Any other name loses its underscores, and a unit spelled at its
    end (``fmax_eV_per_A``, ``lowest_freq_cm1``) goes in brackets. A force
    column with no unit in its name is in eV/Å, the model's unit.
    """

    words = LABELS["saddle"]
    if column == verdict:
        return words["verdict"]
    if column == image:
        return words["image"]
    if frame is not None and _is_count(frame, column):
        return words["count"]
    label = str(column).strip()
    unit = None
    for pattern, name in _UNITS:
        found = re.search(rf"[\s_(\[]({pattern})[)\]]?$", label, flags=re.IGNORECASE)
        if found:
            unit, label = name, label[: found.start()].rstrip()
            break
    low = label.lower().replace(" ", "_")
    if low in ("fmax", "f_max", "max_force", "largest_force", "force", "max_f", "fmax_img"):
        label, unit = words["force"], unit or "eV/Å"
    elif low in ("energy", "e"):
        label = words["energy"]
    else:
        label = re.sub(r"\bfreq\b", words["frequency"], label.replace("_", " "))
        label = re.sub(r"\beig\b", words["eigenvalue"], label)
        label = label[:1].upper() + label[1:]
    return f"{label} [{unit}]" if unit else label


def _saddle_cell(value) -> str:
    """One cell of the table as text: one format for every number that is not a whole number, no "-0.0000"."""

    if isinstance(value, (float, np.floating)):
        text = LABELS["saddle"]["number"].format(value=float(value))
        return text[1:] if text.startswith("-") and float(text) == 0.0 else text
    return str(value)


def _saddle_line(text: str) -> str:
    """One line of words above or under the table."""

    return f"<p style='margin:4px 0'>{escape(text)}</p>"


def _plain_saddle_table(frame, verdict) -> str:
    """A table with no path column: one list of rows, the verdict first, under a line that says so."""

    order = ([verdict] if verdict is not None else []) + [c for c in frame.columns if c != verdict]
    headers = [saddle_header(c, frame, verdict) for c in order]
    table = _html_table(headers, [[_saddle_cell(v) for v in row] for row in frame[order].itertuples(index=False)])
    return _saddle_line(LABELS["saddle"]["no_path"]) + table


def saddle_table_html(saddle_table) -> str:
    """The learner's table, one block of rows per path, the images in order.

    A line above the table says so when the table has no path or no image
    column. A line under it marks a path as "unexpected" when the table
    holds no yes, or more than one, for that path.
    """

    frame = saddle_frame(saddle_table)
    column = path_column(frame)
    words = LABELS["saddle"]
    line = _saddle_line
    found = saddle_answers(frame)
    verdict = found[0] if found is not None else None
    if column is None:
        return _plain_saddle_table(frame, verdict)
    image = image_column(frame, column)
    # the verdict first, so it stays in view on a narrow screen; then the image number and the numbers
    first = [c for c in (verdict, image) if c is not None]
    others = first + [c for c in frame.columns if c not in (column, *first)]
    head = "".join(f"<th style='text-align:left'>{escape(saddle_header(c, frame, verdict, image))}</th>" for c in others)
    body, notes = [], [] if image is not None else [line(words["no_image"])]
    for value in dict.fromkeys(frame[column].tolist()):  # paths in the order of the table
        block = frame[frame[column] == value]
        if image is not None:
            block = block.sort_values(image, kind="stable")
        body.append(
            f"<tr><th colspan='{len(others)}' style='text-align:left;border-top:1px solid #B8B8B8'>"
            f"{escape(str(value))}</th></tr>"
        )
        for row in block[others].itertuples(index=False):
            body.append(
                "<tr>" + "".join(f"<td style='text-align:left'>{escape(_saddle_cell(v))}</td>" for v in row) + "</tr>"
            )
        if found is not None:
            count = sum(_answers(block[found[0]].tolist()) or [])
            if count == 0:
                notes.append(line(words["none"].format(path=value)))
            elif count > 1:
                notes.append(line(words["several"].format(path=value, count=count)))
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    return table + "".join(notes)


def show_saddle_table(saddle_table) -> None:
    """Show the learner's table grouped by path, the images of each path in order.

    ``saddle_table`` is a pandas DataFrame or a list of rows, one row per
    image of ``relaxed_paths``, with: a column that names the path (the names
    in ``reaction_names``; call it ``path``), a column with the image number
    counted from 0 within each path (``image``), a column of yes or no
    (``saddle``), and the numbers behind the answer.
    """

    _display_html(saddle_table_html(saddle_table))
