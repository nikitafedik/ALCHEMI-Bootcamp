# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The packed batch rows of the agentic notebook 02 (cell 10), drawn live.

All atom rows as one strip coloured by molecule, ``batch_idx`` under it,
molecule boundaries marked, and each molecule's name (or formula) with its atom
count on the strip. Works for 1 to 12 molecules and up to about 600 atom rows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from . import common
from .data_records import PROJECT_ROOT

_STYLES = {
    "light": PROJECT_ROOT / "shared" / "alchemi-deck.mplstyle",
    "dark": PROJECT_ROOT / "shared" / "alchemi-deck-dark.mplstyle",
}
# One colour per molecule, cycled: NV greens and kit greys that stay apart from their
# neighbours in both themes. Text on each block uses the matching ink.
_MOLECULE_COLOURS = {
    "light": ["#4B7A00", "#9DC85A", "#5C6770", "#C9D88F", "#8A949B", "#76B900", "#3E464C", "#B8C2C9"],
    "dark": ["#9FD13A", "#4B7A00", "#9AA3AB", "#5F7A3E", "#C9CED3", "#76B900", "#6E7F57", "#C2CF9A"],
}
_INK = {"light": {"ink": "#1A1A1A", "mute": "#6F6F6F", "rule": "#D4D4D4", "wash": "#F4F4F4", "bg": "#FFFFFF"},
        "dark": {"ink": "#F3F4F6", "mute": "#9AA3AB", "rule": "#3A444D", "wash": "#252C32", "bg": "#1A1F23"}}
INDEX_LABEL = "batch_idx"


def _theme(theme: str | None) -> str:
    theme = theme or common.current_theme() or "light"
    return "dark" if theme == "dark" else "light"


_FORMULA = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def _pretty(label: str) -> str:
    """A chemical formula gets subscript counts (C10H12O2 -> C₁₀H₁₂O₂); names stay as given."""
    if not _FORMULA.match(label) or not any(ch.isdigit() for ch in label):
        return label
    return label.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _readable_on(hex_colour: str) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#1A1A1A" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#FFFFFF"


def draw_batch_rows(labels: Sequence[str], counts: Sequence[int], batch_idx: Sequence[int],
                    symbols: Sequence[str] | None = None, theme: str | None = None) -> Figure:
    """Draw the packed rows of one batch and return the Figure.

    labels: one name or formula per molecule, in batch order. counts: atoms per
    molecule. batch_idx: the molecule index of every atom row. symbols: the
    element of every atom row (written on the rows when there are few enough).
    theme: "light" or "dark"; by default the course theme.
    """
    if len(labels) != len(counts):
        raise ValueError("labels and counts must have one entry per molecule")
    total = int(sum(counts))
    if len(batch_idx) != total:
        raise ValueError(f"batch_idx has {len(batch_idx)} rows, counts add up to {total}")
    if symbols is not None and len(symbols) != total:
        raise ValueError("symbols must have one entry per atom row")
    th = _theme(theme)
    ink = _INK[th]
    colours = _MOLECULE_COLOURS[th]
    with plt.style.context(str(_STYLES[th])):
        fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=150)
    ax.set_xlim(-0.02 * total, total * 1.02)
    ax.set_ylim(-1.15, 2.75)
    ax.axis("off")

    starts = [0]
    for c in counts:
        starts.append(starts[-1] + int(c))
    # names sit above the strip; neighbours that would touch go to a second line
    char_w = total / 92.0  # data units per character at this width, roughly
    last_end = [-1e9, -1e9, -1e9]   # up to three label lines
    top_level = 0
    for i, (lab, c) in enumerate(zip(labels, counts)):
        x0, x1 = starts[i], starts[i + 1]
        col = colours[i % len(colours)]
        ax.add_patch(Rectangle((x0, 0), c, 1, facecolor=col, edgecolor="none"))
        # thin row separators when each row is at least about 2 px wide
        if total <= 260:
            for k in range(x0 + 1, x1):
                ax.plot([k, k], [0.06, 0.94], color=ink["bg"], lw=0.35, alpha=0.55)
        if symbols is not None and total <= 60:
            for k in range(x0, x1):
                ax.text(k + 0.5, 0.5, symbols[k], ha="center", va="center", fontsize=6.5,
                        color=_readable_on(col))
        # index band under the strip
        ax.add_patch(Rectangle((x0, -0.95), c, 0.62, facecolor=ink["wash"], edgecolor=ink["rule"], lw=0.8))
        ax.text((x0 + x1) / 2, -0.64, str(i), ha="center", va="center", fontsize=9, color=ink["ink"],
                family="monospace")
        # label: name (bold) over its atom count, centred on the block; a block whose
        # label would touch its left neighbour's moves up one line, with a thin leader
        half = max(len(lab), len(str(c))) * char_w / 2
        cx = (x0 + x1) / 2
        level = next((k for k in range(len(last_end)) if cx - half > last_end[k] + char_w), len(last_end) - 1)
        last_end[level] = cx + half
        top_level = max(top_level, level)
        base = 1.08 + 0.54 * level
        if level:
            ax.plot([cx, cx], [1.0, base - 0.02], color=ink["rule"], lw=0.8)
        ax.text(cx, base, str(c), ha="center", va="bottom", fontsize=8, color=ink["mute"])
        ax.text(cx, base + 0.24, _pretty(lab), ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=ink["ink"])
    # boundaries across strip and index band
    for s in starts:
        ax.plot([s, s], [-0.95, 1.0], color=ink["ink"], lw=1.4)
    ax.text(-0.03 * total, -0.64, INDEX_LABEL, ha="right", va="center", fontsize=9, family="monospace",
            color=ink["mute"])
    for s in (0, total):
        ax.text(s, -1.12, str(s), ha="center", va="top", fontsize=8, color=ink["mute"])
    # height follows the label lines actually used, so a short set has no empty band on top
    ax.set_ylim(-1.15, 1.75 + 0.54 * top_level)
    fig.set_size_inches(7.2, 1.85 + 0.33 * top_level)
    fig.patch.set_alpha(0.0)
    fig.set_layout_engine("none")   # the course style turns on constrained layout; place the axes by hand
    ax.set_position([0.14, 0.04, 0.84, 0.94])   # room on the left for the batch_idx label
    return fig
