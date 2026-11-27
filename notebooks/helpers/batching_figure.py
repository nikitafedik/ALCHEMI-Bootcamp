# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cell 11 of the Core quick tour: what batching gains, in two panes.

Left pane: the time for the learner's own workload (the cell 9 selection,
``batch_multiplier`` copies, a fixed number of optimizer steps) as one batch,
without batching on the same GPU (timed live, one system at a time), and
without batching on a CPU (stored). Right pane: throughput against batch size
for stored cuEq + compile runs on three GPUs, with the learner's batch as one
dot.
"""

from __future__ import annotations

import csv
from base64 import b64encode
from functools import lru_cache
from html import escape
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from . import common
from .data_records import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Learner-visible wording of the figure.
# ---------------------------------------------------------------------------
LEFT_TITLE = "Time to complete, lower is better"
RIGHT_TITLE = "Throughput, higher is better"
BAR_YOU = "Batch of {structures}"
BAR_GPU = "Batch of 1"
BAR_CPU = "CPU run (cached)"
LEFT_XLABEL = "Time for {structures} structures × {steps} steps [s]"
SPEEDUP = "{factor}× faster than batch of 1"
G2_XLABEL = "Structures in the batch"
G2_YLABEL = "Throughput [structure-steps/s]"
G2_LINE_CUEQ = "{gpu}, cuEq + compile"
G2_YOU = "Your batch"
MEMORY_CHIP = "{gb:.1f} GB"
ATOMS_NOTE = "{atoms:,} atoms"
# Memory chips: one shared chip at batch 1 (the largest of the lines), one at each line's last size.
GPU_LINE = "Your run: {gpu}, {memory} GB."
CPU_LINE = "CPU: {cpu}, cached."
NO_GPU_NOTE = "No GPU run was measured here."

# Optimizer steps per structure in the left pane, the same as the stored runs.
STEPS = 50

# ---------------------------------------------------------------------------
# Stored data (notebooks/data/core00, see its README).
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "notebooks" / "data" / "core00"
CPU_SINGLES = DATA_DIR / "cpu_singles.csv"
GRAPH2_CUEQ = DATA_DIR / "graph2_cueq.csv"
CPU_SINGLES_COLUMNS = (
    "key",
    "display_name",
    "atoms",
    "steps",
    "time_s",
    "min_s",
    "max_s",
    "repeats",
    "cpu",
    "threads",
    "source",
)
GRAPH2_CUEQ_COLUMNS = (
    "gpu",
    "route",
    "structures",
    "atoms",
    "structure_steps_per_s",
    "min",
    "max",
    "repeats",
    "peak_reserved_gb",
    "source",
)
_GPU_LABELS = {
    "NVIDIA L40S": "L40S",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA B200": "B200",
}

# The course is light only. Course figure standard: 920 x 400 px at 100 px per inch,
# saved at 2x without a tight crop; text 15 px (titles), 13 px (labels, values, legend),
# 11 px (ticks, chips, footnote); pt = px x 0.72.
_STYLE = PROJECT_ROOT / "shared" / "alchemi-figure.mplstyle"
_FIGSIZE = (9.2, 4.0)
# Axes boxes in px of the 920 x 400 figure: x0, y0 from the bottom, width, height.
_LEFT_BOX = (142, 112, 240, 252)
_RIGHT_BOX = (488, 112, 300, 252)
BODY_PT = 13 * 0.72
SMALL_PT = 11 * 0.72
_PALETTE = {
    "nv": "#76B900",
    "deep": "#4B7A00",
    "ink": "#1A1A1A",
    "ink2": "#4A4A4A",
    "mid": "#8A8A8A",
    "ref": "#B8B8B8",
    "ring": "#FFFFFF",
}


def _read(path: Any, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"{path} must have columns {columns}")
        return list(reader)


@lru_cache(maxsize=1)
def stored_cpu_singles() -> dict[str, Any]:
    """Return the stored CPU time per system for ``STEPS`` steps, by display name."""

    rows = _read(CPU_SINGLES, CPU_SINGLES_COLUMNS)
    if any(int(row["steps"]) != STEPS for row in rows):
        raise ValueError(f"{CPU_SINGLES} must hold {STEPS}-step times")
    return {
        "cpu": rows[0]["cpu"],
        "seconds": {row["display_name"]: float(row["time_s"]) for row in rows},
    }


@lru_cache(maxsize=1)
def stored_graph2() -> dict[str, list[tuple[int, float, float, float, float, int]]]:
    """Return the cuEq + compile lines, ``{gpu: [(n, rate, low, high, peak_gb, atoms)]}``."""

    lines: dict[str, list[tuple[int, float, float, float, float, int]]] = {}
    for row in _read(GRAPH2_CUEQ, GRAPH2_CUEQ_COLUMNS):
        if row["route"] == "plain":
            continue
        lines.setdefault(_GPU_LABELS[row["gpu"]], []).append(
            (
                int(row["structures"]),
                float(row["structure_steps_per_s"]),
                float(row["min"]),
                float(row["max"]),
                float(row["peak_reserved_gb"]),
                int(row["atoms"]),
            )
        )
    return {gpu: sorted(points) for gpu, points in lines.items()}


def learner_run(relaxation: Any) -> dict[str, Any] | None:
    """Return the learner's structures, rate (structure-steps/s) and GPU, or ``None``."""

    preview = relaxation.get("preview", {})
    gpu = preview.get("gpu_name")
    total = preview.get("total_s")
    steps = preview.get("steps_run")
    if not gpu or not total or not steps:
        return None
    copies = int(relaxation["batch_multiplier"])
    structures = len(relaxation["labels"]) * copies
    memory = preview.get("gpu_memory_gb")
    if memory is None:
        try:
            import torch

            memory = round(torch.cuda.get_device_properties(0).total_memory / 1024**3)
        except Exception:  # no CUDA device in this process
            memory = None
    return {
        "labels": list(relaxation["labels"]),
        "copies": copies,
        "structures": structures,
        "rate": structures * float(steps) / float(total),
        "gpu": str(gpu),
        "memory": memory,
    }


class _OneSystem:
    """A picker stand-in that hands one prepared structure to the relaxation."""

    def __init__(self, label: str, structure: Any) -> None:
        self.display_name = label
        self._structure = structure

    def build(self) -> Any:
        return self._structure.copy()

    def selected_specs(self) -> list[Any]:
        return [self]


def time_without_batching(relaxation: Any, *, steps: int = STEPS) -> dict[str, float]:
    """Time each picked system alone on this GPU for ``steps`` L-BFGS steps.

    Each system is the same distorted structure cell 9 relaxed, run as a batch
    of one with a force threshold it cannot reach, so every run takes exactly
    ``steps`` steps. Returns seconds per system, timed like ``total_s``.
    """

    import torch

    from .relaxation_lbfgs import load_mace_model, relax_selected_structures_lbfgs

    device = torch.device("cuda", torch.cuda.current_device())
    # The same speed settings as cell 9, so both bars use one route.
    settings = relaxation["preview"].get("speed_settings", {})
    model = load_mace_model(
        device,
        use_cueq=bool(settings.get("enable_cueq")),
        use_compile=bool(settings.get("compile_model")),
    )
    seconds = {}
    for label, structure in relaxation["preview"]["reference_structures"].items():
        run = relax_selected_structures_lbfgs(
            _OneSystem(label, structure),
            device=device,
            fmax=1e-12,  # eV/Å, never reached: a fixed number of steps
            max_steps=steps,
            distortion_std_angstrom=0.0,  # already distorted in cell 9
            batch_multiplier=1,
            player_factory=lambda **_: None,
            model=model,
        )
        if run["preview"]["steps_run"] != steps:
            raise RuntimeError(f"{label} ran {run['preview']['steps_run']} steps, not {steps}")
        seconds[label] = float(run["preview"]["total_s"])
    del model
    torch.cuda.empty_cache()
    return seconds


def _time_text(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds >= 10 else f"{seconds:.1f} s"


def _factor_text(factor: float) -> str:
    return f"{factor:.0f}" if factor >= 10 else f"{factor:.1f}"


def gpu_label_name(name: str) -> str:
    """Shorten a device name, e.g. ``RTX PRO 6000``."""

    import re

    cleaned = re.sub(
        r"\bNVIDIA\b|Workstation Edition|Max-Q|Server Edition|\bBlackwell\b|"
        r"\bAda Generation\b|\bHBM\d\w*\b|-?\b\d+\s?GB\b|-SXM\d*|[-\s]PCIE\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())


def _left_bars(
    you: dict[str, Any] | None, singles: dict[str, float] | None, labels: list[str], copies: int
) -> list[tuple[str, float, str]]:
    """Return ``(label, seconds, kind)`` for the left pane, top bar first."""

    cpu = stored_cpu_singles()
    bars = []
    if you is not None:
        bars.append(
            (
                BAR_YOU.format(structures=you["structures"]),
                you["structures"] * STEPS / you["rate"],
                "you",
            )
        )
        if singles:
            bars.append(
                (
                    BAR_GPU,
                    sum(singles[label] for label in labels) * copies,
                    "gpu",
                )
            )
    if all(label in cpu["seconds"] for label in labels):
        bars.append(
            (
                BAR_CPU,
                sum(cpu["seconds"][label] for label in labels) * copies,
                "cpu",
            )
        )
    return bars


def _draw_left(
    axis: Any,
    you: dict[str, Any] | None,
    singles: dict[str, float] | None,
    labels: list[str],
    copies: int,
    pal: dict[str, str],
) -> None:
    bars = _left_bars(you, singles, labels, copies)
    axis.set_title(LEFT_TITLE, loc="left")
    if not bars:  # no GPU run and no stored CPU time for these systems
        axis.set_yticks([])
        axis.text(0.5, 0.5, NO_GPU_NOTE, transform=axis.transAxes, ha="center", va="center",
                  color=pal["ink2"])
        return
    colors = {"you": pal["nv"], "gpu": pal["mid"], "cpu": pal["ink2"]}
    longest = max(seconds for _, seconds, _ in bars)
    for index, (_, seconds, kind) in enumerate(bars):
        y = len(bars) - 1 - index
        axis.barh(y, seconds, color=colors[kind], height=0.55)
        text = _time_text(seconds)
        if kind == "you" and len(bars) > 1 and bars[1][2] == "gpu":
            text += "   " + SPEEDUP.format(factor=_factor_text(bars[1][1] / seconds))
        axis.text(
            seconds + longest * 0.015,
            y,
            text,
            va="center",
            ha="left",
            fontweight="bold" if kind == "you" else "normal",
            color=pal["deep"] if kind == "you" else pal["ink"],
        )
    axis.set_yticks(range(len(bars)), [label for label, *_ in bars][::-1])
    axis.tick_params(axis="y", labelsize=BODY_PT)  # category names are labels: 13 px
    axis.set_xlabel(LEFT_XLABEL.format(structures=len(labels) * copies, steps=STEPS))
    axis.set_xlim(0, longest * 1.18)
    axis.set_ylim(-0.6, len(bars) - 0.4)
    axis.grid(axis="y", visible=False)
    axis.set_title(LEFT_TITLE, loc="left")
    for tick in axis.get_yticklabels():
        if bars[0][2] == "you" and tick.get_text() == bars[0][0]:
            tick.set_color(pal["deep"])
            tick.set_fontweight("bold")
    if you is None:
        axis.text(
            0.98,
            0.95,
            NO_GPU_NOTE,
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=pal["ink2"],
        )


def _line(axis: Any, points: Any, color: str, label: str) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    axis.plot(xs, ys, color=color, lw=2.2, marker="o", ms=4, label=label)
    for x, _, low, high, *_ in points:  # repeat range, visible where repeats spread
        axis.plot([x, x], [low, high], color=color, lw=1.2, alpha=0.9)
        for value in (low, high):
            axis.plot([x * 0.94, x * 1.06], [value, value], color=color, lw=1.2)


def _draw_right(axis: Any, you: dict[str, Any] | None, pal: dict[str, str]) -> None:
    colors = {"L40S": pal["ref"], "H100": pal["mid"], "B200": pal["ink"]}
    lines = stored_graph2()
    for gpu, points in lines.items():
        _line(axis, points, colors[gpu], G2_LINE_CUEQ.format(gpu=gpu))
    top = max(8e3, 2.2 * max(p[1] for points in lines.values() for p in points))
    right_edge = max(700.0, *(points[-1][0] * 1.6 for points in lines.values()))
    if you is not None:
        x, y = you["structures"], you["rate"]
        top = max(top, y * 1.5)
        right_edge = max(right_edge, x * 1.6)
        axis.plot(
            [x], [y], marker="o", ms=12, mfc=pal["nv"], mec=pal["ring"], mew=2, zorder=8
        )
    else:
        axis.text(
            0.02,
            0.95,
            NO_GPU_NOTE,
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=pal["ink2"],
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks([t for t in (1, 8, 32, 128, 512, 2048, 8192) if t <= right_edge])
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_xlabel(G2_XLABEL)
    axis.set_ylabel(G2_YLABEL)
    axis.set_xlim(0.75, right_edge)
    axis.set_ylim(20, top)
    axis.set_yticks([30, 100, 300, 1000, 3000])
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axis.set_title(RIGHT_TITLE, loc="left")
    # Legend in the empty lower right, highest line first; then chips, the atom note and the
    # dot's label, each placed clear of the ones before.
    handles, names = axis.get_legend_handles_labels()
    order = sorted(range(len(handles)), key=lambda i: -max(handles[i].get_ydata()))
    axis.legend(
        [handles[i] for i in order], [names[i] for i in order], loc="lower right",
        frameon=False, fontsize=BODY_PT, handlelength=1.8, borderaxespad=0.3,
    )
    # The largest batch on the figure: an arrow with its total atom count, placed before the
    # chips so that they keep clear of its words and its arrow.
    x, y, *_, atoms = max((p for points in lines.values() for p in points), key=lambda p: p[0])
    _place_text(
        axis, ATOMS_NOTE.format(atoms=atoms), x, y, _ARROW_SPOTS, color=pal["ink"], fontsize=BODY_PT,
        arrowprops={"arrowstyle": "->", "color": pal["ink"], "lw": 1.0, "shrinkA": 2, "shrinkB": 5},
    )
    firsts = [points[0] for points in lines.values() if points[0][0] == 1]
    chips = [(max(firsts, key=lambda p: p[1])[:2], max(p[4] for p in firsts), pal["ink2"])] if firsts else []
    chips += [(points[-1][:2], points[-1][4], colors[gpu]) for gpu, points in lines.items()]
    for (x, y), gb, color in chips:
        _place_text(
            axis, MEMORY_CHIP.format(gb=gb), x, y, _CHIP_SPOTS, color=color, fontsize=SMALL_PT,
            bbox={"boxstyle": "round,pad=0.25,rounding_size=0.6", "fc": pal["ring"],
                  "ec": color, "lw": 1.0},
        )
    if you is not None:
        _place_text(axis, G2_YOU, you["structures"], you["rate"], _LABEL_SPOTS, color=pal["deep"])


# Offsets (points) and alignments to try for a memory chip, best first.
_CHIP_SPOTS = (
    ((0, -10), "center", "top"),
    ((0, 10), "center", "bottom"),
    ((0, -26), "center", "top"),
    ((-14, -22), "right", "top"),
    ((14, -22), "left", "top"),
    ((10, -8), "left", "top"),
    ((-10, -8), "right", "top"),
    ((10, 8), "left", "bottom"),
    ((-10, 8), "right", "bottom"),
    ((10, -24), "left", "top"),
    ((10, 24), "left", "bottom"),
    ((-10, -24), "right", "top"),
    ((-10, 24), "right", "bottom"),
)
# Offsets (points) and alignments to try for the atom-count note, best first.
_ARROW_SPOTS = (
    ((14, -48), "left", "top"),  # below the line, in the empty margin right of the pane
    ((-25, -40), "right", "top"),
    ((-60, -95), "right", "top"),
    ((-40, -110), "center", "top"),
    ((-90, -80), "right", "top"),
    ((-55, -28), "right", "top"),
    ((-20, -60), "right", "top"),
    ((-40, -42), "right", "top"),
    ((-45, 28), "right", "bottom"),
    ((-30, -55), "center", "top"),
    ((-55, -20), "right", "center"),
    ((-40, 42), "right", "bottom"),
)
# Offsets (points) and alignments to try for the dot's label, best first.
_LABEL_SPOTS = (
    ((-12, 10), "right", "bottom"),
    ((12, 10), "left", "bottom"),
    ((-12, -10), "right", "top"),
    ((12, -10), "left", "top"),
    ((-14, 0), "right", "center"),
    ((14, 0), "left", "center"),
)


def _extent(text: Any, renderer: Any) -> Any:
    """Return a label's screen box, including its rounded box when it has one."""

    from matplotlib.text import Text

    if hasattr(text, "update_positions"):
        text.update_positions(renderer)  # an annotation places its words at draw time
    box = Text.get_window_extent(text, renderer)  # the words only, not an annotation's arrow
    patch = text.get_bbox_patch()
    if patch is None:
        return box
    style = patch.get_boxstyle()
    pad = getattr(style, "pad", 0.3) * text.get_fontsize() * text.figure.dpi / 72
    return box.padded(pad + patch.get_linewidth() * text.figure.dpi / 72)


def _place_text(
    axis: Any, text: str, x: float, y: float, spots: Any, **style: Any
) -> None:
    """Put a label near (x, y) where it touches no other label and the fewest lines."""

    import numpy as np

    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    others = [_extent(t, renderer) for t in axis.texts]
    arrows = []  # points along each existing annotation arrow, in pixels
    for t in axis.texts:
        if getattr(t, "arrow_patch", None) is not None:
            box = _extent(t, renderer)
            head = np.array(axis.transData.transform(t.xy))
            tail = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
            arrows.append([tail + (head - tail) * k for k in np.linspace(0, 1, 40)])
    segments = []
    for line in axis.lines:
        if line.get_markersize() == 12:
            continue
        pts = axis.transData.transform(np.column_stack(line.get_data()))
        for p0, p1 in zip(pts[:-1], pts[1:], strict=True):
            segments.extend(p0 + (p1 - p0) * t for t in np.linspace(0, 1, 24))
    segments = np.array(segments) if segments else np.empty((0, 2))
    best = None
    for offset, ha, va in spots:
        candidate = axis.annotate(
            text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va,
            annotation_clip=False, **style,
        )
        box = _extent(candidate, renderer).expanded(1.04, 1.12)
        score = 100 * sum(box.overlaps(other) for other in others)
        score += 100 * sum(any(box.contains(*p) for p in arrow) for arrow in arrows)
        if "arrowprops" in style:  # the arrow must not run through another label
            tip = axis.transData.transform((x, y))
            start = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
            path = [start + (tip - start) * t for t in np.linspace(0, 1, 40)]
            score += 100 * sum(any(o.contains(*p) for p in path) for o in others)
        inside = axis.get_window_extent(renderer)
        if box.y0 < inside.y0 or box.y1 > inside.y1 or box.x0 < inside.x0:
            score += 50
        if len(segments):
            score += int(
                (
                    (segments[:, 0] >= box.x0)
                    & (segments[:, 0] <= box.x1)
                    & (segments[:, 1] >= box.y0)
                    & (segments[:, 1] <= box.y1)
                ).sum()
            )
        if best is None or score < best[0]:
            if best is not None:
                best[1].remove()
            best = (score, candidate)
        else:
            candidate.remove()


def _bottom_line(you: dict[str, Any] | None, labels: list[str] | None = None) -> str:
    """Return the line under the figure: the learner's GPU, then the cached CPU.

    With ``labels``, the CPU line is left out when a picked system has no
    stored CPU time, because the pane then draws no CPU bar.
    """

    parts = []
    if you is not None:
        name = gpu_label_name(you["gpu"])
        parts.append(
            GPU_LINE.format(gpu=name, memory=you["memory"]) if you["memory"] else f"{name}."
        )
    cpu = stored_cpu_singles()
    if labels is None or all(label in cpu["seconds"] for label in labels):
        parts.append(CPU_LINE.format(cpu=cpu["cpu"]))
    return "\n".join(parts)


def build_batching_figure(relaxation: Any, *, singles: dict[str, float] | None = None) -> Any:
    """Draw both panes, side by side, for one cell 9 run.

    ``singles`` holds the no-batch GPU seconds per system for ``STEPS`` steps;
    ``None`` leaves the no-batch GPU bar out.
    """

    common.register_course_fonts()
    pal = _PALETTE
    you = learner_run(relaxation)
    labels = list(relaxation["labels"])
    copies = int(relaxation["batch_multiplier"])
    with plt.rc_context():
        plt.style.use(str(_STYLE))
        figure, (left, right) = plt.subplots(1, 2, figsize=_FIGSIZE)
        figure.set_layout_engine("none")  # fixed axes boxes keep every figure the same size
        for axis, (x0, y0, w, h) in ((left, _LEFT_BOX), (right, _RIGHT_BOX)):
            axis.set_position([x0 / 920, y0 / 400, w / 920, h / 400])
        _draw_left(left, you, singles, labels, copies, pal)
        _draw_right(right, you, pal)
        # The run line sits under the left pane, left-aligned with its axis, below the x label.
        left.annotate(
            _bottom_line(you, labels),
            xy=(0, 0),
            xycoords=("axes fraction", left.xaxis.label),
            xytext=(0, -9),
            textcoords="offset points",
            ha="left",
            va="top",
            family="monospace",
            fontsize=SMALL_PT,
            linespacing=1.5,
            color=pal["ink2"],
            annotation_clip=False,
        )
    return figure


def batching_figure_png(relaxation: Any, *, singles: dict[str, float] | None = None) -> bytes:
    """Return the figure as a 2x PNG (1840 x 800 px), drawn and saved in the course style."""

    common.register_course_fonts()  # before the style names the course font
    with plt.style.context(str(_STYLE)):  # fonts resolve at save time
        figure = build_batching_figure(relaxation, singles=singles)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)
    return buffer.getvalue()


def plot_relaxation_time_perspective(relaxation: Any) -> None:
    """Time the picked systems without batching, then show both panes."""

    from IPython.display import HTML, display

    singles = time_without_batching(relaxation) if learner_run(relaxation) else None
    payload = b64encode(batching_figure_png(relaxation, singles=singles)).decode("ascii")
    structures = len(relaxation["labels"]) * int(relaxation["batch_multiplier"])
    alt = escape(
        f"{LEFT_TITLE}: {LEFT_XLABEL.format(structures=structures, steps=STEPS)}; "
        f"{RIGHT_TITLE}: {G2_YLABEL}",
        quote=True,
    )
    display(
        HTML(
            f'<img src="data:image/png;base64,{payload}" alt="{alt}" '
            'style="display:block;box-sizing:border-box;width:100%;max-width:920px;'
            'height:auto;margin:0">'
        )
    )
