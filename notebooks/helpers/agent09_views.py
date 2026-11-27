# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Figures, tables and every line shown to the learner by Agent 09, Small-batch distilled (liquid SiC).

All words a learner reads from the helper are in this file: the session
lines, the labels of figures and tables, and the one line under a result.
They are DRAFTS until the educator approves them on the board. Every figure
has a ``build_*`` function that returns a Matplotlib figure (used by the
tests); ``show`` draws it as one picture, 920 px wide, in the course figure
style (light only).
"""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path

import numpy as np

__all__ = [
    "LABELS",
    "SESSION_TEXT",
    "build_force_figure",
    "build_loss_figure",
    "build_parity_figure",
    "build_speed_figure",
    "build_structure_figure",
    "errors_line",
    "html_table",
    "show",
    "speed_line",
    "stability_line",
    "structure_line",
    "student_line",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STYLE = _PROJECT_ROOT / "shared" / "alchemi-figure.mplstyle"
DISPLAY_PX = 920
# Colours of the course figure style: green main result, grey baseline, ink reference.
GREEN, GREY, LIGHT_GREY, INK = "#76B900", "#6F6F6F", "#B8B8B8", "#1A1A1A"

SESSION_TEXT = (
    "Give these to your agent when you start the session:\n"
    "  notebook    {notebook}\n"
    "  helpers     from helpers import agent09 as helpers\n"
    "  work in     notebooks/agent09_work/\n"
    "  check with  python -m helpers.agent09 check <step>\n"
    "  each file   runs in a notebook cell (%load), after the earlier files\n"
    "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
)

# The frame types of the held-out density, in the order of the tables.
FRAME_TYPES = {
    "melt_aimd": "Melting run",
    "eqlMelt_aimd": "Equilibrated liquid",
    "quench_aimd": "Quench and amorphous",
    "amorph_aimd": "Quench and amorphous",
}
ALL_FRAMES = "All held-out frames"

# Axis labels are "Quantity [unit]".
LABELS = {
    "errors": {
        "xlabel": "DFT force, one component [eV/Å]",
        "ylabel": "Model's force, one component [eV/Å]",
        "published": "MACE-MPA-0",
        "finetuned": "fine-tuned",
        "student": "student",
        "frames_kind": "Frames",
        "count": "Count",
        # Two header rows: the quantity over the models' names.
        "energy": "Energy error [meV/atom]",
        "force": "Force error [eV/Å]",
        "alt": (
            "Parity plot of forces on the held-out frames: the model's force component against the DFT force "
            "component, one point per atom and direction, with the diagonal of a perfect match."
        ),
    },
    "loss": {
        "xlabel": "Training step",
        "ylabel": "Training loss",
        "alt": "Line plot of the training loss against the training step.",
    },
    "store": {
        "count": "{structures} structures, {atoms} atoms",
        "headers": ["Field", "Belongs to", "Shape", "Type"],
        "levels": {"atom": "atoms", "edge": "pairs", "system": "structure"},
    },
    "stability": {
        "headers": [
            "Density [g/cm³]", "Copies", "Target temperature [K]", "Mean temperature [K]",
            "Shortest distance [Å]", "Mean squared displacement at the end [Å²]", "Still liquid",
        ],
        "of": "{done} of {total}",
        "not_judged": "not judged",
    },
    "structure": {
        "xlabel": "Distance r [Å]",
        "ylabel": "g(r)",
        "teacher": "teacher's sampling run",
        "student": "student's run",
        "total": "All pairs",
        "pairs": {"SiSi": "Si-Si", "SiC": "Si-C", "CC": "C-C"},
        "headers": ["Pairs", "Jensen-Shannon divergence", "First peak, teacher [Å]", "First peak, student [Å]"],
        "alt": (
            "Radial distribution functions g(r) of the teacher's sampling run and of the student's run, "
            "one panel for all pairs and one for each pair of elements."
        ),
    },
    "student": {
        "xlabel": "Teacher's force on the atom [eV/Å]",
        "ylabel": "Student's force on the atom [eV/Å]",
        "frames": "Frames",
        "mean_force": "Mean force, teacher [eV/Å]",
        "mean_error": "Mean force error, student [eV/Å]",
        "energy_gap": "Energy difference, root mean square [meV/atom]",
        "alt": (
            "Scatter plot: the length of the student's force on each atom of its own frames against the length "
            "of the teacher's force, with the diagonal of a perfect match."
        ),
    },
    "speed": {
        "xlabel": "MD steps per second",
        "teacher": "teacher (fine-tuned MACE-MPA-0)",
        "student": "student",
        "alt": "Two bars: MD steps per second of the same run with the teacher and with the student.",
    },
}

# The one line under a result. A result off the expected relation is "unexpected", never wrong.
ERRORS_LEAST_FALL = 0.02  # a fall under 2% reads as no change
FINETUNE_FELL = (
    "On all {frames} held-out frames the energy error fell from {e0:.1f} to {e1:.1f} meV/atom and the force "
    "error from {f0:.3f} to {f1:.3f} eV/Å."
)
FINETUNE_UNEXPECTED = (
    "Unexpected: on all {frames} held-out frames the energy error went from {e0:.1f} to {e1:.1f} meV/atom and "
    "the force error from {f0:.3f} to {f1:.3f} eV/Å; at least one did not fall."
)
FINETUNE_TYPE_ROSE = " Unexpected: on the {kind} frames the {what} error rose."
STUDENT_DFT = (
    "Against DFT the student's force error is {fs:.3f} eV/Å, between the fine-tuned model's {ff:.3f} and the "
    "published model's {fp:.3f}; its energy error is {es:.1f} meV/atom."
)
STUDENT_DFT_UNEXPECTED = (
    "Unexpected: against DFT the student's force error is {fs:.3f} eV/Å, above the published model's {fp:.3f}; "
    "its energy error is {es:.1f} meV/atom."
)
STUDENT_CLOSE = (
    "On its own frames the student's forces differ from the teacher's by {error:.2f} eV/Å on average, {share:.0f}% "
    "of the teacher's mean force; its energies differ by {energy:.0f} meV/atom (root mean square)."
)
STUDENT_UNEXPECTED = (
    "Unexpected: on its own frames the student's forces differ from the teacher's by {error:.2f} eV/Å on "
    "average, {share:.0f}% of the teacher's mean force; its energies differ by {energy:.0f} meV/atom."
)
STUDENT_FAR_SHARE = 0.5  # above this share of the teacher's mean force the comparison is marked unexpected
STABILITY_ALL = "{done} of {total} copies are still liquid at the end of the run; StabilityMonitor found no divergence."
STABILITY_SOME = (
    "Unexpected: {done} of {total} copies are still liquid at the end of the run ({why})."
)
STABILITY_SHORT = (
    "The run has {steps} steps; whether a copy is still liquid is judged from {least} steps on. "
    "StabilityMonitor found {divergence}."
)
DIVERGENCE_NONE = "no divergence"
DIVERGENCE_AT = "a divergence at step {step}"
STRUCTURE_FOLLOWS = (
    "The student's structure follows the teacher's: the divergence over all pairs is {js:.1e} (at most "
    "{bound:.1e}), and every first peak of a pair of elements is within {tol:.1f} Å of the teacher's."
)
STRUCTURE_UNEXPECTED = "Unexpected: the student's structure departs from the teacher's ({why})."
STRUCTURE_SHORT = (
    "Too few frames to judge the structure: {student} from the student and {teacher} from the teacher; "
    "at least {least} each."
)
SPEED_LINE = (
    "On the same {copies} cells of 216 atoms the student runs {ratio:.0f} times as many MD steps per second as "
    "the teacher ({student:.1f} against {teacher:.2f})."
)
SPEED_UNEXPECTED = (
    "Unexpected: on the same {copies} cells the student runs {student:.2f} MD steps per second, the teacher "
    "{teacher:.2f}."
)
# How show_speed timed it: on a GPU three runs of 50 steps per model, on a CPU one run of 3 steps.
SPEED_RUNS = " Timed after one warm-up: the median of {runs} runs of {steps} steps per model."
SPEED_RUN_ONE = " Timed after one warm-up: one run of {steps} steps per model."


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


def errors_line(rows: dict, models: list[str]) -> str:
    """The line under the error table.

    ``rows`` maps a frame kind to ``{"frames": n, model: {"energy": meV/atom, "force": eV/Å}}``;
    ``models`` are the model keys in table order ("published", "finetuned", "student").
    """

    total = rows[ALL_FRAMES]
    if models[:2] == ["published", "finetuned"] and "student" not in models:
        before, after = total["published"], total["finetuned"]
        fell = after["force"] < (1.0 - ERRORS_LEAST_FALL) * before["force"] and round(after["energy"], 1) < round(
            before["energy"], 1
        )
        text = (FINETUNE_FELL if fell else FINETUNE_UNEXPECTED).format(
            frames=total["frames"], e0=before["energy"], e1=after["energy"], f0=before["force"], f1=after["force"]
        )
        for kind, row in rows.items():
            if kind == ALL_FRAMES:
                continue
            for what, rose in (
                ("energy", round(row["finetuned"]["energy"], 1) > round(row["published"]["energy"], 1)),
                ("force", round(row["finetuned"]["force"], 3) > round(row["published"]["force"], 3)),
            ):
                if rose:
                    text += FINETUNE_TYPE_ROSE.format(kind=kind.lower(), what=what)
        return text
    if "student" in models:
        student, finetuned, published = total["student"], total.get("finetuned"), total["published"]
        good = student["force"] <= published["force"]
        template = STUDENT_DFT if good and finetuned is not None else STUDENT_DFT_UNEXPECTED
        return template.format(
            fs=student["force"], ff=(finetuned or published)["force"], fp=published["force"], es=student["energy"]
        )
    return ""


def student_line(error: float, teacher_force: float, energy_gap: float) -> str:
    """The line under the student-against-teacher table."""

    share = error / teacher_force if teacher_force > 0 else float("inf")
    text = STUDENT_UNEXPECTED if share > STUDENT_FAR_SHARE else STUDENT_CLOSE
    return text.format(error=error, share=100.0 * share, energy=energy_gap)


def stability_line(verdict: dict) -> str:
    """The line under the stability table; ``verdict`` comes from ``helpers.agent09.still_liquid``."""

    divergence = (
        DIVERGENCE_NONE if verdict["first_divergence_step"] is None
        else DIVERGENCE_AT.format(step=verdict["first_divergence_step"])
    )
    if not verdict["judged"]:
        return STABILITY_SHORT.format(steps=verdict["steps"], least=verdict["least_steps"], divergence=divergence)
    done, total = verdict["n_liquid"], verdict["copies"]
    if done == total and verdict["first_divergence_step"] is None:
        return STABILITY_ALL.format(done=done, total=total, steps=verdict["steps"])
    return STABILITY_SOME.format(done=done, total=total, steps=verdict["steps"], why="; ".join(verdict["reasons"]))


def structure_line(verdict: dict) -> str:
    """The line under the structure figure; ``verdict`` comes from ``helpers.agent09.follows_teacher``."""

    if not verdict["judged"]:
        return STRUCTURE_SHORT.format(
            student=verdict["student_frames"], teacher=verdict["teacher_frames"], least=verdict["least_frames"]
        )
    if verdict["follows"]:
        return STRUCTURE_FOLLOWS.format(js=verdict["js"], bound=verdict["bound"], tol=verdict["peak_tolerance"])
    return STRUCTURE_UNEXPECTED.format(why="; ".join(verdict["reasons"]))


def speed_line(result: dict) -> str:
    """The line under the speed figure."""

    teacher, student = result["teacher_steps_per_s"], result["student_steps_per_s"]
    if student > teacher:
        text = SPEED_LINE.format(copies=result["copies"], ratio=student / teacher, student=student, teacher=teacher)
    else:
        text = SPEED_UNEXPECTED.format(copies=result["copies"], student=student, teacher=teacher)
    runs = len(result.get("teacher_runs_s") or ())
    if runs and "steps" in result:
        text += (SPEED_RUNS if runs > 1 else SPEED_RUN_ONE).format(runs=runs, steps=result["steps"])
    return text


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _course_style():
    """The course figure style (920 px wide) for everything drawn inside."""

    import matplotlib.pyplot as plt

    from .common import register_course_fonts

    register_course_fonts()  # before the style names the course font
    with plt.style.context(str(_STYLE)):
        yield


def _in_course_style(build):
    @functools.wraps(build)
    def wrapper(*args, **kwargs):
        with _course_style():
            return build(*args, **kwargs)

    return wrapper


def _new_figure(*args, **kwargs):
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


def show(figure, alt: str) -> None:
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


def html_table(headers: list[str], rows: list[list], groups: list[tuple[str, int]] | None = None) -> str:
    """A plain left-aligned HTML table, the form of the other agentic notebooks.

    ``groups`` puts a header row above ``headers``: the first ``len(headers) - sum(spans)`` headers span both
    rows, then each ``(label, span)`` stands over the last columns in order (short headers that do not wrap).
    """

    from html import escape

    th = "<th style='text-align:left'{}>{}</th>"
    if not groups:
        head = "<tr>" + "".join(th.format("", escape(str(h))) for h in headers) + "</tr>"
    else:
        single = len(headers) - sum(span for _, span in groups)
        top = "".join(th.format(" rowspan='2'", escape(str(h))) for h in headers[:single])
        top += "".join(th.format(f" colspan='{span}'", escape(str(label))) for label, span in groups)
        head = f"<tr>{top}</tr><tr>" + "".join(th.format("", escape(str(h))) for h in headers[single:]) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td style='text-align:left'>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


@_in_course_style
def build_loss_figure(steps, losses):
    """The training loss against the step, on a log axis when every loss is positive."""

    steps = np.asarray(steps, dtype=float).reshape(-1)
    losses = np.asarray(losses, dtype=float).reshape(-1)
    if steps.size != losses.size or losses.size == 0:
        raise ValueError(
            f"show_loss: the loss history should hold one loss per step; found {losses.size} "
            f"losses and {steps.size} steps"
        )
    figure, axis = _new_figure(figsize=(9.2, 3.6))
    axis.plot(steps, losses, color=GREEN)
    if np.isfinite(losses).all() and (losses > 0).all():
        axis.set_yscale("log")
    axis.set_xlim(left=0)
    axis.set_xlabel(LABELS["loss"]["xlabel"])
    axis.set_ylabel(LABELS["loss"]["ylabel"])
    return figure


@_in_course_style
def build_parity_figure(series):
    """One point per force component: the model's value against the DFT value.

    ``series`` is a list of ``(name, reference, model)``; the first is drawn in
    grey (the published model), the second in green (the fine-tuned model).
    """

    labels = LABELS["errors"]
    cleaned = []
    for name, reference, model in series:
        reference = np.asarray(reference, dtype=float).reshape(-1)
        model = np.asarray(model, dtype=float).reshape(-1)
        if reference.size != model.size:
            raise ValueError(
                f"show_errors: {model.size} model values against {reference.size} reference values; "
                "both should have one value per atom and direction"
            )
        finite = np.isfinite(reference) & np.isfinite(model)
        cleaned.append((name, reference[finite], model[finite]))
    values = np.concatenate([np.concatenate([r, m]) for _, r, m in cleaned]) if cleaned else np.zeros(1)
    top = float(np.abs(values).max()) * 1.05 if values.size and np.abs(values).max() > 0 else 1.0
    figure, axis = _new_figure(figsize=(9.2, 4.6))
    axis.plot([-top, top], [-top, top], color=INK, linewidth=1.0, zorder=1)
    for index, (name, reference, model) in enumerate(cleaned):
        axis.scatter(
            reference, model, s=3, color=GREY if index == 0 else GREEN, alpha=0.3, linewidths=0,
            rasterized=True, zorder=2 + index, label=name,
        )
    axis.set_xlim(-top, top)
    axis.set_ylim(-top, top)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, axis="both")
    axis.set_xlabel(labels["xlabel"])
    axis.set_ylabel(labels["ylabel"])
    if len(cleaned) > 1:
        legend = axis.legend(loc="upper left", frameon=False, markerscale=4)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
    return figure


@_in_course_style
def build_force_figure(teacher_force, student_force):
    """One point per atom and frame: the student's force magnitude against the teacher's."""

    teacher = np.asarray(teacher_force, dtype=float).reshape(-1)
    student = np.asarray(student_force, dtype=float).reshape(-1)
    if teacher.size != student.size:
        raise ValueError(
            f"show_student_vs_teacher: {student.size} student forces against {teacher.size} "
            "teacher forces; both should have one value per atom and frame"
        )
    finite = np.isfinite(teacher) & np.isfinite(student)
    top = float(max(teacher[finite].max(), student[finite].max())) * 1.05 if finite.any() else 1.0
    figure, axis = _new_figure(figsize=(9.2, 4.6))
    axis.plot([0, top], [0, top], color=INK, linewidth=1.0, zorder=1)
    axis.scatter(
        teacher[finite], student[finite], s=3, color=GREEN, alpha=0.25, linewidths=0, rasterized=True, zorder=2,
    )
    axis.set_xlim(0, top)
    axis.set_ylim(0, top)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, axis="both")
    axis.set_xlabel(LABELS["student"]["xlabel"])
    axis.set_ylabel(LABELS["student"]["ylabel"])
    return figure


@_in_course_style
def build_structure_figure(curves):
    """g(r) of the teacher (grey) and the student (green), one panel per curve.

    ``curves`` is a list of ``(title, r, g_teacher, g_student)``.
    """

    labels = LABELS["structure"]
    count = len(curves)
    if count == 0:
        raise ValueError("show_structure: no curves to draw")
    columns = 2 if count > 1 else 1
    rows = -(-count // columns)
    figure, axes = _new_figure(rows, columns, figsize=(9.2, 3.0 * rows), squeeze=False)
    for index, (title, r, teacher, student) in enumerate(curves):
        axis = axes[index // columns][index % columns]
        axis.plot(r, teacher, color=GREY, linewidth=2.2, label=labels["teacher"])
        axis.plot(r, student, color=GREEN, linewidth=1.4, label=labels["student"])
        axis.set_title(title, loc="left")
        axis.set_xlim(0, float(np.max(r)))
        axis.set_ylim(bottom=0)
        axis.set_xlabel(labels["xlabel"])
        axis.set_ylabel(labels["ylabel"])
        if index == 0:
            axis.legend(loc="upper right", frameon=False)
    for index in range(count, rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    return figure


@_in_course_style
def build_speed_figure(teacher_rate, student_rate):
    """Two bars: MD steps per second with the teacher and with the student."""

    labels = LABELS["speed"]
    values = [float(teacher_rate), float(student_rate)]
    figure, axis = _new_figure(figsize=(9.2, 2.2))
    bars = axis.barh([labels["teacher"], labels["student"]], values, color=[GREY, GREEN], height=0.55)
    axis.invert_yaxis()
    axis.bar_label(bars, labels=[f"{v:.3g}" for v in values], padding=6)
    axis.set_xlim(0, max(values) * 1.15 if max(values) > 0 else 1.0)
    axis.grid(False)
    axis.set_xlabel(labels["xlabel"])
    return figure
