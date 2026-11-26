# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

import IPython.display as ipy_display
from helpers import batching_figure as bf
from helpers import common, module00


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


LABELS = ["Ibuprofen", "Azoxystrobin", "Permethrin", "Capsaicin", "Fullerene (C₆₀)"]
SINGLES = {label: 0.65 for label in LABELS}


def _relaxation(
    gpu: str | None = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    total: float = 8.0,
    steps: int = 50,
    copies: int = 12,
    labels: list[str] = LABELS,
) -> dict:
    preview = {"device": "cuda:0" if gpu else "CPU"}
    if gpu:
        preview.update({"gpu_name": gpu, "total_s": total, "steps_run": steps, "gpu_memory_gb": 96})
    return {"labels": list(labels), "batch_multiplier": copies, "preview": preview}


def test_stored_csv_schemas_and_values() -> None:
    for path, columns in (
        (bf.CPU_SINGLES, bf.CPU_SINGLES_COLUMNS),
        (bf.GRAPH2_CUEQ, bf.GRAPH2_CUEQ_COLUMNS),
    ):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames) == columns
            assert list(reader)
    cpu = bf.stored_cpu_singles()
    assert set(LABELS) <= set(cpu["seconds"])
    assert len(cpu["seconds"]) == 12
    assert all(seconds > 0 for seconds in cpu["seconds"].values())
    lines = bf.stored_graph2()
    assert set(lines) == {"L40S", "H100", "B200"}
    for points in lines.values():
        xs = [p[0] for p in points]
        assert xs == sorted(set(xs))
        assert all(low <= rate <= high for _, rate, low, high, *_ in points)
        assert all(gb > 0 and atoms > 0 for *_, gb, atoms in points)


def test_learner_rate_is_structure_steps_per_second() -> None:
    run = bf.learner_run(_relaxation(total=8.0, steps=50))
    assert run["structures"] == 60
    assert run["rate"] == pytest.approx(60 * 50 / 8.0)
    assert bf.learner_run(_relaxation(gpu=None)) is None


@pytest.mark.parametrize("copies", [12, 4])
def test_left_pane_bars_follow_the_learners_workload(copies) -> None:
    structures = len(LABELS) * copies
    total, steps = 8.0, 150
    figure = bf.build_batching_figure(
        _relaxation(total=total, steps=steps, copies=copies), singles=SINGLES
    )
    left, right = figure.axes[:2]
    widths = [patch.get_width() for patch in left.patches]
    rate = structures * steps / total
    you = structures * bf.STEPS / rate
    gpu = sum(SINGLES.values()) * copies
    cpu = sum(bf.stored_cpu_singles()["seconds"][label] for label in LABELS) * copies
    assert widths == pytest.approx([you, gpu, cpu])
    assert left.get_xlabel() == bf.LEFT_XLABEL.format(structures=structures, steps=bf.STEPS)
    ticks = [t.get_text() for t in left.get_yticklabels()][::-1]
    assert ticks == [
        bf.BAR_YOU.format(structures=structures),
        bf.BAR_GPU,
        bf.BAR_CPU,
    ]
    factor = bf._factor_text(gpu / you)
    assert any(bf.SPEEDUP.format(factor=factor) in t.get_text() for t in left.texts)
    assert left.get_title(loc="left") == bf.LEFT_TITLE
    assert right.get_title(loc="left") == bf.RIGHT_TITLE
    dot = [line for line in right.lines if line.get_markersize() == 12][0]
    assert list(dot.get_xdata()) == [structures]
    assert dot.get_ydata()[0] == pytest.approx(rate)
    plt.close(figure)


def test_right_pane_has_cueq_lines_only() -> None:
    figure = bf.build_batching_figure(_relaxation(), singles=SINGLES)
    right = figure.axes[1]
    assert right.get_xscale() == "log" and right.get_yscale() == "log"
    assert right.get_xlabel() == bf.G2_XLABEL and right.get_ylabel() == bf.G2_YLABEL
    labels = {text.get_text() for text in right.texts}
    lines = bf.stored_graph2()
    chips = {bf.MEMORY_CHIP.format(gb=max(points[0][4] for points in lines.values()))}
    chips |= {bf.MEMORY_CHIP.format(gb=points[-1][4]) for points in lines.values()}
    chip_texts = [t for t in right.texts if t.get_text().endswith(" GB")]
    assert len(chip_texts) == 1 + len(lines)  # one shared chip at batch 1, one per line end
    legend = [t.get_text() for t in right.get_legend().get_texts()]
    assert legend == ["B200, cuEq + compile", "H100, cuEq + compile", "L40S, cuEq + compile"]
    assert labels == {bf.G2_YOU} | chips | {bf.ATOMS_NOTE.format(atoms=max(
        (p for points in bf.stored_graph2().values() for p in points), key=lambda p: p[0])[5])}
    assert not [l for l in right.lines if l.get_linestyle() == "--"]
    plt.close(figure)


def test_gpu_line_under_the_figure_uses_the_short_name() -> None:
    figure = bf.build_batching_figure(_relaxation(), singles=SINGLES)
    left = figure.axes[0]
    line = [t for t in left.texts if t.get_text().startswith("Your run:")]
    assert len(line) == 1
    assert line[0].get_text() == "Your run: RTX PRO 6000, 96 GB.\nCPU: AMD EPYC 7313P, cached."
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    box = line[0].get_window_extent(renderer)
    assert box.y1 < left.xaxis.label.get_window_extent(renderer).y0  # below the x label
    assert box.x0 >= 0 and box.y0 >= 0  # inside the figure
    assert abs(box.x0 - left.get_window_extent(renderer).x0) < 2  # left-aligned with the axis
    plt.close(figure)


def test_without_singles_the_gpu_bar_is_left_out() -> None:
    figure = bf.build_batching_figure(_relaxation(), singles=None)
    ticks = [t.get_text() for t in figure.axes[0].get_yticklabels()][::-1]
    assert ticks[0] == bf.BAR_YOU.format(structures=60) and len(ticks) == 2
    plt.close(figure)


def test_without_a_gpu_run_the_cpu_bar_and_a_note_remain() -> None:
    figure = bf.build_batching_figure(_relaxation(gpu=None))
    left, right = figure.axes[:2]
    assert len(left.patches) == 1
    assert bf.NO_GPU_NOTE in {t.get_text() for t in left.texts}
    assert bf.NO_GPU_NOTE in {t.get_text() for t in right.texts}
    assert not [l for l in right.lines if l.get_markersize() == 12]
    plt.close(figure)


def test_the_figure_is_light() -> None:
    figure = bf.build_batching_figure(_relaxation(), singles=SINGLES)
    assert matplotlib.colors.to_hex(figure.get_facecolor()) == "#ffffff"
    plt.close(figure)


def test_gpu_label_name_is_short() -> None:
    assert bf.gpu_label_name("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition") == "RTX PRO 6000"
    assert bf.gpu_label_name("NVIDIA RTX PRO 6000 Blackwell Server Edition") == "RTX PRO 6000"
    assert bf.gpu_label_name("NVIDIA H100 80GB HBM3") == "H100"
    assert bf.gpu_label_name("NVIDIA L40S") == "L40S"
    assert bf.gpu_label_name("NVIDIA RTX 6000 Ada Generation") == "RTX 6000"
    assert bf.gpu_label_name("NVIDIA A100-SXM4-40GB") == "A100"
    assert bf.gpu_label_name("NVIDIA A100 80GB PCIe") == "A100"


def test_a_large_batch_keeps_the_dot_inside_the_right_pane() -> None:
    figure = bf.build_batching_figure(
        _relaxation(total=10.0, steps=50, copies=400), singles=SINGLES
    )
    right = figure.axes[1]
    dot = [line for line in right.lines if line.get_markersize() == 12][0]
    assert dot.get_xdata()[0] == 2000
    assert right.get_xlim()[1] > 2000
    plt.close(figure)


def test_module00_routes_to_the_two_pane_plot(monkeypatch) -> None:
    from IPython.display import HTML

    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda obj: shown.append(obj))
    monkeypatch.setattr(bf, "time_without_batching", lambda relaxation: SINGLES)
    module00.plot_relaxation_time_perspective(_relaxation())
    assert shown and isinstance(shown[0], HTML)
    assert "data:image/png;base64," in shown[0].data


@pytest.mark.parametrize("rate", [250.0, 354.0, 470.0])
def test_your_batch_label_touches_no_other_label(rate) -> None:
    figure = bf.build_batching_figure(_relaxation(total=60 * 50 / rate, steps=50), singles=SINGLES)
    axis = figure.axes[1]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    label = [t for t in axis.texts if t.get_text() == bf.G2_YOU]
    assert len(label) == 1
    box = label[0].get_window_extent(renderer)
    for other in axis.texts:
        if other is not label[0]:
            assert not box.overlaps(other.get_window_extent(renderer)), other.get_text()
    plt.close(figure)
