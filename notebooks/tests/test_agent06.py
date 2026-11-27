# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent 06 helper: session, step contracts, result views and check messages. No GPU, no Toolkit run."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

import IPython.display as ipy_display
from helpers import agent06
from helpers import agent06_check as check
from helpers import agent06_views as views

PLAN = _NOTEBOOK_DIR / "alchemi-agent-06-hooked-in-at-every-step.plan.md"
NOTEBOOK = _NOTEBOOK_DIR / "alchemi-agent-06-hooked-in-at-every-step.ipynb"
DATA = _NOTEBOOK_DIR / "data" / "agent06"
SOLUTIONS = _NOTEBOOK_DIR.parent / "solutions" / "agent-06"

COUNTS = [4, 3, 5]  # atoms of the three test complexes
SPLITS = [2, 1, 2]  # their fragment A sizes
N_STEPS = 200
FRAMES = 20  # one saved frame every 10 steps


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _names(monkeypatch):
    monkeypatch.setattr(agent06, "COMPLEX_NAMES", ["pair one", "pair two", "pair three"])


@pytest.fixture
def shown(monkeypatch):
    """The objects the views pass to IPython's display()."""

    items = []
    monkeypatch.setattr(ipy_display, "display", lambda *objs, **kw: items.extend(objs))
    return items


# ---------------------------------------------------------------------------
# Synthetic complexes, stores and namespaces
# ---------------------------------------------------------------------------


def _complexes():
    """Three small pairs: fragment A around the origin, fragment B 3 Å along x."""

    rng = np.random.default_rng(0)
    out = []
    for count, split in zip(COUNTS, SPLITS):
        positions = rng.normal(scale=0.3, size=(count, 3))
        positions[split:, 0] += 3.0
        out.append(
            SimpleNamespace(
                num_nodes=count,
                positions=positions,
                atomic_masses=np.ones(count),
                atomic_numbers=np.full(count, 6),
            )
        )
    return out


class _Store:
    """A store of saved frames: ``trajectories[k]`` has shape (frames, atoms, 3)."""

    def __init__(self, trajectories):
        self.trajectories = trajectories

    def read(self):
        records = []
        for f in range(len(self.trajectories[0])):
            for k, trajectory in enumerate(self.trajectories):
                numbers = np.full(trajectory.shape[1], 6)
                records.append(SimpleNamespace(positions=trajectory[f], atomic_numbers=numbers))
        return SimpleNamespace(to_data_list=lambda: records)


def _trajectories(complexes, frames=FRAMES, frozen=None, stop=None, drift=None):
    """Random walks from the start. ``frozen``: mask per complex; ``stop``: frame from which complex k stands still;
    ``drift``: per complex, a shift of fragment B along x per frame."""

    rng = np.random.default_rng(1)
    out = []
    for k, data in enumerate(complexes):
        steps = rng.normal(scale=0.05, size=(frames, *data.positions.shape))
        if drift is not None:
            steps[:, SPLITS[k] :, 0] += drift[k]
        if frozen is not None:
            steps[:, frozen[k]] = 0.0
        if stop is not None and stop[k] is not None:
            steps[stop[k] + 1 :] = 0.0
        out.append(data.positions[None] + np.cumsum(steps, axis=0))
    return out


def _ns(**extra):
    complexes = _complexes()
    ns = {
        "complexes": complexes,
        "fragment_a_sizes": list(SPLITS),
        "run_settings": dict(temperature=1000.0, dt=1.0, friction=0.01, n_steps=N_STEPS),
    }
    ns.update(extra)
    return ns


def _log(steps=range(0, N_STEPS, 10), repeat=1):
    rows = [
        {"step": s, "graph_idx": g, "status": 0, "energy": -1.0 * g, "fmax": 0.1, "temperature": 900.0}
        for _ in range(repeat)
        for s in steps
        for g in range(3)
    ]
    return pd.DataFrame(rows)


class _OwnHook:
    stage = "AFTER_STEP"
    frequency = 10


def _built_in(name):
    return type(name, (), {"__module__": "nvalchemi.dynamics.hooks"})()


# ---------------------------------------------------------------------------
# The plan, the notebook and the contracts agree
# ---------------------------------------------------------------------------


def _plan_text() -> str:
    if not PLAN.exists():
        pytest.skip("plan file not present")
    return PLAN.read_text()


@pytest.fixture
def notebook_cells():
    """The cells of the notebook."""

    if not NOTEBOOK.exists():
        pytest.skip("notebook not present")
    return json.loads(NOTEBOOK.read_text())["cells"]


def test_every_plan_table_matches_its_contract():
    found = {}
    for row in re.finditer(r"\| output \| (.*?), in `agent06_work/(\w+)\.py` \|", _plan_text()):
        found[row.group(2)] = re.findall(r"`(\w+)`", row.group(1))
    assert set(found) == set(check.STEPS)
    for step, spec in check.STEPS.items():
        assert spec.file == f"{step}.py"
        assert found[step] == list(spec.names), step


def _code(cells) -> str:
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_notebook_loads_the_step_files_in_order(notebook_cells):
    code = _code(notebook_cells)
    assert re.findall(r"%load agent06_work/(\w+\.py)", code) == [s.file for s in check.STEPS.values()]


def test_every_helper_call_in_the_notebook_exists(notebook_cells):
    code = _code(notebook_cells)
    calls = set(re.findall(r"helpers\.(\w+)\(", code))
    assert calls == set(agent06.__all__)
    for name in calls:
        assert callable(getattr(agent06, name)), name


def test_notebook_calls_use_the_contract_names(notebook_cells):
    code = _code(notebook_cells)
    assert "helpers.plot_interaction(saved_step, interaction_energy, start_interaction_energy)" in code
    assert "helpers.show_run(plain_store, plain_log)" in code
    assert "helpers.plot_separation(saved_step, interaction_energy, separation, nearest_contact)" in code
    assert "helpers.plot_displacement(plain_store, frozen_store, frozen_mask)" in code
    assert "helpers.show_retired(stop_step, final_status, retire_store)" in code


def test_notebook_has_no_saved_outputs_and_numbered_cell_ids(notebook_cells):
    cells = notebook_cells
    assert all(re.fullmatch(r"a06-\d\d", c["id"]) for c in cells)
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in cells if c["cell_type"] == "code")


def test_setup_cell_of_the_check_is_the_notebooks(notebook_cells):
    setup = next("".join(c["source"]) for c in notebook_cells if c["cell_type"] == "code")
    code = [line.split("#")[0].strip() for line in setup.splitlines()]
    assert [line for line in code if line] == check.SETUP_CELL.splitlines()


def test_step_order_and_the_optional_step():
    order = list(check.STEPS)
    assert order == ["interaction", "separation_hook", "freeze", "retire"]
    for step, spec in check.STEPS.items():
        assert all(order.index(a) < order.index(step) for a in spec.after), step
    assert check.OPTIONAL_STEPS == ("freeze",)
    assert "freeze" not in check.STEPS["retire"].after


def test_sample_files_exist_for_every_step():
    if not SOLUTIONS.exists():
        pytest.skip("solutions folder not present")
    for spec in check.STEPS.values():
        assert (SOLUTIONS / spec.file).is_file(), spec.file


def test_default_complexes_are_saved_neutral_pairs():
    """Three saved structures; every pair and every fragment is neutral; the loader reads them."""

    structures, names, sizes = agent06.load_default_complexes()
    assert [len(s) for s in structures][:2] == [25, 14] and sizes[:2] == [13, 7]
    assert len(structures) == 3 and 12 <= len(structures[2]) <= 30 and 0 < sizes[2] < len(structures[2])
    assert names[:2] == ["phenol - N-methylacetamide", "propyne - methyl azide"]
    assert all(int(s.info["charge"]) == 0 and not s.pbc.any() for s in structures)
    assert len(agent06.START_INTERACTION_EV) == 3 and all(v < 0 for v in agent06.START_INTERACTION_EV)
    for file in agent06.DEFAULT_COMPLEX_FILES:
        header = (DATA / file).read_text().splitlines()[1]
        assert "charge=0 charge_a=0 charge_b=0" in header and "NCI Atlas" in header
    assert (DATA / "README.md").is_file()
    assert set("".join(s.get_chemical_formula() for s in structures)) <= set("HCNOFSl0123456789")


def test_no_helper_hands_out_a_looping_distance_function():
    assert not hasattr(agent06, "fragment_com_separations")
    assert "fragment_com_separations" not in agent06.__all__


# ---------------------------------------------------------------------------
# Session and work folder
# ---------------------------------------------------------------------------


def test_show_session_text(capsys):
    agent06.show_session()
    assert capsys.readouterr().out == (
        "Give these to your agent when you start the session:\n"
        "  notebook    notebooks/alchemi-agent-06-hooked-in-at-every-step.ipynb\n"
        "  helpers     from helpers import agent06 as helpers\n"
        "  work in     notebooks/agent06_work/\n"
        "  check with  python -m helpers.agent06 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py\n"
    )


def test_work_dir_is_notebooks_agent06_work(tmp_path, monkeypatch):
    assert agent06.WORK_DIR == _NOTEBOOK_DIR / "agent06_work"
    monkeypatch.setattr(agent06, "WORK_DIR", tmp_path / "agent06_work")
    assert agent06.make_work_dir().is_dir()


def test_stop_step_convention_is_in_the_docstring():
    assert "-1" in agent06.prepare_complexes.__doc__


# ---------------------------------------------------------------------------
# Saved frames
# ---------------------------------------------------------------------------


def test_store_frames_groups_by_complex():
    complexes = _complexes()
    frames = views.store_frames(_Store(_trajectories(complexes)))
    assert [p.shape for p in frames["positions"]] == [(FRAMES, n, 3) for n in COUNTS]
    assert [len(z) for z in frames["numbers"]] == COUNTS


def test_store_frames_rejects_a_partial_frame_and_a_non_store():
    complexes = _complexes()
    store = _Store(_trajectories(complexes))
    records = store.read().to_data_list()[:-1]
    broken = SimpleNamespace(read=lambda: SimpleNamespace(to_data_list=lambda: records))
    with pytest.raises(ValueError, match="multiple of 3"):
        views.store_frames(broken)
    with pytest.raises(TypeError, match="read"):
        views.store_frames([1, 2, 3])


def test_identical_frames_are_recognised():
    complexes = _complexes()
    same = [np.repeat(c.positions[None], FRAMES, axis=0) for c in complexes]
    assert views.frames_all_equal(same)
    assert not views.frames_all_equal(_trajectories(complexes))
    assert not views.frames_all_equal([c.positions[None] for c in complexes])  # one frame says nothing


def test_cpu_store_wrapper_is_installed_once(monkeypatch):
    sinks = pytest.importorskip("nvalchemi.dynamics.sinks")
    monkeypatch.setattr(sinks.HostMemory, "write", sinks.HostMemory.write)
    agent06._copy_saved_frames_on_cpu()
    first = sinks.HostMemory.write
    agent06._copy_saved_frames_on_cpu()
    assert sinks.HostMemory.write is first and first._agent06_copies


# ---------------------------------------------------------------------------
# Result views
# ---------------------------------------------------------------------------


def _texts(figure) -> list[str]:
    out = [t.get_text() for t in figure.texts]
    for axis in figure.axes:
        out += [axis.get_title(loc=loc) for loc in ("left", "center", "right")]
        out += [axis.get_xlabel(), axis.get_ylabel()]
        out += [t.get_text() for t in axis.texts]
        legend = axis.get_legend()
        out += [t.get_text() for t in legend.get_texts()] if legend else []
    return [t for t in out if t]


def _separation_inputs():
    step = np.arange(0, 200, 10)
    separation = np.linspace(4.0, 5.0, 20)[:, None] + np.array([0.0, 0.5, 1.0])
    nearest = np.full((20, 3), 2.5)
    nearest[7, 1] = 3.76
    nearest[3, 0] = 3.1
    return step, separation, nearest


def test_separation_has_one_panel_per_complex_and_marks_the_peak():
    figure = views.build_separation_figure(*_separation_inputs())
    assert len(figure.axes) == 3
    texts = _texts(figure)
    assert {"pair one", "pair two", "pair three"} <= set(texts)
    assert {"fragment separation", "nearest contact", "MD step", "Distance [Å]"} <= set(texts)
    assert {"3.10 Å", "3.76 Å", "2.50 Å"} <= set(texts)


def test_separation_data_gives_the_largest_nearest_contact():
    data = views.separation_data(*_separation_inputs())
    assert data["peak"].tolist() == [3.1, 3.76, 2.5]
    assert data["peak_step"].tolist()[:2] == [30, 70]


def test_separation_with_the_interaction_energy_has_two_rows_and_a_line():
    step, separation, nearest = _separation_inputs()
    energy = np.linspace(-10.0, -4.0, 20)[:, None] * np.ones(3)
    figure = views.build_separation_figure(step, separation, nearest, energy)
    assert len(figure.axes) == 6
    texts = _texts(figure)
    assert {"Interaction energy", "Distance [Å]", "3.76 Å", "pair two"} <= set(texts)
    assert views.separation_line(views.separation_data(step, separation, nearest)) == (
        "Largest nearest contact: 3.10, 3.76 and 2.50 Å."
    )
    with pytest.raises(ValueError, match="on the same saved steps"):
        views.build_separation_figure(step, separation, nearest, energy[:5])


def _interaction_inputs(unit="kcal/mol"):
    step = np.arange(0, 200, 10)
    start_ev = np.array([-0.5, -0.13, -0.4])
    series = (start_ev + np.linspace(0.0, 0.2, 20)[:, None]) * views.UNITS[unit]
    return step, series, start_ev * views.UNITS[unit], start_ev


@pytest.mark.parametrize("unit", ["kcal/mol", "eV", "kJ/mol", "Hartree"])
def test_interaction_reads_the_unit_and_draws_the_starting_value(unit, monkeypatch):
    step, series, start, start_ev = _interaction_inputs(unit)
    monkeypatch.setattr(agent06, "START_INTERACTION_EV", list(start_ev))
    assert views.read_unit(start, start_ev) == unit
    figure = views.build_interaction_figure(step, series, start)
    assert len(figure.axes) == 3
    texts = _texts(figure)
    assert {f"Interaction energy [{unit}]", "MD step", "along the run", "starting structure", "pair one"} <= set(texts)
    dashed = [line for line in figure.axes[1].lines if line.get_linestyle() == "--"]
    assert len(dashed) == 1 and dashed[0].get_ydata()[0] == pytest.approx(start[1])
    line = views.interaction_line(views.interaction_data(step, series, start))
    assert line.startswith("Mean along the run: ") and line.endswith(f" {unit}.") and "At the starting structure: " in line


def test_interaction_without_a_known_unit_names_none_and_rejects_bad_shapes(monkeypatch):
    step, series, start, start_ev = _interaction_inputs()
    monkeypatch.setattr(agent06, "START_INTERACTION_EV", list(start_ev))
    assert views.read_unit(start * 1.5, start_ev) is None
    assert views.read_unit(-start, start_ev) is None
    assert views.read_unit(start[:2], start_ev) is None
    figure = views.build_interaction_figure(step, series * 1.5, start * 1.5)
    assert "Interaction energy" in _texts(figure)
    assert views.interaction_line(views.interaction_data(step, series * 1.5, start * 1.5)).endswith("-13.8.")
    with pytest.raises(ValueError, match="saved steps, complexes"):
        views.build_interaction_figure(step, series.T, start)
    series[2, 1] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        views.build_interaction_figure(step, series, start)


def test_separation_rejects_wrong_shapes_and_bad_values():
    step, separation, nearest = _separation_inputs()
    with pytest.raises(ValueError, match="saved steps, complexes"):
        views.build_separation_figure(step, separation.T, nearest.T)
    nearest[0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        views.build_separation_figure(step, separation, nearest)


def test_displacement_separates_frozen_and_free_atoms():
    complexes = _complexes()
    masks = [np.arange(n) < s for n, s in zip(COUNTS, SPLITS)]
    mask = np.concatenate(masks)
    plain = _Store(_trajectories(complexes))
    frozen = _Store(_trajectories(complexes, frozen=masks))
    data = views.displacement_data(plain, frozen, mask)
    assert data["plain"].shape == data["frozen"].shape == (FRAMES, sum(COUNTS))
    assert data["frozen"][:, mask].max() == 0.0
    assert data["frozen"][:, ~mask].max() > 0.05 and data["plain"][:, mask].max() > 0.05
    figure = views.build_displacement_figure(plain, frozen, mask)
    texts = _texts(figure)
    assert {"plain run", "frozen run", "frozen atoms", "free atoms"} <= set(texts)
    assert "Distance from start [Å]" in texts


def test_displacement_rejects_a_mask_of_the_wrong_length():
    complexes = _complexes()
    store = _Store(_trajectories(complexes))
    with pytest.raises(ValueError, match="one entry per atom"):
        views.displacement_data(store, store, np.zeros(5, dtype=bool))


def test_displacement_says_so_when_the_store_kept_one_state(shown):
    complexes = _complexes()
    same = _Store([np.repeat(c.positions[None], FRAMES, axis=0) for c in complexes])
    views.plot_displacement(same, same, np.zeros(sum(COUNTS), dtype=bool))
    assert len(shown) == 1 and shown[0].data == views.LABELS["same_frames"]


def test_retired_rows_show_the_step_or_ran_to_the_end():
    rows = views.retired_rows(np.array([-1, 440, -1]), np.array([0, 1, 0]))
    assert rows == [["pair one", "ran to the end"], ["pair two", 440], ["pair three", "ran to the end"]]
    with pytest.raises(ValueError, match="one per complex"):
        views.retired_rows(np.array([-1, 440]), np.array([0, 1, 0]))


def test_retired_line_says_what_happened():
    assert views.retired_line(np.array([-1, 440, -1])) == "pair two was retired at step 440. The others ran to the end."
    assert views.retired_line(np.array([-1, -1, -1])) == "No complex was retired: every complex ran to the end."
    two = views.retired_line(np.array([120, 440, -1]))
    assert two.startswith("Unexpected: 2 complexes were retired (pair one, pair two).")
    assert views.retired_line(np.array([120, 440, 30])).startswith("Unexpected: 3 complexes were retired")


def test_log_table_takes_a_frame_or_a_path_and_names_the_complex(tmp_path):
    log = _log()
    table = views.log_table(log)
    assert list(table.columns)[:3] == ["step", "graph_idx", "complex"]
    assert table["complex"].tolist()[:3] == ["pair one", "pair two", "pair three"]
    assert "complex" not in log.columns  # the learner's table is left as it is
    path = tmp_path / "log.csv"
    log.to_csv(path, index=False)
    assert len(views.log_table(path)) == len(views.log_table(str(path))) == len(log)
    with pytest.raises(TypeError, match="DataFrame or the path"):
        views.log_table(42)


def test_log_table_prints_counters_as_whole_numbers():
    log = _log().astype({"step": float, "graph_idx": float, "status": float})  # as the log CSV reads back
    table = views.log_table(log)
    for column in ("step", "graph_idx", "status"):
        assert table[column].dtype == np.int64, column
    html = table.to_html(index=False)
    assert "<td>10</td>" in html and "10.0" not in html and "<td>0.0</td>" not in html
    odd = _log().astype({"step": float})
    odd.loc[0, "step"] = 0.5  # not a whole number: left as it is
    assert views.log_table(odd)["step"].dtype == float


def test_configure_tutorial_silences_the_aimnet2_cutoff_warning_in_kernel_order():
    """The setup cell runs configure_tutorial, then loads AIMNet2; the cutoff notice must not print.

    Toolkit's models.base puts a "once" rule for every UserWarning in front of earlier filters
    when it is first imported, so this runs in a fresh interpreter, in the notebook's order.
    """

    import subprocess

    pytest.importorskip("nvalchemi.models.aimnet2")
    code = (
        "import types, warnings\n"
        "from helpers import agent06 as helpers\n"
        "helpers.configure_tutorial()\n"
        "import torch\n"
        "from nvalchemi.models.aimnet2 import AIMNet2Wrapper\n"
        "t = torch.tensor(5.0, requires_grad=True)\n"
        "AIMNet2Wrapper._extract_cutoff(types.SimpleNamespace(aev=types.SimpleNamespace(rc_s=t, rc_v=t)))\n"
        "warnings.warn_explicit('Converting a tensor with requires_grad=True to a scalar may lead to x',\n"
        "                       UserWarning, 'aimnet2.py', 354, module='nvalchemi.models.aimnet2')\n"
        "warnings.warn_explicit('another notice', UserWarning, 'aimnet2.py', 1, module='nvalchemi.models.aimnet2')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=_NOTEBOOK_DIR, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "requires_grad=True to a scalar" not in result.stderr
    assert "another notice" in result.stderr  # the filter is narrow: other notices still print


def test_public_plotters_show_one_picture_with_alt_text(shown):
    complexes = _complexes()
    step, separation, nearest = _separation_inputs()
    energy = np.linspace(-10.0, -4.0, 20)[:, None] * np.ones(3)
    views.plot_interaction(step, energy, energy[0])
    views.plot_separation(step, energy, separation, nearest)
    views.plot_separation(step, separation, nearest)  # the three-argument form
    mask = np.concatenate([np.arange(n) < s for n, s in zip(COUNTS, SPLITS)])
    store = _Store(_trajectories(complexes))
    views.plot_displacement(store, store, mask)
    pictures = [item for item in shown if "<img " in getattr(item, "data", "")]
    lines = [item.data for item in shown if "<img " not in getattr(item, "data", "")]
    assert len(pictures) == 4 and len(lines) == 3
    alts = (
        views.LABELS["interaction"]["alt"],
        views.LABELS["separation"]["alt_energy"],
        views.LABELS["separation"]["alt"],
        views.LABELS["displacement"]["alt"],
    )
    for item, alt in zip(pictures, alts):
        assert item.data.count("<img ") == 1
        assert f'alt="{alt}"' in item.data
        assert "max-width:920px" in item.data
    assert lines[0].startswith("Mean along the run: ")
    assert lines[1] == lines[2] == "Largest nearest contact: 3.10, 3.76 and 2.50 Å."


def test_figures_use_the_course_font_and_width():
    figure = views.build_separation_figure(*_separation_inputs())
    assert figure.get_figwidth() == pytest.approx(9.2)
    label = figure.axes[0].xaxis.label
    with views._course_style():  # the family is resolved when the figure is drawn
        assert label.get_fontname() == "Open Sans"
    assert label.get_fontsize() == pytest.approx(9.36)
    assert figure.axes[0].title.get_fontweight() == "bold"


def test_show_run_and_show_retired_show_a_player_and_a_table(shown, monkeypatch):
    monkeypatch.setattr(views, "_player", lambda frames, names: ("player", names))
    complexes = _complexes()
    store = _Store(_trajectories(complexes))
    views.show_run(store, _log())
    assert shown[0] == ("player", ["pair one", "pair two", "pair three"])
    assert "<table" in shown[1].data and "temperature" in shown[1].data
    del shown[:]
    views.show_retired(np.array([-1, 60, -1]), np.array([0, 1, 0]), store)
    assert "Retired at step" in shown[0].data and "ran to the end" in shown[0].data
    assert shown[1].data == "pair two was retired at step 60. The others ran to the end."
    assert shown[2][0] == "player"


def test_every_word_on_a_view_is_in_labels():
    words = [v for group in views.LABELS.values() for v in (group.values() if isinstance(group, dict) else [group])]
    assert all(isinstance(w, str) and w for w in words)
    assert "·" not in "".join(words) and "—" not in "".join(words)


# ---------------------------------------------------------------------------
# Check messages: interaction (the first run: its frames, its log, the interaction energy)
# ---------------------------------------------------------------------------


def _plain_ns(**extra):
    ns = _ns()
    ns.update(plain_store=_Store(_trajectories(ns["complexes"])), plain_log=_log())
    ns.update(extra)
    return ns


def test_plain_store_and_log_pass_with_a_frame_or_a_path(tmp_path):
    assert check.check_plain_store_and_log(_plain_ns()) == []
    path = tmp_path / "plain_log.csv"
    _log().to_csv(path, index=False)
    assert check.check_plain_store_and_log(_plain_ns(plain_log=str(path))) == []


def test_plain_store_and_log_messages():
    ns = _plain_ns()
    lines = check.check_plain_store_and_log
    same = _Store([np.repeat(c.positions[None], FRAMES, axis=0) for c in ns["complexes"]])
    assert "every saved frame is the same" in lines(_plain_ns(plain_store=same))[0]
    two = _Store(_trajectories(ns["complexes"])[:2] + [np.zeros((FRAMES, 9, 3))])
    assert "[4, 3, 5]" in lines(_plain_ns(plain_store=two))[0]
    assert "second run added" in lines(_plain_ns(plain_log=_log(repeat=2)))[0]
    assert "no temperature column" in lines(_plain_ns(plain_log=_log().drop(columns="temperature")))[0]
    assert "ends at step 90" in lines(_plain_ns(plain_log=_log(steps=range(0, 100, 10))))[0]
    assert "no such file" in lines(_plain_ns(plain_log="missing.csv"))[0]
    assert "DataFrame or the path" in lines(_plain_ns(plain_log=[1]))[0]


START_EV = np.array([-0.5, -0.13, -0.4])  # the test model's interaction energies at the start, eV


def _fake_interaction_ev(ns, positions):
    """A stand-in for the composed model: the start value, weaker as fragment B drifts along x."""

    out = []
    for k, (data, split, xyz) in enumerate(zip(ns["complexes"], ns["fragment_a_sizes"], positions)):
        shift = xyz[:, split:, 0].mean(axis=1) - data.positions[split:, 0].mean()
        out.append(START_EV[k] + 0.05 * shift)
    return np.stack(out, axis=1)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(check, "_interaction_ev", _fake_interaction_ev)


def _interaction_ns(unit="kcal/mol", every=10, **extra):
    ns = _plain_ns()
    step = np.arange(0, N_STEPS, every)
    frames = store = ns["plain_store"].trajectories
    if every != 10:  # records on other steps than the saved frames: no frame to compare with
        frames = [np.repeat(t, 10 // every, axis=0) for t in store]
    ns.update(
        saved_step=step,
        interaction_energy=_fake_interaction_ev(ns, frames) * views.UNITS[unit],
        start_interaction_energy=START_EV * views.UNITS[unit],
    )
    ns.update(extra)
    return ns


@pytest.mark.parametrize("unit", ["kcal/mol", "eV", "kJ/mol", "Hartree"])
def test_interaction_passes_in_each_unit_and_with_any_route(model, unit):
    built_in = [_built_in("SnapshotHook"), _built_in("LoggingHook"), _built_in("NeighborListHook")]
    assert check.check_interaction(_interaction_ns(unit), {"hooks": [_OwnHook(), *built_in]}) == []
    assert check.check_interaction(_interaction_ns(unit), {"hooks": built_in}) == []  # no hook of its own
    assert check.check_interaction(_interaction_ns(unit, every=5), {}) == []


def test_interaction_wants_the_built_in_hooks_for_frames_and_log(model):
    lines = check.check_interaction(_interaction_ns(), {"hooks": [_OwnHook(), _built_in("SnapshotHook")]})
    assert lines == ["interaction.py registers no built-in LoggingHook to write the log"]


def test_interaction_messages(model):
    ns = _interaction_ns()
    lines = check.check_interaction(_interaction_ns(start_interaction_energy=ns["start_interaction_energy"] * 1.5), {})
    assert len(lines) == 1 and "start_interaction_energy is [-17.30, -4.50, -13.84]" in lines[0]
    assert "[-11.53, -3.00, -9.22] kcal/mol" in lines[0] and "eV" in lines[0] and "kJ/mol" in lines[0]
    assert "[-0.0184, -0.0048, -0.0147] Hartree" in lines[0]
    assert "minus the energies of its two fragments" in lines[0]
    flipped = check.check_interaction(_interaction_ns(start_interaction_energy=-ns["start_interaction_energy"]), {})
    assert "start_interaction_energy is" in flipped[0]
    lines = check.check_interaction(_interaction_ns(interaction_energy=ns["interaction_energy"][:, :2]), {})
    assert "interaction_energy has shape (20, 2)" in lines[0] and "(20, 3)" in lines[0]
    lines = check.check_interaction(_interaction_ns(start_interaction_energy=ns["start_interaction_energy"][:2]), {})
    assert "start_interaction_energy has shape (2,)" in lines[0]
    bad = ns["interaction_energy"].copy()
    bad[4, 2] = np.nan
    assert "not finite" in check.check_interaction(_interaction_ns(interaction_energy=bad), {})[0]
    flat = np.repeat(ns["start_interaction_energy"][None], 20, axis=0)
    assert "does not change during the run" in check.check_interaction(_interaction_ns(interaction_energy=flat), {})[0]
    assert "saved_step has shape" in check.check_interaction(_interaction_ns(saved_step=5), {})[0]
    assert "does not increase" in check.check_interaction(_interaction_ns(saved_step=ns["saved_step"][::-1]), {})[0]
    assert "no temperature column" in check.check_interaction(_interaction_ns(plain_log=_log().drop(columns="temperature")), {})[0]


def test_interaction_reports_results_off_the_expected_relation_as_unexpected(model):
    ns = _interaction_ns()
    energy = ns["interaction_energy"]
    late = check.check_interaction(_interaction_ns(interaction_energy=energy + np.array([2.0, 0.0, 0.0])), {})
    assert late and all(line.startswith("unexpected: ") for line in late)
    assert "the first record is" in late[0] and "begins at the starting value" in late[0]
    held = energy.copy()
    held[5:, 1] += 40.0  # fragments held at their starting energies: the series climbs
    lines = check.check_interaction(_interaction_ns(interaction_energy=held), {})
    assert all(line.startswith("unexpected: ") for line in lines)
    assert any("interaction_energy runs from" in line and "Both fragments are computed" in line for line in lines)
    coarse = _interaction_ns()
    coarse.update(saved_step=coarse["saved_step"][::2], interaction_energy=energy[::2])
    lines = check.check_interaction(coarse, {})
    assert len(lines) == 1 and lines[0].startswith("unexpected: saved_step advances by 20 steps; the step records every 10 steps.")
    off = energy.copy()
    off[-1, 2] += 1.0
    lines = check.check_interaction(_interaction_ns(interaction_energy=off), {})
    assert len(lines) == 1 and lines[0].startswith("unexpected: at step 190 interaction_energy is")
    assert "from the frame saved at that record" in lines[0]


# ---------------------------------------------------------------------------
# Check messages: separation_hook
# ---------------------------------------------------------------------------


def _separation_ns(every=10, **extra):
    ns = _ns()
    start_sep, start_near = check._start_distances(ns)
    step = np.arange(0, N_STEPS, every)
    grow = np.linspace(0.0, 1.0, len(step))[:, None]
    ns.update(
        saved_step=step,
        interaction_energy=np.zeros((len(step), 3)),
        separation=start_sep + grow,
        nearest_contact=start_near + grow,
    )
    ns.update(extra)
    return ns


def test_separation_hook_passes():
    assert check.check_separation_hook(_separation_ns(), {"hooks": [_OwnHook()]}) == []
    assert check.check_separation_hook(_separation_ns(every=5), {"hooks": [_OwnHook()]}) == []
    assert check.check_separation_hook(_separation_ns(every=20), {"hooks": [_OwnHook()]}) == []


def test_separation_hook_messages():
    ns = _separation_ns()
    lines = check.check_separation_hook(_separation_ns(separation=ns["separation"][:, :2]), {})
    assert "(20, 2)" in lines[0] and "(20, 3)" in lines[0]
    swapped = ns["separation"][:, [1, 0, 2]] + np.array([0.0, 1.0, 0.0])
    assert "in the order of complexes" in check.check_separation_hook(_separation_ns(separation=swapped), {})[0]
    in_nm = ns["nearest_contact"] / 10.0
    assert "nearest_contact starts at" in check.check_separation_hook(_separation_ns(nearest_contact=in_nm), {})[0]
    bad = ns["separation"].copy()
    bad[3, 1] = np.inf
    assert "not finite" in check.check_separation_hook(_separation_ns(separation=bad), {})[0]
    short = _separation_ns()
    short["run_settings"]["n_steps"] = 1000
    assert "ends at step 190" in check.check_separation_hook(short, {})[0]
    assert "shape (20, 3)" in check.check_separation_hook(_separation_ns(saved_step=ns["separation"]), {})[0]
    lines = check.check_separation_hook(_separation_ns(interaction_energy=np.zeros((40, 3))), {})
    assert lines == [
        "separation has 20 records and interaction_energy 40; record the distances on the steps of the interaction energy"
    ]
    flat = np.repeat(ns["separation"][:1], 20, axis=0)
    assert "does not change during the run" in check.check_separation_hook(_separation_ns(separation=flat), {})[0]


def test_separation_hook_wants_a_hook_of_its_own():
    lines = check.check_separation_hook(_separation_ns(), {"hooks": [_built_in("SnapshotHook")]})
    assert len(lines) == 1 and lines[0].startswith("separation_hook.py registers no hook of its own")


def test_separation_from_a_run_that_did_not_follow_the_first_is_unexpected():
    ns = _ns()
    walk = _trajectories(ns["complexes"])
    masses = [c.atomic_masses.reshape(-1, 1) for c in ns["complexes"]]
    pairs = [check._distances(p, m, s) for p, m, s in zip(walk, masses, SPLITS)]
    separation = np.stack([d[0] for d in pairs], axis=1)
    nearest = np.stack([d[1] for d in pairs], axis=1)
    same = _separation_ns(separation=separation, nearest_contact=nearest, plain_store=_Store(walk))
    assert check.check_separation_hook(same, {"hooks": [_OwnHook()]}) == []
    other = _separation_ns(plain_store=_Store(walk))
    lines = check.check_separation_hook(other, {"hooks": [_OwnHook()]})
    assert len(lines) == 1 and lines[0].startswith("unexpected: the nearest contacts differ by up to")
    assert "starts the same way" in lines[0]


# ---------------------------------------------------------------------------
# Check messages: freeze
# ---------------------------------------------------------------------------


def _freeze_ns(**extra):
    ns = _ns()
    masks = [np.arange(n) < s for n, s in zip(COUNTS, SPLITS)]
    ns.update(
        frozen_mask=np.concatenate(masks),
        frozen_store=_Store(_trajectories(ns["complexes"], frozen=masks)),
        plain_store=_Store(_trajectories(ns["complexes"])),
    )
    ns.update(extra)
    return ns


def test_freeze_passes():
    assert check.check_freeze(_freeze_ns(), {"hooks": [_built_in("FreezeAtomsHook")]}) == []


def test_freeze_messages():
    ns = _freeze_ns()
    assert "should be boolean" in check.check_freeze(_freeze_ns(frozen_mask=ns["frozen_mask"].astype(int)), {})[0]
    assert "one entry per atom" in check.check_freeze(_freeze_ns(frozen_mask=np.ones(3, dtype=bool)), {})[0]
    lines = check.check_freeze(_freeze_ns(frozen_mask=~ns["frozen_mask"]), {})
    assert "marks 7 atoms" in lines[0] and "5 atoms" in lines[0]
    moving = _Store(_trajectories(ns["complexes"]))
    assert "a frozen atom moved" in check.check_freeze(_freeze_ns(frozen_store=moving), {})[0]
    everything = [np.ones(n, dtype=bool) for n in COUNTS]
    still = _trajectories(ns["complexes"], frozen=everything)
    still[0][-1, -1, 0] += 1e-9  # not literally identical frames
    assert "free atoms did not move" in check.check_freeze(_freeze_ns(frozen_store=_Store(still)), {})[-1]
    fewer = _Store([t[:10] for t in ns["frozen_store"].trajectories])
    assert "holds 10 saved frames; the plain run saved 20" in check.check_freeze(_freeze_ns(frozen_store=fewer), {})[0]
    assert "no FreezeAtomsHook" in check.check_freeze(_freeze_ns(), {"hooks": [_OwnHook()]})[0]


def test_freeze_names_a_mask_given_per_complex():
    per_complex = [np.arange(n) < s for n, s in zip(COUNTS, SPLITS)]
    for mask in (per_complex, tuple(per_complex), [m.tolist() for m in per_complex]):
        lines = check.check_freeze(_freeze_ns(frozen_mask=mask), {})
        assert lines == ["frozen_mask holds 3 separate masks of 4, 3, 5 entries; one entry per atom of the batch gives one mask of shape (12,)"]
    torch = pytest.importorskip("torch")
    flat = np.concatenate(per_complex)
    for mask in (flat.tolist(), list(flat), list(torch.tensor(flat))):  # one mask over the batch, item by item
        assert check.check_freeze(_freeze_ns(frozen_mask=mask), {"hooks": [_built_in("FreezeAtomsHook")]}) == []
    torch_lines = check.check_freeze(_freeze_ns(frozen_mask=[torch.tensor(m) for m in per_complex]), {})
    assert torch_lines[0].startswith("frozen_mask holds 3 separate masks of 4, 3, 5 entries")


# ---------------------------------------------------------------------------
# Check messages: retire. A stopped complex against one that ran out of steps.
# ---------------------------------------------------------------------------

DRIFT = [0.0, 0.08, 0.0]  # complex 2 moves apart; the others stay together


def _retire_ns(stop=(None, 6, None), stop_step=(-1, 60, -1), status=(0, 1, 0), **extra):
    ns = _ns()
    ns.update(
        stop_step=np.array(stop_step),
        final_status=np.array(status),
        retire_store=_Store(_trajectories(ns["complexes"], stop=list(stop), drift=DRIFT)),
    )
    ns.update(extra)
    return ns


def test_retire_passes_when_one_complex_stops_and_the_others_run_on():
    assert check.check_retire(_retire_ns(), {}) == []


def test_retire_reads_the_frames_and_notes_a_status_that_does_not_tell():
    """On a fused stage every complex ends with the exit status; the frames tell who was stopped."""

    lines = check.check_retire(_retire_ns(status=(1, 1, 1)), {})
    assert len(lines) == 1 and lines[0].startswith("unexpected: final_status is [1, 1, 1]")


def test_retire_with_two_or_three_retired_is_unexpected_and_passes():
    for stop, steps, count in (((3, 6, None), (30, 60, -1), 2), ((3, 6, 4), (30, 60, 40), 3)):
        status = tuple(int(s >= 0) for s in steps)
        lines = check.check_retire(_retire_ns(stop=stop, stop_step=steps, status=status), {})
        notes = [line for line in lines if line.startswith("unexpected: ")]
        assert any(f"{count} complexes were retired" in line and "the step retires one" in line for line in notes)
        assert "complex 1 at step 30, complex 2 at step 60" in notes[0]


def test_retire_notes_a_run_that_did_not_follow_the_plot():
    ns = _retire_ns()
    frames = ns["retire_store"].trajectories
    masses = [c.atomic_masses.reshape(-1, 1) for c in ns["complexes"]]
    nearest = np.stack([check._distances(p, m, s)[1] for p, m, s in zip(frames, masses, SPLITS)], axis=1)
    assert check.check_retire(_retire_ns(nearest_contact=nearest), {}) == []
    lines = check.check_retire(_retire_ns(nearest_contact=nearest + 0.5), {})
    assert len(lines) == 1 and lines[0].startswith("unexpected: before the first stop the nearest contacts differ by up to 0.50 Å")
    assert check.check_retire(_retire_ns(nearest_contact=nearest[:7]), {}) == []  # other records: nothing to compare


def test_retire_tells_a_stop_from_the_end_of_the_run():
    lines = check.check_retire(_retire_ns(stop=(None, None, None), stop_step=(N_STEPS, N_STEPS, N_STEPS), status=(1, 1, 1)), {})
    assert len(lines) == 3 and all("ran out of steps" in line and "use -1" in line for line in lines)


def test_retire_messages():
    assert check.check_retire(_retire_ns(stop=(None, None, None), stop_step=(-1, -1, -1)), {}) == [
        "no complex was retired; lower your limit"
    ]
    lines = check.check_retire(_retire_ns(stop=(None, None, None)), {})
    assert "complex 2 is marked retired at step 60 and still moves" in lines[0]
    lines = check.check_retire(_retire_ns(stop_step=(-1, -1, -1)), {})
    assert "complex 2 stands still from saved frame 6 on" in lines[0] and "stop_step is -1" in lines[0]
    lines = check.check_retire(_retire_ns(stop_step=(-1, 150, -1)), {})
    assert "stop_step is 150" in lines[0] and "about step 60" in lines[0]
    assert "shape (2,)" in check.check_retire(_retire_ns(stop_step=(-1, 60)), {})[0]
    assert "whole MD steps" in check.check_retire(_retire_ns(stop_step=(-1.0, 60.0, -1.0)), {})[0]
    assert "final_status has shape (3, 1)" in check.check_retire(_retire_ns(status=((0,), (1,), (0,))), {})[0]


def test_retire_wants_one_limit_for_every_complex():
    """Complex 1 is retired early while complex 2 runs on and moves much further apart."""

    lines = check.check_retire(_retire_ns(stop=(2, None, None), stop_step=(20, -1, -1), status=(1, 0, 0)), {})
    assert "complex 2 ran to the end with a nearest contact of up to" in lines[0]
    assert "one limit holds for every complex" in lines[0]


def test_retire_reports_a_store_that_kept_one_state():
    ns = _retire_ns()
    same = _Store([np.repeat(c.positions[None], FRAMES, axis=0) for c in ns["complexes"]])
    lines = check.check_retire(_retire_ns(retire_store=same), {})
    assert lines == [f"retire_store: {check.SAME_LINE}"]


def test_still_from():
    walk = np.cumsum(np.ones((6, 2, 3)), axis=0)
    assert check.still_from(walk) is None
    stopped = walk.copy()
    stopped[3:] = stopped[3]
    assert check.still_from(stopped) == 3
    assert check.still_from(np.ones((6, 2, 3))) == 0


# ---------------------------------------------------------------------------
# Running a check: files, order, the second run
# ---------------------------------------------------------------------------


def _work(tmp_path, **files):
    work = tmp_path / "agent06_work"
    work.mkdir(parents=True)
    for name, text in files.items():
        (work / f"{name}.py").write_text(text)
    return work


def test_main_prints_usage_for_an_unknown_step(capsys):
    assert check.main(["check", "nonsense"]) == 2
    out = capsys.readouterr().out
    assert "usage: python -m helpers.agent06 check <step>" in out
    assert "interaction, separation_hook, freeze, retire" in out


def test_a_step_is_named_by_stem_file_or_path():
    assert check.resolve_step("retire") == check.resolve_step("retire.py") == "retire"
    assert check.resolve_step("agent06_work/interaction.py") == "interaction"


def test_missing_files_messages(tmp_path):
    work = _work(tmp_path)
    assert "the step file interaction.py is missing" in check.check_step("interaction", work, _ns())[0]
    assert "the earlier step file interaction.py is missing" in check.check_step("retire", work, _ns())[0]


def test_a_failing_file_is_reported_in_one_line(tmp_path):
    work = _work(tmp_path, interaction="raise RuntimeError('boom\\nsecond line')")
    assert check.check_step("interaction", work, _ns()) == ["interaction: running interaction.py failed: RuntimeError: boom"]
    work2 = _work(tmp_path / "b", interaction="x = 1", separation_hook="y = 2", retire="z = 3")
    (work2 / "interaction.py").write_text("raise ValueError('early')")
    assert check.check_step("retire", work2, _ns()) == [
        "retire: an earlier file failed: interaction.py: ValueError: early"
    ]


def test_missing_names_and_undefined_names(tmp_path):
    work = _work(tmp_path, interaction="saved_step = interaction_energy = start_interaction_energy = plain_store = 1")
    assert check.check_step("interaction", work, _ns()) == ["interaction: plain_log is missing; the step asks for it"]
    work2 = _work(tmp_path / "b", interaction="plain_store = save_every")
    assert "uses save_every, which no earlier cell defines" in check.check_step("interaction", work2, _ns())[0]


def test_a_script_name_such_as_file_is_named_and_removed(tmp_path):
    """``__file__`` exists in a script and not in a notebook cell: the line says to remove it."""

    text = "from pathlib import Path\nhere = Path(__file__).parent\n"
    work = _work(tmp_path, interaction=text)
    assert check.check_step("interaction", work, _ns()) == [
        "interaction: interaction.py uses __file__, which does not exist in a notebook cell; remove that line"
    ]


FIRST_NAMES = "saved_step = interaction_energy = start_interaction_energy = plain_store = plain_log = object()\n"


def test_a_later_file_that_assigns_an_earlier_name_again_passes_with_one_unexpected_line(tmp_path):
    """A file that re-runs the first step inside itself passes; one note names what it assigned again."""

    again = "interaction_energy = object()\nplain_store = object()\nseparation = nearest_contact = 1\n"
    ns = _ns()
    work = _work(tmp_path, interaction=FIRST_NAMES, separation_hook=again)
    check_lines = []
    original = check.STEPS["separation_hook"]
    check.STEPS["separation_hook"] = check.Step(original.file, original.names, original.after, ())
    try:
        check_lines = check.check_step("separation_hook", work, ns)
    finally:
        check.STEPS["separation_hook"] = original
    assert check_lines == [
        "separation_hook: OK (2 names)",
        "separation_hook: unexpected: separation_hook.py assigned interaction_energy, plain_store again; "
        "the cells above now show this run",
    ]
    reads = "my_energy = interaction_energy\nseparation = nearest_contact = 1\n"
    lines = check.check_step("separation_hook", _work(tmp_path / "b", interaction=FIRST_NAMES, separation_hook=reads), _ns())
    assert not any("again" in line for line in lines)


def test_the_optional_file_is_not_held_to_the_earlier_names(tmp_path):
    files = dict(
        interaction=FIRST_NAMES,
        separation_hook="separation = nearest_contact = object()\n",
        freeze="plain_store = object()\nfrozen_mask = frozen_store = 1\n",
        retire="stop_step = final_status = retire_store = 1\n",
    )
    lines = check.check_step("retire", _work(tmp_path, **files), _ns())
    assert not any("freeze.py" in line for line in lines)


def test_the_line_under_the_interaction_plot_in_hartree_has_four_decimals(monkeypatch):
    step, series, start, start_ev = _interaction_inputs("Hartree")
    monkeypatch.setattr(agent06, "START_INTERACTION_EV", list(start_ev))
    line = views.interaction_line(views.interaction_data(step, series, start))
    assert line.endswith("At the starting structure: -0.0184, -0.0048 and -0.0147 Hartree.")


def test_the_step_file_runs_twice_and_the_second_failure_is_named(tmp_path):
    text = "runs = globals().get('runs', 0) + 1\nassert runs < 2, 'already ran'\n"
    work = _work(tmp_path, interaction=text)
    line = check.check_step("interaction", work, _ns())[0]
    assert line.startswith("interaction: running interaction.py a second time failed: AssertionError")


def test_the_optional_step_runs_only_if_it_is_there_and_never_blocks(tmp_path):
    files = dict(interaction="order = ['plain']", separation_hook="order.append('separation')")
    retire = "order.append('retire')\nstop_step = final_status = retire_store = None\n"
    ns = _ns()
    check.check_step("retire", _work(tmp_path, retire=retire, **files), ns)
    assert ns["order"] == ["plain", "separation", "retire", "retire"]
    ns = _ns()
    work = _work(tmp_path / "b", retire=retire, freeze="order.append('freeze')\nraise RuntimeError('mine')", **files)
    check.check_step("retire", work, ns)
    assert ns["order"] == ["plain", "separation", "freeze", "retire", "retire"]


def test_files_run_in_the_notebook_folder_and_pass_with_real_arrays(tmp_path):
    """A step file writes to ``agent06_work/...`` relative to the notebook folder."""

    text = (
        "import numpy as np\n"
        "open('agent06_work/out.txt', 'w').write('x')\n"
        "stop_step = np.array([-1, 60, -1]); final_status = np.array([0, 1, 0]); retire_store = make_store()\n"
    )
    ns = _ns()
    store = _Store(_trajectories(ns["complexes"], stop=[None, 6, None], drift=DRIFT))
    ns["make_store"] = lambda: store
    work = _work(tmp_path, interaction="", separation_hook="", retire=text)
    assert check.check_step("retire", work, ns) == ["retire: OK (3 names)"]
    assert (work / "out.txt").exists()


def test_an_unexpected_line_follows_ok_and_the_check_exits_zero(tmp_path, capsys, monkeypatch):
    text = (
        "import numpy as np\n"
        "stop_step = np.array([30, 60, -1]); final_status = np.array([1, 1, 0]); retire_store = make_store()\n"
    )
    ns = _ns()
    store = _Store(_trajectories(ns["complexes"], stop=[3, 6, None], drift=[0.08, 0.08, 0.0]))
    ns["make_store"] = lambda: store
    work = _work(tmp_path, interaction="", separation_hook="", retire=text)
    lines = check.check_step("retire", work, ns)
    assert lines[0] == "retire: OK (3 names)"
    assert len(lines) == 2 and lines[1].startswith("retire: unexpected: 2 complexes were retired")
    monkeypatch.setattr(check, "check_step", lambda step, work_dir=None: lines)
    assert check.main(["check", "retire"]) == 0
    assert capsys.readouterr().out.splitlines() == lines
    monkeypatch.setattr(check, "check_step", lambda step, work_dir=None: ["retire: no complex was retired; lower your limit"])
    assert check.main(["check", "retire"]) == 1
