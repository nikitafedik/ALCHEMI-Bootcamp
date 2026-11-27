# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent 03 helper: loaders, step contracts, plotters and check messages. No GPU needed."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

import IPython.display as ipy_display
from helpers import _agent03_loaders, agent03, agent03_check

PLAN = _NOTEBOOK_DIR / "alchemi-agent-03-ready-made.plan.md"
# Draft exercises that are not in the plan or the notebook (the timestep scan left 03).
DRAFT_STEPS = {"timestep_scan"}
PLANNED_STEPS = {k: v for k, v in agent03_check.STEPS.items() if k not in DRAFT_STEPS}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def shown(monkeypatch):
    """The objects the plotters pass to IPython's display()."""

    items = []
    monkeypatch.setattr(ipy_display, "display", lambda *objs, **kw: items.extend(objs))
    return items


def _texts(figure) -> list[str]:
    out = [t.get_text() for t in figure.texts]
    for axis in figure.axes:
        out += [axis.get_title(loc=loc) for loc in ("left", "center", "right")]
        out += [axis.get_xlabel(), axis.get_ylabel()]
        out += [t.get_text() for t in axis.texts]
        out += [t.get_text() for t in axis.get_xticklabels()]
    return [t for t in out if t]


# ---------------------------------------------------------------------------
# The plan: names, files and helper calls
# ---------------------------------------------------------------------------


def _plan_text() -> str:
    if not PLAN.exists():
        pytest.skip("plan file not present")
    return PLAN.read_text()


def test_every_plan_table_matches_its_contract():
    found: dict[str, list[str]] = {}
    for row in re.finditer(r"\| output\s*\| (.*?), in `agent03_work/(\w+)\.py` \|", _plan_text()):
        found[row.group(2)] = re.findall(r"`(\w+)`", row.group(1))
    assert {f"{s}.py" for s in found} == {spec.file for spec in PLANNED_STEPS.values()}
    for step, spec in PLANNED_STEPS.items():
        assert found[spec.file.removesuffix(".py")] == list(spec.names), step


def test_load_cells_follow_the_step_order():
    loads = re.findall(r"%load agent03_work/(\w+\.py)", _plan_text())
    assert loads == [spec.file for spec in PLANNED_STEPS.values()]


def test_every_helper_call_in_the_plan_exists():
    calls = set(re.findall(r"helpers\.(\w+)\(", _plan_text()))
    assert {
        "configure_tutorial",
        "make_work_dir",
        "show_session",
        "show_crystal_grid",
        "plot_force_histograms",
        "plot_batch_speedup",
        "plot_start_temperature",
        "plot_energy_drift",
        "show_speed_table",
    } <= calls
    for name in calls:
        assert callable(getattr(agent03, name)), name


def test_plan_calls_use_these_arguments():
    plan = _plan_text()
    assert "helpers.show_crystal_grid(crystal_batch, start_max_force)" in plan
    assert "helpers.plot_force_histograms(start_max_force, relaxed_max_force, force_threshold)" in plan
    assert "helpers.plot_batch_speedup(batch_seconds, single_seconds, len(start_max_force))" in plan
    assert "helpers.plot_start_temperature(start_temperature)" in plan
    assert "helpers.plot_energy_drift(time_fs, crystal_energy, md_batch)" in plan
    assert "helpers.show_speed_table(speed_table, setting_roles)" in plan


def test_setup_cell_runs_the_planned_lines():
    lines = agent03_check.SETUP_CELL.splitlines()
    assert lines[0] == "from helpers import agent03 as helpers"
    assert lines[1] == "helpers.configure_tutorial()"
    assert lines[2] == "work_dir = helpers.make_work_dir()"
    assert lines[-1] == "helpers.show_session()"


def test_after_lists_follow_step_order():
    order = list(agent03_check.STEPS)
    assert order == [
        "relax_build", "relax_run", "md_build", "md_run", "timestep_scan", "temperature_ladder", "speed_settings"
    ]
    for step, spec in agent03_check.STEPS.items():
        assert all(order.index(a) < order.index(step) for a in spec.after), step
    assert agent03_check.STEPS["speed_settings"].after == ["relax_build"]
    assert agent03_check.STEPS["timestep_scan"].after == ["relax_build"]
    assert agent03_check.STEPS["temperature_ladder"].after == ["relax_build"]


def test_contract_units_and_shapes():
    names = {n: s for step in agent03_check.STEPS.values() for n, s in step.names.items()}
    assert names["start_max_force"].shape == ("n_structures",)
    assert names["start_max_force"].unit == "eV/Å"
    assert names["single_seconds"].shape == (3,)
    assert names["crystal_energy"].shape == ("n_saved", "n_structures")
    assert names["crystal_energy"].unit == "eV"
    assert names["time_fs"].increasing and names["time_fs"].unit == "fs"
    assert names["start_temperature"].unit == "K" and names["start_temperature"].positive


# ---------------------------------------------------------------------------
# Session and work folder
# ---------------------------------------------------------------------------


def test_show_session_text(capsys):
    agent03.show_session()
    assert capsys.readouterr().out == (
        "Give these to your agent when you start the session:\n"
        "  notebook    notebooks/alchemi-agent-03-ready-made.ipynb\n"
        "  helpers     from helpers import agent03 as helpers\n"
        "  work in     notebooks/agent03_work/\n"
        "  check with  python -m helpers.agent03 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n"
        "              what it reports; <step> is the file name without .py\n"
    )


def test_every_step_name_is_its_file_name_without_py():
    for step, spec in agent03_check.STEPS.items():
        assert spec.file == f"{step}.py"


def test_make_work_dir_creates_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(agent03, "WORK_DIR", tmp_path / "agent03_work")
    path = agent03.make_work_dir()
    assert path.is_dir() and path.name == "agent03_work"


def test_default_work_dir_is_notebooks_agent03_work():
    assert agent03.WORK_DIR == _NOTEBOOK_DIR / "agent03_work"


# ---------------------------------------------------------------------------
# Crystal loader
# ---------------------------------------------------------------------------


def test_285_crystals_are_built():
    frames = _agent03_loaders.build_crystals()
    assert len(frames) == 285
    assert len({f.positions.tobytes() for f in frames}) == 285  # each has its own rattle seed


def test_load_crystals_shapes():
    frames = agent03.load_crystals(200)
    assert len(frames) == 200
    sizes = [len(f) for f in frames]
    assert 8 <= min(sizes) and max(sizes) <= 128
    assert len(set(sizes)) > 10  # varied size
    for f in frames:
        assert f.pbc.all()
        assert abs(np.linalg.det(np.asarray(f.cell))) > 1e-6
        assert "charge" not in f.info


def test_load_crystals_default_is_200():
    assert len(agent03.load_crystals()) == 200


def test_load_crystals_is_deterministic():
    a, b = agent03.load_crystals(200), agent03.load_crystals(200)
    assert [x.info["label"] for x in a] == [x.info["label"] for x in b]
    assert all(np.array_equal(x.positions, y.positions) for x, y in zip(a, b))


def test_load_crystals_picks_evenly_from_the_built_set():
    built = _agent03_loaders.build_crystals()
    picked = agent03.load_crystals(200)
    where = {f.positions.tobytes(): i for i, f in enumerate(built)}
    index = [where[f.positions.tobytes()] for f in picked]
    assert index[0] == 0 and index[-1] == 284
    assert index == sorted(set(index))  # no repeats
    assert max(np.diff(index)) <= 2 and min(np.diff(index)) >= 1
    for i, f in zip(index, picked):
        assert np.array_equal(f.positions, built[i].positions)


def test_load_crystals_other_counts():
    assert len(agent03.load_crystals(5)) == 5
    assert len(agent03.load_crystals(1)) == 1
    assert len(agent03.load_crystals(1000)) == 285
    with pytest.raises(ValueError):
        agent03.load_crystals(0)


def test_rattle_is_0_1_angstrom():
    frame = _agent03_loaders.build_crystals()[0]
    base_label, cell = _agent03_loaders._base_cells()[0]
    reps = tuple(int(x) for x in frame.info["label"].split("-")[-1].split("x"))
    ideal = cell.repeat(reps)
    assert len(ideal) == len(frame)
    shift = (frame.positions - ideal.positions).ravel()
    assert 0.07 < shift.std() < 0.13
    assert np.array_equal(frame.cell, ideal.cell)  # cell unchanged


# ---------------------------------------------------------------------------
# Plotters on small fake inputs
# ---------------------------------------------------------------------------

FORCE_START = np.array([0.4, 2.0, 9.5, 0.12, 20.0, 1.1, 3.3, 0.9])
FORCE_RELAXED = np.array([0.049, 0.031, 0.0499, 0.02, 0.045, 0.011, 0.038, 0.04])


def test_grid_has_one_box_per_crystal_and_a_force_colorbar():
    figure = agent03.build_crystal_grid_figure(SimpleNamespace(num_graphs=8), FORCE_START)
    assert any("Largest force [eV/Å]" == t for t in _texts(figure))
    mesh = figure.axes[0].collections[0]
    assert mesh.get_array().count() == 8  # masked cells are not boxes


def test_grid_with_200_crystals_uses_rows_of_25():
    force = np.linspace(0.1, 20, 200)
    figure = agent03.build_crystal_grid_figure(SimpleNamespace(num_graphs=200), force)
    assert figure.axes[0].get_xlim() == (0, 25) and figure.axes[0].get_ylim() == (8, 0)


def test_grid_reddest_box_is_the_largest_force():
    figure = agent03.build_crystal_grid_figure(SimpleNamespace(num_graphs=8), FORCE_START)
    mesh = figure.axes[0].collections[0]
    rgba = mesh.to_rgba(mesh.get_array())
    red = rgba.reshape(-1, 4)[:, 0] - rgba.reshape(-1, 4)[:, 1]
    assert int(np.argmax(red)) == int(np.argmax(FORCE_START))


def test_grid_rejects_a_mismatch():
    with pytest.raises(ValueError, match="holds 3 crystals"):
        agent03.build_crystal_grid_figure(SimpleNamespace(num_graphs=3), FORCE_START)


def test_histogram_left_is_relaxed_and_right_is_the_start():
    figure = agent03.build_force_histograms_figure(FORCE_START, FORCE_RELAXED, 0.05)
    left, right = figure.axes
    assert "after relaxation" in left.get_xlabel() and "start" in right.get_xlabel()
    assert left.get_ylabel() == "Number of crystals"
    data = agent03.histogram_data(FORCE_START, FORCE_RELAXED, 0.05)
    assert data["relaxed"].sum() == 8 and data["start"].sum() == 8


def test_histogram_threshold_is_a_bin_edge_and_a_dashed_line():
    figure = agent03.build_force_histograms_figure(FORCE_START, FORCE_RELAXED, 0.05)
    data = agent03.histogram_data(FORCE_START, FORCE_RELAXED, 0.05)
    assert np.isclose(data["edges"], 0.05).any()
    for axis in figure.axes:
        lines = [ln for ln in axis.lines if ln.get_linestyle() == "--"]
        assert len(lines) == 1 and lines[0].get_xdata()[0] == pytest.approx(0.05)
    assert "threshold" in _texts(figure)


def test_histogram_relaxed_all_below_threshold():
    data = agent03.histogram_data(FORCE_START, FORCE_RELAXED, 0.05)
    assert data["relaxed_above"] == 0
    edge = int(np.argmin(np.abs(data["edges"] - 0.05)))
    assert data["relaxed"][:edge].sum() == 8 and data["relaxed"][edge:].sum() == 0
    assert data["start"][edge:].sum() == 8  # every start value is above the line


def test_histogram_counts_relaxed_values_above_the_line():
    relaxed = FORCE_RELAXED.copy()
    relaxed[0] = 0.0502
    assert agent03.histogram_data(FORCE_START, relaxed, 0.05)["relaxed_above"] == 1


def test_histogram_survives_zero_forces_and_tensors():
    torch = pytest.importorskip("torch")
    figure = agent03.build_force_histograms_figure(
        torch.tensor([0.0, 1.0, 5.0]), torch.tensor([0.0, 0.01, 0.04]), torch.tensor(0.05)
    )
    assert len(figure.axes) == 2


def test_speedup_projects_the_one_by_one_bar():
    data = agent03.speedup_data(2.0, [10.0, 20.0, 30.0], 200)
    assert data == {"batched": 2.0, "one_by_one": 20.0 * 200, "n_timed": 3}
    figure = agent03.build_batch_speedup_figure(2.0, [10.0, 20.0, 30.0], 200)
    heights = [p.get_height() for p in figure.axes[0].patches]
    assert heights == pytest.approx([2.0, 4000.0])
    figure.canvas.draw()
    texts = _texts(figure)
    assert "Time [s]" in texts
    assert "batched" in texts
    assert "one by one\n(projected from 3 structures)" in texts
    assert "2.00 s" in texts and "4000 s" in texts


def test_speedup_takes_tensors():
    torch = pytest.importorskip("torch")
    data = agent03.speedup_data(torch.tensor(1.5), torch.tensor([1.0, 2.0, 3.0]), 10)
    assert data["batched"] == 1.5 and data["one_by_one"] == 20.0


def test_start_temperature_axis_starts_at_zero():
    temperature = np.full(200, 300.0) + np.linspace(-2e-4, 2e-4, 200)
    figure = agent03.build_start_temperature_figure(temperature)
    axis = figure.axes[0]
    assert axis.get_ylim()[0] == 0 and axis.get_ylim()[1] > 300
    assert len(axis.lines[0].get_xdata()) == 200
    assert axis.get_ylabel() == "Starting temperature [K]"


def _scan_inputs(n_saved=11, dts=(0.5, 1.0, 2.0, 4.0)):
    """A fake timestep scan: replica_dt, saved_step, total_energy (n_saved, n_replicas)."""

    step = np.arange(n_saved) * 10
    dt = np.array(dts)
    drift = 1e-4 * (dt[None, :] ** 2) * np.arange(n_saved)[:, None]
    return dt, step, -100.0 + drift


def _ladder_inputs(n_saved=31, temps=(300.0, 840.0, 1380.0, 1920.0, 2460.0, 3000.0)):
    """A fake ladder: time_fs, replica_temperature, replica_energy (n_saved, n_temperatures) eV/atom.

    Each copy's energy rises from the lattice value to 1.5 kT per atom above it, so hotter copies
    sit above cooler ones.
    """

    time = np.arange(n_saved) * 20.0
    t = np.array(temps)
    rise = 1.5 * 8.617e-5 * t[None, :] * (1.0 - np.exp(-np.arange(n_saved) / 5.0))[:, None]
    return time, t, -5.4 + rise


def _drift_inputs(scale=1e-3, n_saved=11, n=4):
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 1, (n_saved, n)).cumsum(axis=0)
    energy = -100.0 + scale * (steps - steps[0])  # eV, row 0 is the start
    atoms = SimpleNamespace(num_nodes_per_graph=np.array([8, 16, 32, 64]))
    return np.arange(n_saved, dtype=float), energy, atoms


def test_energy_drift_is_per_atom_from_the_start_in_mev():
    time, energy, atoms = _drift_inputs()
    data = agent03.drift_data(time, energy, atoms)
    assert data["drift"].shape == (11, 4)
    assert np.all(data["drift"][0] == 0)
    expected = (energy[:, 2] - energy[0, 2]) / 32 * 1000.0
    assert data["drift"][:, 2] == pytest.approx(expected)


def test_energy_drift_draws_one_thin_line_per_crystal():
    figure = agent03.build_energy_drift_figure(*_drift_inputs())
    axis = figure.axes[0]
    thin = [ln for ln in axis.lines if ln.get_linewidth() < 1.0 and len(ln.get_xdata()) == 11]
    assert len(thin) == 4  # one line per crystal (the zero line has two points)
    assert axis.get_xlabel() == "Time [fs]"
    assert axis.get_ylabel() == "Change in total energy [meV/atom]"


def test_energy_drift_axis_has_a_floor_and_grows_with_the_data():
    small = agent03.build_energy_drift_figure(*_drift_inputs(scale=1e-6)).axes[0].get_ylim()
    assert small == (-agent03.DRIFT_AXIS_FLOOR_MEV, agent03.DRIFT_AXIS_FLOOR_MEV)
    big_time, big_energy, atoms = _drift_inputs(scale=5.0)
    drift = np.abs(agent03.drift_data(big_time, big_energy, atoms)["drift"]).max()
    assert drift > agent03.DRIFT_AXIS_FLOOR_MEV
    low, high = agent03.build_energy_drift_figure(big_time, big_energy, atoms).axes[0].get_ylim()
    assert high == pytest.approx(drift * 1.15) and low == -high


def test_energy_drift_rejects_a_wrong_shape():
    time, energy, atoms = _drift_inputs()
    with pytest.raises(ValueError, match="shape"):
        agent03.build_energy_drift_figure(time, energy[:, :3], atoms)


@pytest.mark.parametrize(
    "call",
    [
        lambda: agent03.show_crystal_grid(SimpleNamespace(num_graphs=8), FORCE_START),
        lambda: agent03.plot_force_histograms(FORCE_START, FORCE_RELAXED, 0.05),
        lambda: agent03.plot_batch_speedup(2.0, [10.0, 20.0, 30.0], 8),
        lambda: agent03.plot_start_temperature(np.full(8, 300.0)),
        lambda: agent03.plot_energy_drift(*_drift_inputs()),
        lambda: agent03.plot_timestep_scan(*_scan_inputs()),
        lambda: agent03.plot_temperature_ladder(*_ladder_inputs()),
    ],
)
def test_public_plotters_show_one_picture_with_alt_text(call, shown):
    call()
    assert len(shown) == 1
    html = shown[0].data
    assert html.count("<img") == 1 and 'alt="' in html and "alt=\"\"" not in html
    assert "max-width:920px" in html
    assert plt.get_fignums() == []  # the figure is closed


def test_figures_use_the_course_font_and_size():
    figure = agent03.build_start_temperature_figure(np.full(8, 300.0))
    assert tuple(figure.get_size_inches()) == (9.2, 4.0)
    with agent03._course_style():  # the family is resolved when the figure is drawn
        assert figure.axes[0].xaxis.label.get_fontname() == "Open Sans"
        assert figure.axes[0].yaxis.label.get_fontname() == "Open Sans"


# ---------------------------------------------------------------------------
# Speed table
# ---------------------------------------------------------------------------


def _proposal_table(**changes):
    """The contract's table: four cases (neither, cuEquivariance only, compile only, both)."""

    pd = pytest.importorskip("pandas")
    columns = {
        "cueq": [False, True, False, True],
        "compile": [False, False, True, True],
        "first_call_s": [16.6, 1.8, 15.7, 2.3],
        "repeat_s": [16.3, 1.8, 14.7, 1.3],
    }
    columns.update(changes)
    return pd.DataFrame(columns)


ROLES = {"cueq": "GPU kernels for the tensor products", "compile": "torch.compile"}


def test_speed_table_rows_show_four_cases_and_both_times():
    headers, rows = agent03.speed_table_rows(_proposal_table())
    assert headers == ["cuEquivariance", "Compile", "First call [s]", "Repeat [s]"]
    assert rows[0] == ["off", "off", "16.6", "16.3"]
    assert rows[3] == ["on", "on", "2.3", "1.3"]


def test_speed_table_view_names_missing_columns():
    table = _proposal_table().drop(columns=["repeat_s"])
    with pytest.raises(ValueError, match="missing repeat_s"):
        agent03.speed_table_rows(table)


def test_speed_table_view_keeps_extra_columns_out():
    table = _proposal_table(note=["", "", "", "x"], speedup=[1.0, 9.0, 1.1, 12.5])
    headers, rows = agent03.speed_table_rows(table)
    assert len(headers) == 4 and all(len(r) == 4 for r in rows)


def test_show_speed_table_shows_roles_and_escapes_text(shown):
    roles = {"cueq": "GPU kernels <fast>", "compile": "torch.compile & more"}
    agent03.show_speed_table(_proposal_table(), roles)
    html = shown[0].data
    assert "GPU kernels &lt;fast&gt;" in html and "torch.compile &amp; more" in html
    assert html.count("<table>") == 2 and "<th style='text-align:left'>Role</th>" in html


def test_roles_keys_are_matched_without_case():
    html = agent03.speed_table_html(_proposal_table(), {"CUEQ": "a", "Compile": "b"})
    assert html.count("<td style='text-align:left'>cuEquivariance</td>") == 1
    assert "<td style='text-align:left'>Compile</td>" in html


# ---------------------------------------------------------------------------
# The step check: messages (see notes/builder-03/check_messages.md)
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_nvalchemi():
    pytest.importorskip("nvalchemi")
    return pytest.importorskip("torch")


def _check(step, name, ns, n_crystals=3, **extra):
    spec = agent03_check.STEPS[step].names[name]
    syms = {"n_structures": n_crystals, **extra}
    return agent03_check._check_name(step, name, spec, ns, syms)


def test_missing_name(torch_nvalchemi):
    assert _check("relax_build", "mace_model", {}) == [
        "relax_build: mace_model is missing; the step file should create it"
    ]


def test_wrong_kinds(torch_nvalchemi):
    torch = torch_nvalchemi
    assert _check("relax_build", "crystal_batch", {"crystal_batch": [1, 2]}) == [
        "relax_build: crystal_batch should be a Batch; found list of length 2"
    ]
    assert _check("relax_build", "mace_model", {"mace_model": torch.nn.Linear(1, 1)}) == [
        "relax_build: mace_model should be a Toolkit model; found Linear"
    ]
    assert _check("relax_build", "start_max_force", {"start_max_force": "abc"}) == [
        "relax_build: start_max_force should be numbers in eV/Å; found str"
    ]
    assert _check("relax_run", "force_threshold", {"force_threshold": torch.ones(2)}) == [
        "relax_run: force_threshold should be one number in eV/Å; found Tensor of shape (2,)"
    ]


def test_wrong_shape_and_bad_values(torch_nvalchemi):
    torch = torch_nvalchemi
    assert _check("relax_build", "start_max_force", {"start_max_force": torch.ones(2)}) == [
        "relax_build: start_max_force should have shape (n_structures,) = (3,) in eV/Å; "
        "found shape (2,)"
    ]
    assert _check("relax_build", "start_max_force", {"start_max_force": torch.tensor([1.0, float("nan"), 2.0])}) == [
        "relax_build: start_max_force should be finite everywhere; found 1 of 3 values that are not (nan or inf)"
    ]
    assert _check("relax_run", "batch_seconds", {"batch_seconds": 0.0}) == [
        "relax_run: batch_seconds should be above 0 everywhere; found 1 of 1 values at or below 0 (smallest 0)"
    ]
    assert _check("md_run", "time_fs", {"time_fs": torch.tensor([0.0, 1.0, 1.0])}, n_saved=3) == [
        "md_run: time_fs should increase from one saved row to the next; found rows that do not"
    ]
    assert _check(
        "md_run", "crystal_energy", {"crystal_energy": torch.zeros(3, 2)}, n_saved=3
    ) == [
        "md_run: crystal_energy should have shape (n_saved, n_structures) = (3, 3) in eV; "
        "found shape (3, 2)"
    ]


def _batch(sizes=(2, 3)):
    pytest.importorskip("torch")
    nvalchemi_data = pytest.importorskip("nvalchemi.data")
    from ase import Atoms

    atoms = [
        Atoms("H" * n, positions=np.arange(3 * n, dtype=float).reshape(n, 3), cell=np.eye(3) * 5, pbc=True)
        for n in sizes
    ]
    return nvalchemi_data.Batch.from_data_list([nvalchemi_data.AtomicData.from_atoms(a) for a in atoms])


def test_md_build_messages(torch_nvalchemi):
    torch = torch_nvalchemi
    crystal_batch = _batch((2, 3))
    md_batch = _batch((2, 3))
    ns = {"crystal_batch": crystal_batch, "md_batch": md_batch, "n_crystals": 2}
    md_batch.velocities = None
    problems = agent03_check._contract_extras("md_build", ns, {})
    assert "md_build: md_batch should carry velocities; found none" in problems
    md_batch = _batch((2, 3))  # a batch from ASE atoms starts with zero velocities
    ns["md_batch"] = md_batch
    md_batch.status = torch.ones(2, 1, dtype=torch.long)
    problems = agent03_check._contract_extras("md_build", ns, {})
    assert problems == [
        "md_build: no graph of md_batch should be marked finished before the MD starts; "
        "found 2 of 2 marked finished (status 1 or more, left over from the relaxation)"
    ]
    md_batch.status = torch.zeros(2, 1, dtype=torch.long)
    assert agent03_check._contract_extras("md_build", ns, {}) == []
    ns["md_batch"] = _batch((2, 4))
    assert agent03_check._same_graphs("md_build", "md_batch", ns) == [
        "md_build: md_batch should hold the same 2 graphs as crystal_batch "
        "(same atom counts, same order); found 2 graphs"
    ]


def test_relax_build_messages(torch_nvalchemi):
    batch = _batch((2, 3))
    problems = agent03_check._contract_extras("relax_build", {"crystal_batch": batch, "n_crystals": 5}, {})
    assert problems == [
        "relax_build: crystal_batch should have 5 graphs (n_crystals); found 2"
    ]


def test_relax_run_threshold_message_allows_one_percent(torch_nvalchemi):
    torch = torch_nvalchemi
    ns = {
        "relaxed_max_force": torch.tensor([0.0504, 0.0300]),
        "force_threshold": 0.05,
    }
    assert agent03_check._contract_extras("relax_run", ns, {}) == []
    ns["relaxed_max_force"] = torch.tensor([0.0600, 0.0300, 0.2])
    assert agent03_check._contract_extras("relax_run", ns, {}) == [
        "relax_run: every relaxed_max_force should be at or below force_threshold (0.05 eV/Å); "
        "found 2 of 3 above it (largest 0.2 eV/Å)"
    ]


def _force_case(torch):
    """Two graphs (2 and 3 atoms). Largest vector length 5 and 2; largest component 4 and 2."""
    forces = torch.zeros(5, 3)
    forces[0] = torch.tensor([3.0, 4.0, 0.0])
    forces[2] = torch.tensor([0.0, 0.0, 2.0])
    return forces


def test_largest_force_is_the_vector_length(torch_nvalchemi, monkeypatch):
    torch = torch_nvalchemi
    forces = _force_case(torch)
    monkeypatch.setattr(agent03_check, "prepare_model_batch", lambda batch, model: batch)
    calls = []

    def model(batch):
        calls.append(1)
        return {"forces": forces}

    ns = {"crystal_batch": _batch((2, 3)), "mace_model": model, "n_crystals": 2}
    ns["start_max_force"] = torch.tensor([5.0, 2.0])
    assert agent03_check._contract_extras("relax_build", ns, {}) == []
    assert calls == [1]  # no stored forces, so one forward call
    ns["start_max_force"] = torch.tensor([5.04, 2.0])  # inside 1 percent
    assert agent03_check._contract_extras("relax_build", ns, {}) == []
    ns["start_max_force"] = torch.tensor([4.0, 2.0])  # the largest Cartesian component
    assert agent03_check._contract_extras("relax_build", ns, {}) == [
        "relax_build: start_max_force should be the largest force on any atom, measured as the "
        "length of the force vector, in eV/Å; found values that differ for 1 of 2 crystals "
        "(for example 4 instead of 5)"
    ]


def test_largest_force_uses_stored_forces_and_falls_back(torch_nvalchemi, monkeypatch):
    torch = torch_nvalchemi
    forces = _force_case(torch)
    monkeypatch.setattr(agent03_check, "prepare_model_batch", lambda batch, model: batch)
    calls = []

    def model(batch):
        calls.append(1)
        return {"forces": forces}

    batch = _batch((2, 3))
    batch.forces = forces
    ns = {"relaxed_batch": batch, "mace_model": model, "relaxed_max_force": torch.tensor([5.0, 2.0])}
    assert agent03_check._contract_extras("relax_run", ns, {}) == []
    assert calls == []  # the stored forces agree, no forward call
    # Stored forces one step old: the fresh forces decide.
    batch.forces = forces * 0.5
    assert agent03_check._contract_extras("relax_run", ns, {}) == []
    assert calls == [1]
    ns["relaxed_max_force"] = torch.tensor([4.0, 1.0])
    problems = agent03_check._contract_extras("relax_run", ns, {})
    assert problems == [
        "relax_run: relaxed_max_force should be the largest force on any atom, measured as the "
        "length of the force vector, in eV/Å; found values that differ for 2 of 2 crystals "
        "(for example 4 instead of 5)"
    ]


def test_largest_force_check_never_fails_a_file_it_cannot_judge(torch_nvalchemi, monkeypatch):
    torch = torch_nvalchemi
    monkeypatch.setattr(agent03_check, "prepare_model_batch", lambda batch, model: batch)

    def broken(batch):
        raise RuntimeError("no forces")

    ns = {"relaxed_batch": _batch((2, 3)), "mace_model": broken, "relaxed_max_force": torch.tensor([9.0, 9.0])}
    assert agent03_check._force_is_vector_length("relax_run", "relaxed_max_force", "relaxed_batch", ns) == []
    ns["relaxed_max_force"] = torch.tensor([9.0])  # wrong shape is reported by another check
    assert agent03_check._force_is_vector_length("relax_run", "relaxed_max_force", "relaxed_batch", ns) == []


def test_atoms_moved_message(torch_nvalchemi):
    md_batch = _batch((2, 3))
    ctx = {
        "md_positions": (
            md_batch.positions.detach().cpu().double().clone(),
            md_batch.batch_idx.detach().cpu().long().clone(),
            md_batch.num_graphs,
            2,
        ),
        "batch_ids": {},
    }
    ns = {"md_batch": md_batch}
    assert agent03_check._atoms_moved("md_run", ns, ctx) == [
        "md_run: no atom moved in 2 of 2 crystals during the MD; "
        "md_batch may still carry the relaxation's finished marks"
    ]
    md_batch.positions = md_batch.positions + 0.1
    assert agent03_check._atoms_moved("md_run", ns, ctx) == []


def _watch(times, dt=1.0):
    return SimpleNamespace(hooks={1: [None, dt, list(times)]})


def test_time_axis_message(torch_nvalchemi):
    torch = torch_nvalchemi
    ce = torch.zeros(4, 2)
    on_time = {"time_fs": torch.tensor([0.0, 1.0, 2.0, 3.0]), "crystal_energy": ce}
    ctx = {"dyn_watch": _watch([1.0, 2.0, 3.0])}  # one extra first row for time 0
    assert agent03_check._time_axis("md_run", on_time, ctx) == []
    late = {"time_fs": torch.tensor([0.0, 1.0, 2.0, 5.0]), "crystal_energy": ce}
    assert agent03_check._time_axis("md_run", late, ctx) == [
        "md_run: time_fs should give the time of each saved energy row; the last row was "
        "recorded at 3 fs but time_fs says 5 fs"
    ]
    one_step_early = {"time_fs": torch.tensor([0.0, 1.0, 2.0, 2.0 + 1.0]), "crystal_energy": ce}
    assert agent03_check._time_axis("md_run", one_step_early, ctx) == []
    # no hook matches the rows: the check stays silent
    assert agent03_check._time_axis("md_run", late, {"dyn_watch": _watch([1.0, 2.0])}) == []


def test_settings_words():
    torch = pytest.importorskip("torch")
    assert agent03_check._settings_words((False, False, torch.float32)) == (
        "cuEquivariance off, not compiled, float32"
    )
    assert agent03_check._settings_words((True, True, torch.float64)) == (
        "cuEquivariance on, compiled, float64"
    )


def test_main_prints_usage_for_an_unknown_step(capsys):
    assert agent03_check.main(["x", "check", "nope"]) == 2
    assert "usage: python -m helpers.agent03 check <relax_build|relax_run|md_build|md_run|timestep_scan|temperature_ladder|speed_settings>" in (
        capsys.readouterr().err
    )


def test_missing_files_messages(tmp_path, monkeypatch, capsys, torch_nvalchemi):
    monkeypatch.setattr(agent03, "WORK_DIR", tmp_path)
    problems, _ = agent03_check.check_step("relax_build")
    assert problems == [f"relax_build: the step file relax_build.py is missing in {tmp_path}"]
    problems, _ = agent03_check.check_step("relax_run")
    assert problems == [f"relax_run: the earlier step file relax_build.py is missing in {tmp_path}"]


def test_a_failing_step_file_is_reported_in_one_line(tmp_path, monkeypatch, capsys, torch_nvalchemi):
    monkeypatch.setattr(agent03, "WORK_DIR", tmp_path)
    (tmp_path / "relax_build.py").write_text("raise ValueError('no crystals here\\nsecond line')\n")
    problems, _ = agent03_check.check_step("relax_build")
    assert problems[0] == "relax_build: relax_build.py stopped with ValueError: no crystals here"


def test_a_file_that_uses_dunder_file_is_told_to_remove_that_line(tmp_path, monkeypatch, capsys, torch_nvalchemi):
    monkeypatch.setattr(agent03, "WORK_DIR", tmp_path)
    (tmp_path / "relax_build.py").write_text("from pathlib import Path\nhere = Path(__file__).parent\n")
    problems, _ = agent03_check.check_step("relax_build")
    assert problems[0] == (
        "relax_build: relax_build.py uses __file__, which does not exist in a notebook cell; remove that line"
    )
    # another missing name keeps the general line
    (tmp_path / "relax_build.py").write_text("print(not_defined_anywhere)\n")
    problems, _ = agent03_check.check_step("relax_build")
    assert problems[0] == (
        "relax_build: relax_build.py stopped with NameError: name 'not_defined_anywhere' is not defined"
    )
    # in an earlier file the line names the earlier step
    (tmp_path / "relax_build.py").write_text("here = __file__\n")
    problems, _ = agent03_check.check_step("relax_run")
    assert problems == [
        "relax_run: could not run the earlier steps; relax_build: relax_build.py uses __file__, "
        "which does not exist in a notebook cell; remove that line"
    ]


# ---------------------------------------------------------------------------
# Draft exercise: timestep scan (plot, contract, messages)
# ---------------------------------------------------------------------------


def test_timestep_scan_contract():
    spec = agent03_check.STEPS["timestep_scan"]
    assert spec.file == "timestep_scan.py"
    assert spec.after == ["relax_build"]
    assert list(spec.names) == ["replica_dt", "saved_step", "total_energy"]
    dt, step, energy = (spec.names[n] for n in spec.names)
    assert dt.shape == ("n_replicas",) and dt.unit == "fs" and dt.positive and dt.finite
    assert step.shape == ("n_saved",) and step.increasing and step.integer
    assert energy.shape == ("n_saved", "n_replicas") and energy.unit == "eV"
    assert not energy.finite  # a copy that blew up may turn non-finite


def _scan_lines(axis):
    """The data lines of the scan plot (not the zero line)."""

    return [ln for ln in axis.get_lines() if ln.get_linewidth() == 1.8]


def test_timestep_scan_draws_one_line_per_timestep_against_time():
    dt, step, energy = _scan_inputs()
    figure = agent03.build_timestep_scan_figure(dt, step, energy)
    axis = figure.axes[0]
    lines = _scan_lines(axis)
    assert len(lines) == 4
    assert axis.get_yscale() == "symlog"
    ends = sorted(float(ln.get_xdata()[-1]) for ln in lines)
    assert ends == pytest.approx(sorted(step[-1] * dt))  # time = saved_step x that copy's dt
    names = _texts(figure)
    assert all(f"{v:g} fs" in names for v in dt)
    assert "Time [fs]" in names and "Change in total energy [meV]" in names


def test_timestep_scan_change_is_in_mev_from_the_first_row():
    dt, step, energy = _scan_inputs()
    data = agent03.timestep_scan_data(dt, step, energy)
    assert data["change"][0] == pytest.approx(0.0)
    assert data["change"][-1, 0] == pytest.approx((energy[-1, 0] - energy[0, 0]) * 1000.0)


def test_timestep_scan_marks_a_copy_that_blew_up():
    dt, step, energy = _scan_inputs()
    energy[6:, 3] = np.nan
    figure = agent03.build_timestep_scan_figure(dt, step, energy)
    axis = figure.axes[0]
    assert len(axis.collections) == 1  # one cross, on the copy that blew up
    longest = max(len(ln.get_xdata()) for ln in _scan_lines(axis))
    shortest = min(len(ln.get_xdata()) for ln in _scan_lines(axis))
    assert (longest, shortest) == (11, 6)  # drawn to the last finite point


def test_timestep_scan_takes_tensors_and_rejects_a_wrong_shape():
    torch = pytest.importorskip("torch")
    dt, step, energy = _scan_inputs()
    agent03.build_timestep_scan_figure(torch.tensor(dt), torch.tensor(step), torch.tensor(energy))
    with pytest.raises(ValueError, match="shape"):
        agent03.build_timestep_scan_figure(dt, step, energy[:, :3])


def test_timestep_scan_alt_text_is_a_marked_placeholder():
    assert agent03.LABELS["timestep_scan"]["alt"].startswith("PLACEHOLDER")


def test_timestep_scan_name_checks(torch_nvalchemi):
    torch = torch_nvalchemi

    def check(name, value, **syms):
        spec = agent03_check.STEPS["timestep_scan"].names[name]
        return agent03_check._check_name("timestep_scan", name, spec, {name: value}, syms)

    syms = {"n_replicas": 3, "n_saved": 4}
    assert check("replica_dt", torch.tensor([0.5, 1.0, 2.0]), **syms) == []
    assert check("replica_dt", torch.tensor([0.5, 1.0, 2.0]).reshape(3, 1), **syms) == [
        "timestep_scan: replica_dt should have shape (n_replicas,) = (3,) in fs; found shape (3, 1)"
    ]
    assert check("replica_dt", torch.tensor([0.5, -1.0, 2.0]), **syms) == [
        "timestep_scan: replica_dt should be above 0 everywhere; found 1 of 3 values at or below 0 (smallest -1)"
    ]
    assert check("saved_step", torch.tensor([0, 10, 20, 30]), **syms) == []
    assert check("saved_step", torch.tensor([0, 10, 10, 30]), **syms) == [
        "timestep_scan: saved_step should increase from one saved row to the next; found rows that do not"
    ]
    assert check("saved_step", torch.tensor([0.0, 10.5, 20.0, 30.0]), **syms) == [
        "timestep_scan: saved_step should hold whole numbers (step counts); found values with a fraction"
    ]
    assert check("total_energy", torch.zeros(4, 3), **syms) == []
    assert check("total_energy", torch.zeros(3, 4), **syms) == [
        "timestep_scan: total_energy should have shape (n_saved, n_replicas) = (4, 3) in eV; "
        "found shape (3, 4)"
    ]
    blown = torch.zeros(4, 3)
    blown[2:, 1] = float("nan")
    assert check("total_energy", blown, **syms) == []  # a copy that blew up is allowed


def test_timestep_scan_symbols_come_from_its_own_names(torch_nvalchemi):
    torch = torch_nvalchemi
    ns = {
        "n_crystals": 200,
        "time_fs": torch.zeros(501),  # left by md_run, not the scan's own
        "saved_step": torch.zeros(101),
        "replica_dt": torch.ones(6),
    }
    assert agent03_check._symbols(ns, "timestep_scan") == {"n_structures": 200, "n_saved": 101, "n_replicas": 6}
    assert agent03_check._symbols(ns, "md_run")["n_saved"] == 501


def test_timestep_scan_value_messages(torch_nvalchemi):
    torch = torch_nvalchemi

    def values(dt, energy, saved=(0, 10, 20, 30)):
        ns = {"replica_dt": torch.tensor(dt), "saved_step": torch.tensor(saved), "total_energy": energy}
        return agent03_check._scan_values("timestep_scan", ns, {})

    ok = torch.zeros(4, 3)
    assert values([0.5, 1.0, 2.0], ok) == []
    assert values([1.0, 1.0, 1.0], ok) == [
        "timestep_scan: replica_dt should hold two or more different timesteps, one for each copy, "
        "in fs; found 1 different timestep among 3"
    ]
    assert values([1.0], torch.zeros(4, 1)) == [
        "timestep_scan: replica_dt should hold two or more different timesteps, one for each copy, "
        "in fs; found 1 different timestep among 1"
    ]
    assert values([0.5, 1.0], torch.zeros(1, 2), saved=(0,)) == [
        "timestep_scan: saved_step should hold two or more saved rows; found 1"
    ]
    bad_start = torch.zeros(4, 3)
    bad_start[0, 2] = float("nan")
    assert values([0.5, 1.0, 2.0], bad_start) == [
        "timestep_scan: the first row of total_energy should be finite for every copy; found 1 of 3 copies that are not"
    ]
    all_nan = torch.zeros(4, 3)
    all_nan[2:, :] = float("inf")
    assert values([0.5, 1.0, 2.0], all_nan) == [
        "timestep_scan: total_energy should stay finite and within 10 eV per copy of the first row "
        "for at least one copy; found none"
    ]
    one_blown = torch.zeros(4, 3)
    one_blown[2:, 2] = float("nan")
    assert values([0.5, 1.0, 2.0], one_blown) == []


# --- the copies: same crystal, periodic, each with its own timestep -----------


def _ase_copies(atoms, n=3):
    nvalchemi_data = pytest.importorskip("nvalchemi.data")
    return nvalchemi_data.Batch.from_data_list(
        [nvalchemi_data.AtomicData.from_atoms(atoms.copy()) for _ in range(n)]
    )


def _engine(dt_fs):
    torch = pytest.importorskip("torch")
    from nvalchemi.dynamics._units import fs_to_internal_time

    return SimpleNamespace(_dt_init=fs_to_internal_time(torch.tensor(dt_fs)))


def _scan_ctx(torch, batch, dt_fs):
    runs = [(_engine(dt_fs), agent03_check._start_state(batch))]
    return {"run_watch": SimpleNamespace(runs=runs)}


def _copies(torch, batch, dt=(0.5, 1.0, 2.0), engine_dt=None):
    ns = {"replica_dt": torch.tensor(dt)}
    ctx = _scan_ctx(torch, batch, list(engine_dt or dt))
    return agent03_check._scan_copies("timestep_scan", ns, ctx)


def test_copies_of_a_crystal_from_the_set_pass(torch_nvalchemi):

    batch = _batch((3, 3, 3))  # graphs of H atoms with a cell and pbc
    assert _copies(torch_nvalchemi, batch) == []


def test_copies_of_a_crystal_that_is_not_in_the_set_pass(torch_nvalchemi):
    from ase.build import bulk

    batch = _ase_copies(bulk("Cu", cubic=True))
    assert _copies(torch_nvalchemi, batch) == []
    batch = _ase_copies(bulk("Si"), n=3)
    assert _copies(torch_nvalchemi, batch) == []


def test_copies_of_a_molecule_fail_with_the_periodic_line(torch_nvalchemi):
    from ase.build import molecule

    batch = _ase_copies(molecule("H2O"))
    assert _copies(torch_nvalchemi, batch) == [
        "timestep_scan: the copies should be a periodic crystal (a cell and periodic boundaries on all "
        "three axes); found copies that carry no cell or no periodic boundaries; MACE-MPA-0 is a "
        "model for materials"
    ]


def test_copies_with_one_open_axis_fail(torch_nvalchemi):
    from ase.build import bulk

    atoms = bulk("Cu", cubic=True)
    atoms.pbc = [True, True, False]
    problems = _copies(torch_nvalchemi, _ase_copies(atoms))
    assert len(problems) == 1 and "found 3 of 3 copies without periodic boundaries" in problems[0]


def test_copies_must_be_the_same_crystal(torch_nvalchemi):
    from ase.build import bulk
    nvalchemi_data = pytest.importorskip("nvalchemi.data")

    cu, al = bulk("Cu", cubic=True), bulk("Al", cubic=True)
    mixed = nvalchemi_data.Batch.from_data_list(
        [nvalchemi_data.AtomicData.from_atoms(a) for a in (cu, cu, al)]
    )
    assert _copies(torch_nvalchemi, mixed) == [
        "timestep_scan: every copy should be the same crystal; found copies with different atoms"
    ]
    shifted = cu.copy()
    shifted.positions[0] += 0.3
    moved = nvalchemi_data.Batch.from_data_list(
        [nvalchemi_data.AtomicData.from_atoms(a) for a in (cu, cu, shifted)]
    )
    assert _copies(torch_nvalchemi, moved) == [
        "timestep_scan: every copy should start from the same positions; found copies that differ"
    ]
    bigger = nvalchemi_data.Batch.from_data_list(
        [nvalchemi_data.AtomicData.from_atoms(a) for a in (cu, cu, cu.repeat((2, 1, 1)))]
    )
    assert _copies(torch_nvalchemi, bigger) == [
        "timestep_scan: every copy should be the same crystal; found copies with different atom counts"
    ]


def test_every_copy_must_run_with_its_own_timestep(torch_nvalchemi):
    torch = torch_nvalchemi
    batch = _batch((3, 3, 3))
    assert _copies(torch, batch, engine_dt=[1.0]) == [  # one number for every copy
        "timestep_scan: replica_dt should be the timestep each copy ran with (fs); the run used "
        "1 but replica_dt says 0.5, 1, 2"
    ]
    assert _copies(torch, batch, engine_dt=[0.5, 1.0, 4.0]) == [
        "timestep_scan: replica_dt should be the timestep each copy ran with (fs); the run used "
        "0.5, 1, 4 but replica_dt says 0.5, 1, 2"
    ]
    # the engine holds a tensor in fs: the same values pass
    assert _copies(torch, batch, engine_dt=[0.5, 1.0, 2.0]) == []


def test_copies_must_run_in_one_batch(torch_nvalchemi):
    torch = torch_nvalchemi
    ns = {"replica_dt": torch.tensor([0.5, 1.0, 2.0])}
    ctx = _scan_ctx(torch, _batch((3,)), [0.5])
    assert agent03_check._scan_copies("timestep_scan", ns, ctx) == [
        "timestep_scan: the copies should run together in one batch, one graph for each entry of "
        "replica_dt (3); found runs of 1 graphs"
    ]


def test_copies_check_is_silent_when_no_run_was_seen(torch_nvalchemi):
    torch = torch_nvalchemi
    ns = {"replica_dt": torch.tensor([0.5, 1.0])}
    assert agent03_check._scan_copies("timestep_scan", ns, {}) == []
    assert agent03_check._scan_copies("timestep_scan", ns, {"run_watch": SimpleNamespace(runs=[])}) == []


def test_copies_with_an_empty_cell_and_open_boundaries_are_counted(torch_nvalchemi):
    torch = torch_nvalchemi
    from ase.build import bulk

    batch = _ase_copies(bulk("Cu", cubic=True))
    ctx = _scan_ctx(torch, batch, [0.5, 1.0, 2.0])
    state = ctx["run_watch"].runs[0][1]
    state["pbc"] = torch.zeros(3, 3, dtype=torch.bool)
    state["cell"] = torch.zeros(3, 3, 3, dtype=torch.float64)
    problems = agent03_check._scan_copies("timestep_scan", {"replica_dt": torch.tensor([0.5, 1.0, 2.0])}, ctx)
    assert problems == [
        "timestep_scan: the copies should be a periodic crystal (a cell and periodic boundaries on all "
        "three axes); found 3 of 3 copies without periodic boundaries on all three axes and 3 with an "
        "empty cell; MACE-MPA-0 is a model for materials"
    ]


def test_copies_marked_finished_are_reported(torch_nvalchemi):
    torch = torch_nvalchemi
    from ase.build import bulk

    batch = _ase_copies(bulk("Cu", cubic=True))
    ctx = _scan_ctx(torch, batch, [0.5, 1.0, 2.0])
    ctx["run_watch"].runs[0][1]["finished"] = 3
    problems = agent03_check._scan_copies("timestep_scan", {"replica_dt": torch.tensor([0.5, 1.0, 2.0])}, ctx)
    assert problems == [
        "timestep_scan: no copy should be marked finished before the run starts; found 3 of 3 marked "
        "finished (status 1 or more, left over from the relaxation)"
    ]


# --- blow-ups by size, copies that did not move, the capped plot ------------------


def test_blown_up_rows_use_size_as_well_as_finiteness():
    energy = np.zeros((5, 4))
    energy[3:, 0] = 1e5          # finite, but far off: a blow-up
    energy[2:, 1] = np.nan       # not finite
    energy[:, 2] = 0.002         # a stable copy
    energy[4, 3] = 10.5          # leaves the 10 eV limit at the last row
    rows = agent03.scan_blowup_rows(energy - energy[0], agent03.SCAN_BLOWUP_EV_PER_COPY)
    assert list(rows) == [3, 2, 5, 4]
    assert list(agent03.scan_blowup_rows(energy - energy[0], np.array([1.0, 1.0, 1.0, 20.0]))) == [3, 2, 5, 5]


def test_check_fails_when_every_copy_blows_up_even_if_finite(torch_nvalchemi):
    torch = torch_nvalchemi

    def values(energy, ctx=None):
        ns = {"replica_dt": torch.tensor([0.5, 1.0, 2.0]), "saved_step": torch.tensor([0, 10, 20, 30]),
              "total_energy": energy}
        return agent03_check._scan_values("timestep_scan", ns, ctx or {})

    huge = torch.zeros(4, 3)
    huge[2:, :] = 1e8  # finite everywhere
    assert values(huge) == [
        "timestep_scan: total_energy should stay finite and within 10 eV per copy of the first row "
        "for at least one copy; found none"
    ]
    one_ok = huge.clone()
    one_ok[:, 0] = 0.01
    assert values(one_ok) == []
    # with the run known, the limit is 1 eV per atom: 3 copies of 8 atoms
    ctx = {"run_watch": SimpleNamespace(runs=[(None, agent03_check._start_state(_batch((8, 8, 8))))])}
    mid = torch.zeros(4, 3)
    mid[2:, :] = 6.0  # under 10 eV for the copy, under 8 eV (1 eV x 8 atoms): passes
    assert values(mid, ctx) == []
    mid[2:, :] = 9.0  # over 8 eV
    assert values(mid, ctx) == [
        "timestep_scan: total_energy should stay finite and within 1 eV per atom of the first row "
        "for at least one copy; found none"
    ]


def _moved_state(torch, batch, shift):
    ctx = _scan_ctx(torch, batch, [0.5, 1.0, 2.0])
    state = ctx["run_watch"].runs[0][1]
    state["end_pos"] = state["pos"] + shift
    return ctx


def test_copies_that_did_not_move_are_reported(torch_nvalchemi):
    torch = torch_nvalchemi
    from ase.build import bulk

    batch = _ase_copies(bulk("Cu", cubic=True))  # 3 copies of 4 atoms
    ns = {"replica_dt": torch.tensor([0.5, 1.0, 2.0])}
    pos = agent03_check._start_state(batch)["pos"]
    ctx = _moved_state(torch, batch, torch.zeros_like(pos))
    assert agent03_check._scan_copies("timestep_scan", ns, ctx) == [
        "timestep_scan: the atoms of 3 of 3 copies did not move during the run, so their energy cannot drift"
    ]
    # one copy moves, two stay
    shift = torch.zeros_like(pos)
    shift[:4] = 0.01
    assert agent03_check._scan_copies("timestep_scan", ns, _moved_state(torch, batch, shift)) == [
        "timestep_scan: the atoms of 2 of 3 copies did not move during the run, so their energy cannot drift"
    ]
    # one atom of each copy moving is enough
    shift = torch.zeros_like(pos)
    shift[0::4] = 0.01
    assert agent03_check._scan_copies("timestep_scan", ns, _moved_state(torch, batch, shift)) == []
    # finished marks add the hint
    ctx = _moved_state(torch, batch, torch.zeros_like(pos))
    ctx["run_watch"].runs[0][1]["finished"] = 3
    lines = agent03_check._scan_copies("timestep_scan", ns, ctx)
    assert lines[-1] == (
        "timestep_scan: the atoms of 3 of 3 copies did not move during the run, so their energy cannot "
        "drift; the copies may still carry the relaxation's finished marks"
    )
    # no end positions recorded: silent
    assert agent03_check._scan_copies("timestep_scan", ns, _scan_ctx(torch, batch, [0.5, 1.0, 2.0])) == []


def test_run_watch_records_end_positions(torch_nvalchemi):
    from nvalchemi.dynamics.base import BaseDynamics

    batch = _batch((3, 3))
    original = BaseDynamics.run

    def fake_run(engine, b, *a, **k):
        b.positions = b.positions + 0.5
        return b

    BaseDynamics.run = fake_run
    try:
        with agent03_check._WatchEngineRuns() as watch:
            BaseDynamics.run(SimpleNamespace(), batch)
    finally:
        BaseDynamics.run = original
    state = watch.runs[0][1]
    assert float((state["end_pos"] - state["pos"]).abs().min()) == pytest.approx(0.5)


def test_plot_caps_a_finite_blow_up_and_marks_it():
    dt, step, energy = _scan_inputs()
    energy[6:, 3] = -100.0 + 1e8   # finite, huge: would stretch the axis to 1e11 meV
    energy[3:, 2] = -100.0 + 50.0  # over the 10 eV limit from row 3
    figure = agent03.build_timestep_scan_figure(dt, step, energy)
    axis = figure.axes[0]
    assert sorted(len(ln.get_xdata()) for ln in _scan_lines(axis)) == [3, 6, 11, 11]
    assert len(axis.collections) == 2  # one cross for each blown-up copy
    top = max(float(np.max(ln.get_ydata())) for ln in _scan_lines(axis))
    assert top < 1e4 + 1  # nothing above the limit (10 eV = 1e4 meV) is drawn
    assert axis.get_ylim()[1] < 1e5


def test_timestep_scan_alt_text_says_crystal():
    assert "crystal" in agent03.LABELS["timestep_scan"]["alt"]
    assert "molecule" not in agent03.LABELS["timestep_scan"]["alt"]


# ---------------------------------------------------------------------------
# Draft exercise: temperature ladder (default crystal, view, contract, messages)
# ---------------------------------------------------------------------------

KB = 8.617e-5  # eV/K


def test_ladder_default_crystal_is_the_saved_copper_supercell():
    atoms = agent03.load_ladder_crystal()
    assert len(atoms) == 64 and set(atoms.get_chemical_symbols()) == {"Cu"}
    assert atoms.pbc.all()
    assert atoms.cell.lengths() == pytest.approx([10.211] * 3, abs=1e-3)
    widths = 1.0 / np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    assert widths.min() > 6.0  # every plane spacing is above the model's 6 Å cutoff
    from ase.build import bulk

    built = bulk("Cu", "fcc").repeat((4, 4, 4))
    assert np.abs(atoms.positions - built.positions).max() < 1e-6  # the file is what the README says
    assert agent03.load_ladder_crystal() is not atoms


def test_ladder_contract():
    spec = agent03_check.STEPS["temperature_ladder"]
    assert spec.file == "temperature_ladder.py" and spec.after == ["relax_build"]
    assert list(spec.names) == ["time_fs", "replica_temperature", "replica_energy"]
    time, temps, energy = (spec.names[n] for n in spec.names)
    assert time.shape == ("n_saved",) and time.unit == "fs" and time.increasing
    assert temps.shape == ("n_temperatures",) and temps.unit == "K" and temps.positive
    assert energy.shape == ("n_saved", "n_temperatures") and energy.unit == "eV/atom"
    assert not energy.finite  # a copy that blew up may turn non-finite


def test_setup_cell_names_the_ladder_setting_and_reads_the_overrides(monkeypatch):
    assert f"n_temperatures = {agent03_check.SETTINGS['n_temperatures']}" in agent03_check.SETUP_CELL.splitlines()
    assert "n_crystals = 200" in agent03_check.SETUP_CELL.splitlines()
    monkeypatch.setenv("ALCHEMI_AGENT03_N_CRYSTALS", "4")
    monkeypatch.setenv("ALCHEMI_AGENT03_N_TEMPERATURES", "3")
    lines = agent03_check.setup_cell().splitlines()
    assert "n_crystals = 4" in lines and "n_temperatures = 3" in lines


def test_ladder_view_is_energy_above_the_first_row_in_mev_per_atom():
    time, temps, energy = _ladder_inputs()
    data = agent03.ladder_data(time, temps, energy)
    assert data["above"][0] == pytest.approx(0.0)
    assert data["above"][-1, 2] == pytest.approx((energy[-1, 2] - energy[0, 2]) * 1000.0)
    assert data["notes"] == [] and list(data["blown"]) == [31] * 6
    figure = agent03.build_temperature_ladder_figure(time, temps, energy)
    axis = figure.axes[0]
    lines = _scan_lines(axis)
    assert len(lines) == 6  # one line for each temperature
    names = _texts(figure)
    assert all(f"{t:g} K" in names for t in temps)
    assert "Time [fs]" in names
    assert "Potential energy above the relaxed crystal [meV/atom]" in names
    # hotter copies end above cooler ones, and the lines run from blue (cold) to red (hot)
    by_end = sorted(lines, key=lambda ln: ln.get_ydata()[-1])
    reds = [ln.get_color()[0] for ln in by_end]
    assert reds == sorted(reds) and reds[0] < 0.1 and reds[-1] > 0.8
    assert float(by_end[-1].get_ydata()[-1]) == pytest.approx(1.5 * KB * 3000.0 * 1000.0, rel=0.01)


def test_ladder_view_takes_tensors_and_rejects_a_wrong_shape():
    torch = pytest.importorskip("torch")
    time, temps, energy = _ladder_inputs()
    agent03.build_temperature_ladder_figure(torch.tensor(time), torch.tensor(temps), torch.tensor(energy))
    with pytest.raises(ValueError, match=r"shape \(n_saved, n_temperatures\) = \(31, 6\); found \(31, 5\)"):
        agent03.build_temperature_ladder_figure(time, temps, energy[:, :5])


def test_ladder_view_notes_an_out_of_order_pair_and_still_draws(capsys, shown):
    time, temps, energy = _ladder_inputs()
    energy[:, [1, 2]] = energy[:, [2, 1]]  # the 840 K column now holds the 1380 K energies
    notes = agent03.ladder_data(time, temps, energy)["notes"]
    assert notes == [
        "unexpected: over the last third of the run, the copy at 1380 K sits below the copy at "
        "840 K (108 against 177 meV per atom)"
    ]
    agent03.plot_temperature_ladder(time, temps, energy)
    assert len(shown) == 1 and capsys.readouterr().out.strip() == notes[0]


def test_ladder_order_is_judged_on_the_last_third_only():
    time, temps, energy = _ladder_inputs()
    early = energy.copy()
    early[1:15, [0, 5]] = early[1:15, [5, 0]]  # out of order early in the run only
    assert agent03.ladder_data(time, temps, early)["notes"] == []


def test_ladder_marks_a_copy_that_blew_up_and_judges_the_rest():
    time, temps, energy = _ladder_inputs()
    energy[10:, 5] = np.nan       # not finite
    energy[20:, 4] = -5.4 + 1e5   # finite and far off
    data = agent03.ladder_data(time, temps, energy)
    assert list(data["blown"]) == [31, 31, 31, 31, 20, 10]
    assert data["notes"] == []  # the order is judged over the copies that survive
    figure = agent03.build_temperature_ladder_figure(time, temps, energy)
    axis = figure.axes[0]
    assert len(axis.collections) == 2  # one cross for each blown-up copy
    assert sorted(len(ln.get_xdata()) for ln in _scan_lines(axis)) == [10, 20, 31, 31, 31, 31]
    assert max(float(np.nanmax(ln.get_ydata())) for ln in _scan_lines(axis)) < 1e4


def test_ladder_blowup_limit_keeps_the_hottest_intact_copies():
    # Intact copies of the advisor's trial sat 3.5 and 4.7 eV per atom above the relaxed crystal.
    assert agent03.LADDER["blowup_ev_per_atom"] > 4.7 * 2
    time, temps, energy = _ladder_inputs()
    energy[5:, 5] = -5.4 + 4.7
    assert list(agent03.ladder_data(time, temps, energy)["blown"]) == [31] * 6


def test_ladder_notes_when_every_copy_blew_up():
    time, temps, energy = _ladder_inputs()
    energy[3:, :] = np.inf
    assert agent03.ladder_data(time, temps, energy)["notes"] == [
        "unexpected: every copy blew up, so the order of the energies cannot be judged"
    ]


def test_ladder_notes_whole_copy_energies():
    time, temps, energy = _ladder_inputs()
    whole = energy * 64
    notes = agent03.ladder_data(time, temps, whole)["notes"]
    assert notes == [
        "replica_energy should be in eV per atom; its first row is about -345.6 eV, the size of the "
        "energy of a whole copy; divide each copy's energy by its number of atoms"
    ]
    known = agent03.ladder_data(time, temps, whole, n_atoms=64)
    assert known["notes"][0].startswith("replica_energy should be in eV per atom") and " of 64 atoms" in known["notes"][0]
    assert known["above"][-1, 2] == pytest.approx((energy[-1, 2] - energy[0, 2]) * 1000.0)


def test_ladder_notes_a_start_that_is_not_the_relaxed_crystal():
    time, temps, energy = _ladder_inputs()
    cooled = energy.copy()
    cooled[1:, 0] -= 0.02  # the coldest copy drops 20 meV per atom below the first row
    assert agent03.ladder_data(time, temps, cooled)["notes"] == [
        "unexpected: the energy of 1 of 6 copies falls below the first row, so the copies did not "
        "start from a relaxed crystal; the zero of the plot is the first row"
    ]
    late = energy[3:]  # the first saved row is already warm and differs between the copies
    notes = agent03.ladder_data(time[3:], temps, late)["notes"]
    assert len(notes) == 1 and notes[0].startswith(
        "unexpected: the first row of replica_energy differs between the copies by "
    )


def _thermostat(torch, target_k, n):
    """A stand-in for a finished NVTLangevin run: the engine and what the watch reads from it."""
    from nvalchemi.dynamics import KB_EV

    target = torch.as_tensor(target_k, dtype=torch.float64).reshape(-1)
    held = target.expand(n) if target.numel() == 1 else target
    return {"target_init_K": held.clone(), "target_end_K": (held * KB_EV).clone() / KB_EV}


def _ladder_ns_ctx(torch, temps=(300.0, 1000.0, 3000.0), atoms=None, engine_k=None, shift=0.05, extra=None):
    """Names and watch record of a ladder file that ran copies of one crystal in one batch."""
    from ase.build import bulk

    k = len(temps)
    batch = _ase_copies(atoms if atoms is not None else bulk("Cu", cubic=True), n=k)
    state = agent03_check._start_state(batch)
    state["end_pos"] = state["pos"] + shift
    state["optimizer"] = False
    state["exchange"] = False
    state |= _thermostat(torch, list(temps) if engine_k is None else engine_k, k)
    state |= extra or {}
    time, t, energy = _ladder_inputs(n_saved=13, temps=temps)
    ns = {"time_fs": torch.tensor(time), "replica_temperature": torch.tensor(t), "replica_energy": torch.tensor(energy)}
    ctx = {"notes": [], "run_watch": SimpleNamespace(runs=[(SimpleNamespace(), state)])}
    return ns, ctx, state


def test_ladder_passes_for_copies_of_one_crystal_at_their_own_temperatures(torch_nvalchemi):
    ns, ctx, _ = _ladder_ns_ctx(torch_nvalchemi)
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == []
    assert ctx["notes"] == []


def test_ladder_fails_when_no_atom_moved(torch_nvalchemi):
    torch = torch_nvalchemi
    ns, ctx, _ = _ladder_ns_ctx(torch, shift=0.0)
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [
        "temperature_ladder: the atoms of 3 of 3 copies did not move during the run"
    ]
    ns, ctx, _ = _ladder_ns_ctx(torch, shift=0.0, extra={"finished": 3})
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [
        "temperature_ladder: the atoms of 3 of 3 copies did not move during the run; the copies may "
        "still carry the relaxation's finished marks"
    ]
    ns["replica_energy"] = torch.full((13, 3), -4.09)  # flat lines: no "unexpected" note beside the failure
    ctx["notes"].clear()
    assert len(agent03_check._ladder_contract("temperature_ladder", ns, ctx)) == 1 and ctx["notes"] == []


def test_ladder_motion_is_judged_on_the_dynamics_not_on_a_relaxation_before_it(torch_nvalchemi):
    torch = torch_nvalchemi
    ns, ctx, md = _ladder_ns_ctx(torch, shift=0.0, extra={"finished": 3})
    relax = dict(md, optimizer=True, finished=0)
    relax["end_pos"] = relax["pos"] + 0.2  # the relaxation moved the atoms, the thermostat run did not
    ctx["run_watch"].runs.insert(0, (SimpleNamespace(), relax))
    problems = agent03_check._ladder_contract("temperature_ladder", ns, ctx)
    assert problems == [
        "temperature_ladder: the atoms of 3 of 3 copies did not move during the run; the copies may "
        "still carry the relaxation's finished marks"
    ]


def test_ladder_fails_on_a_flat_energy_even_when_no_run_was_seen(torch_nvalchemi):
    torch = torch_nvalchemi
    ns, _, _ = _ladder_ns_ctx(torch)
    ns["replica_energy"] = torch.full((13, 3), -4.09)
    assert agent03_check._ladder_contract("temperature_ladder", ns, {"notes": []}) == [
        "temperature_ladder: replica_energy of 3 of 3 copies does not change along the run, so their "
        "atoms did not move"
    ]
    # with the watch, the line about the atoms is printed once, not twice
    ns2, ctx, _ = _ladder_ns_ctx(torch, shift=0.0)
    ns2["replica_energy"] = torch.full((13, 3), -4.09)
    assert len(agent03_check._ladder_contract("temperature_ladder", ns2, ctx)) == 1


def test_ladder_fails_when_one_temperature_was_set_for_all_copies(torch_nvalchemi):
    torch = torch_nvalchemi
    # temperature of shape (1,) or a scalar: the engine holds 500 K for every copy
    ns, ctx, _ = _ladder_ns_ctx(torch, engine_k=[500.0])
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [
        "temperature_ladder: replica_temperature should be the temperature each copy's thermostat "
        "held (K); the run used 500, 500, 500 but replica_temperature says 300, 1000, 3000"
    ]
    # the same value for every copy in replica_temperature as well
    ns, ctx, _ = _ladder_ns_ctx(torch, temps=(500.0, 500.0, 500.0))
    problems = agent03_check._ladder_contract("temperature_ladder", ns, ctx)
    assert problems[0] == (
        "temperature_ladder: replica_temperature should hold a different temperature for each copy, "
        "in K; found 1 different temperature among 3"
    )


def test_ladder_fails_when_copies_are_not_one_batch_of_one_periodic_crystal(torch_nvalchemi):
    torch = torch_nvalchemi
    from ase.build import molecule

    ns, ctx, state = _ladder_ns_ctx(torch)
    state["n"] = 1  # the file ran one copy at a time
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [
        "temperature_ladder: the copies should run together in one batch, one graph for each "
        "temperature (n_temperatures = 3); found runs of 1 graphs"
    ]
    ns, ctx, _ = _ladder_ns_ctx(torch, atoms=molecule("H2O"))
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [
        "temperature_ladder: the copies should be a periodic crystal (a cell and periodic boundaries "
        "on all three axes); found copies that carry no cell or no periodic boundaries; MACE-MPA-0 "
        "is a model for materials"
    ]
    ns, ctx, state = _ladder_ns_ctx(torch)
    state["cell"][2] *= 1.1
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [
        "temperature_ladder: every copy should be the same crystal; found copies with different cells"
    ]
    ns, ctx, state = _ladder_ns_ctx(torch)
    state["z"][-1] = 13
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [
        "temperature_ladder: every copy should be the same crystal; found copies with different atoms"
    ]


def test_ladder_fails_on_a_first_row_that_is_not_finite(torch_nvalchemi):
    torch = torch_nvalchemi
    ns, ctx, _ = _ladder_ns_ctx(torch)
    ns["replica_energy"][0, 1] = float("nan")
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [
        "temperature_ladder: the first row of replica_energy should be finite for every copy; found "
        "1 of 3 copies that are not"
    ]


def test_ladder_out_of_order_is_a_note_and_the_step_passes(torch_nvalchemi):
    torch = torch_nvalchemi
    ns, ctx, _ = _ladder_ns_ctx(torch)
    ns["replica_energy"] = ns["replica_energy"][:, [0, 2, 1]]
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == []
    assert len(ctx["notes"]) == 1 and ctx["notes"][0].startswith(
        "unexpected: over the last third of the run, the copy at 3000 K sits below the copy at 1000 K"
    )


def test_ladder_accepts_another_route_and_compares_kinetic_temperatures(torch_nvalchemi):
    torch = torch_nvalchemi
    from nvalchemi.dynamics import KB_EV

    # replica exchange rebinds the targets, so they are not read; the end velocities are compared
    ns, ctx, state = _ladder_ns_ctx(torch, extra={"exchange": True})
    n = state["idx"].numel()
    masses = torch.full((n, 1), 63.5, dtype=torch.float64)
    temps = torch.tensor([300.0, 1000.0, 3000.0], dtype=torch.float64)

    def velocities(kelvin):  # every atom at the speed of its copy's temperature
        speed = torch.sqrt(3.0 * KB_EV * kelvin[state["idx"]] / 63.5)
        return torch.stack([speed, torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)], dim=1)

    state |= {"masses": masses, "end_vel": velocities(temps)}
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == [] and ctx["notes"] == []
    state["end_vel"] = velocities(temps.flip(0))  # the hottest copy is the coldest in the run
    ctx["notes"].clear()
    assert agent03_check._ladder_contract("temperature_ladder", ns, ctx) == []
    assert ctx["notes"] and all(note.startswith("unexpected: at the end of the run the copy at ") for note in ctx["notes"])


def test_ladder_name_checks_use_n_temperatures(torch_nvalchemi):
    torch = torch_nvalchemi
    ns = {"n_crystals": 4, "n_temperatures": 6, "time_fs": torch.arange(5.0)}
    syms = agent03_check._symbols(ns, "temperature_ladder")
    assert syms == {"n_structures": 4, "n_temperatures": 6, "n_saved": 5}
    spec = agent03_check.STEPS["temperature_ladder"].names["replica_temperature"]
    ns["replica_temperature"] = torch.tensor([300.0, 600.0, 900.0])
    assert agent03_check._check_name("temperature_ladder", "replica_temperature", spec, ns, syms) == [
        "temperature_ladder: replica_temperature should have shape (n_temperatures,) = (6,) in K; "
        "found shape (3,)"
    ]


def test_ladder_text_never_speaks_of_melting():
    words = " ".join(str(v) for v in agent03.MESSAGES.values()) + " " + " ".join(
        str(v) for v in agent03.LABELS["temperature_ladder"].values()
    ) + (agent03.plot_temperature_ladder.__doc__ or "") + (agent03.load_ladder_crystal.__doc__ or "")
    assert "melt" not in words.lower() and "liquid" not in words.lower()


def test_run_watch_tells_a_relaxation_from_dynamics(torch_nvalchemi):
    from nvalchemi.dynamics import LBFGS, NVTLangevin

    lbfgs, nvt = object.__new__(LBFGS), object.__new__(NVTLangevin)
    assert agent03_check._is_optimizer(lbfgs) and not agent03_check._is_optimizer(nvt)
    assert agent03_check._is_optimizer(SimpleNamespace(sub_stages=[(0, lbfgs)]))
    assert not agent03_check._is_optimizer(SimpleNamespace(sub_stages=[(0, lbfgs), (1, nvt)]))


# ---------------------------------------------------------------------------
# speed_settings: the contract and its messages
# ---------------------------------------------------------------------------


def _speed(table, roles=ROLES, watch=None):
    ctx = {"notes": [], "watch": watch}
    problems = agent03_check._speed_contract("speed_settings", {"speed_table": table, "setting_roles": roles}, ctx)
    return problems, ctx["notes"]


def test_speed_contract_passes_on_four_cases_with_two_times_and_two_roles():
    assert _speed(_proposal_table()) == ([], [])
    # column names and role keys match without regard to case; extra columns are ignored
    pd = pytest.importorskip("pandas")
    table = _proposal_table(note=["", "", "", ""]).rename(columns={"cueq": "CuEq", "repeat_s": "Repeat_s"})
    assert _speed(table, {"CUEQ": "a", "Compile": "b"}) == ([], [])
    assert isinstance(table, pd.DataFrame)


def test_speed_contract_fails_on_a_missing_or_repeated_case():
    table = _proposal_table().iloc[:3]
    assert _speed(table)[0] == ["speed_settings: speed_table is missing the case cuEquivariance on, compile on"]
    twice = _proposal_table(cueq=[False, True, False, False])
    assert _speed(twice)[0] == [
        "speed_settings: speed_table should hold each case once; found cuEquivariance off, compile on 2 times",
        "speed_settings: speed_table is missing the case cuEquivariance on, compile on",
    ]


def test_speed_contract_fails_on_missing_or_bad_times():
    assert _speed(_proposal_table().drop(columns=["first_call_s"]))[0] == [
        "speed_settings: speed_table should have the columns cueq, compile, first_call_s and repeat_s; "
        "missing first_call_s"
    ]
    nan = _proposal_table(repeat_s=[16.3, float("nan"), 14.7, 1.3])
    assert _speed(nan)[0] == [
        "speed_settings: repeat_s should be a time in s, finite and above 0, for every case; found nan "
        "for cuEquivariance on, compile off"
    ]
    zero = _proposal_table(first_call_s=[16.6, 1.8, 0.0, 2.3])
    assert "found 0.0 for cuEquivariance off, compile on" in _speed(zero)[0][0]


def test_speed_contract_fails_on_a_table_that_is_not_a_table_and_on_bad_flags():
    assert _speed([[1, 2], [3, 4]])[0] == [
        "speed_settings: speed_table should be a table (pandas DataFrame); found list of length 2"
    ]
    words = _proposal_table(cueq=["no", "yes", "no", "yes"])
    assert _speed(words)[0] == [
        "speed_settings: the cueq column of speed_table should hold True or False; found 'no'"
    ]


def test_speed_contract_fails_on_missing_roles():
    assert _speed(_proposal_table(), {})[0] == [
        "speed_settings: setting_roles should be a dict with a short text for each of the keys cueq "
        "and compile; found an empty dict"
    ]
    assert _speed(_proposal_table(), {"cueq": "kernels"})[0] == [
        "speed_settings: setting_roles should be a dict with a short text for each of the keys cueq "
        "and compile; found the keys cueq"
    ]
    assert _speed(_proposal_table(), {"cueq": "kernels", "compile": "  "})[0] == [
        "speed_settings: setting_roles should be a dict with a short text for each of the keys cueq "
        "and compile; found no text for compile"
    ]
    assert "found list of length 2" in _speed(_proposal_table(), ["a", "b"])[0][0]


def test_speed_both_slower_than_neither_is_a_note_and_the_view_prints_it(capsys, shown):
    slow = _proposal_table(repeat_s=[1.0, 0.8, 0.9, 1.5])
    problems, notes = _speed(slow)
    assert problems == [] and notes == [
        "unexpected: with cuEquivariance and compile both on, a repeat took 1.5 s, longer than the "
        "1 s with both off"
    ]
    agent03.show_speed_table(slow, ROLES)
    assert capsys.readouterr().out.strip() == notes[0]


def test_speed_notes_a_case_no_model_was_called_with():
    torch = pytest.importorskip("torch")
    seen = {1: (None, (False, False, torch.float32)), 2: (None, (True, False, torch.float32))}
    problems, notes = _speed(_proposal_table(), watch=SimpleNamespace(models=seen))
    assert problems == [] and len(notes) == 2
    assert notes[0] == (
        "unexpected: no model with cuEquivariance off, compile on was called while the file ran, so "
        "that row may come from other settings (models called: cuEquivariance off, not compiled, "
        "float32; cuEquivariance on, not compiled, float32)"
    )
    assert _speed(_proposal_table(), watch=SimpleNamespace(models={}))[1] == []  # no model seen: silent


# ---------------------------------------------------------------------------
# The reading under the force histograms; the crystals timed one by one
# ---------------------------------------------------------------------------


def test_force_histograms_print_the_reading_after_the_run(capsys, shown):
    agent03.plot_force_histograms(FORCE_START, FORCE_RELAXED, 0.05)
    assert capsys.readouterr().out.strip() == (
        "The largest force of every relaxed crystal is below the threshold (dashed line)."
    )
    relaxed = np.array(FORCE_RELAXED, dtype=float).copy()
    relaxed[:2] = 0.2
    agent03.plot_force_histograms(FORCE_START, relaxed, 0.05)
    assert capsys.readouterr().out.strip() == "unexpected: 2 relaxed crystals are above the threshold"
    assert len(shown) == 2


def test_one_by_one_timing_notes_crystals_that_are_not_of_average_size(torch_nvalchemi):
    torch = torch_nvalchemi
    crystal_batch = _batch((8, 16, 24, 32, 120))  # mean 40 atoms

    def notes(sizes):
        runs = [(None, agent03_check._start_state(crystal_batch))]  # the batched run
        runs += [(None, agent03_check._start_state(_batch((n,)))) for n in sizes]
        ns = {"crystal_batch": crystal_batch, "single_seconds": torch.tensor([1.0, 2.0, 3.0])}
        return agent03_check._single_sizes_notes(ns, SimpleNamespace(runs=runs))

    assert notes([120, 8, 40, 40, 40]) == []  # warm-up runs first; the last three are the timed ones
    assert notes([32, 40, 48]) == []
    assert notes([120, 120, 120]) == [
        "unexpected: the crystals timed one by one hold 120, 120, 120 atoms, while the batch averages "
        "40 atoms per crystal; the projected one-by-one time depends on that choice"
    ]
    assert notes([8, 8, 16])[0].startswith("unexpected: the crystals timed one by one hold 8, 8, 16 atoms")
    assert notes([40, 40]) == []  # fewer one-crystal runs than timings: silent
    assert agent03_check.STEPS["relax_run"].watch_runs


def test_rows_that_repeat_after_the_first_fail_the_ladder_and_the_md_run():
    # SnapshotHook with a HostMemory sink (Toolkit 0.3.0) saves the batch's last energy in every
    # frame; the plot then shows a step to a flat line (validation of 26 Nov 2026).
    time, temps, energy = _ladder_inputs(n_saved=4, temps=(100.0, 3000.0))
    stuck = energy.copy()
    stuck[1:] = energy[-1]
    assert agent03_check._rows_repeat("temperature_ladder", "replica_energy", {"replica_energy": stuck}) == [
        "temperature_ladder: the 3 saved rows of replica_energy after the first hold exactly the same "
        "numbers, so they were not recorded along the run; keep a copy of the energies each time a row is saved"
    ]
    assert agent03_check._rows_repeat("temperature_ladder", "replica_energy", {"replica_energy": energy}) == []
    assert agent03_check._rows_repeat("md_run", "crystal_energy", {"crystal_energy": stuck}) != []
    flat = np.full((4, 2), -5.4)  # nothing moved: the flat-line check reports it, not this one
    assert agent03_check._rows_repeat("md_run", "crystal_energy", {"crystal_energy": flat}) == []


def test_ladder_view_prints_no_order_note_for_equal_lines():
    time, temps, _ = _ladder_inputs(n_saved=10, temps=(100.0, 3000.0))
    assert agent03.ladder_data(time, temps, np.full((10, 2), -4.0979))["notes"] == []


def test_ladder_time_axis_is_a_note(torch_nvalchemi):
    # Two validation trials (26 Nov 2026) labelled the saved rows one save off; the ladder notes it.
    energy = np.zeros((4, 2))
    ctx = {"dyn_watch": _watch([2.0, 42.0, 82.0], dt=2.0)}  # rows after 1, 21, 41 steps of 2 fs, plus time 0
    on_time = {"time_fs": np.array([0.0, 2.0, 42.0, 82.0]), "replica_energy": energy}
    assert agent03_check._ladder_time_axis_notes(on_time, ctx) == []
    off = {"time_fs": np.array([0.0, 40.0, 80.0, 120.0]), "replica_energy": energy}
    assert agent03_check._ladder_time_axis_notes(off, ctx) == [
        "unexpected: time_fs should give the time of each saved energy row; the last row was "
        "recorded at 82 fs but time_fs says 120 fs"
    ]
    assert agent03_check._ladder_time_axis_notes(off, {}) == []  # no watch: silent


def test_configure_tutorial_drops_only_the_mace_default_model_line(monkeypatch, capsys):
    """MACE prints "Using medium MPA-0 model as default ..." at every model load; configure_tutorial drops that line."""

    foundations_models = pytest.importorskip("mace.calculators.foundations_models")
    monkeypatch.setattr(agent03, "_configure_tutorial", lambda: None)  # the course style is tested elsewhere

    def fake_download(model=None):
        print("Using medium MPA-0 model as default MACE-MP model, to use previous (before 3.10) default model please specify 'medium' as model argument")
        print("Cached MACE model to /tmp/x")
        return "/tmp/x"

    monkeypatch.setattr(foundations_models, "download_mace_mp_checkpoint", fake_download)
    agent03.configure_tutorial()
    agent03.configure_tutorial()  # a second call does not wrap twice
    wrapped = foundations_models.download_mace_mp_checkpoint
    assert getattr(wrapped, "_alchemi_quiet", False) and wrapped.__wrapped__ is fake_download
    assert wrapped("medium-mpa-0") == "/tmp/x"
    out = capsys.readouterr().out
    assert "Using medium MPA-0 model" not in out
    assert out == "Cached MACE model to /tmp/x\n"


def test_configure_tutorial_ignores_the_speed_step_notices_by_message_and_module(monkeypatch):
    """The two notices of the speed step (L40S run, 27 Nov) are ignored; other warnings of the same modules show."""

    import warnings

    monkeypatch.setattr(agent03, "_configure_tutorial", lambda: None)
    copy_notice = "To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone()"
    tf32_notice = (
        "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. "
        "Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
    )

    def shown(message, module):
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn_explicit(message, UserWarning, "x.py", 1, module=module, registry={})
        return [str(w.message) for w in caught]

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        agent03.configure_tutorial()
        assert shown(copy_notice, "mace.modules.models") == []
        assert shown(tf32_notice, "torch._inductor.compile_fx") == []
        assert shown("something else", "mace.modules.models") == ["something else"]
        assert shown(copy_notice, "my_own_code") == [copy_notice]
        assert shown(tf32_notice, "my_own_code") == [tf32_notice]


def test_torch_jit_notice_is_matched_by_message_and_place():
    import warnings

    from helpers.model_support import is_torch_jit_notice

    def record(message, category, filename):
        return warnings.WarningMessage(message, category, filename, 1)

    notice = "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`."
    jit_file = "/venv/lib/python3.12/site-packages/torch/jit/_script.py"
    assert is_torch_jit_notice(record(DeprecationWarning(notice), DeprecationWarning, jit_file))
    assert not is_torch_jit_notice(record(DeprecationWarning(notice), DeprecationWarning, "/my/code.py"))
    assert not is_torch_jit_notice(record(DeprecationWarning("other"), DeprecationWarning, jit_file))
    assert not is_torch_jit_notice(record(UserWarning(notice), UserWarning, jit_file))


def test_configure_tutorial_imports_mace_inside_the_course_guard():
    """The first import of mace raises torch's jit notice behind a "once" rule; the course guard
    (helpers.model_support.suppress_known_mace_load_noise) drops it and shows every other warning.

    A fresh interpreter, because the notices are raised only by the first import. The guard is
    replaced by one that raises the two warnings itself, so the test does not depend on what the
    installed mace raises.
    """

    import subprocess

    pytest.importorskip("mace.calculators.foundations_models")
    code = (
        "import contextlib, warnings\n"
        "from helpers import agent03, model_support\n"
        "guard = model_support.suppress_known_mace_load_noise\n"
        "@contextlib.contextmanager\n"
        "def noisy():\n"
        "    with guard():\n"
        "        warnings.simplefilter('once')\n"
        "        warnings.warn_explicit('`torch.jit.script` is deprecated. Please switch.', DeprecationWarning,\n"
        "                               '/venv/site-packages/torch/jit/_script.py', 1488)\n"
        "        warnings.warn_explicit('another notice', UserWarning, 'x.py', 1)\n"
        "        yield\n"
        "model_support.suppress_known_mace_load_noise = noisy\n"
        "agent03._quiet_mace_default_line()\n"
        "from mace.calculators import foundations_models\n"
        "print('wrapped', getattr(foundations_models.download_mace_mp_checkpoint, '_alchemi_quiet', False))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=_NOTEBOOK_DIR, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "wrapped True" in result.stdout
    assert "torch.jit.script" not in result.stderr
    assert "another notice" in result.stderr
