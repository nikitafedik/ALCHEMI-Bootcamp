# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent 07 helper: the notebook against its helpers, the views, the store and the step check.

No model is loaded and no GPU is needed: the step files of these tests run stand-in stages
that call their hooks the way a Toolkit stage does.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import agent07, agent07_check

LIVE_NOTEBOOK = _NOTEBOOK_DIR / "alchemi-agent-07-one-out-one-in.ipynb"
# A07_NOTEBOOK names another notebook to test (the candidate notebook of plan 3.1).
NOTEBOOK = Path(os.environ.get("A07_NOTEBOOK") or LIVE_NOTEBOOK)


# ---------------------------------------------------------------------------
# The notebook
# ---------------------------------------------------------------------------


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text())["cells"]


def _code() -> str:
    return "\n".join("".join(c["source"]) for c in _cells() if c["cell_type"] == "code")


def test_notebook_cells_have_ids_and_no_outputs():
    cells = _cells()
    assert [c["id"] for c in cells] == [f"a07-{n:02d}" for n in range(1, len(cells) + 1)]
    assert all(not c["outputs"] and c["execution_count"] is None for c in cells if c["cell_type"] == "code")


def test_notebook_calls_only_helpers_that_exist():
    called = set(re.findall(r"helpers\.([A-Za-z_]+)", _code()))
    assert called, "the notebook calls no helper"
    missing = sorted(name for name in called if not hasattr(agent07, name))
    assert not missing, f"the notebook calls helpers that do not exist: {missing}"


def test_notebook_loads_the_files_the_check_knows():
    loaded = re.findall(r"^%load agent07_work/(\w+\.py)$", _code(), flags=re.M)
    assert loaded == [agent07_check.STEPS[s].file for s in ("fused", "refill", "staged", "relax_md")]


def test_text_names_every_output_the_check_asks_for():
    text = "\n".join("".join(c["source"]) for c in _cells() if c["cell_type"] == "markdown")
    for step in agent07_check.STEPS.values():
        for name in step.names:
            assert f"`{name}`" in text, name


def test_setup_cell_and_check_agree():
    """The names the setup cell sets are the ones the check sets, with the same values."""

    setup = next("".join(c["source"]) for c in _cells() if "helpers.prepare_study" in "".join(c["source"]))
    check = agent07_check.setup_source("full")
    for name in ("batch_multiplier", "max_batch_size", "max_atoms"):
        value = re.search(rf"^{name} = (\S+)", setup, flags=re.M).group(1)
        assert re.search(rf"^{name} = {re.escape(value)}$", check, flags=re.M), name
    if "stage_settings" not in setup:
        pytest.skip(
            f"{NOTEBOOK.name} has no stage_settings line yet: it waits for the educator. "
            "Run with A07_NOTEBOOK=<candidate notebook> to test the cell the helpers are built for"
        )
    ns: dict = {}
    exec(re.search(r"^stage_settings = .*$", setup, flags=re.M).group(0), ns)
    assert ns["stage_settings"] == agent07.STAGE_SETTINGS == agent07.stage_settings()


def test_setup_source_sizes():
    full, small = agent07_check.setup_source("full"), agent07_check.setup_source("small")
    assert "batch_multiplier = 26\n" in full and "max_batch_size = 1308\n" in full and "max_atoms = 20000\n" in full
    assert "nvt_steps=600" in full and "nve_steps=200" in full and "energy_span" not in full
    assert "batch_multiplier = 0.1\n" in small and "nvt_steps=200" in small and "nve_steps=50" in small
    compile(small, "setup", "exec")
    compile(full, "setup", "exec")


def test_small_size_follows_the_environment(monkeypatch):
    monkeypatch.setenv("ALCHEMI_AGENT07_SMALL", "1")
    assert agent07_check.setup_source() == agent07_check.setup_source("small")
    monkeypatch.delenv("ALCHEMI_AGENT07_SMALL")
    assert agent07_check.setup_source() == agent07_check.setup_source("full")


# ---------------------------------------------------------------------------
# Session lines and views
# ---------------------------------------------------------------------------


def test_show_session(capsys):
    agent07.show_session()
    out = capsys.readouterr().out
    assert "notebooks/alchemi-agent-07-one-out-one-in.ipynb" in out
    assert "python -m helpers.agent07 check <step>" in out
    assert "notebooks/agent07_work/" in out
    lines = out.splitlines()
    check = lines.index("  check with  python -m helpers.agent07 check <step>")
    assert lines[check + 1] == "  each file   runs in a notebook cell (%load), after the earlier files"
    assert lines[check + 2] == "  always      run the check from notebooks/ after you write a file, and fix"
    assert lines[check + 3] == "              what it reports; <step> is the file name without .py"


def test_every_step_is_its_file_name_without_py():
    """The session line says <step> is the file name without .py."""

    assert all(spec.file == f"{step}.py" for step, spec in agent07_check.STEPS.items())


def test_work_dir_is_made():
    assert agent07.make_work_dir() == _NOTEBOOK_DIR / "agent07_work"
    assert agent07.WORK_DIR.is_dir()


stage_trace = torch.tensor([[120, 300], [540, 700], [200, 420]])
relax_md_trace = torch.tensor([[30, 130], [75, 175]])


def test_stage_changes_names_the_stages_by_the_trace(capsys):
    agent07.show_stage_changes(stage_trace)
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["trajectories", "3"]
    assert out[1].startswith("earliest switch, NVT to NVE") and out[1].endswith("after 120 MD steps")
    assert out[2].startswith("latest switch, NVT to NVE") and out[2].endswith("after 540 MD steps")
    assert out[3].endswith("by 420 MD steps")
    assert "batch" not in "\n".join(out)

    agent07.show_stage_changes(relax_md_trace)
    out = capsys.readouterr().out
    assert "earliest switch, relaxation to MD" in out and "after 30 steps" in out
    assert "NVT" not in out and "MD steps" not in out


def test_stage_changes_other_names(capsys):
    other = torch.tensor([[5, 9], [7, 9]])
    agent07.show_stage_changes(other)
    assert "first stage to second stage" in capsys.readouterr().out
    agent07.show_stage_changes(other, stages=("relaxation", "MD"))
    assert "relaxation to MD" in capsys.readouterr().out


def test_stage_changes_marks_one_step_as_unexpected(capsys):
    agent07.show_stage_changes(torch.tensor([[50, 90], [50, 80]]))
    assert "Unexpected" in capsys.readouterr().out


def test_stage_changes_rejects_another_shape():
    with pytest.raises(ValueError, match="one row per trajectory and two columns"):
        agent07.show_stage_changes(torch.zeros(4))


def _batch(ids, atoms=6, **extra):
    ids = list(ids)
    return SimpleNamespace(
        simulation_id=torch.tensor(ids).reshape(-1, 1),
        num_graphs=len(ids),
        num_nodes=atoms * len(ids),
        num_nodes_per_graph=torch.full((len(ids),), atoms),
        **extra,
    )


class _Store:
    """A stand-in store with what the views read."""

    def __init__(self, ids, admitted):
        self._ids, self.admitted_inflight = list(ids), admitted

    def read(self):
        return _batch(self._ids)


def test_show_refill_counts(capsys):
    table = agent07.show_refill(_Store(range(10), 4), [None] * 10)
    assert list(table.columns) == ["Finished", "Admitted inflight"]
    assert table.iloc[0].tolist() == [10, 4]
    assert capsys.readouterr().out == ""
    agent07.show_refill(_Store(range(10), 0), [None] * 10)
    assert "Unexpected" in capsys.readouterr().out
    table = agent07.show_refill([_batch(range(10))], [None] * 10)
    assert table.iloc[0].tolist() == [10, "not recorded"]


def test_show_staged(capsys):
    table = agent07.show_staged([_batch(range(5)), _batch(range(5, 10))], [None] * 10)
    assert table.iloc[0].tolist() == [2, 10, 10]
    assert capsys.readouterr().out == ""
    agent07.show_staged([_batch(range(5))], [None] * 10)
    assert "5 of 10" in capsys.readouterr().out


def test_compare_routes(capsys):
    table = agent07.compare_routes(
        staged=([_batch(range(5)), _batch(range(5, 10))], 30.04),
        fused=([_batch(range(10))], 20.0),
        inflight=(_Store(range(10), 6), 10.0),
    )
    assert list(table.columns) == ["Route", "Finished", "Admitted inflight", "Elapsed [s]"]
    assert table["Route"].tolist() == ["Run in stages", "Fused stages", "Fused stages + refill"]
    assert table["Finished"].tolist() == [10, 10, 10]
    assert table["Admitted inflight"].tolist() == [0, 0, 6]
    assert table["Elapsed [s]"].tolist() == [30.0, 20.0, 10.0]
    assert capsys.readouterr().out == ""


def test_tables_show_without_the_index_column():
    """Render check of 27 Nov: the views' tables had a bold 0, 1, 2 index column."""

    tables = [
        agent07.show_refill(_Store(range(3), 1), [None] * 3),
        agent07.show_staged([_batch(range(3))], [None] * 3),
        agent07.compare_routes(staged=([_batch(range(3))], 1.0), fused=([_batch(range(3))], 1.0),
                               inflight=(_Store(range(3), 1), 1.0)),
    ]
    for table in tables:
        assert "<th>0</th>" not in table._repr_html_() and "<tbody>" in table._repr_html_()
        assert not repr(table).splitlines()[1].startswith("0 ")
        assert isinstance(table.iloc[[0]], type(table))


def test_compare_routes_marks_unexpected(capsys):
    agent07.compare_routes(
        staged=([_batch(range(9))], 1.0), fused=([_batch(range(10))], 1.0), inflight=(_Store(range(10), 0), 1.0)
    )
    out = capsys.readouterr().out
    assert "did not finish the same number" in out and "no trajectory entered" in out


def test_views_reject_a_count_in_place_of_batches():
    with pytest.raises(TypeError, match="finished batches of a route"):
        agent07.compare_routes(staged=(7, 1.0), fused=(7, 1.0), inflight=(_Store(range(3), 1), 1.0))


# ---------------------------------------------------------------------------
# The store, with real Toolkit batches (no model)
# ---------------------------------------------------------------------------


def _real_batch(ids):
    from nvalchemi.data import AtomicData, Batch

    items = []
    for i in ids:
        item = AtomicData(positions=torch.rand(3, 3), atomic_numbers=torch.tensor([8, 1, 1]))
        item.add_system_property("simulation_id", torch.tensor([[i]], dtype=torch.long))
        items.append(item)
    return Batch.from_data_list(items)


def test_store_works_on_cpu_and_counts_late_entries():
    store = agent07.CudaCompletionSink(capacity=6, device="cpu")
    first = _real_batch([0, 1, 2])
    store.write(first, mask=torch.tensor([True, False, False]))
    assert store.admitted_inflight == 0
    second = _real_batch([1, 2, 3])
    store.write(second, mask=torch.tensor([True, True, True]))
    store.write(_real_batch([4, 5]), mask=torch.tensor([True, True]))
    assert len(store) == 6
    assert store.admitted_inflight == 3
    assert store.occupancy == [(3, 9), (3, 9), (2, 6)]
    assert sorted(store.read().simulation_id.reshape(-1).tolist()) == [0, 1, 2, 3, 4, 5]
    table = agent07.show_refill(store, [None] * 6)
    assert table.iloc[0].tolist() == [6, 3]
    store.zero()
    assert len(store) == 0 and store.admitted_inflight is None and store.occupancy == []


def test_store_device_defaults_to_the_run_device():
    store = agent07.CudaCompletionSink(capacity=2)
    assert store._device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    store.write(_real_batch([0, 1]), mask=torch.tensor([True, True]))
    assert len(store) == 2


def test_store_refuses_more_than_its_capacity():
    store = agent07.CudaCompletionSink(capacity=1, device="cpu")
    with pytest.raises(RuntimeError, match="capacity"):
        store.write(_real_batch([0, 1]), mask=torch.tensor([True, True]))


# ---------------------------------------------------------------------------
# The step check, on stand-in stages
# ---------------------------------------------------------------------------

TOTAL, ATOMS = 12, 10

_FAKES = '''
import torch
from types import SimpleNamespace


def fake_batch(ids):
    ids = list(ids)
    return SimpleNamespace(simulation_id=torch.tensor(ids).reshape(-1, 1), num_graphs=len(ids),
                           num_nodes=10 * len(ids), num_nodes_per_graph=torch.full((len(ids),), 10))


class FakeStage:
    def __init__(self, n_steps):
        self.n_steps, self.hooks = n_steps, []

    def register_hook(self, hook):
        self.hooks.append(hook)

    def run(self, batch, steps=1, active=None):
        ctx = SimpleNamespace(batch=batch, active_graph_mask=active)
        for _ in range(steps):
            for hook in self.hooks:
                hook(ctx, None)
        return batch


def groups():
    return [range(start, min(start + max_batch_size, len(trajectories)))
            for start in range(0, len(trajectories), max_batch_size)]
'''

FUSED = _FAKES + '''
def build_stages():
    return FakeStage(stage_settings["nvt_steps"]), FakeStage(stage_settings["nve_steps"])


fused_batches = [fake_batch(g) for g in groups()]
fused_seconds = 2.5
stage_trace = torch.stack([torch.arange(len(trajectories)) * 10 + 100,
                           torch.arange(len(trajectories)) * 10 + 300], dim=1)
'''

STAGED = '''
staged_batches = []
for g in groups():
    nvt, nve = build_stages()
    b = fake_batch(g)
    nvt.run(b, steps=3)
    nve.run(b, steps=2)
    staged_batches.append(b)
staged_seconds = 4.0
'''

REFILL = '''
nvt, nve = build_stages()
order = list(range(len(trajectories)))
active = order[:max_batch_size]
waiting = order[max_batch_size:]
while active:
    b = fake_batch(active)
    nvt.run(b, steps=3)
    nve.run(b, steps=2)
    active = active[1:] + waiting[:1]      # one out, one in
    waiting = waiting[1:]
completed = [fake_batch(order)]
inflight_seconds = 3.0
'''

RELAX_MD = _FAKES + '''
relax_md_batches = [fake_batch(g) for g in groups()]
relax_md_trace = torch.stack([torch.arange(len(trajectories)) + 20,
                              torch.arange(len(trajectories)) + 120], dim=1)
'''

GOOD = {"fused": FUSED, "staged": STAGED, "refill": REFILL, "relax_md": RELAX_MD}


def _namespace(total=TOTAL, max_batch_size=4, max_atoms=40):
    def item(i):
        value = SimpleNamespace(
            num_nodes=ATOMS,
            positions=torch.full((ATOMS, 3), float(i)),
            velocities=torch.zeros(ATOMS, 3),
            status=torch.zeros(1, 1, dtype=torch.long),
        )
        return value

    return {
        "__name__": "__main__",
        "trajectories": tuple(item(i) for i in range(total)),
        "max_batch_size": max_batch_size,
        "max_atoms": max_atoms,
        "stage_settings": dict(agent07.STAGE_SETTINGS),
    }


def _run(tmp_path, step, files=None, **ns_kwargs):
    for name, text in {**GOOD, **(files or {})}.items():
        if text is not None:
            (tmp_path / agent07_check.STEPS[name].file).write_text(text)
    return agent07_check.check_namespace(step, _namespace(**ns_kwargs), tmp_path)


@pytest.mark.parametrize("step", ["fused", "refill", "staged", "relax_md"])
def test_check_passes_on_a_good_file(tmp_path, step):
    problems, notes = _run(tmp_path, step)
    assert problems == []
    assert [n for n in notes if "not the course store" not in n] == []


def test_staged_passes_without_a_status_column(tmp_path):
    """A staged run written with plain engines has no ``status``; the check counts the steps."""

    assert "status" not in STAGED
    assert _run(tmp_path, "staged") == ([], [])


def _one(tmp_path, step, files, expected, **ns_kwargs):
    problems, _ = _run(tmp_path, step, files, **ns_kwargs)
    assert len(problems) == 1, problems
    assert expected in problems[0], problems[0]
    assert problems[0].startswith(("fused: ", "refill: ", "staged: ", "relax_md: "))


def test_missing_file(tmp_path):
    (tmp_path / "fused.py").write_text(FUSED)
    problems, _ = agent07_check.check_namespace("staged", _namespace(), tmp_path)
    assert problems == [f"staged: staged.py is missing in {tmp_path}"]


def test_file_that_stops(tmp_path):
    _one(tmp_path, "fused", {"fused": "raise ValueError('boom')"}, "fused.py stops with ValueError: boom")


def test_a_file_that_uses_dunder_file(tmp_path):
    """A %load cell has no __file__; the check names the line to remove."""

    _one(tmp_path, "fused", {"fused": "from pathlib import Path\nhere = Path(__file__).parent\n" + FUSED},
         "fused: fused.py uses __file__, which does not exist in a notebook cell; remove that line")


def test_missing_name(tmp_path):
    problems, _ = _run(tmp_path, "fused", {"fused": FUSED + "\ndel fused_seconds, stage_trace\n"})
    assert problems == ["fused: fused_seconds is missing", "fused: stage_trace is missing"]


def test_fused_batches_as_a_count(tmp_path):
    _one(tmp_path, "fused", {"fused": FUSED + "\nfused_batches = 3\n"}, "fused_batches must be the finished batches")


def test_a_store_whose_read_stops(tmp_path):
    """Validation of 26 Nov: Toolkit's HostMemory.read() stopped on a size mismatch, and the
    check said "got HostMemory" as if it were no store at all."""

    broken = RELAX_MD + (
        "\nclass Store:\n"
        "    def read(self):\n"
        "        raise RuntimeError('Sizes of tensors must match')\n"
        "relax_md_batches = Store()\n"
    )
    _one(tmp_path, "relax_md", {"relax_md": broken},
         "relax_md_batches is a store (Store), but its read() stops with RuntimeError: Sizes of tensors must match")


def test_fused_batches_not_once_each(tmp_path):
    _one(tmp_path, "fused", {"fused": FUSED + "\nfused_batches = fused_batches[:2]\n"},
         "fused_batches holds 8 trajectories, 8 distinct; expected 12, each once")


def test_fused_in_one_batch_is_over_capacity(tmp_path):
    _one(tmp_path, "fused", {"fused": FUSED + "\nfused_batches = [fake_batch(range(len(trajectories)))]\n"},
         "one batch holds 12 trajectories and 120 atoms; the capacity is max_batch_size = 4 and max_atoms = 40")


def test_seconds_must_be_a_number(tmp_path):
    _one(tmp_path, "fused", {"fused": FUSED + "\nfused_seconds = 'fast'\n"}, "fused_seconds must be the run time in seconds")
    _one(tmp_path, "staged", {"staged": STAGED + "\nstaged_seconds = 0\n"}, "staged_seconds must be the run time in seconds")


def test_trace_shape_and_order(tmp_path):
    _one(tmp_path, "fused", {"fused": FUSED + "\nstage_trace = stage_trace[:5]\n"}, "stage_trace has shape (5, 2), expected (12, 2)")
    _one(tmp_path, "fused", {"fused": FUSED + "\nstage_trace = stage_trace.flip(1)\n"}, "12 a finish step before the switch step (row 0: switch 300, finish 100)")
    _one(tmp_path, "relax_md", {"relax_md": RELAX_MD + "\nrelax_md_trace = relax_md_trace.T\n"}, "relax_md_trace has shape (2, 12)")


def test_a_stage_left_before_its_window(tmp_path):
    """A rule on one step's temperature lets a trajectory leave NVT before `window` steps
    (validation of 26 Nov: the owner's first fused trial switched one trajectory after 8 MD
    steps with window 50). A mean over the window cannot pass that soon."""

    _one(tmp_path, "fused", {"fused": FUSED + "\nstage_trace[3] = torch.tensor([8, 208])\n"},
         "1 of 12 trajectories left NVT after fewer than `window` = 100 MD steps in it (row 3: 8)")
    # counted from 0, the earliest pass of a mean rule is window - 1: passes
    problems, _ = _run(tmp_path, "fused", {"fused": FUSED + "\nstage_trace[3] = torch.tensor([99, 299])\n"})
    assert problems == []


def test_nve_runs_its_fixed_length(tmp_path):
    """NVE has no stopping rule: every trajectory runs `nve_steps` MD steps in it (decision of
    26 Nov, "Fixed NVE length"). One step of slack either way for traces that count from 0."""

    _one(tmp_path, "fused", {"fused": FUSED + "\nstage_trace[0, 1] = stage_trace[0, 0] + 20\n"},
         "1 of 12 trajectories ran NVE for other than `nve_steps` = 200 MD steps (row 0: 20)")
    _one(tmp_path, "fused", {"fused": FUSED + "\nstage_trace[2:5, 1] = stage_trace[2:5, 0] + 500\n"},
         "3 of 12 trajectories ran NVE for other than `nve_steps` = 200 MD steps (row 2: 500)")
    for length in (199, 201):
        problems, notes = _run(tmp_path, "fused", {"fused": FUSED + f"\nstage_trace[:, 1] = stage_trace[:, 0] + {length}\n"})
        assert problems == [] and notes == []


def test_build_stages_must_return_a_pair(tmp_path):
    _one(tmp_path, "fused", {"fused": FUSED + "\ndef build_stages():\n    return FakeStage(1)\n"}, "build_stages() must take no arguments")
    _one(tmp_path, "fused", {"fused": FUSED + "\ndef build_stages(model):\n    return 1, 2\n"}, "build_stages() must take no arguments")


def test_a_route_must_not_change_the_inputs(tmp_path):
    _one(tmp_path, "staged", {"staged": STAGED + "\ntrajectories[0].status += 2\n"}, "staged.py changed `trajectories`")


def test_a_route_must_use_build_stages(tmp_path):
    _one(tmp_path, "staged", {"staged": "staged_batches = [fake_batch(g) for g in groups()]\nstaged_seconds = 1.0\n"},
         "staged.py did not call build_stages() from step 1")


def test_staged_run_that_skips_the_second_stage(tmp_path):
    _one(tmp_path, "staged", {"staged": STAGED.replace("    nve.run(b, steps=2)\n", "")},
         "12 of 12 trajectories did not take a step in both stages (0 without NVT, 12 without NVE)")


def test_staged_run_with_inactive_trajectories(tmp_path):
    """A second stage that treats every trajectory as done counts as not run."""

    broken = STAGED.replace("nve.run(b, steps=2)", "nve.run(b, steps=2, active=torch.zeros(b.num_graphs, dtype=torch.bool))")
    _one(tmp_path, "staged", {"staged": broken}, "12 without NVE")


def test_refill_over_capacity(tmp_path):
    broken = REFILL.replace("active = order[:max_batch_size]", "active = order[:max_batch_size + 2]").replace(
        "waiting = order[max_batch_size:]", "waiting = order[max_batch_size + 2:]")
    _one(tmp_path, "refill", {"refill": broken}, "the active batch held up to 6 trajectories and 60 atoms")


def test_refill_that_did_not_refill(tmp_path):
    """Fixed groups written to a store are not a refill."""

    broken = (
        "nvt, nve = build_stages()\n"
        "for g in groups():\n"
        "    b = fake_batch(g)\n"
        "    nvt.run(b, steps=3)\n"
        "    nve.run(b, steps=2)\n"
        "completed = [fake_batch(range(len(trajectories)))]\n"
        "inflight_seconds = 3.0\n"
    )
    _one(tmp_path, "refill", {"refill": broken}, "the run did not refill")


def test_notes_are_unexpected_and_pass(tmp_path):
    same = FUSED + "\nstage_trace = torch.tensor([[150, 350]] * len(trajectories))\n"
    problems, notes = _run(tmp_path, "fused", {"fused": same})
    assert problems == [] and len(notes) == 1 and "every trajectory switched at step 150: unexpected" in notes[0]

    capped = FUSED + "\nstage_trace[:3, 0] = 600\nstage_trace[:3, 1] = 800\n"
    problems, notes = _run(tmp_path, "fused", {"fused": capped})
    assert problems == [] and notes == [
        "fused: 3 of 12 trajectories left NVT at its step cap of 600 without passing the rule: unexpected"
    ]

    problems, notes = _run(tmp_path, "refill", max_batch_size=64, max_atoms=834)
    assert problems == [] and any("fit in the first batch" in n for n in notes)

    other_cap = FUSED.replace('FakeStage(stage_settings["nvt_steps"])', "FakeStage(2000)")
    problems, notes = _run(tmp_path, "fused", {"fused": other_cap})
    assert problems == [] and any("gives NVT a step cap of 2000; stage_settings has 600" in n for n in notes)

    other_length = FUSED.replace('FakeStage(stage_settings["nve_steps"])', "FakeStage(500)")
    problems, notes = _run(tmp_path, "fused", {"fused": other_length})
    assert problems == [] and notes == [
        "fused: build_stages() gives NVE a length of 500 steps; stage_settings has nve_steps = 200"
    ]


def test_fixed_nve_length_is_not_unexpected(tmp_path, monkeypatch):
    """Every trajectory ends NVE at `nve_steps`: that is the design, not a cap, so no note."""

    settings = dict(agent07.STAGE_SETTINGS, nve_steps=2)  # the stand-in runs NVE 2 steps per pass
    monkeypatch.setattr(agent07, "STAGE_SETTINGS", settings)
    for step in ("fused", "refill"):
        fused = FUSED.replace("* 10 + 300", "* 10 + 102")
        problems, notes = _run(tmp_path, step, {"fused": fused})
        assert problems == [], (step, problems)
        assert not [n for n in notes if "NVE" in n or "unexpected" in n], (step, notes)


def test_check_follows_the_study_size(tmp_path):
    """Nothing in the check is fixed to 410, 64 or 834."""

    for total, cap in ((5, 2), (30, 7)):
        for step in GOOD:
            problems, _ = _run(tmp_path, step, total=total, max_batch_size=cap, max_atoms=cap * ATOMS)
            assert problems == [], (total, cap, step, problems)
    source = Path(agent07_check.__file__).read_text()
    assert "410" not in source and "TOTAL" not in source


def test_main(monkeypatch, capsys):
    assert agent07_check.main(["check"]) == 2
    assert agent07_check.main(["check", "nothing"]) == 2
    assert "steps: fused, refill, staged, relax_md" in capsys.readouterr().out
    monkeypatch.setattr(agent07_check, "check_step", lambda step: ([], ["fused: a note"]))
    assert agent07_check.main(["check", "fused"]) == 0
    assert capsys.readouterr().out == "fused: OK\nfused: a note\n"
    monkeypatch.setattr(agent07_check, "check_step", lambda step: (["fused: a problem"], []))
    assert agent07_check.main(["check", "agent07_work/fused.py"]) == 1
    assert capsys.readouterr().out == "fused: a problem\n"


def test_every_line_is_in_one_place():
    """Lines a learner or an agent reads live in LINES of the two modules."""

    for module in (agent07, agent07_check):
        assert all(isinstance(v, str) and v for v in module.LINES.values())
