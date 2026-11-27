# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent 02 helper and notebook: the step contracts match the notebook's tables, no GPU needed."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import agent02, agent02_checks
from helpers import agent02_text as T

# A02_NOTEBOOK points the notebook tests at another copy (revision 26 Nov: the candidate, before it is promoted)
NOTEBOOK = Path(os.environ.get("A02_NOTEBOOK") or _NOTEBOOK_DIR / "alchemi-agent-02-data-and-batching.ipynb")

# steps that have a check but no cells in the notebook: build_routes while it waits for the educator (a02q-build-routes);
# once its cells are in, the step it replaces (own_batch, PROVISIONAL switch in agent02_text.py)
_HAS_BUILD_ROUTES = "agent02_work/build_routes.py" in NOTEBOOK.read_text()
NOT_IN_NOTEBOOK_YET = {"own_batch"} if _HAS_BUILD_ROUTES and T.PROVISIONAL_BUILD_ROUTES_REPLACES_OWN_BATCH else {"build_routes"}


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text())["cells"]


def _tables() -> dict[str, list[str]]:
    """File stem -> names in the output row of the step table that names it."""

    found: dict[str, list[str]] = {}
    for cell in _cells():
        text = "".join(cell["source"])
        for row in re.finditer(r"\| output\s*\| (.*?), in `agent02_work/(\w+)\.py` \|", text):
            found[row.group(2)] = re.findall(r"`(\w+)`", row.group(1))
    return found


def test_every_table_matches_its_contract():
    tables = _tables()
    assert set(tables) | NOT_IN_NOTEBOOK_YET == set(agent02_checks.STEPS)
    for step, names in tables.items():
        assert tuple(names) == agent02_checks.STEPS[step].names, step


def test_load_cells_name_the_contract_files():
    loads = [
        "".join(c["source"]).strip().removeprefix("%load agent02_work/")
        for c in _cells()
        if c["cell_type"] == "code" and "".join(c["source"]).startswith("%load")
    ]
    assert loads == [agent02_checks.STEPS[s].file for s in agent02_checks.STEPS if s not in NOT_IN_NOTEBOOK_YET]


def test_setup_cell_has_the_four_planned_lines():
    setup = next("".join(c["source"]) for c in _cells() if "make_work_dir" in "".join(c["source"]))
    lines = [ln.split("#")[0].strip() for ln in setup.splitlines()]
    planned = [
        "from helpers import agent02 as helpers",
        "helpers.configure_tutorial()",
        "work_dir = helpers.make_work_dir()",
        "helpers.show_session()",
    ]
    if _HAS_BUILD_ROUTES:  # its input row names n_structures (revision 26 Nov)
        planned.insert(3, f"n_structures = {T.PROVISIONAL_N_STRUCTURES}")
    assert lines == planned
    # the check runs the same setup, without the line that only prints
    assert [ln for ln in agent02_checks.SETUP.splitlines()] == [ln for ln in planned if ln != "helpers.show_session()"] or not _HAS_BUILD_ROUTES


def test_check_setup_defines_n_structures_as_the_provisional_size(tmp_path, monkeypatch, capsys):
    """A build_routes file that uses n_structures without defining it passes the check (the setup cell defines it)."""

    monkeypatch.setattr(agent02_checks, "STEPS", {
        "a": agent02_checks.Step("a.py", ("x",)),
        "build_routes": agent02_checks.Step("build_routes.py", ("build_table",), after=("a",)),
    })
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "build_routes.py", f"assert n_structures == {T.PROVISIONAL_N_STRUCTURES}\nimport pandas as pd\nbuild_table = pd.DataFrame({{'route': ['CPU then moved', 'directly on the GPU', 'tensors'], 'best_s': [0.05, 0.08, 0.008]}})")
    assert agent02_checks.run_check("build_routes", tmp_path) == 0
    assert "build_routes: OK" in capsys.readouterr().out


def test_show_session_text(capsys):
    agent02.show_session()
    out = capsys.readouterr().out
    assert "check with  python -m helpers.agent02 check <step>" in out
    assert out.startswith("Give these to your agent when you start the session:")
    lines = out.splitlines()
    assert lines[-3:] == [
        "  each file   runs in a notebook cell (%load), after the earlier files",
        "  always      run the check from notebooks/ after you write a file, and fix",
        "              what it reports; <step> is the file name without .py",
    ]
    values = [ln[14:] for ln in lines[1:]]  # every value starts in the same column as the lines above
    assert all(v and not v.startswith(" ") and ln[:2] == "  " for v, ln in zip(values, lines[1:]))
    assert lines[-1][:14] == " " * 14  # the last line continues the one above it


def test_every_step_name_is_its_file_name_without_py():
    for step, spec in agent02_checks.STEPS.items():
        assert spec.file == f"{step}.py", step


def test_after_lists_follow_notebook_order():
    order = list(agent02_checks.STEPS)
    for step, spec in agent02_checks.STEPS.items():
        assert all(order.index(a) < order.index(step) for a in spec.after), step
        assert "my_batch" not in spec.after  # the optional Try it is never required


def test_lesson_molecules_have_147_atoms():
    pytest.importorskip("rdkit")
    counts = [len(agent02.load_molecule(name)) for name in agent02.MOLECULES]
    assert counts == [24, 21, 33, 20, 49]
    assert sum(counts) == 147


def test_unknown_molecule_is_refused():
    pytest.importorskip("rdkit")
    with pytest.raises(ValueError, match="Unknown molecule"):
        agent02.load_molecule("water")


def test_check_accepts_names_paths_and_numbers(tmp_path, capsys):
    """An unknown step is reported; a number, a stem, a file name and a path all name a step."""

    assert agent02_checks.run_check("nonsense", tmp_path) == 2
    assert "unknown step 'nonsense'" in capsys.readouterr().out
    for spelled, expected in (("2", "batch"), ("batch", "batch"), ("batch.py", "batch"), ("agent02_work/batch.py", "batch")):
        assert agent02_checks.resolve_step(spelled) == expected
    assert agent02_checks.resolve_step("99") == "99"
    # PROVISIONAL (a02q-build-routes): build_routes replaces own_batch, so step 4 is build_routes and step 6 the store
    if T.PROVISIONAL_BUILD_ROUTES_REPLACES_OWN_BATCH:
        assert [agent02_checks.resolve_step(str(n)) for n in range(1, 8)] == ["atomic_data", "batch", "my_batch", "build_routes", "model_call", "zarr_roundtrip", "7"]
    else:
        assert [agent02_checks.resolve_step(str(n)) for n in range(1, 8)] == list(agent02_checks.STEPS)
    assert agent02_checks.resolve_step("own_batch") == "own_batch"  # its check stays, by name


def _light_steps(monkeypatch):
    """Three light steps: a, an optional step opt, and c, which comes after both."""

    steps = {
        "a": agent02_checks.Step("a.py", ("x",)),
        "opt": agent02_checks.Step("opt.py", ("y",), after=("a",)),
        "c": agent02_checks.Step("c.py", ("z",), after=("a",)),
    }
    monkeypatch.setattr(agent02_checks, "STEPS", steps)
    monkeypatch.setattr(agent02_checks, "OPTIONAL_STEPS", ("opt",))


def _write(root: Path, name: str, text: str) -> None:
    (root / "agent02_work").mkdir(exist_ok=True)
    (root / "agent02_work" / name).write_text(text)


def test_optional_step_runs_before_later_steps_when_it_exists(tmp_path, monkeypatch, capsys):
    _light_steps(monkeypatch)
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "opt.py", "y = 2\ndef helper():\n    return 3")
    _write(tmp_path, "c.py", "z = helper()")
    assert agent02_checks.run_check("c", tmp_path) == 0
    assert "c: OK" in capsys.readouterr().out


def test_error_in_the_optional_file_is_not_reported_for_later_steps(tmp_path, monkeypatch, capsys):
    _light_steps(monkeypatch)
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "opt.py", "raise RuntimeError('half written')")
    _write(tmp_path, "c.py", "z = 3")
    assert agent02_checks.run_check("c", tmp_path) == 0
    assert "half written" not in capsys.readouterr().out


def test_missing_optional_file_gives_the_name_error_line(tmp_path, monkeypatch, capsys):
    _light_steps(monkeypatch)
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "c.py", "z = helper()")
    assert agent02_checks.run_check("c", tmp_path) == 1
    assert "c: c.py uses helper, which is not defined in an earlier file; define it in this file" in capsys.readouterr().out


def test_a_file_that_uses_dunder_file_is_told_to_remove_that_line(tmp_path, monkeypatch, capsys):
    _light_steps(monkeypatch)
    _write(tmp_path, "a.py", "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nx = 1")
    assert agent02_checks.run_check("a", tmp_path) == 1
    out = capsys.readouterr().out
    assert "a: a.py uses __file__, which does not exist in a notebook cell; remove that line" in out
    assert "define it in this file" not in out


def test_build_table_check(capsys):
    pd = pytest.importorskip("pandas")
    problems: list[str] = []

    def add(key, **fields):
        problems.append(T.line(key, step="build_routes", **fields))

    good = pd.DataFrame({"route": ["CPU then moved", "directly on the GPU", "tensors"], "mean_s": [0.5, 0.4, 0.1], "std_s": [0.01, 0.01, 0.0]})
    agent02_checks._check_build_table(good, add)
    assert problems == []
    agent02_checks._check_build_table(good.head(2), add)
    agent02_checks._check_build_table(pd.DataFrame({"route": ["a", "b", "c"]}), add)
    agent02_checks._check_build_table(pd.DataFrame({"mean_s": [0.5, 0.4, 0.1]}), add)
    assert problems == [
        "build_routes: build_table has 2 rows; one row for each of the three routes gives at least 3",
        "build_routes: build_table has no column of times (numbers above zero)",
        "build_routes: build_table does not name its three routes in a text column",
    ]


def _table(pd, cpu, gpu, tensors, names=True):
    routes = ["ASE objects on the CPU, then .to(device)", "ASE objects directly on the GPU", "tensors, Batch.from_raw_dicts"]
    if not names:
        routes = ["one", "two", "three"]
    return pd.DataFrame({"route": routes, "best_s": [cpu, gpu, tensors], "worst_s": [cpu * 1.1, gpu * 1.1, tensors * 1.1]})


def test_build_route_order_usual_unexpected_and_unnamed():
    pd = pytest.importorskip("pandas")
    usual = agent02_checks.build_route_order(_table(pd, 0.05, 0.08, 0.008))
    assert usual["status"] == "usual" and usual["marked"] == []
    swapped = agent02_checks.build_route_order(_table(pd, 0.08, 0.05, 0.008))
    assert swapped["status"] == "unexpected" and swapped["marked"] == ["cpu_then_moved", "direct_gpu"]
    slow_tensors = agent02_checks.build_route_order(_table(pd, 0.05, 0.08, 0.1))
    assert slow_tensors["status"] == "unexpected" and slow_tensors["marked"] == ["cpu_then_moved", "tensors"]
    assert agent02_checks.build_route_order(_table(pd, 0.05, 0.08, 0.008, names=False))["status"] == "unnamed"


def test_unexpected_order_is_a_note_and_the_check_still_passes(tmp_path, monkeypatch, capsys):
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(agent02_checks, "STEPS", {
        "a": agent02_checks.Step("a.py", ("x",)),
        "build_routes": agent02_checks.Step("build_routes.py", ("build_table",), after=("a",)),
    })
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "build_routes.py", "import pandas as pd\nbuild_table = pd.DataFrame({'route': ['CPU then moved', 'directly on the GPU', 'tensors'], 'best_s': [0.08, 0.05, 0.008]})")
    assert agent02_checks.run_check("build_routes", tmp_path) == 0
    out = capsys.readouterr().out
    assert T.line("order_note", step="build_routes") in out and "build_routes: OK" in out


def test_show_build_routes_marks_rows_and_names_the_order(monkeypatch):
    pd = pytest.importorskip("pandas")
    shown = []
    import IPython.display as ipy_display

    monkeypatch.setattr(ipy_display, "display", lambda *objects, **kw: shown.extend(objects))
    agent02.show_build_routes(_table(pd, 0.05, 0.08, 0.008))
    assert "(unexpected)" not in shown[0].data and "as usual" in shown[1].data
    shown.clear()
    agent02.show_build_routes(_table(pd, 0.08, 0.05, 0.008))
    assert "(unexpected)" in shown[0].data and "unexpected" in shown[1].data
    shown.clear()
    agent02.show_build_routes(_table(pd, 0.05, 0.08, 0.008, names=False))
    assert shown[0].data == T.CHECK_LINES["table_unnamed"].split(": ", 1)[1]


def test_build_route_times_within_ten_percent_are_a_tie_shown_as_unexpected():
    pd = pytest.importorskip("pandas")
    tie = agent02_checks.build_route_order(_table(pd, 0.050, 0.054, 0.008))  # CPU then moved vs directly on the GPU: 7 % apart
    assert tie["status"] == "unexpected" and tie["marked"] == ["cpu_then_moved", "direct_gpu"]
    clear = agent02_checks.build_route_order(_table(pd, 0.050, 0.060, 0.008))  # 17 % apart
    assert clear["status"] == "usual"


def test_saved_defaults_load_and_fit_the_model_elements():
    pytest.importorskip("ase")
    aimnet2_elements = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"}
    molecule = agent02.load_default_molecule()
    assert len(molecule) == 26 and molecule.get_chemical_formula() == "C10H13NO2"
    five = {agent02.load_molecule(name).get_chemical_formula() for name in agent02.MOLECULES}
    assert molecule.get_chemical_formula() not in five
    default_set = agent02.load_default_set()
    assert [len(a) for a in default_set] == [9, 11, 8]
    for atoms in [molecule, *default_set]:
        assert set(atoms.get_chemical_symbols()) <= aimnet2_elements


# ---------------------------------------------------------------------------
# Owner round: any molecule order, an appended store, the field table, the text module
# ---------------------------------------------------------------------------
def _five_batch():
    pytest.importorskip("rdkit")
    import torch
    from nvalchemi.data import AtomicData, Batch

    data = [AtomicData.from_atoms(agent02.load_molecule(name)) for name in agent02.MOLECULES]
    batch = Batch.from_data_list(data)
    batch.energy = torch.arange(5, dtype=torch.float64).reshape(5, 1) * -1000.5
    batch.forces = torch.randn(batch.num_nodes, 3)
    return batch


def _store(tmp_path, name, batch, appends=0):
    from nvalchemi.data.datapipes import AtomicDataZarrReader, AtomicDataZarrWriter, Dataset

    path = str(tmp_path / name)
    AtomicDataZarrWriter(path).write(batch)
    for _ in range(appends):
        AtomicDataZarrWriter(path).append(batch)
    dataset = Dataset(AtomicDataZarrReader(path), device="cpu")
    return path, dataset.load_batches([list(range(len(dataset)))])[0]


@pytest.mark.parametrize("appends,expected_tail", [(0, False), (1, True)])
def test_saved_and_read_compare_for_a_new_and_an_appended_store(tmp_path, appends, expected_tail):
    batch = _five_batch()
    _, loaded = _store(tmp_path, f"s{appends}.zarr", batch, appends)
    verdict = agent02_checks.compare_saved_and_read(batch, loaded)
    assert verdict["same"] and verdict["tail"] is expected_tail and verdict["differs"] == []


def test_saved_and_read_differ_when_the_values_differ(tmp_path):
    batch = _five_batch()
    _, loaded = _store(tmp_path, "d.zarr", batch)
    batch.forces = batch.forces + 1.0
    verdict = agent02_checks.compare_saved_and_read(batch, loaded)
    assert not verdict["same"] and verdict["differs"] == ["forces"]


def test_show_store_last_line_for_new_and_appended_stores(tmp_path, monkeypatch):
    import IPython.display as ipy_display

    batch = _five_batch()
    for appends in (0, 1):
        path, loaded = _store(tmp_path, f"v{appends}.zarr", batch, appends)
        shown = []
        monkeypatch.setattr(ipy_display, "display", lambda *objects, _s=shown, **kw: _s.extend(objects))
        agent02.show_store(path, batch, loaded)
        assert shown[-1].data == T.STORE_EQUAL
        assert shown[0].data == T.STORE_COUNT.format(structures=5 * (appends + 1), atoms=147 * (appends + 1))
        assert "Unit" in shown[1].data and "eV/Å" in shown[1].data


def test_show_store_alone_lists_the_structures(tmp_path, monkeypatch):
    import IPython.display as ipy_display

    batch = _five_batch()
    path, _ = _store(tmp_path, "own.zarr", batch)
    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda *objects, **kw: shown.extend(objects))
    agent02.show_store(path)
    assert all(header in shown[1].data for header in T.STORE_PER_STRUCTURE_HEADERS)
    assert shown[1].data.count("<tr>") == 6  # header and five structures


def test_field_table_has_a_unit_for_every_row_and_the_two_structure_and_atom_rows_the_model_fills(monkeypatch):
    pytest.importorskip("rdkit")
    import IPython.display as ipy_display
    from nvalchemi.data import AtomicData

    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda *objects, **kw: shown.extend(objects))
    agent02.show_fields(AtomicData.from_atoms(agent02.load_molecule("caffeine")))
    table = shown[-1].data
    rows = [r for r in table.split("<tr>")[2:]]
    assert rows and all("<td style='text-align:left'>-</td>" not in r.split("</td>")[2] for r in rows)  # no row has the dash as its unit
    assert T.NOT_SET in table and "structure" in table and "atoms" in table
    assert "energy" in table and "forces" in table


def test_every_check_line_is_used_and_every_used_line_exists():
    import re

    source = (Path(agent02_checks.__file__)).read_text()
    used = set(re.findall(r'(?:line|add)\("([a-z_0-9]+)"', source))
    assert used <= set(T.CHECK_LINES), used - set(T.CHECK_LINES)
    assert set(T.CHECK_LINES) - used == set(), set(T.CHECK_LINES) - used


def test_batch_check_accepts_any_order_of_the_five_molecules(tmp_path, capsys):
    pytest.importorskip("rdkit")
    _write(tmp_path, "atomic_data.py", 'from nvalchemi.data import AtomicData\ncaffeine_data = AtomicData.from_atoms(helpers.load_molecule("caffeine"))')
    order = ["ibuprofen", "caffeine", "aspirin", "paracetamol", "capsaicin"]
    _write(tmp_path, "batch.py", f"from nvalchemi.data import AtomicData, Batch\nmolecule_data = [AtomicData.from_atoms(helpers.load_molecule(n)) for n in {order}]\nbatch = Batch.from_data_list(molecule_data)")
    assert agent02_checks.run_check("batch", tmp_path) == 0
    assert "batch: OK" in capsys.readouterr().out
    _write(tmp_path, "batch.py", "from nvalchemi.data import AtomicData, Batch\nmolecule_data = [AtomicData.from_atoms(helpers.load_molecule(n)) for n in ['caffeine', 'aspirin']]\nbatch = Batch.from_data_list(molecule_data)")
    assert agent02_checks.run_check("batch", tmp_path) == 1
    out = capsys.readouterr().out
    assert "molecule_data has atom counts [24, 21]" in out and "in any order" in out


# ---------------------------------------------------------------------------
# Takeover round (26 Nov): the provisional choices, the sample of build_routes, names a Try it must keep
# ---------------------------------------------------------------------------
_SAMPLES = _NOTEBOOK_DIR.parent / "solutions" / "agent-02"


def test_provisional_choices_are_what_the_task_names():
    assert T.PROVISIONAL_BUILD_ROUTES_REPLACES_OWN_BATCH is True
    assert (T.PROVISIONAL_N_STRUCTURES, T.PROVISIONAL_WARMUPS, T.PROVISIONAL_TIMED_RUNS) == (1000, 1, 5)
    assert T.BUILD_ROUTE_TEXT["tensors"][1] == "quicker still, but nothing is filled in or checked for you"
    assert T.PROVISIONAL_TENSOR_FIELDS == ("positions", "atomic_numbers")
    assert T.PROVISIONAL_UNEXPECTED_ORDER_PASSES is True
    everything = Path(T.__file__).read_text() + Path(agent02.__file__).read_text() + Path(agent02_checks.__file__).read_text()
    assert "the fields to you" not in everything  # the phrase the educator rejected


def test_build_routes_sample_follows_the_provisional_size_and_runs_without_a_gpu():
    source = (_SAMPLES / "build_routes.py").read_text()
    assert f"n_structures = {T.PROVISIONAL_N_STRUCTURES}\n" in source
    assert f"range({T.PROVISIONAL_TIMED_RUNS})" in source
    assert source.count("build()  ") == T.PROVISIONAL_WARMUPS  # one warm-up build per route
    assert 'torch.device("cuda")' not in source and "torch.cuda.is_available()" in source
    assert "\n        batch = " not in source and "\nbatch = " not in source  # it does not rebind the lesson's batch
    for field in T.PROVISIONAL_TENSOR_FIELDS:
        assert f'"{field}": ' in source


def test_load_many_default_size_is_the_provisional_one():
    import inspect

    assert inspect.signature(agent02.load_many).parameters["n_structures"].default == T.PROVISIONAL_N_STRUCTURES


def test_show_build_routes_prints_size_repeats_and_the_fields_of_the_tensor_route(monkeypatch):
    pd = pytest.importorskip("pandas")
    import IPython.display as ipy_display

    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda *objects, **kw: shown.extend(objects))
    table = _table(pd, 0.05, 0.08, 0.008)
    table["structures"], table["atoms"], table["timed_runs"] = 1000, 29400, 5
    table["fields"] = ["atom_categories, atomic_masses, atomic_numbers, positions, velocities"] * 2 + ["atomic_numbers, positions"]
    agent02.show_build_routes(table)
    assert shown[0].data == "One batch of 1,000 structures, 29,400 atoms; 5 timed builds per route."
    assert "from tensors (positions and atomic numbers only)" in shown[1].data
    assert "quicker still, but nothing is filled in or checked for you" in shown[1].data
    shown.clear()
    table.loc[2, "fields"] = "atom_categories, atomic_masses, atomic_numbers, positions, velocities"  # five fields given as tensors
    agent02.show_build_routes(table)
    assert "positions and atomic numbers only" not in shown[1].data and "from tensors" in shown[1].data
    shown.clear()
    agent02.show_build_routes(_table(pd, 0.05, 0.08, 0.008))  # no fields column, route names silent: no fields claimed
    assert "positions and atomic numbers only" not in shown[0].data and "from tensors" in shown[0].data
    shown.clear()
    named = _table(pd, 0.05, 0.08, 0.008)  # no fields column, the tensor route's own name gives its fields
    named["route"] = named["route"].astype(str).str.replace("tensors", "tensors (positions and atomic numbers only)")
    agent02.show_build_routes(named)
    assert "from tensors (positions and atomic numbers only)" in shown[0].data
    shown.clear()
    named["route"] = named["route"].str.replace("(positions and atomic numbers only)", "(positions, atomic numbers, masses)", regex=False)
    agent02.show_build_routes(named)  # the name lists more fields: no mark
    assert "positions and atomic numbers only" not in shown[0].data and "from tensors" in shown[0].data


def test_a_try_it_file_that_rebinds_an_earlier_name_is_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent02_checks, "STEPS", {
        "a": agent02_checks.Step("a.py", ("batch",)),
        "build_routes": agent02_checks.Step("build_routes.py", ("build_table",), after=("a",)),
    })
    _write(tmp_path, "a.py", "batch = [1, 2, 3]")
    table = "import pandas as pd\nbuild_table = pd.DataFrame({'route': ['CPU then moved', 'directly on the GPU', 'tensors'], 'best_s': [0.05, 0.08, 0.008]})"
    _write(tmp_path, "build_routes.py", table + "\nbatch = [4, 5]")
    assert agent02_checks.run_check("build_routes", tmp_path) == 1
    assert "build_routes: build_routes.py changed batch, which the later steps use; give what you build here another name" in capsys.readouterr().out
    _write(tmp_path, "build_routes.py", table + "\nbuilt = [4, 5]")
    assert agent02_checks.run_check("build_routes", tmp_path) == 0


def test_a_later_step_fails_when_a_try_it_file_before_it_rebinds_an_earlier_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent02_checks, "STEPS", {
        "a": agent02_checks.Step("a.py", ("batch",)),
        "build_routes": agent02_checks.Step("build_routes.py", ("build_table",), after=("a",)),
        "c": agent02_checks.Step("c.py", ("n",), after=("a",)),
    })
    monkeypatch.setattr(agent02_checks, "OPTIONAL_STEPS", ("build_routes",))
    _write(tmp_path, "a.py", "batch = [1, 2, 3]")
    _write(tmp_path, "build_routes.py", "batch = [4, 5]")
    _write(tmp_path, "c.py", "n = len(batch)")
    assert agent02_checks.run_check("c", tmp_path) == 1
    assert "c: build_routes.py changed batch, which the later steps use; give what you build here another name" in capsys.readouterr().out
    _write(tmp_path, "build_routes.py", "built = [4, 5]")
    assert agent02_checks.run_check("c", tmp_path) == 0


def test_unexpected_order_fails_only_when_the_provisional_switch_is_off(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent02_checks, "STEPS", {
        "a": agent02_checks.Step("a.py", ("x",)),
        "build_routes": agent02_checks.Step("build_routes.py", ("build_table",), after=("a",)),
    })
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "build_routes.py", "import pandas as pd\nbuild_table = pd.DataFrame({'route': ['CPU then moved', 'directly on the GPU', 'tensors'], 'best_s': [0.08, 0.05, 0.008]})")
    monkeypatch.setattr(agent02_checks, "PROVISIONAL_UNEXPECTED_ORDER_PASSES", False)
    assert agent02_checks.run_check("build_routes", tmp_path) == 1
    assert "unexpected order" in capsys.readouterr().out


def test_configure_tutorial_quiets_the_two_library_warnings():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent02.configure_tutorial()
        warnings.warn("AtomicData field 'energy' was cast from torch.float64 to torch.float32 to match positions", UserWarning)
        warnings.warn("Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.", UserWarning)
        warnings.warn("another warning stays visible", UserWarning)
    assert [str(w.message) for w in caught if "another" in str(w.message) or "cast from" in str(w.message) or "requires_grad" in str(w.message)] == ["another warning stays visible"]


def test_time_column_is_never_a_size_column_and_milliseconds_are_read_as_such():
    pd = pytest.importorskip("pandas")
    routes = ["ASE objects on CPU, then moved to GPU", "ASE objects directly on GPU", "directly from tensors"]
    # the size columns come first, the time column has a plain name (as a learner's agent may write it)
    table = pd.DataFrame({"route": routes, "structures": [1000] * 3, "atoms": [29400] * 3, "runs": [3] * 3, "seconds": [0.05, 0.08, 0.008]})
    assert agent02_checks.build_route_times(table) == [0.05, 0.08, 0.008]
    assert agent02_checks.build_route_order(table)["status"] == "usual"
    in_ms = pd.DataFrame({"route": routes, "n_structures": [1000] * 3, "time_ms": [50.0, 80.0, 8.0]})
    assert agent02_checks.build_route_times(in_ms) == pytest.approx([0.05, 0.08, 0.008])
    only_sizes = pd.DataFrame({"route": routes, "structures": [1000] * 3, "atoms": [29400] * 3})
    assert agent02_checks.build_route_times(only_sizes) == []
    problems = []
    agent02_checks._check_build_table(only_sizes, lambda key, **f: problems.append(key))
    assert problems == ["table_no_times"]
    as_rows = table.to_dict("records")  # a list of rows is a table too
    assert agent02_checks.build_route_order(as_rows)["status"] == "usual"
