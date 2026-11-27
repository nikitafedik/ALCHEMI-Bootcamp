# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent 08 helper: the notebook against its contracts, the reactions, the views and the check.

No GPU needed. One test loads the model on the CPU (``test_model_gives_second_derivatives``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
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
from helpers import agent08, agent08_check, agent08_views

NOTEBOOK = _NOTEBOOK_DIR / "alchemi-agent-08-paths-of-least-resistance.ipynb"
NAMES = ["ammonia inversion", "ethane rotation", "HCN to HNC", "malonaldehyde proton transfer"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def shown(monkeypatch):
    """The objects the views pass to IPython's display()."""

    items = []
    monkeypatch.setattr(ipy_display, "display", lambda *objs, **kw: items.extend(objs))
    return items


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text())["cells"]


def _text(cell: dict) -> str:
    return "".join(cell["source"])


# ---------------------------------------------------------------------------
# The notebook: names, files and helper calls
# ---------------------------------------------------------------------------


def test_every_table_matches_its_contract():
    found: dict[str, list[str]] = {}
    for cell in _cells():
        for row in re.finditer(r"\| output\s*\| (.*?), in `agent08_work/(\w+)\.py` \|", _text(cell)):
            found[row.group(2)] = re.findall(r"`(\w+)`", row.group(1))
    assert list(found) == list(agent08_check.STEPS)
    for step, names in found.items():
        assert tuple(names) == agent08_check.STEPS[step].names, step


def test_load_cells_follow_the_step_order():
    loads = [_text(c).strip() for c in _cells() if c["cell_type"] == "code" and _text(c).startswith("%load")]
    assert loads == [f"%load agent08_work/{spec.file}" for spec in agent08_check.STEPS.values()]


def test_setup_cell_is_what_the_check_runs():
    setup = next(_text(c) for c in _cells() if "make_work_dir" in _text(c))
    assert setup == agent08_check.SETUP_CELL


def test_every_helper_call_in_the_notebook_exists():
    calls = set()
    for cell in _cells():
        if cell["cell_type"] == "code":
            calls |= set(re.findall(r"helpers\.(\w+)\(", _text(cell)))
    assert calls == {
        "configure_tutorial", "make_work_dir", "load_model", "load_reactions", "show_session",
        "show_reactions", "show_paths", "show_neb_log", "plot_energy_profiles", "show_saddle_table",
    }
    for name in calls:
        assert callable(getattr(agent08, name)), name
        assert name in agent08.__all__, name


def test_view_calls_take_the_names_their_step_returns():
    code = "\n".join(_text(c) for c in _cells() if c["cell_type"] == "code")
    assert "helpers.show_reactions(reactants, products, reaction_names)" in code
    assert "helpers.show_paths(paths, reaction_names)" in code
    assert "helpers.show_neb_log(neb_log, reaction_names)" in code
    assert "helpers.plot_energy_profiles(relaxed_paths, path_energy, reaction_names, barriers, converged)" in code
    assert "helpers.show_saddle_table(saddle_table)" in code


def test_cells_have_ids_and_no_saved_outputs():
    cells = _cells()
    ids = [c["id"] for c in cells]
    # numbered cells; the top animation and the roadmap strip carry ids of their own
    assert all(re.fullmatch(r"a08-(\d\d|top|strip)", i) for i in ids)
    numbered = [i for i in ids if i[4:].isdigit()]
    assert numbered == sorted(numbered) and len(set(ids)) == len(ids)
    assert ids[:5] == ["a08-01", "a08-top", "a08-02", "a08-strip", "a08-03"]
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
        assert "—" not in _text(cell) and "·" not in _text(cell)


def test_step_tables_have_the_common_form():
    """Inputs separated by semicolons, outputs by commas, the file last."""

    rows = [m.group(1) for c in _cells() for m in re.finditer(r"\| input\s*\| (.*?) \|", _text(c))]
    assert rows == ["`reactants`; `products`; `neb_settings`", "`paths`; `model`; `neb_settings`", "`relaxed_paths`; `model`"]


def test_the_check_runs_the_settings_the_notebook_shows():
    assert agent08_check.NEB_SETTINGS_LINE in agent08_check.SETUP_CELL
    ns: dict = {}
    exec(agent08_check.NEB_SETTINGS_LINE, ns)
    assert ns["neb_settings"] == {"num_images": 8, "fmax": 0.003, "regular_fmax": 0.03, "n_steps": 600}
    assert ns["neb_settings"]["num_images"] % 2 == 0  # no image starts on the saddle of a symmetric path
    # the chain that keeps the saddle test from being circular: ends < NEB stop < saddle tolerance
    assert agent08.RELAX_FMAX < ns["neb_settings"]["fmax"] < agent08_check.SADDLE_FORCE_TOLERANCE


def test_after_lists_follow_step_order():
    order = list(agent08_check.STEPS)
    for step, spec in agent08_check.STEPS.items():
        assert list(spec.after) == order[: order.index(step)], step


def test_show_session_text(capsys):
    agent08.show_session()
    assert capsys.readouterr().out == (
        "Give these to your agent when you start the session:\n"
        "  notebook    notebooks/alchemi-agent-08-paths-of-least-resistance.ipynb\n"
        "  helpers     from helpers import agent08 as helpers\n"
        "  work in     notebooks/agent08_work/\n"
        "  check with  python -m helpers.agent08 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py\n"
    )


def test_every_step_name_is_its_file_name_without_py():
    """The session line says <step> is the file name without .py."""

    for step, spec in agent08_check.STEPS.items():
        assert spec.file == f"{step}.py"


def test_make_work_dir_creates_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(agent08, "WORK_DIR", tmp_path / "agent08_work")
    assert agent08.make_work_dir() == tmp_path / "agent08_work"
    assert (tmp_path / "agent08_work").is_dir()


def test_default_work_dir_is_notebooks_agent08_work():
    assert agent08.WORK_DIR == _NOTEBOOK_DIR / "agent08_work"
    assert (_NOTEBOOK_DIR / agent08.NOTEBOOK.removeprefix("notebooks/")).exists()


# ---------------------------------------------------------------------------
# The reactions
# ---------------------------------------------------------------------------


def test_four_reactions_with_matched_atoms():
    assert list(agent08.REACTIONS) == NAMES
    formulas = []
    for build in agent08.REACTIONS.values():
        a, b = build()
        assert a.get_chemical_symbols() == b.get_chemical_symbols()
        formulas.append(a.get_chemical_formula())
    assert formulas == ["H3N", "C2H6", "CHN", "C3H4O2"]


def test_ammonia_product_is_the_mirror_image_with_h_kept():
    a, b = agent08.REACTIONS["ammonia inversion"]()
    assert np.allclose(a.positions[1:], b.positions[1:])
    plane = a.positions[1:].mean(0)
    normal = np.cross(a.positions[2] - a.positions[1], a.positions[3] - a.positions[1])
    assert np.isclose((a.positions[0] - plane) @ normal, -(b.positions[0] - plane) @ normal)


def test_ethane_turn_is_split_between_the_methyls():
    a, b = agent08.REACTIONS["ethane rotation"]()
    assert np.allclose(a.positions[:2], b.positions[:2])  # the carbons stay
    moved = np.linalg.norm(a.positions - b.positions, axis=1)
    assert np.allclose(moved[2:], moved[2], atol=1e-3) and moved[2] > 0.5  # every H moves the same way
    assert np.allclose(a.get_all_distances()[0, 2:5], b.get_all_distances()[0, 2:5])  # each methyl turns rigidly


def test_hcn_keeps_a_turned_axis_so_the_h_goes_around_the_bond():
    a, b = agent08.REACTIONS["HCN to HNC"]()
    axis_a, axis_b = a.positions[2] - a.positions[1], b.positions[2] - b.positions[1]
    cosine = axis_a @ axis_b / np.linalg.norm(axis_a) / np.linalg.norm(axis_b)
    assert np.isclose(np.degrees(np.arccos(cosine)), 60.0, atol=0.5)
    middle_h = 0.5 * (a.positions[0] + b.positions[0])
    bond = 0.5 * (a.positions[1:] + b.positions[1:])
    assert np.linalg.norm(middle_h - bond, axis=1).min() > 0.7  # Å, clear of both heavy atoms


def test_lay_over_puts_a_turned_copy_back():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(6, 3))
    turn = agent08._rotate_about(q, q.mean(0), np.array([0.3, -1.0, 0.5]), 77.0) + np.array([1.0, 2.0, -3.0])
    assert np.allclose(agent08._lay_over(turn, q), q, atol=1e-8)


def test_unknown_reaction_is_refused():
    with pytest.raises(ValueError, match="Unknown reaction 'water splitting'"):
        agent08.load_reactions(model=None, names=["water splitting"])


def _aimnet2_weights() -> bool:
    """True when the notebook's AIMNet2 checkpoint is on this machine, so loading it downloads nothing."""

    try:
        from aimnet.calculators import model_registry as registry
    except Exception:
        return False
    agent08.configure_tutorial()  # sets AIMNET_CACHE_DIR to the course runtime where the course prepared it
    try:
        name = registry.resolve_registry_model_name(agent08.MODEL_NAME)
        file = registry.load_model_registry()["models"][name]["file"]
    except Exception:
        return False
    bundled = Path(registry.__file__).parent / "assets" / file
    cache = Path(os.environ.get("AIMNET_CACHE_DIR") or Path.home() / ".cache" / "aimnet")
    return (cache / file).is_file() or bundled.is_file()


def _needs_aimnet2() -> None:
    pytest.importorskip("aimnet")
    if not _aimnet2_weights():
        pytest.skip("AIMNet2 weights are not on this machine")


def test_the_model_without_d3_is_the_default():
    """The notebook's model is AIMNet2 with its Coulomb term, without D3, whatever the environment says."""

    import inspect

    assert inspect.signature(agent08.load_model).parameters["d3"].default is False
    assert "os.environ" not in inspect.getsource(agent08.load_model)
    assert "AIMNet2 with its Coulomb term" in agent08.load_model.__doc__
    assert agent08.REACTIONS_FILE.name == "reactions-v2.extxyz"


def test_the_shipping_model_reads_the_saved_ends_and_gives_a_dense_hessian():
    """The notebook's model: the saved ends pass the loader's limit as read, and compute_hessian works."""

    torch = pytest.importorskip("torch")
    _needs_aimnet2()
    from helpers.model_support import prepare_model_batch
    from nvalchemi.data import AtomicData, Batch

    agent08.configure_tutorial()
    model = agent08.load_model("cpu")
    reactants, products = agent08._saved_ends(NAMES)
    assert max(agent08.largest_forces(model, reactants + products)) <= agent08.RELAX_FMAX  # no relaxation needed
    batch = Batch.from_data_list([AtomicData.from_atoms(reactants[0], device="cpu")], device="cpu")
    hessian = model.compute_hessian(prepare_model_batch(batch, model))["hessian"]
    assert bool(torch.isfinite(hessian).all())


def _setup_in_a_fresh_interpreter() -> subprocess.CompletedProcess:
    """The setup cell's model and loader in a new Python, from notebooks/, as the kernel runs them."""

    code = (
        "from helpers import agent08 as helpers\n"
        "helpers.configure_tutorial()\n"
        "model = helpers.load_model('cpu')\n"
        "helpers.load_reactions(model)\n"
        "print('groups', len(model.groups))\n"
    )
    full = {k: v for k, v in os.environ.items() if not k.startswith("ALCHEMI_AGENT08")}
    full.update(CUDA_VISIBLE_DEVICES="")
    return subprocess.run(
        [sys.executable, "-c", code], cwd=_NOTEBOOK_DIR, env=full, capture_output=True, text=True, timeout=600, check=False
    )


def test_the_setup_model_is_one_group_and_prints_no_aimnet2_warning():
    """The notebook's model is one group (AIMNet2 + Coulomb), and the Toolkit aimnet2.py warning
    "Converting a tensor with requires_grad=True to a scalar" does not reach stderr."""

    _needs_aimnet2()
    done = _setup_in_a_fresh_interpreter()
    assert done.returncode == 0, done.stderr[-2000:]
    assert "groups 1" in done.stdout
    assert "Converting a tensor with requires_grad=True to a scalar" not in done.stderr


def test_configure_tutorial_installs_the_aimnet2_warning_filter():
    import warnings

    with warnings.catch_warnings():
        agent08.configure_tutorial()
        hits = [f for f in warnings.filters if f[0] == "ignore" and f[1] is not None
                and f[1].match("Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.")]
        assert hits and hits[0][3] is not None and hits[0][3].match("nvalchemi.models.aimnet2")


def test_the_d3_variant_refuses_a_dense_hessian():
    """With ``d3=True``, compute_hessian refuses the D3 term (no second derivatives in Toolkit 0.3.0)."""

    _needs_aimnet2()
    from helpers.model_support import d3_parameter_file, prepare_model_batch

    agent08.configure_tutorial()
    try:
        d3_parameter_file()
    except RuntimeError as exc:
        pytest.skip(f"the D3 tables of the course runtime are not prepared here: {exc}")
    from nvalchemi.data import AtomicData, Batch

    model = agent08.load_model("cpu", d3=True)
    a, _ = agent08.REACTIONS["ammonia inversion"]()
    batch = Batch.from_data_list([AtomicData.from_atoms(a, device="cpu")], device="cpu")
    with pytest.raises(Exception, match="DFTD3ModelWrapper does not support derivative operation"):
        model.compute_hessian(prepare_model_batch(batch, model))


def test_load_reactions_says_the_ends_are_already_laid_over():
    assert "already laid over each other" in agent08.load_reactions.__doc__


def test_saved_default_reactions_match_the_builders():
    """data/agent08/reactions-v2.extxyz: two frames per reaction, the atoms of the builders, in their order."""

    from ase.io import read

    frames = read(agent08.REACTIONS_FILE, index=":")
    assert [(f.info["reaction"], f.info["end"]) for f in frames] == [
        (name, end) for name in NAMES for end in ("reactant", "product")
    ]
    reactants, products = agent08._saved_ends(NAMES)
    for name, a, b in zip(NAMES, reactants, products):
        built, _ = agent08.REACTIONS[name]()
        assert a.get_chemical_symbols() == b.get_chemical_symbols() == built.get_chemical_symbols(), name
    assert agent08._saved_ends(["no such reaction"]) is None
    assert (agent08.DATA_DIR / "README.md").is_file()


def test_relax_refuses_ends_that_did_not_get_there(monkeypatch):
    """The loader measures what it relaxed and raises when a force is still above the limit."""

    _needs_aimnet2()
    agent08.configure_tutorial()
    model = agent08.load_model("cpu", d3=False)
    a, b = agent08.REACTIONS["ammonia inversion"]()
    with pytest.raises(RuntimeError, match="did not relax: the largest force on an atom is .* after 1 steps"):
        agent08._relax(model, [a, b], fmax=1.0e-9, max_steps=1)


def test_loader_gives_tight_ends_from_the_saved_file():
    _needs_aimnet2()
    agent08.configure_tutorial()
    model = agent08.load_model("cpu", d3=False)
    reactants, products, names = agent08.load_reactions(model)
    assert names == NAMES
    assert max(agent08.largest_forces(model, reactants + products)) <= agent08.RELAX_FMAX == 0.0005


def test_model_gives_second_derivatives():
    """The Coulomb term of the notebook's model accepts Toolkit's Hessian request."""

    torch = pytest.importorskip("torch")
    _needs_aimnet2()
    from nvalchemi.data import AtomicData, Batch

    agent08.configure_tutorial()
    model = agent08.load_model("cpu", d3=False)
    a, _ = agent08.REACTIONS["ammonia inversion"]()
    batch = Batch.from_data_list([AtomicData.from_atoms(a, device="cpu")], device="cpu")
    from helpers.model_support import prepare_model_batch

    batch = prepare_model_batch(batch, model)  # compute_hessian reads the neighbor data of the batch
    out = model(batch)
    assert tuple(out["energy"].reshape(-1).shape) == (1,) and tuple(out["forces"].shape) == (4, 3)
    hessian = model.compute_hessian(batch)["hessian"]
    assert hessian.numel() == (3 * 4) ** 2 and bool(torch.isfinite(hessian).all())


# ---------------------------------------------------------------------------
# A batch of paths for the views and the check, without Toolkit
# ---------------------------------------------------------------------------


class FakeBatch:
    """The few fields of a Batch of paths that the views and the check read."""

    def __init__(self, images_per_path, atoms_per_path, positions=None, **fields):
        counts = [n for n, m in zip(atoms_per_path, images_per_path) for _ in range(m)]
        self.group_layout = SimpleNamespace(num_graphs_per_group=np.asarray(images_per_path))
        self.num_nodes_per_graph = np.asarray(counts)
        self.num_graphs, self.num_nodes = len(counts), sum(counts)
        self.atomic_numbers = np.ones(sum(counts), dtype=int)
        rng = np.random.default_rng(1)
        self.positions = rng.normal(size=(sum(counts), 3)) if positions is None else positions
        for name, value in fields.items():
            setattr(self, name, value)

    def clone(self):
        return self


def _write_log(path: Path, rows: list[tuple]) -> Path:
    lines = ["step,group_idx,status,fmax,energy_barrier,highest_interior_image_idx,path_length"]
    lines += [f"{float(s)},{float(g)},{float(st)},{f},0.1,2.0,1.0" for s, g, st, f in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def neb_log(tmp_path):
    """Two paths: the first stops at step 2 and keeps its row; the second runs out at step 4."""

    rows = []
    for step in range(5):
        rows.append((step, 0, 2 if step >= 2 else 1, [1.0, 0.3, 0.04, 0.04, 0.04][step]))
        rows.append((step, 1, 1, [2.0, 1.0, 0.6, 0.5, 0.44][step]))
    return _write_log(tmp_path / "neb_log.csv", rows)


# ---------------------------------------------------------------------------
# The views
# ---------------------------------------------------------------------------


def test_stop_is_the_first_finished_row_not_the_last(neb_log):
    data = agent08_views.neb_log_data(neb_log, ["a", "b"])
    assert [d["stop_step"] for d in data] == [2, 4]
    assert [d["stop_force"] for d in data] == [0.04, 0.44]
    assert list(data[0]["steps"]) == [0, 1, 2]  # the repeated rows after the stop are left out


def test_neb_log_accepts_a_string_and_refuses_other_files(neb_log, tmp_path):
    assert agent08_views.neb_log_rows(str(neb_log), ["a", "b"])[1] == [["a", 2, "0.0400"], ["b", 4, "0.4400"]]
    with pytest.raises(FileNotFoundError, match="does not exist"):
        agent08_views.read_neb_log(tmp_path / "missing.csv")
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not an NEB diagnostics file"):
        agent08_views.read_neb_log(other)
    with pytest.raises(ValueError, match="reaction_names has 3 names; there are 2 paths"):
        agent08_views.neb_log_data(neb_log, ["a", "b", "c"])


def test_neb_log_figure_has_one_panel_per_path_and_a_dot_at_the_stop(neb_log):
    figure = agent08_views.build_neb_log_figure(neb_log, ["a", "b"])
    assert [axis.get_title(loc="left") for axis in figure.axes] == ["a", "b"]
    assert figure.axes[0].get_ylabel() == "Force [eV/Å]" and figure.axes[0].get_xlabel() == "Step"
    assert figure.axes[0].get_yscale() == "log"
    dot = figure.axes[0].lines[1]
    assert list(dot.get_xdata()) == [2] and list(dot.get_ydata()) == [0.04]
    assert figure.axes[0].get_ylim() == figure.axes[1].get_ylim()


def test_profiles_are_relative_to_the_first_image_in_kcal_per_mol():
    torch = pytest.importorskip("torch")
    batch = FakeBatch([3, 4], [2, 3])
    energy = torch.tensor([[-10.0], [-9.9], [-10.0], [-5.0], [-4.0], [-4.5], [-4.8]])  # eV, (n, 1)
    data = agent08_views.profile_data(batch, energy, ["a", "b"], torch.tensor([0.1, 1.0]), [torch.tensor(True), torch.tensor(False)])
    assert np.allclose(data[0]["energy"], [0.0, 0.1 * 23.0605, 0.0], atol=1e-4)
    assert data[1]["top"] == 1 and np.isclose(data[1]["barrier"], 23.0605)
    assert [d["converged"] for d in data] == [True, False]


def test_profiles_refuse_a_wrong_length():
    batch = FakeBatch([3, 4], [2, 3])
    with pytest.raises(ValueError, match="path_energy has 6 values; relaxed_paths has 7 images"):
        agent08_views.profile_data(batch, np.zeros(6), ["a", "b"], [0, 0], [True, True])
    with pytest.raises(ValueError, match="barriers has 3 values; there are 2 paths"):
        agent08_views.profile_data(batch, np.zeros(7), ["a", "b"], [0, 0, 0], [True, True])


def test_profile_titles_give_the_barrier_and_the_state():
    batch = FakeBatch([3, 3], [2, 3])
    figure = agent08_views.build_energy_profiles_figure(
        batch, np.array([0, 0.5, 0, 0, 0.2, 0.1]), ["a", "b"], np.array([0.5, 0.2]), np.array([True, False])
    )
    assert [axis.get_title(loc="left") for axis in figure.axes] == [
        "a\n11.53 kcal/mol, converged",
        "b\n4.61 kcal/mol, not converged",
    ]
    assert figure.axes[0].get_ylabel() == "Energy [kcal/mol]" and figure.axes[1].get_xlabel() == "Image"


def test_figures_use_the_course_font_and_size(neb_log):
    figure = agent08_views.build_neb_log_figure(neb_log, ["a", "b"])
    assert figure.get_size_inches()[0] == pytest.approx(9.2)
    axis = figure.axes[0]
    with agent08_views._course_style():  # the family is resolved when the figure is drawn
        assert axis.xaxis.label.get_fontname() == "Open Sans"
        assert axis.yaxis.label.get_fontname() == "Open Sans"
    assert axis.xaxis.label.get_fontsize() == pytest.approx(9.36)


def test_plots_show_one_picture_with_alt_text_and_the_table(neb_log, shown):
    agent08_views.show_neb_log(neb_log, ["a", "b"])
    picture, table = (item.data for item in shown)
    assert picture.count("<img ") == 1 and "max-width:920px" in picture
    assert f'alt="{agent08_views.LABELS["neb_log"]["alt"]}"' in picture
    assert "<th style='text-align:left'>Stopped at step</th>" in table and "<td style='text-align:left'>0.4400</td>" in table


def test_paths_table_has_one_row_per_path():
    batch = FakeBatch([9, 9], [4, 8])
    assert agent08_views.paths_rows(batch, ["a", "b"]) == (["Path", "Images", "Atoms"], [["a", 9, 4], ["b", 9, 8]])
    with pytest.raises(ValueError, match="has no groups"):
        agent08_views.images_per_path(SimpleNamespace())


def test_player_has_one_card_per_path_and_counts_images():
    markup = agent08_views.paths_player_markup(FakeBatch([5, 7], [4, 8]), ["first path", "second path"])
    assert markup.count("<iframe") == 1 and "mountPlayer" in markup
    assert "first path" in markup and "second path" in markup
    assert "replace(/^Step/,&quot;Image&quot;)" in markup


def test_reactions_view_mounts_both_ends_of_every_reaction():
    pairs = [build() for build in agent08.REACTIONS.values()]
    markup = agent08_views.reactions_markup([a for a, _ in pairs], [b for _, b in pairs], NAMES)
    assert markup.count("<iframe") == 1 and markup.count("AlchemiViewer.mountStructure(") == 8
    for word in NAMES + ["reactant", "product"]:
        assert word in markup
    with pytest.raises(ValueError, match="4 reactants and 3 products"):
        agent08_views.reactions_markup([a for a, _ in pairs], [b for _, b in pairs][:3], NAMES)


def test_saddle_table_is_shown_as_it_is_grouped_by_path():
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame(
        {"path": ["a", "a", "b <c>"], "image": [0, 1, 0], "fmax": [0.001665, 0.5, 0.25], "saddle": ["no", "yes", "no"]}
    )
    markup = agent08_views.saddle_table_html(table)
    assert markup.count("colspan='3'") == 2 and "b &lt;c&gt;" in markup
    assert "0.0017" in markup and markup.count("<td style='text-align:left'>yes</td>") == 1
    heads = re.findall(r"<th style='text-align:left'>([^<]*)</th>", markup.split("<tbody>")[0])
    assert heads == ["Saddle point", "Image", "Largest force [eV/Å]"]  # the verdict first
    assert "unexpected: no image of b &lt;c&gt; is a saddle point in this table" in markup
    assert "unexpected: no image of a " not in markup
    rows = [{"n": 1, "saddle": "no"}, {"n": 2, "saddle": "yes"}]  # no path column: one plain table and a line
    plain = agent08_views.saddle_table_html(rows)
    assert plain.count("<tr>") == 3 and plain.startswith("<p style='margin:4px 0'>saddle_table has no column that names the path")


def test_saddle_view_headers_have_units_and_one_number_format():
    """Readable headers with units, one number format, the verdict first (visible at 920 px)."""

    pd = pytest.importorskip("pandas")
    sample = pd.DataFrame({
        "path": ["a"], "image": [0], "energy_eV": [-1539.774862], "fmax_eV_per_A": [2.2e-5], "stationary": ["yes"],
        "lowest_freq_cm1": [-1002.605], "second_freq_cm1": [1416.7], "n_imaginary": [1], "saddle": ["yes"],
    })
    markup = agent08_views.saddle_table_html(sample)
    heads = re.findall(r"<th style='text-align:left'>([^<]*)</th>", markup.split("<tbody>")[0])
    assert heads == ["Saddle point", "Image", "Energy [eV]", "Largest force [eV/Å]", "Stationary",
                     "Lowest frequency [cm⁻¹]", "Second frequency [cm⁻¹]", "Imaginary modes"]
    cells = re.findall(r"<td style='text-align:left'>([^<]*)</td>", markup)
    assert cells == ["yes", "0", "-1539.7749", "0.0000", "yes", "-1002.6050", "1416.7000", "1"]
    trial = pd.DataFrame({"path": ["a"], "image": [0], "largest_force": [0.0294], "imaginary_modes": [0], "saddle": ["no"]})
    heads = re.findall(r"<th style='text-align:left'>([^<]*)</th>", agent08_views.saddle_table_html(trial).split("<tbody>")[0])
    assert heads == ["Saddle point", "Image", "Largest force [eV/Å]", "Imaginary modes"]


def test_saddle_view_sorts_the_images_of_each_path_and_counts_the_yes_rows():
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame(
        {"reaction": ["a"] * 3, "k": [2, 0, 1], "force": [0.3, 0.1, 0.2], "saddle": ["yes", "no", "yes"]}
    )
    markup = agent08_views.saddle_table_html(table)
    assert markup.index("0.1") < markup.index("0.2") < markup.index("0.3")
    assert "unexpected: 2 images of a are saddle points in this table" in markup
    assert agent08_views.image_column(table, "reaction") == "k"
    assert agent08_views.path_column(table, ["a", "b"]) == "reaction" and agent08_views.path_column(table, ["x"]) is None


def test_barrier_table_has_one_row_per_path_and_marks_a_path_that_is_not_converged():
    batch = FakeBatch([3, 3], [2, 2])
    energy = np.array([0.0, 0.5, 0.1, 1.0, 1.2, 0.9])
    headers, rows = agent08_views.profile_rows(batch, energy, ["a", "b"], [0.5, 0.2], [True, False])
    assert headers == ["Path", "Barrier [kcal/mol]", "Converged", ""]
    assert rows == [
        ["a", "11.53", "yes", ""],
        ["b", "4.61", "no", "unexpected: the path ran out of steps, so this barrier is an estimate"],
    ]


# ---------------------------------------------------------------------------
# The check: measuring the top of a path
# ---------------------------------------------------------------------------


def _parabola(places):
    """One atom on a line; E = -(x - 1)^2 eV, so the top is at x = 1 Å."""

    positions = [np.array([[x, 0.0, 0.0]]) for x in places]
    energy = np.array([-((x - 1.0) ** 2) for x in places])
    forces = [np.array([[2.0 * (x - 1.0), 0.0, 0.0]]) for x in places]  # F = -dE/dx
    return positions, energy, forces


def test_top_on_the_saddle_is_at_distance_zero():
    top = agent08_check.path_top(*_parabola([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert top["image"] == 2 and top["force"] == 0.0
    assert top["distance"] == pytest.approx(0.0, abs=1e-6) and top["below"] == pytest.approx(0.0, abs=1e-6)


def test_top_that_stopped_short_is_measured_against_the_fitted_path():
    top = agent08_check.path_top(*_parabola([0.0, 0.4, 0.8, 1.2001, 1.6, 2.0]))
    assert top["image"] == 2 and top["force"] == pytest.approx(0.4)
    assert top["distance"] == pytest.approx(0.2, abs=0.01)
    assert top["below"] == pytest.approx(0.04 * 23.0605, rel=0.02)


# ---------------------------------------------------------------------------
# The check: messages
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_batches(monkeypatch):
    monkeypatch.setattr(agent08_check, "_is_batch", lambda value: isinstance(value, FakeBatch))


def _paths_ns(images=(5, 5), **fields):
    reactants = [np.zeros((2, 3)), np.zeros((3, 3))]
    products = [np.ones((2, 3)), np.ones((3, 3))]
    blocks = []
    for count, a, b in zip(images, reactants, products):
        blocks += [a + (b - a) * k / (count - 1) for k in range(count)]
    positions = np.concatenate(blocks)
    n_nodes = len(positions)
    full = {"energy": _Sized((sum(images), 1)), "forces": _Sized((n_nodes, 3)), "velocities": _Sized((n_nodes, 3))}
    full.update(fields)
    paths = FakeBatch(list(images), [2, 3], positions, **{k: v for k, v in full.items() if v is not None})
    return {
        "paths": paths,
        "reaction_names": ["a", "b"],
        "reactants": [_Ends(r) for r in reactants],
        "products": [_Ends(p) for p in products],
        "neb_settings": {"num_images": 5, "fmax": 0.05, "regular_fmax": 0.2, "n_steps": 100},
    }


class _Sized:
    def __init__(self, shape):
        self.shape = shape

    def numel(self):
        return int(np.prod(self.shape))


class _Ends:
    def __init__(self, positions):
        self.positions = positions

    def __len__(self):
        return len(self.positions)


def test_paths_contract_passes_a_straight_line_with_its_fields(fake_batches):
    assert agent08_check._check_paths(_paths_ns(), {}) == ([], [])


def test_paths_messages(fake_batches):
    ns = _paths_ns()
    ns["paths"] = [1, 2]
    assert agent08_check._check_paths(ns, {})[0] == ["paths is list; it should be one Batch"]

    ns = _paths_ns(velocities=None)
    assert agent08_check._check_paths(ns, {})[0] == [
        "paths has no velocities; interpolate_paths drops energy, forces and velocities, "
        "and the NEB run reads all three. Attach them to paths"
    ]

    ns = _paths_ns(images=(3, 5))
    assert agent08_check._check_paths(ns, {})[0] == ["a has 3 images; neb_settings asks for 5, both ends included"]

    ns = _paths_ns()
    ns["reaction_names"] = ["a", "b", "c"]
    assert agent08_check._check_paths(ns, {})[0] == ["paths holds 2 paths; the 3 reactions give 3, all in one Batch"]

    ns = _paths_ns()
    ns["products"][1].positions = ns["products"][1].positions + 0.25
    (line,) = agent08_check._check_paths(ns, {})[0]
    assert line.startswith("the last image of b is not the product the loader gave (largest difference 0.250 Å)")
    assert line.endswith("build the path without aligning them again")

    ns = _paths_ns(forces=_Sized((7, 3)))
    assert agent08_check._check_paths(ns, {})[0] == ["paths.forces has shape (7, 3); one row per atom gives (25, 3)"]


class FakeModel:
    """Returns the energies and forces it was built with, as Toolkit models do."""

    def __init__(self, energy, forces):
        self.energy, self.forces = np.asarray(energy, dtype=float), np.asarray(forces, dtype=float)

    def __call__(self, batch):
        return {"energy": self.energy.reshape(-1, 1), "forces": self.forces}


_GOOD_LOG = [(0, 0, 1, 1.0), (0, 1, 1, 2.0), (1, 0, 2, 0.04), (1, 1, 1, 0.5), (2, 0, 2, 0.04), (2, 1, 2, 0.03)]
_ENERGY = np.array([-10.0, -9.9, -9.8, -9.9, -10.0, -5.0, -4.6, -4.0, -4.5, -4.8])


def _neb_ns(tmp_path, rows=_GOOD_LOG, **changes):
    """The merged step: two paths of five images, barriers 0.2 and 1.0 eV, both converged at 0.05 eV/Å."""

    ns = {
        "relaxed_paths": FakeBatch([5, 5], [2, 3]),
        "model": FakeModel(_ENERGY, np.zeros((25, 3))),
        "neb_log": _write_log(tmp_path / "neb_log.csv", rows),
        "reaction_names": ["a", "b"],
        "neb_settings": {"num_images": 5, "fmax": 0.05, "regular_fmax": 0.2, "n_steps": 100},
        "path_energy": _ENERGY.reshape(-1, 1),
        "barriers": np.array([0.2, 1.0]),
        "converged": np.array([True, True]),
    }
    ns.update(changes)
    return ns


def _ctx(**changes):
    ctx = {
        "groups": [5, 5],
        "neb_calls": 1,
        "neb": SimpleNamespace(climbing=object(), fmax=0.05, convergence_hook=None),
        "start": None,
    }
    ctx.update(changes)
    return ctx


def test_neb_contract_passes_and_reports_every_path(fake_batches, tmp_path):
    problems, notes = agent08_check._check_neb(_neb_ns(tmp_path), _ctx())
    assert problems == []
    assert notes == [
        "a: barrier 4.61 kcal/mol, converged at step 1 with force 0.0400 eV/Å",
        "b: barrier 23.06 kcal/mol, converged at step 2 with force 0.0300 eV/Å",
    ]


def test_a_path_that_ran_out_of_steps_passes_as_unexpected(fake_batches, tmp_path):
    """Done when asks for a barrier and a converged yes or no per path, not for convergence."""

    cut_off = [(0, 0, 1, 1.0), (0, 1, 1, 2.0), (1, 0, 2, 0.04), (1, 1, 1, 0.44)]
    ns = _neb_ns(tmp_path, cut_off, converged=np.array([True, False]))
    problems, notes = agent08_check._check_neb(ns, _ctx())
    assert problems == []
    assert notes[1] == (
        "b: barrier 23.06 kcal/mol, not converged: unexpected. The path ended at step 1 with force "
        "0.4400 eV/Å, above 0.05; its barrier is an estimate"
    )
    # a stage cap ends the path as "finished"; the flag still follows the force
    capped = cut_off[:3] + [(1, 1, 2, 0.44)]
    ns = _neb_ns(tmp_path, capped)
    assert agent08_check._check_neb(ns, _ctx())[0] == [
        "converged is True for b; its last force in the log is 0.4400 eV/Å and the run stops a path at 0.05 eV/Å"
    ]


def test_neb_messages(fake_batches, tmp_path):
    ns = _neb_ns(tmp_path)
    assert agent08_check._check_neb(ns, _ctx(neb_calls=2))[0] == [
        "neb.py called NEB.run with model 2 times; one call relaxes every path in the Batch"
    ]
    assert agent08_check._check_neb(ns, _ctx(neb_calls=0, neb=None))[0] == [
        "neb.py made no NEB.run call with model; the step asks for one NEB call on the Batch of paths"
    ]
    no_climb = SimpleNamespace(climbing=None, fmax=0.05, convergence_hook=None)
    assert agent08_check._check_neb(ns, _ctx(neb=no_climb))[0] == [
        "the NEB ran without a climbing image; pass climbing=ClimbingImageConfig(...) to NEB"
    ]
    assert "relaxed_paths has paths of [5, 5] images; paths has [9, 9]" in agent08_check._check_neb(ns, _ctx(groups=[9, 9]))[0][0]
    same = [image.copy() for image in agent08_check._images(ns["relaxed_paths"])]
    assert agent08_check._check_neb(ns, _ctx(start=same))[0] == [
        "relaxed_paths equals the straight-line paths; the NEB did not move any image"
    ]
    ns["neb_log"] = tmp_path / "missing.csv"
    assert "does not exist; it is the diagnostics file of the NEB run" in agent08_check._check_neb(ns, _ctx())[0][0]
    ns["neb_log"] = 3.0
    assert agent08_check._check_neb(ns, _ctx())[0] == [
        "neb_log is float; it should be the path of the diagnostics file the NEB run wrote"
    ]


def test_barrier_messages(fake_batches, tmp_path):
    def lines(**changes):
        return agent08_check._check_neb(_neb_ns(tmp_path, **changes), _ctx())[0]

    assert lines(path_energy=np.zeros(4)) == ["path_energy has 4 values; one per image of relaxed_paths gives 10"]
    kcal = _ENERGY.reshape(-1, 1) * 23.0605
    assert lines(path_energy=kcal)[0].startswith("path_energy differs from the model's energy of the images by up to 2")
    assert lines(barriers=np.array([0.2, 1.0]) * 23.0605) == [
        "the barrier of a is 4.612; the forward barrier, the highest image between the ends minus the first image, is 0.2000 eV",
        "the barrier of b is 23.06; the forward barrier, the highest image between the ends minus the first image, is 1.0000 eV",
    ]
    assert lines(barriers=np.array([0.2])) == ["barriers has 1 values; one per path gives 2"]
    assert lines(converged=[True, False]) == [
        "converged is False for b; its last force in the log is 0.0300 eV/Å and the run stops a path at 0.05 eV/Å"
    ]


def test_a_stop_force_other_than_the_settings_is_a_note(fake_batches, tmp_path):
    """A learner who tries another stop (Fine control) still passes; the check says what it saw."""

    neb = SimpleNamespace(climbing=object(), fmax=0.1, convergence_hook=None)
    problems, notes = agent08_check._check_neb(_neb_ns(tmp_path), _ctx(neb=neb))
    assert problems == [] and notes[0] == "the run stops a path at 0.1 eV/Å; neb_settings has fmax 0.05"


def test_a_second_run_from_finished_paths_is_reported(fake_batches, tmp_path):
    first = agent08_views.neb_log_data(_write_log(tmp_path / "first.csv", _GOOD_LOG), ["a", "b"])
    ns = _neb_ns(tmp_path, [(0, 0, 2, 0.04), (0, 1, 2, 0.03)])
    problems, notes = agent08_check._check_neb(ns, _ctx(first_log=first))
    assert len(problems) == 1 and problems[0].startswith("running neb.py a second time left a log with a single step")
    assert "paths.clone()" in problems[0]
    assert notes[1].startswith("b: barrier 23.06 kcal/mol, converged at step 2")  # the notes come from the first run


def _hook_neb(*criteria):
    """An NEB given convergence_hook=...: its fmax stays at the default and is not used."""

    hook = SimpleNamespace(criteria=list(criteria))
    return SimpleNamespace(climbing=object(), fmax=0.05, convergence_hook=hook)


def _criterion(threshold, key="forces", reduce_op="norm", custom_op=None):
    return SimpleNamespace(key=key, threshold=threshold, reduce_op=reduce_op, reduce_dims=-1, custom_op=custom_op)


def test_converged_uses_the_threshold_of_a_convergence_hook(fake_batches, tmp_path):
    """NEB(convergence_hook=ConvergenceHook.from_fmax(0.1, by_group=True)) stops paths at 0.1, not at neb.fmax."""

    log = [(0, 0, 1, 1.0), (0, 1, 1, 2.0), (1, 0, 2, 0.08), (1, 1, 2, 0.3)]
    ns = _neb_ns(tmp_path, log, converged=np.array([True, False]))
    ctx = _ctx(neb=_hook_neb(_criterion(0.1)))
    assert agent08_check._check_neb(ns, ctx)[0] == []
    ns["converged"] = np.array([False, True])
    assert agent08_check._check_neb(ns, ctx)[0] == [
        "converged is False for a; its last force in the log is 0.0800 eV/Å and the run stops a path at 0.1 eV/Å",
        "converged is True for b; its last force in the log is 0.3000 eV/Å and the run stops a path at 0.1 eV/Å",
    ]


def test_a_stop_rule_the_check_cannot_read_fails_with_one_line(fake_batches, tmp_path):
    """A wrong converged cannot pass unseen behind a hook the log cannot be compared with."""

    for neb in (_hook_neb(_criterion(0.1), _criterion(1e-6, key="energy_change")), _hook_neb(_criterion(0.1, reduce_op="max"))):
        assert agent08_check._run_threshold(neb) is None
        problems, notes = agent08_check._check_neb(_neb_ns(tmp_path), _ctx(neb=neb))
        assert problems == [
            "the check cannot read the force at which this NEB stops a path, so it cannot test converged; "
            "pass fmax=neb_settings['fmax'] to NEB"
        ]
        assert notes[0] == "a stopped at step 1 with force 0.0400 eV/Å"


def test_run_threshold_reads_a_real_convergence_hook():
    pytest.importorskip("nvalchemi")
    from nvalchemi.dynamics import ConvergenceHook

    neb = SimpleNamespace(fmax=0.05, convergence_hook=ConvergenceHook.from_fmax(0.1, by_group=True))
    assert agent08_check._run_threshold(neb) == 0.1
    assert agent08_check._run_threshold(SimpleNamespace(fmax=0.02, convergence_hook=None)) == 0.02


def test_verdict_column_is_the_one_named_saddle():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"stationary": ["yes", "no"], "saddle": ["no", "no"], "n_imaginary": [0, 1], "f": [0.1, 0.2]})
    assert agent08_check.saddle_answers(frame) == ("saddle", [False, False])
    assert agent08_check.saddle_answers(frame.rename(columns={"saddle": "verdict"})) == ("verdict", [False, False])
    assert agent08_check.saddle_answers(pd.DataFrame({"is_saddle": [True, False]})) == ("is_saddle", [True, False])
    assert agent08_check.saddle_answers(pd.DataFrame({"n_imaginary": [0, 1]})) is None  # 0 and 1 are counts


def _saddle_ns(answers, **columns):
    """Two paths of five one-atom images on the parabola; the top of each is image 2."""

    pd = pytest.importorskip("pandas")
    positions, energy, forces = _parabola([0.0, 0.5, 1.0, 1.5, 2.0])
    batch = FakeBatch([5, 5], [1, 1], np.concatenate(positions * 2))
    table = {"path": ["a"] * 5 + ["b"] * 5, "image": list(range(5)) * 2, "force": [0.5] * 10,
             "n_imaginary": [0, 0, 1, 0, 0] * 2, "saddle": answers}
    table.update(columns)
    return {
        "relaxed_paths": batch,
        "model": FakeModel(np.concatenate([energy, energy]), np.concatenate(forces * 2)),
        "reaction_names": ["a", "b"],
        "saddle_table": pd.DataFrame({k: v for k, v in table.items() if v is not None}),
    }


_TOPS = ["no", "no", "yes", "no", "no"] * 2


def test_saddle_contract_passes_and_reports_the_top_of_every_path(fake_batches):
    problems, notes = agent08_check._check_saddle(_saddle_ns(_TOPS), {})
    assert problems == []
    assert notes == [
        f"{name}: highest image 2, your answer yes, largest force 0.0000 eV/Å; the top of the fitted "
        "path is 0.00 Å along the path from it and 0.000 kcal/mol higher"
        for name in ("a", "b")
    ]


def test_a_no_for_the_top_image_passes_as_unexpected(fake_batches):
    """A yes or a no for the top image passes alike; the note carries the numbers and the word."""

    problems, notes = agent08_check._check_saddle(_saddle_ns(["no"] * 10), {})
    assert problems == [] and "your answer no" in notes[0]
    assert notes[0].endswith("kcal/mol higher. Unexpected: the top image of a converged path is usually a saddle point.")


def test_a_yes_is_held_to_a_force_limit_of_its_own(fake_batches):
    """The limit is the check's number, measured with a fresh model call, whatever the NEB stopped at."""

    assert agent08_check.SADDLE_FORCE_TOLERANCE == 0.005
    slope = ["no", "yes", "yes", "no", "no"] + _TOPS[5:]  # image 1 sits on the slope of the parabola
    line, not_top = agent08_check._check_saddle(_saddle_ns(slope, n_imaginary=[0, 1, 1, 0, 0] * 2), {})[0]
    assert line.startswith("a, image 1: the table says yes, and the largest force on an atom there is ")
    assert line.endswith("here at or below 0.005 eV/Å whatever force the NEB stopped at")
    assert not_top.startswith("a, image 1: the table says yes, and the image is not higher in energy")


def test_a_yes_must_be_a_top_of_its_path_with_forces_near_zero(fake_batches):
    """A force-only verdict gives a yes to images that sit in the minima. The step asks for both tests;
    the path holds the curvature: a saddle is higher than both neighbours."""

    ns = _saddle_ns(_TOPS)
    flat = ns["model"]
    energy = np.asarray(flat.energy, dtype=float).copy()
    energy[1] = energy[0]  # image 1 of path a in the reactant's well: no force along the line, not a top
    ns["model"] = FakeModel(energy, flat.forces)
    ns["relaxed_paths"].positions[1] = ns["relaxed_paths"].positions[0]
    answers = ["no", "yes", "yes", "no", "no"] + _TOPS[5:]
    ns["saddle_table"]["saddle"] = answers
    problems = agent08_check._check_saddle(ns, {})[0]
    assert "a, image 1: the table says yes, and the image is not higher in energy than both its neighbours on the path; " \
        "a saddle point on a path is a top: the images on both sides are lower" in problems


def test_a_yes_fails_when_the_tables_own_count_of_imaginary_modes_is_not_one(fake_batches):
    def lines(counts, name="n_imaginary"):
        return agent08_check._check_saddle(_saddle_ns(_TOPS, **{name: counts}), {})[0]

    assert lines([0, 0, 1, 0, 0] * 2) == []
    assert lines([0, 0, 2, 0, 0] + [0, 0, 1, 0, 0]) == [
        "a, image 2: the table says yes, and its column n_imaginary gives 2 imaginary modes; "
        "a saddle point has exactly one imaginary frequency"
    ]
    assert len(agent08_check._check_saddle(_saddle_ns(_TOPS, n_imaginary=None, **{"imaginary modes": [0] * 10}), {})[0]) == 2
    assert lines([0] * 10, name="lowest_imaginary_freq") == []  # a frequency, not a count: not read
    assert agent08_check.imaginary_count_column(__import__("pandas").DataFrame({"n_imag": [1], "x": [0.1]})) == "n_imag"


def _rigid_and_soft_hessian(masses):
    """A 3-atom Hessian: stiff in the vibrations, negative along the rigid rotations (a structure off its minimum)."""

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.3, 1.1, 0.4]])
    w = np.repeat(np.sqrt(masses), 3)
    r = positions - (masses[:, None] * positions).sum(0) / masses.sum()
    rot = np.stack([np.cross(np.broadcast_to(e, (3, 3)), r).reshape(-1) * w for e in np.eye(3)], axis=1)
    q, _ = np.linalg.qr(rot)
    hm = 2.0 * np.eye(9) - 3.0 * q @ q.T  # eigenvalues +2 (vibrations, translations) and -1 (rotations)
    return positions, hm * np.outer(w, w)


def test_projected_frequencies_drop_the_rigid_rotations():
    masses = np.array([14.0, 1.0, 1.0])
    positions, hessian = _rigid_and_soft_hessian(masses)
    raw = np.linalg.eigvalsh(hessian / np.outer(np.repeat(np.sqrt(masses), 3), np.repeat(np.sqrt(masses), 3)))
    assert int((raw < 0).sum()) == 3  # unprojected: the three rotations look imaginary
    f = agent08_check.projected_frequencies(hessian, positions, masses)
    assert len(f) == 3 and int((f < -agent08_check.IMAGINARY_MAYBE).sum()) == 0
    assert np.allclose(f, np.sqrt(2.0) * agent08_check.CM_PER_SQRT_EV_A2_AMU)


class _HessianModel(FakeModel):
    def compute_hessian(self, batch):  # the check takes the frequencies from image_frequencies, patched below
        raise AssertionError("not called")


def _recount_ns(counts, freqs, monkeypatch):
    ns = _saddle_ns(_TOPS, n_imaginary=counts)
    ns["model"] = _HessianModel(ns["model"].energy, ns["model"].forces)
    monkeypatch.setattr(agent08_check, "image_frequencies", lambda model, batch: freqs)
    return ns


def test_the_check_recounts_the_imaginary_modes_of_every_image(fake_batches, monkeypatch):
    """A table that gives 3 imaginary modes where the projected Hessian gives 1 is told so."""

    one, none = np.array([-500.0, 300.0]), np.array([200.0, 300.0])
    freqs = [none, none, one, one, none] * 2
    ns = _recount_ns([0, 0, 1, 1, 0] * 2, freqs, monkeypatch)
    assert agent08_check._check_saddle(ns, {})[0] == []
    ns = _recount_ns([0, 0, 1, 3, 0] + [0, 0, 1, 3, 0], freqs, monkeypatch)
    assert agent08_check._check_saddle(ns, {})[0] == [
        "the column n_imaginary differs from the check's count of imaginary modes at 2 of 10 images, first at a, "
        "image 3: 3 in the table, 1 in the check, which projects the rigid translations and rotations out of the "
        "model's Hessian before it counts"
    ]
    # a mode between -50 and -10 cm^-1 may be counted either way
    soft = [none, np.array([-30.0, 300.0]), one, none, none] * 2
    for counts in ([0, 0, 1, 0, 0] * 2, [0, 1, 1, 0, 0] * 2):
        assert agent08_check._check_saddle(_recount_ns(counts, soft, monkeypatch), {})[0] == []


def test_a_model_without_a_hessian_gives_a_note_not_a_failure(fake_batches, monkeypatch):
    ns = _recount_ns([0, 0, 1, 0, 0] * 2, "ValueError: no second derivatives", monkeypatch)
    problems, notes = agent08_check._check_saddle(ns, {})
    assert problems == [] and notes[-1] == (
        "the check could not take the model's Hessian (ValueError: no second derivatives), "
        "so it did not recount the imaginary modes"
    )


NO_COUNT = (
    "saddle_table has no column with the count of imaginary modes of each image; add one of whole numbers "
    'with "imaginary" in its name: a saddle point has exactly one imaginary frequency, and the check reads that count'
)


def test_a_table_without_a_count_of_imaginary_modes_fails_with_one_line(fake_batches):
    """A table with no count column must not skip the curvature test."""

    assert agent08_check._check_saddle(_saddle_ns(_TOPS, n_imaginary=None), {})[0] == [NO_COUNT]
    as_float = agent08_check._check_saddle(_saddle_ns(_TOPS, n_imaginary=[0.0, 0.0, 1.0, 0.0, 0.0] * 2), {})[0]
    assert as_float == [NO_COUNT]  # a count is whole numbers
    assert agent08_check._check_saddle(_saddle_ns(_TOPS), {})[0] == []


def test_saddle_messages(fake_batches):
    def lines(answers=_TOPS, **columns):
        return agent08_check._check_saddle(_saddle_ns(answers, **columns), {})[0]

    assert lines(["yes"] + _TOPS[1:]) == [
        "a: the table says yes for the first image, an end of the path; the ends are relaxed minima, "
        "so the answer for an end is no"
    ]
    assert lines(answers=None) == ["saddle_table has no column of yes or no; Done when asks for a yes or no for every image"]
    assert lines(force=None, n_imaginary=None) == [
        "saddle_table holds no numbers behind the answers; Done when asks for them", NO_COUNT
    ]
    assert lines(force=None, n_imaginary=[0, 0, 1, 0, 0] * 2) == []  # a count is a number too
    ns = _saddle_ns(_TOPS)
    ns["saddle_table"] = ns["saddle_table"].iloc[:4]
    assert agent08_check._check_saddle(ns, {})[0] == ["saddle_table has 4 rows; one per image of relaxed_paths gives 10"]


def test_the_table_must_name_the_path_and_the_image_of_every_row(fake_batches):
    """"Read one path from end to end" needs both columns."""

    def lines(**columns):
        return agent08_check._check_saddle(_saddle_ns(_TOPS, **columns), {})[0]

    assert lines(path=None) == [
        "saddle_table has no column that names the path of each row; add one that holds the names in reaction_names"
    ]
    assert lines(path=[0] * 5 + [1] * 5)[0].startswith("saddle_table has no column that names the path")
    assert lines(image=None) == [
        "saddle_table has no column with the image number of each row; add one that counts the images of each path from 0"
    ]
    assert lines(image=[0, 1, 2, 3, 3] + list(range(5))) == [
        "saddle_table has no column with the image number of each row; add one that counts the images of each path from 0"
    ]
    assert lines(path=["a"] * 4 + ["b"] * 6, image=[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]) == [
        "saddle_table does not list every image of a once; the path has images 0 to 4"
    ]
    assert lines(image=list(range(1, 6)) * 2) == []  # counting from 1 is read as well
    assert lines(reaction=["a"] * 5 + ["b"] * 5, path=None, image=None, k=list(range(5)) * 2) == []  # other column names


def test_rows_are_matched_by_path_and_image_when_the_table_is_sorted_differently(fake_batches):
    ns = _saddle_ns(_TOPS)
    ns["saddle_table"] = ns["saddle_table"].iloc[::-1].reset_index(drop=True)
    assert agent08_check._check_saddle(ns, {})[0] == []


# ---------------------------------------------------------------------------
# The check: running the files
# ---------------------------------------------------------------------------


@pytest.fixture
def light(tmp_path, monkeypatch):
    """Three light steps in a temporary work folder, with a setup cell that needs no model."""

    steps = {
        "one": agent08_check.Step("one.py", ("x",)),
        "two": agent08_check.Step("two.py", ("y", "z"), after=("one",)),
    }
    monkeypatch.setattr(agent08_check, "STEPS", steps)
    monkeypatch.setattr(agent08_check, "SETUP_CELL", f"from pathlib import Path\nwork_dir = Path({str(tmp_path)!r})\nprint('setup')")
    monkeypatch.setattr(
        agent08_check,
        "CHECKS",
        {"one": lambda ns, ctx: ([], []), "two": lambda ns, ctx: (["y is small"] if ns["y"] < 2 else [], ["y was read"])},
    )
    monkeypatch.setattr(agent08_check, "_WatchNEB", lambda: _NoWatch())
    return tmp_path


class _NoWatch:
    runs: list = []

    def on(self, model):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _main(step):
    return agent08_check.main(["agent08", "check", step])


def test_ok_line_and_notes(light, capsys):
    (light / "one.py").write_text("x = 1\nprint('from the file')")
    (light / "two.py").write_text("y = x + 1\nz = 0")
    assert _main("two") == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["two: note: y was read", "two: OK (2 names)"]  # the files' prints go to stderr


def test_problem_lines_fail_the_step(light, capsys):
    (light / "one.py").write_text("x = 0")
    (light / "two.py").write_text("y = x + 1\nz = 0")
    assert _main("two") == 1
    assert capsys.readouterr().out.splitlines() == ["two: note: y was read", "two: y is small"]


def test_missing_files_and_names(light, capsys):
    assert _main("one") == 1
    assert capsys.readouterr().out.strip() == f"one: the step file one.py is missing in {light}"
    assert _main("two") == 1
    assert capsys.readouterr().out.strip() == f"two: the earlier step file one.py is missing in {light}"
    (light / "one.py").write_text("x = 1")
    (light / "two.py").write_text("y = 5")
    assert _main("two") == 1
    assert capsys.readouterr().out.strip() == "two: z is missing; the step asks for it"


def test_a_failing_file_is_reported_in_one_line(light, capsys):
    (light / "one.py").write_text("raise RuntimeError('first line\\nsecond line')")
    assert _main("one") == 1
    assert capsys.readouterr().out.strip() == "one: running one.py failed: RuntimeError: first line"
    (light / "two.py").write_text("y = z = 2")
    assert _main("two") == 1
    assert capsys.readouterr().out.strip() == "two: an earlier file failed: one.py: RuntimeError: first line"
    (light / "one.py").write_text("x = undefined_name + 1")
    assert _main("one") == 1
    assert capsys.readouterr().out.strip() == (
        "one: one.py uses undefined_name, which no earlier cell or file defines; define it in this file"
    )


def test_a_file_that_uses_dunder_file_is_told_to_remove_that_line(light, capsys):
    (light / "one.py").write_text("from pathlib import Path\nhere = Path(__file__).parent\nx = 1")
    assert _main("one") == 1
    assert capsys.readouterr().out.strip() == (
        "one: one.py uses __file__, which does not exist in a notebook cell; remove that line"
    )


def test_the_step_file_runs_twice(light, capsys):
    (light / "one.py").write_text("x = 1\nruns = globals().get('runs', 0) + 1\nassert runs < 2, 'not again'")
    assert _main("one") == 1
    assert capsys.readouterr().out.strip() == "one: running one.py a second time failed: AssertionError: not again"


def test_main_prints_usage_for_an_unknown_step(capsys):
    assert agent08_check.main(["agent08", "check", "nonsense"]) == 2
    assert agent08_check.main(["agent08"]) == 2
    assert "usage: python -m helpers.agent08 check <paths|neb|saddle>" in capsys.readouterr().err


def test_check_accepts_names_paths_and_numbers():
    for spelled, expected in (("2", "neb"), ("neb", "neb"), ("neb.py", "neb"), ("agent08_work/saddle.py", "saddle"), ("3", "saddle")):
        assert agent08_check.resolve_step(spelled) == expected
    assert agent08_check.resolve_step("4") == "4"
    assert agent08_check.resolve_step("barriers") == agent08_check.resolve_step("barriers.py") == "neb"  # the merged step


def test_every_printed_line_is_in_one_place():
    """The lines a learner can see: LINES in the check, LABELS in the views, MESSAGES in the helper."""

    source = (Path(agent08_check.__file__)).read_text()
    body = source[source.index("# Small tools") :]
    assert "problems.append(f\"" not in body and "return [f\"{" not in body.replace('return [f"{step}: "', "").replace('return [f"{step}: {line}"', "")
    assert {"unexpected", "none", "several"} <= set(agent08_views.LABELS["profiles"]) | set(agent08_views.LABELS["saddle"])
    assert set(agent08.MESSAGES) == {"ends_not_relaxed", "unknown_reaction"}


def test_neb_watch_counts_runs_and_puts_the_class_back():
    pytest.importorskip("nvalchemi")
    from nvalchemi.dynamics.mep import NEB

    before = NEB.run
    assert "run" not in NEB.__dict__
    with agent08_check._WatchNEB() as watch:
        assert NEB.run is not before and watch.runs == []
        mine, other = object(), object()
        watch.runs += [SimpleNamespace(model=mine), SimpleNamespace(model=other), SimpleNamespace(model=mine)]
        assert len(watch.on(mine)) == 2  # a pre-pass on another model is not counted
    assert NEB.run is before and "run" not in NEB.__dict__


def test_a_setup_cell_that_fails_is_one_line(monkeypatch, capsys):
    """A setup cell that fails gives one line, with no traceback."""

    monkeypatch.setattr(agent08_check, "SETUP_CELL", "raise RuntimeError('D3 parameter file is missing: /x/dftd3_parameters.pt')")
    assert agent08_check.main(["agent08", "check", "paths"]) == 1
    assert capsys.readouterr().out.strip() == (
        "paths: the notebook's setup cell failed: RuntimeError: D3 parameter file is missing: /x/dftd3_parameters.pt"
    )
