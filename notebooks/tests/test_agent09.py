# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent 09, Small-batch distilled (liquid SiC): session, data, rules, views, step contracts and check messages.

No GPU needed. Only the last test loads MACE-MPA-0; it skips when the weights are not on this machine.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import agent09 as helpers
from helpers import agent09_check as check
from helpers import agent09_views as views

CUT = _NOTEBOOK_DIR / "data" / "agent09" / "sic-tgap" / "cut"
SOLUTIONS = _NOTEBOOK_DIR.parent / "solutions" / "agent-09"
NOTEBOOK = _NOTEBOOK_DIR.parent / helpers.NOTEBOOK


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("ALCHEMI_AGENT09_WORK_DIR", str(tmp_path))
    return tmp_path


# --- session and setup -------------------------------------------------------


def test_setup_cell_is_ten_lines_and_names_every_budget():
    lines = helpers.SETUP_CELL.splitlines()
    assert len(lines) <= 10
    assert lines[0].startswith("# ")
    budget_line = next(line for line in lines if line.startswith("ft_steps"))
    assert budget_line.split("=")[0].replace(" ", "").split(",") == list(helpers.BUDGET_NAMES)
    assert "120, 16, 400, 1500, 800" in budget_line
    assert any(line.startswith("starts, test_starts = helpers.load_starts(n_replicas)") for line in lines)


def test_show_session_text(capsys):
    helpers.show_session()
    out = capsys.readouterr().out
    assert "from helpers import agent09 as helpers" in out
    assert "python -m helpers.agent09 check <step>" in out
    assert "notebooks/agent09_work/" in out


def test_configure_tutorial_installs_the_torch_jit_filter():
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")  # what a kernel shows without the filter
        helpers.configure_tutorial()
        spec = helpers.JIT_DEPRECATION_FILTER
        installed = [
            f for f in warnings.filters
            if f[0] == "ignore" and f[2] is DeprecationWarning and f[1] is not None
            and f[1].pattern == spec["message"] and f[3] is not None and f[3].pattern == spec["module"]
        ]
        assert installed, "configure_tutorial did not install the torch.jit filter"
        seen.clear()
        for module, text in (
            ("torch.jit._script", "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`."),
            ("torch.jit._serialization", "`torch.jit.load` is deprecated. Please switch to `torch.export`."),
        ):
            warnings.warn_explicit(text, DeprecationWarning, "x.py", 1, module=module)
        assert not seen
        # narrow: the same words from another module, or another notice from torch.jit, still show
        warnings.warn_explicit("`torch.jit.script` is deprecated.", DeprecationWarning, "y.py", 1, module="mine")
        warnings.warn_explicit("something else", DeprecationWarning, "x.py", 1, module="torch.jit._script")
        assert len(seen) == 2


def test_every_public_name_exists():
    for name in helpers.__all__:
        assert hasattr(helpers, name), name


def test_steps_follow_their_order_and_have_samples():
    names = list(check.STEPS)
    assert names == ["finetune", "sample_label", "distill", "run_student", "structure"]
    for index, (name, step) in enumerate(check.STEPS.items()):
        assert set(step.after) <= set(names[:index])
        assert (SOLUTIONS / step.file).is_file(), step.file


def test_samples_leave_the_names_of_their_contract():
    for step in check.STEPS.values():
        text = (SOLUTIONS / step.file).read_text()
        for name in step.names:
            assert re.search(rf"^{name}\s*=|^{name},|, {name}\s*=", text, re.M), (step.file, name)


def test_solutions_folder_holds_only_this_design():
    files = sorted(p.name for p in SOLUTIONS.iterdir() if not p.name.startswith("."))
    assert files == sorted([*(s.file for s in check.STEPS.values()), "prompts.md", "solutions.md"])


def _notebook_code_cells() -> list[str]:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def test_notebook_loads_the_same_steps():
    """The notebook has one ``%load`` cell per step, in the order of the check."""

    loads = [re.fullmatch(r"%load agent09_work/(\w+\.py)", cell.strip()) for cell in _notebook_code_cells()]
    assert [m.group(1) for m in loads if m] == [s.file for s in check.STEPS.values()]


def test_notebook_setup_cell_is_the_one_the_check_runs():
    assert helpers.SETUP_CELL in _notebook_code_cells()


# --- the data ----------------------------------------------------------------


def test_cut_has_both_files_a_record_and_a_licence():
    record = json.loads((CUT / "cut-record.json").read_text())
    assert record["train"]["frames"] == 300 and record["held_out"]["frames"] == 60
    assert record["train"]["by_type"] == {"melt_aimd": 100, "eqlMelt_aimd": 100, "quench_aimd": 50, "amorph_aimd": 50}
    assert list(record["held_out"]["by_density"]) == ["2.7"]
    assert "2.7" not in record["train"]["by_density"]
    for name, sha in record["sha256"].items():
        file = CUT / (name.split(" ")[0] + ".gz")
        assert hashlib.sha256(gzip.decompress(file.read_bytes())).hexdigest() == sha, name
    licence = (CUT / "LICENSE").read_text()
    assert "CC BY 4.0" in licence and "10.5281/zenodo.17182227" in licence and "Changes made" in licence
    assert "10.5281/zenodo.17182227" in (CUT.parent / "README.md").read_text()


def test_starts_are_equilibrated_liquid_of_the_training_densities():
    starts, test_starts = helpers.load_starts(16)
    assert len(starts) == len(test_starts) == 16
    for atoms in [*starts, *test_starts]:
        assert len(atoms) == 216 and atoms.pbc.all() and atoms.get_chemical_formula() == "C108Si108"
        assert atoms.info["density"] != 2.7
        density, temperature = helpers.target_temperature(atoms.cell.array)
        assert density == atoms.info["density"] and temperature == atoms.info["temperature_K"]
    # sixteen starts over the five training densities: 6 at 3000 K (2.3 and 2.5 g/cm3), then 4, 3 and 3
    assert [a.info["temperature_K"] for a in starts] == [3000.0] * 6 + [3400.0] * 4 + [3700.0] * 3 + [4000.0] * 3
    assert [a.info["density"] for a in starts] == [a.info["density"] for a in test_starts]
    assert not {a.info["source_index"] for a in starts} & {a.info["source_index"] for a in test_starts}
    small, _ = helpers.load_starts(2)
    assert [a.info["density"] for a in small] == [2.3, 3.2]


def test_density_of_a_course_cell():
    starts, _ = helpers.load_starts(16)
    assert abs(helpers.density_of(starts[0].cell.array) - 2.3) < 0.01
    assert abs(helpers.density_of(starts[-1].cell.array) - 3.2) < 0.01


# --- stores for the rules -----------------------------------------------------


def _store(path, snapshots, copies=2, jitter=0.0, walk=0.0, seed=0):
    """A store written snapshot after snapshot: ``copies`` liquid starts, each moved a little per snapshot."""

    import torch
    from nvalchemi.data import AtomicData
    from nvalchemi.data.datapipes import AtomicDataZarrWriter

    rng = np.random.default_rng(seed)
    starts, _ = helpers.load_starts(copies)
    frames, offsets = [], [np.zeros((216, 3)) for _ in starts]
    for _ in range(snapshots):
        for c, atoms in enumerate(starts):
            offsets[c] = offsets[c] + walk * rng.standard_normal((216, 3))
            moved = atoms.copy()
            moved.positions = atoms.positions + offsets[c] + jitter * rng.standard_normal((216, 3))
            data = AtomicData.from_atoms(moved)
            data.forces = torch.zeros(216, 3)
            data.energy = torch.zeros(1, 1)
            data.velocities = torch.zeros(216, 3)
            frames.append(data)
    AtomicDataZarrWriter(str(path)).write(frames)
    return path


def test_copies_in_store_finds_the_snapshot_size(tmp_path):
    pytest.importorskip("nvalchemi")
    path = _store(tmp_path / "s.zarr", snapshots=3, copies=2, walk=0.05)
    _, batch = helpers.read_store(path)
    assert helpers.copies_in_store(batch) == 2


def test_still_liquid_judges_only_long_runs_and_flags_atoms_that_stay_put(tmp_path):
    pytest.importorskip("nvalchemi")
    path = _store(tmp_path / "s.zarr", snapshots=5, copies=2, jitter=0.01)
    short = helpers.still_liquid(path, steps=40)
    assert not short["judged"] and short["copies"] == 2
    long = helpers.still_liquid(path, steps=400)
    assert long["judged"] and long["n_liquid"] == 0
    assert any("stopped moving apart" in reason for reason in long["reasons"])
    assert any("target temperature" in reason for reason in long["reasons"])
    assert "Unexpected" in views.stability_line(long)
    assert "judged from 200 steps on" in views.stability_line(short)


def _rdf(peak, frames=200, width=0.3):
    import torch
    from nvalchemi.training.distillation.evaluation.stability import RadialDistribution

    edges = torch.linspace(0.0, 6.0, 61, dtype=torch.float64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    g = torch.exp(-((centres - peak) / width) ** 2) + 0.01
    return RadialDistribution(edges=edges, g_r=g, counts=g * 1000, num_frames=frames, num_atoms=frames * 216, r_max=6.0)


def test_follows_teacher_reads_the_divergence_and_the_partial_peaks():
    pytest.importorskip("nvalchemi")
    same = helpers.follows_teacher(_rdf(1.85), _rdf(1.85), {"SiC": (_rdf(1.85), _rdf(1.85))})
    assert same["judged"] and same["follows"] and "follows the teacher's" in views.structure_line(same)
    shifted = helpers.follows_teacher(_rdf(1.85), _rdf(1.85), {"SiSi": (_rdf(2.45), _rdf(2.75))})
    assert not shifted["follows"] and "first Si-Si peak at 2.75 Å" in shifted["reasons"][0]
    apart = helpers.follows_teacher(_rdf(1.85), _rdf(2.95))
    assert not apart["follows"] and "Unexpected" in views.structure_line(apart)
    few = helpers.follows_teacher(_rdf(1.85, frames=6), _rdf(1.85, frames=6))
    assert not few["judged"] and "Too few frames" in views.structure_line(few)


def test_make_student_has_34464_weights_on_a_si_c_store(tmp_path):
    pytest.importorskip("mace")
    path = _store(tmp_path / "s.zarr", snapshots=2, copies=2, jitter=0.05)
    student = helpers.make_student(path)
    assert sum(p.numel() for p in student.parameters()) == 34_464
    again = helpers.make_student(path)
    assert all(bool((a == b).all()) for a, b in zip(student.parameters(), again.parameters()))


# --- lines and figures --------------------------------------------------------


def _rows(published, finetuned, student=None):
    def models(e0, f0, e1, f1, e2=None, f2=None):
        out = {"published": {"energy": e0, "force": f0}, "finetuned": {"energy": e1, "force": f1}}
        if e2 is not None:
            out["student"] = {"energy": e2, "force": f2}
        return out

    return {
        "Melting run": {"frames": 20, **models(*published, *finetuned, *(student or ()))},
        views.ALL_FRAMES: {"frames": 60, **models(*published, *finetuned, *(student or ()))},
    }


def test_errors_line_says_fell_or_unexpected():
    fell = views.errors_line(_rows((62.7, 0.463), (18.0, 0.354)), ["published", "finetuned"])
    assert fell.startswith("On all 60 held-out frames the energy error fell from 62.7 to 18.0")
    rose = views.errors_line(_rows((62.7, 0.463), (70.0, 0.354)), ["published", "finetuned"])
    assert rose.startswith("Unexpected:") and "energy error rose" in rose
    student = views.errors_line(_rows((62.7, 0.463), (18.0, 0.354), (40.0, 0.41)), ["published", "finetuned", "student"])
    assert "between the fine-tuned model's 0.354 and the published model's 0.463" in student


def test_student_and_speed_lines():
    assert views.student_line(0.06, 0.4, 18.0).startswith("On its own frames the student's forces differ")
    assert views.student_line(0.3, 0.4, 18.0).startswith("Unexpected:")
    fast = views.speed_line({"teacher_steps_per_s": 3.6, "student_steps_per_s": 92.0, "copies": 16})
    assert "26 times as many MD steps per second" in fast
    assert views.speed_line({"teacher_steps_per_s": 3.6, "student_steps_per_s": 2.0, "copies": 16}).startswith("Unexpected")
    # the line says how the helper timed it: three runs on a GPU, one on a CPU
    gpu = views.speed_line({"teacher_steps_per_s": 3.6, "student_steps_per_s": 92.0, "copies": 16,
                            "teacher_runs_s": [1.0, 1.1, 1.2], "steps": 50})
    assert gpu.endswith("Timed after one warm-up: the median of 3 runs of 50 steps per model.")
    cpu = views.speed_line({"teacher_steps_per_s": 0.5, "student_steps_per_s": 17.0, "copies": 2,
                            "teacher_runs_s": [6.0], "steps": 3})
    assert cpu.endswith("Timed after one warm-up: one run of 3 steps per model.")
    assert helpers.TIMING_REPEATS == 3 and helpers.TIMING_REPEATS_CPU == 1


def test_error_table_has_two_short_header_rows():
    labels = views.LABELS["errors"]
    keys = ["published", "finetuned", "student"]
    html = views.html_table(
        [labels["frames_kind"], labels["count"], *[labels[k] for k in keys] * 2],
        [["All held-out frames", 60, "62.7", "19.0", "40.0", "0.463", "0.354", "0.433"]],
        [(labels["energy"], 3), (labels["force"], 3)],
    )
    head = html.split("<tbody>")[0]
    assert head.count("<tr>") == 2 and "rowspan='2'>Frames<" in head
    assert "colspan='3'>Energy error [meV/atom]<" in head and "colspan='3'>Force error [eV/Å]<" in head
    assert max(len(re.sub("<[^>]+>", "", cell)) for cell in head.split("</th>")) <= len("Energy error [meV/atom]")
    assert html.count("<td") == 8


def test_no_helper_line_says_wrong():
    text = " ".join(str(v) for k, v in vars(views).items() if k.isupper() and isinstance(v, str))
    assert "wrong" not in text.lower()
    assert "—" not in text and "·" not in text


def test_every_figure_has_an_alternative_text():
    for key in ("errors", "student", "loss", "structure", "speed"):
        assert len(views.LABELS[key]["alt"]) > 20, key


def test_figures_draw():
    views.build_parity_figure([("a", [0.0, 1.0], [0.1, 0.9]), ("b", [0.0, 1.0], [0.0, 1.0])])
    views.build_loss_figure([1, 2, 3], [3.0, 2.0, 1.0])
    views.build_force_figure([1.0, 2.0], [1.1, 1.9])
    r = np.linspace(0.05, 5.95, 60)
    views.build_structure_figure([("All pairs", r, np.ones(60), np.ones(60)), ("Si-C", r, np.ones(60), np.ones(60))])
    views.build_speed_figure(3.6, 92.0)
    with pytest.raises(ValueError):
        views.build_loss_figure([1, 2], [1.0])


# --- check messages ----------------------------------------------------------


def test_main_prints_usage_and_unknown_steps(capsys):
    assert check.main(["x"]) == 2
    assert "finetune|sample_label|distill|run_student|structure" in capsys.readouterr().out
    assert check.main(["x", "check", "nothing"]) == 2
    assert "unknown step 'nothing'" in capsys.readouterr().out


def test_a_missing_step_file_is_reported(work):
    problems, notes = check.check_step("finetune")
    assert problems == [f"finetune: the step file finetune.py is missing in {work}"]
    assert notes == []


def test_a_file_that_uses_dunder_file_is_told_to_remove_that_line(tmp_path):
    path = tmp_path / "run_student.py"
    path.write_text("from pathlib import Path\nhere = Path(__file__).parent\n")
    error = check._run_file({"__name__": "__main__"}, path)
    assert check._error_line("run_student", error, 1) == (
        "run_student: run_student.py uses __file__, which does not exist in a notebook cell; remove that line"
    )


def test_an_undefined_ordinary_name_is_still_asked_for_in_the_file(tmp_path):
    path = tmp_path / "distill.py"
    path.write_text("student_model = trained_student\n")
    error = check._run_file({"__name__": "__main__"}, path)
    assert check._error_line("distill", error, 1).endswith(
        "uses trained_student, which is not defined in an earlier file; define it in this file"
    )


def test_loss_that_does_not_fall_is_unexpected_not_a_problem():
    problems, notes = [], []
    check._loss_lines("distill", "loss_history", {"loss_history": [(5, 1.0), (10, 2.0)]}, problems, notes)
    assert problems == [] and notes[0].startswith("distill: unexpected: the loss in loss_history did not fall")
    problems = []
    check._loss_lines("finetune", "ft_loss_history", {"ft_loss_history": [1.0]}, problems, [])
    assert problems == ["finetune: ft_loss_history needs at least two finite loss values; found 1"]


def test_a_small_store_is_unexpected_and_one_frame_is_a_problem():
    class _Reader:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    problems, notes = [], []
    check._enough("sample_label", "labeled_path", _Reader(12), problems, notes)
    assert problems == [] and notes[0].startswith("sample_label: unexpected: the store at labeled_path holds only 12")
    check._enough("sample_label", "labeled_path", _Reader(1), problems, notes)
    assert problems == ["sample_label: the store at labeled_path holds 1 structure; an MD saves many frames"]


def test_stability_must_be_the_monitor_metrics():
    problems = []
    ns = {"stability": {"first_divergence_step": None}, "test_steps": 800}
    assert not check._is_stability_of_run("run_student", ns, problems)
    assert problems[0].startswith("run_student: stability is dict; it should be the metrics of the StabilityMonitor")


def test_structure_needs_radial_distributions():
    pytest.importorskip("nvalchemi")
    problems = []
    check._check_structure({"teacher_rdf": [1.0], "student_rdf": _rdf(1.85)}, problems, [], {})
    assert problems == [
        "structure: teacher_rdf is list; it should be the RadialDistribution that Toolkit's radial_distribution returns"
    ]


def test_documented_small_setting_has_one_value_per_budget():
    """The small CPU setting written in solutions.md gives every budget of the setup cell a value."""

    text = (SOLUTIONS / "solutions.md").read_text(encoding="utf-8")
    budgets = re.search(r"ALCHEMI_AGENT09_BUDGETS=([\d,]+)", text).group(1)
    assert len(budgets.split(",")) == len(helpers.BUDGET_NAMES)
    assert re.search(r"ALCHEMI_AGENT09_SUBSET=\d+,\d+\b", text)


# --- with the pretrained model -----------------------------------------------


def _mpa_0_weights() -> bool:
    """True when MACE-MPA-0 loads without a download: bundled or in the MACE cache."""

    try:
        from mace.calculators.foundations_models import local_model_path, mace_mp_urls
        from mace.tools.utils import get_cache_dir
    except Exception:
        return False
    cached = "".join(c for c in Path(mace_mp_urls["medium-mpa-0"]).name if c.isalnum() or c == "_")
    return Path(local_model_path).is_file() or (Path(get_cache_dir()) / cached).is_file()


@pytest.mark.skipif(not _mpa_0_weights(), reason="MACE-MPA-0 weights are not on this machine")
def test_prepare_finetuning_scores_the_published_model_by_frame_kind(work, monkeypatch):
    import torch

    monkeypatch.setenv("ALCHEMI_AGENT09_SUBSET", "4,3")
    _, test_path, base = helpers.prepare_finetuning(torch.device("cpu"))
    result = helpers.score(base, test_path)
    assert result["frames"] == 3 and result["atoms"] == 3 * 216
    rows = helpers.errors_by_kind(test_path, {"published": result})
    assert list(rows) == ["Melting run", "Equilibrated liquid", "Quench and amorphous", views.ALL_FRAMES]
    assert all(np.isfinite(r["published"]["energy"]) and r["published"]["force"] > 0 for r in rows.values())
    assert json.loads((work / "reference_energies_4_3.json").read_text())["rank"] == 1
