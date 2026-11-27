# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent 04 owner tests: the saved default, the matrix Coulomb term, the check lines and the samples.

The light tests need no GPU. The sample tests run every step check on the sample
solutions and on each deliberately broken variant; they load AIMNet2 and are
skipped unless ``ALCHEMI_RUN_SAMPLES=1`` (GPU or CPU, a few minutes).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import agent04, agent04_check

SOLUTIONS = _NOTEBOOK_DIR.parent / "solutions" / "agent-04"
README = _NOTEBOOK_DIR / "data" / "agent04" / "README.md"


# ---------------------------------------------------------------------------
# The saved default
# ---------------------------------------------------------------------------


def test_saved_default_is_the_documented_file():
    digest = hashlib.sha256(agent04.DEFAULT_DIPEPTIDE.read_bytes()).hexdigest()
    assert digest in README.read_text()


def test_saved_default_holds_139_neutral_starts_of_22_atoms(capsys):
    frames = agent04.load_dipeptide_grid()
    assert len(frames) == 139
    assert {len(f) for f in frames} == {22}
    assert {int(f.info["charge"]) for f in frames} == {0}
    assert set(frames[0].get_chemical_symbols()) == {"H", "C", "N", "O"}
    assert "139 of 144 starts kept (5 with clashing atoms dropped)" in capsys.readouterr().out


def test_saved_default_equals_a_fresh_build():
    saved = agent04.load_dipeptide_grid()
    built, total = agent04.build_dipeptide_grid(12)
    assert total == 144 and len(built) == len(saved)
    for a, b in zip(saved, built):
        assert (a.info["phi"], a.info["psi"]) == (b.info["phi"], b.info["psi"])
        assert np.allclose(a.positions, b.positions, atol=1e-5)


def test_saved_default_has_no_clash_and_every_angle_on_the_grid():
    frames = agent04.load_dipeptide_grid()
    assert min(agent04._min_distance(f) for f in frames) >= 0.7
    grid = {-180.0 + 30.0 * k for k in range(12)}
    assert {f.info["phi"] for f in frames} <= grid and {f.info["psi"] for f in frames} <= grid


def test_other_grid_sizes_are_built(capsys):
    frames = agent04.load_dipeptide_grid(4)
    assert 0 < len(frames) <= 16
    assert "of 16 starts kept" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The matrix Coulomb term
# ---------------------------------------------------------------------------


def test_coulomb_term_reads_the_matrix_list():
    term = agent04.DirectCoulombAdapter()
    config = term.model_config.neighbor_config
    assert config.format.value == "matrix" and config.cutoff == 15.0 and config.half_list is False
    assert term.model_config.required_inputs == {"partial_charges"}


def test_coulomb_pairs_from_a_matrix_equal_the_brute_force_sum():
    torch = pytest.importorskip("torch")
    from nvalchemi.data import AtomicData, Batch

    positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.0, 0.0], [30.0, 0.0, 0.0]], dtype=torch.float32)
    charges = torch.tensor([0.4, -0.3, 0.1, 0.2], dtype=torch.float32)
    data = AtomicData(positions=positions, atomic_numbers=torch.tensor([1, 8, 1, 1]))
    batch = Batch.from_data_list([data])
    batch.__dict__["partial_charges"] = charges.reshape(-1, 1)  # what AIMNet2 would leave for the Coulomb term
    # the neighbor matrix the pipeline would build: every atom with the others within 15 A, padded with n_atoms
    n = 4
    near = [[j for j in range(n) if j != i and float((positions[i] - positions[j]).norm()) <= 15.0] for i in range(n)]
    width = max(len(row) for row in near)
    matrix = torch.tensor([row + [n] * (width - len(row)) for row in near], dtype=torch.int32)
    batch.__dict__["neighbor_matrix"] = matrix
    batch.__dict__["num_neighbors"] = torch.tensor([len(row) for row in near], dtype=torch.int32)
    term = agent04.DirectCoulombAdapter()
    inputs = term.adapt_input(batch)
    assert inputs["neighbor_pairs"].shape[1] == 2
    energy = term.native(**inputs)
    expected = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = float((positions[i] - positions[j]).norm())
            if r <= 15.0:
                expected += agent04._CourseDirectCoulombAdapter(15.0).native.coulomb_constant * float(charges[i] * charges[j]) / r
    assert float(energy.sum()) == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# The check lines
# ---------------------------------------------------------------------------


def test_every_line_has_its_fields_and_no_forbidden_punctuation():
    for key, line in agent04_check.LINES.items():
        assert "—" not in line and "·" not in line, key
        fields = set(re.findall(r"{(\w+)", line))
        assert fields <= {"step", "n", "file", "folder", "error", "name", "shape", "got", "k", "cutoff", "floor", "term", "mae",
                          "gap", "no_d3", "composed", "base", "threshold", "steps"}, (key, fields)


def test_coulomb_floor_is_the_fine_control_example_range():
    assert agent04_check.MIN_COULOMB_RANGE == 12.0


def test_relax_settings_hold_the_force_threshold():
    settings = agent04.relax_settings()
    assert set(settings) == {"fmax"} and settings["fmax"] == agent04.RELAX_FMAX
    assert 0.0 < settings["fmax"] <= 0.05  # eV/Å
    assert "relax_settings" in agent04.__all__


def test_conformer_sample_takes_the_threshold_from_the_helper():
    assert 'helpers.relax_settings()["fmax"]' in (SOLUTIONS / "conformers.py").read_text()


def test_unexpected_lines_say_the_step_still_passes():
    for key in ("unexpected_order", "unexpected_composed", "unexpected_higher", "unexpected_force"):
        assert "unexpected" in agent04_check.LINES[key] and "still passes" in agent04_check.LINES[key]


def test_missing_file_line_names_the_file_and_folder(tmp_path):
    line = agent04_check.check_step("compose_build", work_dir=tmp_path)[0]
    assert line == agent04_check.LINES["file_missing"].format(step="compose_build", file="compose_build.py", folder=tmp_path)


def test_a_file_that_uses_dunder_file_is_told_to_remove_that_line(tmp_path):
    (tmp_path / "compose_build.py").write_text("from pathlib import Path\nhere = Path(__file__).parent\n")
    lines = agent04_check.check_step("compose_build", work_dir=tmp_path)
    assert lines == [
        "compose_build: compose_build.py uses __file__, which does not exist in a notebook cell; remove that line"
    ], lines


def test_other_undefined_names_still_report_the_error(tmp_path):
    (tmp_path / "compose_build.py").write_text("x = undefined_name\n")
    lines = agent04_check.check_step("compose_build", work_dir=tmp_path)
    assert lines == ["compose_build: compose_build.py stops with NameError: name 'undefined_name' is not defined"], lines


def test_error_lines_say_unexpected_not_wrong():
    for key in ("scan_error", "ncia_error"):
        assert "unexpected" in agent04_check.LINES[key] and "wrong" not in agent04_check.LINES[key]
    assert not [key for key, line in agent04_check.LINES.items() if "wrong" in line]


def test_session_lines_name_the_check_and_the_order_of_files(capsys):
    agent04.show_session()
    lines = capsys.readouterr().out.splitlines()
    assert lines[-4:] == [
        "  check with  python -m helpers.agent04 check <step>",
        "  each file   runs in a notebook cell (%load), after the earlier files",
        "  always      run the check from notebooks/ after you write a file, and fix",
        "              what it reports; <step> is the file name without .py",
    ]


def test_every_step_name_is_its_file_name_without_py():
    for name, step in agent04_check.STEPS.items():
        assert step.file == f"{name}.py", (name, step.file)


def test_samples_exist_for_every_step():
    for step in agent04_check.STEPS.values():
        assert (SOLUTIONS / step.file).exists(), step.file
    assert (SOLUTIONS / "solutions.md").exists()


# ---------------------------------------------------------------------------
# Samples and broken variants (load AIMNet2; skipped unless ALCHEMI_RUN_SAMPLES=1)
# ---------------------------------------------------------------------------

BROKEN = {
    "compose_build__half_batch.py": ("problem", "scan_batch holds 45 structures"),
    "compose_build__charge_lost.py": ("problem", "changed the total charge of 20 of 90 structures"),
    "compose_run__short_range.py": ("problem", "range of 5 Å; use at least 12 Å"),
    "compose_run__forces_outside.py": ("problem", "outside the summed energy"),
    "no_d3__with_d3.py": ("problem", "no_d3_model still has a D3 term"),
    "no_d3__course_coo_adapter.py": ("problem", "Toolkit 0.3.0 cannot reset the neighbor list"),
    "no_d3__second_run_fails.py": ("problem", "stops when it runs a second time"),
    "no_d3__unexpected_order.py": ("note", "unexpected: the error without D3"),
    "no_d3__bare_model.py": ("problem", "no_d3_model has no electrostatics term"),
    "conformers__not_relaxed.py": ("problem", "no atom moved between conformer_batch and relaxed_conformers"),
    "conformers__missing_start.py": ("problem", "the batches hold 10 conformers; the loader gives 139"),
    "conformers__few_steps.py": ("problem", f"conformers still have a force above {agent04.RELAX_FMAX:g} eV/Å"),
}

run_samples = pytest.mark.skipif(os.environ.get("ALCHEMI_RUN_SAMPLES") != "1", reason="set ALCHEMI_RUN_SAMPLES=1 to run the model checks")


def _work_folder(tmp_path: Path, variant: str | None = None) -> Path:
    for name in agent04_check.STEPS.values():
        shutil.copy(SOLUTIONS / name.file, tmp_path / name.file)
    if variant:
        step = variant.split("__")[0]
        shutil.copy(SOLUTIONS / "broken" / variant, tmp_path / agent04_check.STEPS[step].file)
    return tmp_path


@run_samples
@pytest.mark.parametrize("step", list(agent04_check.STEPS))
def test_each_check_passes_on_its_sample(step, tmp_path):
    lines = agent04_check.check_step(step, work_dir=_work_folder(tmp_path))
    assert lines[0] == agent04_check.LINES["ok"].format(step=step, n=len(agent04_check.STEPS[step].names)), lines
    assert not [l for l in lines[1:] if "unexpected" in l], lines


@run_samples
@pytest.mark.parametrize("variant", list(BROKEN))
def test_each_check_flags_its_broken_variant(variant, tmp_path):
    kind, expected = BROKEN[variant]
    step = variant.split("__")[0]
    lines = agent04_check.check_step(step, work_dir=_work_folder(tmp_path, variant))
    text = "\n".join(lines)
    assert expected in text, lines
    if kind == "problem":
        assert ": OK (" not in lines[0], lines
    else:
        assert ": OK (" in lines[0], lines  # unexpected, and the step still passes


@run_samples
def test_fine_control_example_passes_its_check(tmp_path):
    """The Fine control example of the lesson, "a 12 Å cutoff for the coulomb term", passes ``compose_run``."""
    work = _work_folder(tmp_path)
    shutil.copy(SOLUTIONS / "fine_control" / "compose_run__cutoff_12.py", work / "compose_run.py")
    lines = agent04_check.check_step("compose_run", work_dir=work)
    assert lines == [agent04_check.LINES["ok"].format(step="compose_run", n=3)], lines

