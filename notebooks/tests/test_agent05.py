# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pytest
import torch

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

import IPython.display as ipy_display
from helpers import agent05
from nvalchemi.data import AtomicData, Batch


@pytest.fixture(scope="module")
def pairs():
    return agent05.load_pair_scans()


def _molecule_batch(count=None):
    atoms = agent05.load_molecules()[:count]
    return Batch.from_data_list([AtomicData.from_atoms(a) for a in atoms])


def test_default_molecules_fit_the_model_card():
    molecules = agent05.load_molecules()
    assert [m.info["name"] for m in molecules] == ["ethanol", "acetate", "aspirin", "paracetamol", "caffeine"]
    allowed = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
    assert sum(len(m) for m in molecules) == 81
    for molecule in molecules:
        assert len(molecule) <= 30
        assert -2 <= molecule.info["charge"] <= 2
        assert set(molecule.get_chemical_symbols()) <= allowed
    assert molecules[1].info["charge"] == -1


def test_session_text_names_the_notebook_and_the_check(capsys):
    agent05.show_session()
    text = capsys.readouterr().out
    assert "alchemi-agent-05-guest-model.ipynb" in text
    assert "from helpers import agent05 as helpers" in text
    assert "agent05_work" in text
    assert "python -m helpers.agent05 check <step>" in text
    lines = text.splitlines()
    check = lines.index("  check with  python -m helpers.agent05 check <step>")
    assert lines[check + 1] == "  each file   runs in a notebook cell (%load), after the earlier files"
    assert lines[check + 2] == "  always      run the check from notebooks/ after you write a file, and fix"
    assert lines[check + 3] == "              what it reports; <step> is the file name without .py"
    # every step name is its file name without .py
    assert all(spec["file"] == step + ".py" for step, spec in agent05._STEPS.items())


def test_pair_scans_match_stored_energies(pairs):
    batch, labels, energy = pairs
    assert batch.num_graphs == 90 and len(labels) == 30
    assert tuple(energy.shape) == (90, 1)
    assert all(label["idx_a"] == label["idx_ab"] + 1 for label in labels)


def test_pairs_check_passes_flags_and_marks_unexpected(pairs):
    _, labels, composed = pairs
    ns = {"scan_labels": labels, "composed_energy": composed}
    # a copy of the composed energies is on the expected relation: no problem, no note
    assert agent05._problems("aceff_pairs", {**ns, "aceff_energy": composed.clone()}) == ([], [])
    # missing name, wrong shape, not finite: one plain line each
    assert "aceff_energy is missing" in agent05._problems("aceff_pairs", ns)[0][0]
    wrong_shape = agent05._problems("aceff_pairs", {**ns, "aceff_energy": composed.flatten()})[0]
    assert "shape (90, 1)" in wrong_shape[0]
    nan = composed.clone()
    nan[3] = float("nan")
    assert "finite" in agent05._problems("aceff_pairs", {**ns, "aceff_energy": nan})[0][0]
    # energies in the wrong unit are "unexpected" and still pass
    problems, notes = agent05._problems("aceff_pairs", {**ns, "aceff_energy": composed * 23.06})
    assert problems == [] and notes and notes[0].startswith("unexpected:")


def test_relax_check_passes_and_flags():
    batch = _molecule_batch()
    ns = {"relaxed_batch": batch, "force_threshold": 0.05, "relaxed_max_force": torch.full((5,), 0.04)}
    assert agent05._problems("aceff_relax", ns) == ([], [])
    # "below" is strictly below, with no allowance: just under passes, equal and just over fail
    ns["relaxed_max_force"] = torch.full((5,), 0.04997, dtype=torch.float64)
    assert agent05._problems("aceff_relax", ns) == ([], [])
    for value in (0.05, 0.05004, 0.0502):
        ns["relaxed_max_force"] = torch.full((5,), value, dtype=torch.float64)
        problems = agent05._problems("aceff_relax", ns)[0]
        assert len(problems) == 1 and "not below it" in problems[0]
    # a float32 value equal to the threshold as typed sits a little above it, and fails
    ns["relaxed_max_force"] = torch.full((5,), 0.05, dtype=torch.float32)
    assert len(agent05._problems("aceff_relax", ns)[0]) == 1
    # one molecule above the threshold is named, with its value
    ns["relaxed_max_force"] = torch.tensor([0.04, 0.04, 0.2, 0.04, 0.04])
    problem = agent05._problems("aceff_relax", ns)[0][0]
    assert "C9H8O4 (0.2000)" in problem and "not below it" in problem
    # one number instead of one value per molecule
    ns["relaxed_max_force"] = torch.tensor(0.04)
    assert "one value for each molecule" in agent05._problems("aceff_relax", ns)[0][0]
    # a smaller set than the default is the learner's choice: no problem, no note
    short = _molecule_batch(3)
    ns = {"relaxed_batch": short, "force_threshold": 0.05, "relaxed_max_force": torch.full((3,), 0.04)}
    assert agent05._problems("aceff_relax", ns) == ([], [])
    # molecules the file built that did not reach relaxed_batch are "unexpected" and still pass
    for start in (agent05.load_molecules(), _molecule_batch()):
        problems, notes = agent05._problems("aceff_relax", {**ns, "molecules": start})
        assert problems == [] and len(notes) == 1
        assert notes[0].startswith("unexpected:") and "molecules holds 5" in notes[0] and "so 2 did not reach" in notes[0]
    # only names the step's own file created count when the check records them
    assert agent05._problems("aceff_relax", {**ns, "earlier": _molecule_batch(), "_step_names": {"relaxed_batch"}})[1] == []
    # missing names
    assert len(agent05._problems("aceff_relax", {})[0]) == 3


def test_relax_check_notes_a_molecule_above_the_training_size():
    from ase import Atoms

    big = Atoms("C19H30O2", positions=[[1.5 * i, 0.0, 0.0] for i in range(51)])
    big.info["charge"] = 0
    batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in [big, agent05.load_molecules()[0]]])
    ns = {"relaxed_batch": batch, "force_threshold": 0.05, "relaxed_max_force": torch.full((2,), 0.04)}
    problems, notes = agent05._problems("aceff_relax", ns)
    assert problems == [] and len(notes) == 1
    assert notes[0].startswith("note: C19H30O2 (51 atoms) is larger than the 30 atoms")
    assert notes[0].endswith("(the step still passes)")


def test_force_values_print_on_their_side_of_the_threshold():
    # four decimals when they already read on the right side; more when four would show the threshold
    assert agent05._force_text(0.0487, 0.05) == "0.0487"
    assert agent05._force_text(0.04997, 0.05) == "0.04997"
    assert agent05._force_text(0.05004, 0.05) == "0.05004"
    assert agent05._force_text(0.049999, 0.05) == "0.049999"
    assert agent05._force_text(0.05, 0.05) == "0.0500"
    assert agent05._below(0.04997, 0.05) and not agent05._below(0.05, 0.05) and not agent05._below(0.05004, 0.05)


def test_forces_view_says_yes_only_strictly_below(monkeypatch):
    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda obj: shown.append(obj))
    values = torch.tensor([0.0487, 0.04997, 0.05, 0.05004, 0.0325], dtype=torch.float64)
    agent05.show_relaxed_forces(_molecule_batch(), values, 0.05)
    table = shown[0].data
    assert table.iloc[:, 2].tolist() == ["0.0487", "0.04997", "0.0500", "0.05004", "0.0325"]
    assert table.iloc[:, 4].tolist() == ["yes", "yes", "no", "no", "yes"]


def test_a_file_that_uses_file_gets_the_notebook_cell_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent05, "WORK_DIR", tmp_path)
    (tmp_path / "aceff_wrapper.py").write_text("from pathlib import Path\nhere = Path(__file__).parent\n")
    assert agent05._check("aceff_wrapper") == 1
    out = capsys.readouterr().out
    assert (
        "problem: aceff_wrapper: aceff_wrapper.py uses __file__, which does not exist in a notebook cell; "
        "remove that line"
    ) in out
    assert "define it" not in out and "Traceback" not in out


def test_done_when_word_is_shared_by_the_view_and_the_check():
    assert agent05._THRESHOLD_WORD == "below"
    assert agent05._WORDS["forces_heads"][-1] == f"{agent05._THRESHOLD_WORD} the threshold"
    assert f"should be {agent05._THRESHOLD_WORD} force_threshold" in agent05._WORDS["above_threshold"]


def test_parked_loaders_and_views_still_work(monkeypatch):
    # the timestep and proton-hop pieces left this notebook; their helpers are kept unchanged
    aspirin = agent05.load_aspirin()
    assert len(aspirin) == 21 and aspirin.info["charge"] == 0
    malonaldehyde = agent05.load_malonaldehyde()
    hydrogen, oxygen = malonaldehyde.info["hop_pair"]
    assert malonaldehyde.get_chemical_symbols()[hydrogen] == "H"
    assert malonaldehyde.get_chemical_symbols()[oxygen] == "O"
    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda obj: shown.append(obj))
    steps = torch.arange(0, 101, 20)
    energy = torch.zeros(len(steps), 3)
    energy[:, 2] = torch.tensor([0.0, 1.0, float("nan"), float("nan"), float("nan"), float("nan")])
    agent05.plot_timestep_scan(torch.tensor([0.5, 1.0, 4.0]), steps, energy)
    agent05.show_proton_hop(None, torch.full((10, 4), 1.2), torch.linspace(0.9, 2.2, 30), torch.zeros(30))
    assert len(shown) == 2


def test_parked_checks_report_missing_and_wrong_names():
    good = {"replica_dt": torch.tensor([0.5, 1.0]), "saved_step": torch.arange(5), "total_energy": torch.zeros(5, 2)}
    assert agent05._parked_problems("timestep_scan", dict(good)) == []
    missing = {k: v for k, v in good.items() if k != "total_energy"}
    assert any("total_energy" in line for line in agent05._parked_problems("timestep_scan", missing))
    assert any("total_energy" in line for line in agent05._parked_problems("timestep_scan", {**good, "total_energy": torch.zeros(2, 5)}))
    assert agent05._parked_problems("aceff_check", {"aceff_energy": torch.zeros(90, 1), "aceff_max_force": 0.03}) == []
    assert any("aceff_energy" in line for line in agent05._parked_problems("aceff_check", {"aceff_energy": torch.zeros(90), "aceff_max_force": 0.03}))
    assert list(agent05._PARKED_STEPS) == ["aceff_check", "timestep_scan", "proton_hop"]


def test_views_run(pairs, monkeypatch):
    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda obj: shown.append(obj))
    _, labels, composed = pairs
    agent05.plot_interaction_curves(labels, composed=composed, wrapped=composed + 0.01)
    agent05.show_relaxed_forces(_molecule_batch(), torch.full((5,), 0.04), 0.05)
    assert len(shown) == 2
    with pytest.raises(ValueError):
        agent05.show_relaxed_forces(_molecule_batch(), torch.tensor(0.04), 0.05)


def test_step_names_and_usage():
    assert list(agent05._STEPS) == ["aceff_wrapper", "aceff_pairs", "aceff_relax"]
    assert agent05._check("not_a_step") == 2


def test_configure_tutorial_silences_the_torchmdnet_checkpoint_notice_in_kernel_order():
    """Setup cell order: configure_tutorial, then torchmdnet loads AceFF; its "Old-format checkpoint" notice must not print.

    A fresh interpreter, because Toolkit's models.base puts a "once" rule for every UserWarning in
    front of earlier filters when it is first imported.
    """

    import subprocess

    pytest.importorskip("torchmdnet.models.model")
    code = (
        "import warnings\n"
        "from helpers import agent05 as helpers\n"
        "helpers.configure_tutorial()\n"
        "import torchmdnet.models.model\n"
        "import nvalchemi.models.base\n"
        "warnings.warn_explicit(\"Old-format checkpoint detected ('check_errors' found in hyper_parameters). x\",\n"
        "                       UserWarning, 'model.py', 312, module='torchmdnet.models.model')\n"
        "warnings.warn_explicit('another notice', UserWarning, 'model.py', 1, module='torchmdnet.models.model')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=_NOTEBOOK_DIR, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "Old-format checkpoint" not in result.stderr
    assert "another notice" in result.stderr  # the filter is narrow: other notices still print


def test_configure_tutorial_drops_the_hub_token_notice_and_keeps_other_hub_lines():
    """The Hub sends "You are sending unauthenticated requests to the HF Hub" with the first download;
    huggingface_hub logs it on the logger huggingface_hub.utils._http. Only that line is dropped."""

    import logging

    logger = logging.getLogger("huggingface_hub.utils._http")
    before = list(logger.filters)  # an earlier test may have run configure_tutorial already
    logger.filters[:] = [rule for rule in before if not isinstance(rule, agent05._HubTokenNotice)]
    start = len(logger.filters)
    seen = []

    class Collect(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = Collect()
    logger.addHandler(handler)
    try:
        agent05.configure_tutorial()
        agent05.configure_tutorial()  # a second call adds no second filter
        assert len(logger.filters) == start + 1
        logger.warning(
            "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN "
            "to enable higher rate limits and faster downloads."
        )
        logger.warning("Warning: this repository moved")
        logger.error("You are sending unauthenticated requests to the HF Hub and the download failed")
    finally:
        logger.removeHandler(handler)
        logger.filters[:] = before
    assert seen == [
        "Warning: this repository moved",
        "You are sending unauthenticated requests to the HF Hub and the download failed",
    ]
