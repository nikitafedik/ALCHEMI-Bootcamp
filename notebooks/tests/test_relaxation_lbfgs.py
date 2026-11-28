# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html as html_lib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import module00, relaxation_lbfgs, viewers_three


def test_thin_frame_indices_keeps_first_sixty_and_the_last_frame() -> None:
    assert relaxation_lbfgs.thin_frame_indices(1) == [0]
    assert relaxation_lbfgs.thin_frame_indices(60) == list(range(60))
    thinned = relaxation_lbfgs.thin_frame_indices(250)
    assert thinned[:60] == list(range(60))
    assert thinned[-1] == 249
    assert thinned == sorted(set(thinned))
    assert len(thinned) <= relaxation_lbfgs.PLAYER_MAX_FRAMES
    assert len(thinned) > 60


def _atoms_with_forces(scale: float) -> Atoms:
    atoms = molecule("H2O")
    atoms.arrays["forces"] = [[scale, 0.0, 0.0]] * 3
    return atoms


def test_player_shows_real_step_numbers_and_checks_length() -> None:
    frames = [_atoms_with_forces(s) for s in (2.0, 1.0, 0.5)]
    output = viewers_three.show_relaxation_player_three(
        {"Water": frames}, step_labels=[0, 40, 201], converged_steps={"Water": 30}
    ).data
    document = html_lib.unescape(re.search(r'srcdoc="([^"]*)"', output).group(1))
    assert "step_labels:[0,40,201]" in document
    assert "dataset.fixed" not in document
    assert '"stop_frame":1' in document
    with pytest.raises(ValueError, match="step_labels"):
        viewers_three.show_relaxation_player_three(
            {"Water": frames}, step_labels=[0, 40]
        )


def test_module00_keeps_the_fixed_step_run_for_steps_calls(monkeypatch) -> None:
    calls = []
    from helpers import relaxation

    monkeypatch.setattr(
        relaxation,
        "relax_selected_structures",
        lambda *a, **k: calls.append(("fire2", k)) or {},
    )
    module00.relax_selected_structures(object(), device="cpu", steps=3, batch_multiplier=1)
    assert calls and calls[0][0] == "fire2"
    assert calls[0][1]["player_factory"] is viewers_three.show_relaxation_player_three


def _lbfgs_available() -> bool:
    try:
        from nvalchemi.dynamics import LBFGS  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class _Spec:
    structure: Atoms
    display_name: str

    def build(self) -> Atoms:
        return self.structure.copy()


@pytest.mark.skipif(not _lbfgs_available(), reason="LBFGS needs Toolkit 0.3.0")
def test_lbfgs_run_returns_the_cell_9_dictionary_on_cpu() -> None:
    from helpers.catalog import mace_checkpoint_path

    try:
        mace_checkpoint_path("medium-mpa-0")
    except Exception:  # checkpoint not available in this environment
        pytest.skip("MACE checkpoint not available")

    class _Selection:
        def selected_specs(self):
            return [_Spec(molecule("H2O"), "Water"), _Spec(molecule("NH3"), "Ammonia")]

    result = module00.relax_selected_structures(
        _Selection(),
        device="cpu",
        fmax=0.05,
        max_steps=4,
        distortion_std_angstrom=0.12,
        distortion_seed=7,
        batch_multiplier=2,
    )
    preview = result["preview"]
    assert set(result) == {
        "batch",
        "batch_multiplier",
        "labels",
        "preview",
        "player",
    }
    assert preview["warmup_s"] > 0
    assert 0 < preview["total_s"] < 600 and preview["model_load_s"] > 0
    assert preview["steps_run"] <= 4
    assert preview["step_labels"][0] == 0
    assert preview["step_labels"][-1] == preview["steps_run"]
    assert all(
        len(frames) == len(preview["step_labels"])
        for frames in preview["trajectories"].values()
    )
    assert set(preview["force_history"]) == {"Water", "Ammonia"}
    assert "AlchemiViewer.mountPlayer" in html_lib.unescape(result["player"].data)


@pytest.mark.skipif(not _lbfgs_available(), reason="LBFGS needs Toolkit 0.3.0")
def test_lbfgs_run_keeps_frozen_atoms_and_distinct_frames_on_cpu() -> None:
    import numpy as np
    from ase.build import add_adsorbate, fcc111
    from helpers.catalog import mace_checkpoint_path

    try:
        mace_checkpoint_path("medium-mpa-0")
    except Exception:  # checkpoint not available in this environment
        pytest.skip("MACE checkpoint not available")

    slab = fcc111("Cu", size=(2, 2, 3), vacuum=7.0)
    frozen = slab.get_tags() >= 2
    add_adsorbate(slab, "O", height=1.2, position="fcc")
    frozen = np.append(frozen, False)
    slab.set_array("frozen", frozen)
    built = slab.copy()

    class _Selection:
        def selected_specs(self):
            return [_Spec(slab, "Slab"), _Spec(molecule("H2O"), "Water")]

    result = module00.relax_selected_structures(
        _Selection(),
        device="cpu",
        fmax=0.05,
        max_steps=3,
        distortion_std_angstrom=0.20,
        distortion_seed=7,
        batch_multiplier=2,
        warmup=False,
    )
    preview = result["preview"]
    frames = preview["trajectories"]["Slab"]
    start = preview["reference_structures"]["Slab"]

    assert preview["frozen_atoms"] == {"Slab": 8, "Water": 0}
    # The shake leaves the frozen atoms where the builder put them.
    np.testing.assert_array_equal(start.positions[frozen], built.positions[frozen])
    assert np.abs(start.positions[~frozen] - built.positions[~frozen]).max() > 0.05
    for frame in frames:
        np.testing.assert_allclose(
            frame.positions[frozen], built.positions[frozen], rtol=0.0, atol=1e-5
        )
        assert np.abs(frame.arrays["forces"][frozen]).max() == 0.0
    # Free atoms move, and each stored frame is its own step.
    assert np.abs(frames[-1].positions[~frozen] - frames[0].positions[~frozen]).max() > 1e-3
    forces = preview["force_history"]["Slab"]
    assert len(set(np.round(forces, 6))) > 1


def test_molecule_first_mixed_batch_keeps_per_system_cell_and_pbc() -> None:
    import torch
    from ase.build import bulk
    from helpers.relaxation_lbfgs import _with_cell_and_pbc
    from nvalchemi.data import AtomicData, Batch

    water = molecule("H2O")
    crystal = bulk("Cu", cubic=True)
    crystal.pbc = True
    data = [
        _with_cell_and_pbc(AtomicData.from_atoms(structure), structure)
        for structure in (water, crystal, water)
    ]
    batch = Batch.from_data_list(data)

    assert batch.cell.shape == (3, 3, 3)
    assert batch.pbc.shape == (3, 3)
    assert batch.pbc.tolist() == [[False] * 3, [True] * 3, [False] * 3]
    extent = float((water.positions.max(0) - water.positions.min(0)).max())
    assert torch.allclose(
        batch.cell[0], (extent + 12.0) * torch.eye(3, dtype=batch.cell.dtype)
    )
    assert torch.allclose(
        batch.cell[1], torch.as_tensor(crystal.cell.array, dtype=batch.cell.dtype)
    )


def test_srtio3_catalog_entry() -> None:
    from helpers.catalog import opening_performance_catalog

    catalog = opening_performance_catalog()
    keys = list(catalog)
    spec = catalog["srtio3"]
    atoms = spec.build()

    assert keys[0] == "srtio3" and len(keys) == 12
    assert spec.display_name == "Strontium titanate (SrTiO₃)"
    assert (spec.purpose, spec.domain) == (
        "Perovskite photocatalyst",
        "Solar water splitting",
    )
    assert len(atoms) == 40 and bool(atoms.pbc.all())
    assert atoms.get_chemical_formula(mode="hill") == "O24Sr8Ti8"
    assert "Pm-3m" in atoms.info["source"] and "3.905" in atoms.info["source"]
