# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html as html_lib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import module00, viewers, viewers_three


@dataclass(frozen=True)
class _Spec:
    structure: Atoms
    display_name: str
    domain: str
    purpose: str

    def build(self) -> Atoms:
        return self.structure.copy()


def _catalog(count: int = 3) -> dict[str, _Spec]:
    slab = Atoms(
        "PtCO",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 1.0, 3.1]],
        cell=[4.0, 4.0, 8.0],
        pbc=True,
    )
    catalog = {
        "water": _Spec(molecule("H2O"), "Water", "Chemistry", "Small molecule"),
        "slab": _Spec(slab, "CO on Pt", "Catalysis", "Surface"),
    }
    for index in range(count - 2):
        catalog[f"system-{index}"] = _Spec(
            molecule("NH3"), f"System {index}", "Chemistry", "Small molecule"
        )
    return catalog


def _srcdoc(widget_html: str) -> str:
    match = re.search(r'srcdoc="([^"]*)"', widget_html)
    assert match is not None
    return html_lib.unescape(match.group(1))


def _controls(selection) -> dict:
    return {
        key: card.children[2]
        for key, card in zip(
            selection._catalog, selection.widget.children[2].children, strict=True
        )
    }


def test_explorer_builds_cards_with_three_viewer_and_no_network() -> None:
    catalog = _catalog()
    selection = viewers_three.show_structure_explorer_three(catalog, selected="slab")
    grid = selection.widget.children[2]
    first_card = grid.children[0]
    document = _srcdoc(first_card.children[0].value)
    slab_document = _srcdoc(grid.children[1].children[0].value)

    assert isinstance(selection, viewers.StructureSelection)
    assert len(grid.children) == 3
    assert "AlchemiViewer.mountStructure" in document
    assert "window.AlchemiViewer" in document
    assert "THREE" in document
    assert "<script src" not in document
    assert "<link" not in document
    assert not re.search(r"""(src|href)=["']?https?://""", document)
    assert "$3Dmol" not in document
    assert "Water" in document
    payload = json.loads(
        re.search(r"mountStructure\(document.getElementById\('root'\),(.*),\{height", slab_document).group(1)
    )
    assert payload["periodic"] is True
    assert len(payload["cell_edges"]) == 12
    assert payload["cell"][0][0] == 4.0
    assert payload["symbols"] == ["Pt", "C", "O"]
    assert selection.selected_keys == ("slab",)
    assert selection.selected_specs() == [catalog["slab"]]
    assert grid.layout.grid_template_columns == "repeat(4, minmax(0, 1fr))"
    assert "Choose systems for the batch" in selection.widget.children[1].children[0].value
    assert "data-runtime-note" in selection._repr_mimebundle_()["text/html"]


def test_explorer_uses_viewer_classes_and_wording() -> None:
    selection = viewers_three.show_structure_explorer_three(_catalog())
    controls = _controls(selection)
    status = selection._status
    cards = selection.widget.children[2].children

    assert {"av-picker", "av-light"} <= set(selection.widget._dom_classes)
    assert ".av-chip" in selection.widget.children[0].value
    assert "data-jp-theme-light" not in selection.widget.children[0].value  # light only
    assert "av-btn" in controls["water"]._dom_classes
    assert "av-card" in cards[0]._dom_classes
    assert "Choose one or more systems" in status.value
    assert "is-empty" in status._dom_classes
    controls["water"].value = True
    assert status.value == "1 system in batch"
    assert "is-empty" not in status._dom_classes
    assert "is-on" in controls["water"]._dom_classes
    assert "is-selected" in cards[0]._dom_classes
    assert controls["water"].description == "✓ In batch"
    controls["slab"].value = True
    assert status.value == "2 systems in batch"
    controls["water"].value = False
    assert controls["water"].description == "+ Add to batch"
    assert "is-on" not in controls["water"]._dom_classes
    assert "is-selected" not in cards[0]._dom_classes


def test_explorer_disables_other_buttons_at_the_limit() -> None:
    catalog = _catalog(6)
    selection = viewers_three.show_structure_explorer_three(
        catalog, selected=tuple(catalog)[:5]
    )
    controls = _controls(selection)
    keys = tuple(catalog)
    assert controls[keys[5]].disabled is True
    assert controls[keys[0]].disabled is False
    controls[keys[0]].value = False
    assert controls[keys[5]].disabled is False


def test_explorer_limits_match_the_original() -> None:
    catalog = _catalog(6)
    selection = viewers_three.show_structure_explorer_three(catalog)
    controls = _controls(selection)
    keys = tuple(catalog)
    for key in keys[:5]:
        controls[key].value = True
    controls[keys[5]].value = True

    assert selection.selected_keys == keys[:5]
    assert controls[keys[5]].value is False
    with pytest.raises(ValueError, match="select at most 5"):
        viewers_three.show_structure_explorer_three(catalog, selected=keys)
    with pytest.raises(KeyError):
        viewers_three.show_structure_explorer_three(catalog, selected="missing")
    with pytest.raises(ValueError):
        viewers_three.show_structure_explorer_three({})
    with pytest.raises(TypeError):
        viewers_three.show_structure_explorer_three({"water": molecule("H2O")})
    empty = viewers_three.show_structure_explorer_three(catalog)
    with pytest.raises(ValueError, match="Select at least one"):
        empty.selected_specs()


class _Frame:
    """A trajectory frame with positions and per-atom forces."""

    def __init__(self, atoms: Atoms, scale: float) -> None:
        self.atoms = atoms
        self.scale = scale


def _atoms_with_forces(base: Atoms, scale: float) -> Atoms:
    atoms = base.copy()
    atoms.positions += 0.01 * scale
    atoms.arrays["forces"] = np.full((len(atoms), 3), scale)
    return atoms


def test_player_builds_from_a_small_trajectory() -> None:
    water, ammonia = molecule("H2O"), molecule("NH3")
    trajectories = {
        "Water": [_atoms_with_forces(water, s) for s in (2.0, 1.0, 0.5)],
        "Ammonia": [_atoms_with_forces(ammonia, s) for s in (2.0, 1.0, 0.5)],
    }
    output = viewers_three.show_relaxation_player_three(
        trajectories,
        force_history={"Water": [3.0, 1.5, 0.8], "Ammonia": [3.0, 1.5, 0.8]},
        batch_multiplier=16,
        frame_interval_ms=150,
    ).data
    document = _srcdoc(output)

    assert "AlchemiViewer.mountPlayer" in document
    assert "frame_interval_ms:150" in document
    assert not re.search(r"""(src|href)=["']?https?://""", document)
    assert "<script src" not in document
    assert "Batch: 2 systems × 16 copies = 32 structures, 112 atoms" in output
    assert "Static initial/final preview" in output
    assert "data-runtime-note" in output
    assert "fmax 3.000 → 0.800 eV/Å" in output
    payload = json.loads(
        re.search(r"mountPlayer\(document.getElementById\('root'\),(.*),\{height", document).group(1)
    )
    water_item = payload[0]
    assert water_item["name"] == "Water"
    assert water_item["symbols"] == ["O", "H", "H"]
    assert len(water_item["frames"]) == 3
    assert len(water_item["frames"][0]) == 3
    assert water_item["cell"] is None and water_item["cell_edges"] == []
    assert water_item["fmax"] == [round(float(np.sqrt(3) * s), 4) for s in (2.0, 1.0, 0.5)]
    assert len(water_item["atom_forces"][0]) == 3


def test_player_keeps_periodic_cell_and_limits() -> None:
    slab = _catalog()["slab"].build()
    output = viewers_three.show_relaxation_player_three(
        {"CO on Pt": [_atoms_with_forces(slab, 1.0)] * 2},
        reference_structures={"CO on Pt": slab},
    ).data
    payload = json.loads(
        re.search(
            r"mountPlayer\(document.getElementById\('root'\),(.*),\{height",
            _srcdoc(output),
        ).group(1)
    )
    assert len(payload[0]["cell_edges"]) == 12
    assert payload[0]["cell"][2][2] == 8.0

    many = {f"s{i}": [molecule("H2O")] * 2 for i in range(6)}
    with pytest.raises(ValueError, match="at most 5"):
        viewers_three.show_relaxation_player_three(many)
    with pytest.raises(ValueError, match="batch_multiplier"):
        viewers_three.show_relaxation_player_three(
            {"w": [_atoms_with_forces(molecule("H2O"), 1.0)] * 2}, batch_multiplier=0
        )
    with pytest.raises(ValueError, match="frame_interval_ms"):
        viewers_three.show_relaxation_player_three(
            {"w": [_atoms_with_forces(molecule("H2O"), 1.0)] * 2}, frame_interval_ms=5
        )


def test_player_without_forces_falls_back_to_the_3dmol_player() -> None:
    output = viewers_three.show_relaxation_player_three(
        {"Water": [molecule("H2O"), molecule("H2O")]}
    ).data
    assert "Synchronized py3Dmol relaxation player" in output


def test_module00_points_to_the_three_versions() -> None:
    assert module00.show_structure_explorer is viewers_three.show_structure_explorer_three
    assert "show_structure_explorer" in module00.__all__


def test_relax_selected_structures_accepts_player_factory(monkeypatch) -> None:
    pytest.importorskip("nvalchemi")
    from helpers import relaxation

    class _Model:
        def eval(self):
            return self

        def set_config(self, *_):
            pass

    class _Wrapper:
        @staticmethod
        def from_checkpoint(*_, **__):
            return _Model()

    import nvalchemi.models as models

    monkeypatch.setattr(models, "MACEWrapper", _Wrapper)
    monkeypatch.setattr(relaxation, "mace_checkpoint_path", lambda *_: "x")
    monkeypatch.setattr(
        relaxation,
        "run_relaxation_preview",
        lambda *a, **k: {"labels": ["Water"], "elapsed_s": 1.0},
    )
    seen = {}

    def factory(**preview):
        seen.update(preview)
        return "player"

    class _Sel:
        def selected_specs(self):
            return [_catalog()["water"]]

    result = relaxation.relax_selected_structures(
        _Sel(), device="cpu", steps=1, batch_multiplier=1, player_factory=factory
    )
    assert result["player"] == "player"
    assert "model" not in result
    assert "reference_structures" in seen
