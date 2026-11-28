# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CORE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_DIR))

from helpers.catalog import (
    mace_checkpoint_path,
    opening_performance_catalog,
    structure_from_smiles,
)


def test_catalog_returns_fresh_structures() -> None:
    silicon = opening_performance_catalog()["si_diamond"]
    first = silicon.build()
    second = silicon.build()
    first.positions[0, 0] += 1.0

    assert not np.array_equal(first.positions, second.positions)


@pytest.mark.parametrize(
    ("key", "anchor"),
    [("co_pt111_4x4", "C"), ("h2o_cu111_4x4", "O")],
)
def test_opening_surface_adsorbates_are_centered(key: str, anchor: str) -> None:
    structure = opening_performance_catalog()[key].build()
    anchor_index = structure.get_chemical_symbols().index(anchor)

    np.testing.assert_allclose(
        structure.get_scaled_positions(wrap=True)[anchor_index, :2],
        [0.5, 0.5],
        atol=1e-12,
    )


def test_core00_picker_holds_twelve_periodic_materials() -> None:
    catalog = opening_performance_catalog()

    assert list(catalog) == [
        "srtio3",
        "si_diamond",
        "tio2_rutile",
        "zno_wurtzite",
        "fe_bcc",
        "mgo_rocksalt",
        "mg_mof74",
        "graphene_vacancy",
        "co_pt111_4x4",
        "h2o_cu111_4x4",
        "o_ag111_4x4",
        "nh3_ni111_4x4",
    ]
    for key, spec in catalog.items():
        structure = spec.build()
        assert spec.periodic and bool(structure.pbc.any()), key
        assert structure.cell.volume > 0, key
        assert 40 <= len(structure) <= 72, key
        assert spec.how_built and spec.source and spec.license, key
        assert structure.info["source"], key
        assert structure.info["license"] and structure.info["how_built"], key
        assert "canonical_smiles" not in structure.info, key


def test_core00_slabs_mark_the_two_bottom_layers_frozen() -> None:
    catalog = opening_performance_catalog()
    slabs = [key for key in catalog if "111" in key]

    assert len(slabs) == 4
    for key in slabs:
        structure = catalog[key].build()
        frozen = structure.arrays["frozen"]
        heights = structure.positions[:, 2]
        assert int(frozen.sum()) == 32, key
        assert structure.info["frozen_layers"] == 2
        assert heights[frozen].max() < heights[~frozen].min(), key
        assert structure.copy().arrays["frozen"].sum() == 32
    assert all(
        "frozen" not in catalog[key].build().arrays for key in catalog if key not in slabs
    )


def test_oxygen_sits_in_a_hollow_and_ammonia_points_nitrogen_down() -> None:
    catalog = opening_performance_catalog()
    silver = catalog["o_ag111_4x4"].build()
    top = silver.positions[silver.get_tags() == 1]
    nearest = np.sort(np.linalg.norm(top[:, :2] - silver.positions[-1, :2], axis=1))
    np.testing.assert_allclose(nearest[:3], nearest[0], atol=1e-9)

    nickel = catalog["nh3_ni111_4x4"].build()
    symbols = np.array(nickel.get_chemical_symbols())
    assert nickel.positions[symbols == "N", 2].max() < nickel.positions[symbols == "H", 2].min()


def test_graphene_vacancy_is_near_the_centre_of_the_sheet() -> None:
    from ase.build import graphene

    sheet = catalog_sheet = graphene(a=2.46, size=(5, 5, 1), vacuum=7.0)
    vacancy = opening_performance_catalog()["graphene_vacancy"].build()
    assert len(vacancy) == len(catalog_sheet) - 1 == 49
    kept = {tuple(np.round(position, 6)) for position in vacancy.positions}
    missing = [p for p in sheet.positions if tuple(np.round(p, 6)) not in kept]
    assert len(missing) == 1
    center = 0.5 * (sheet.cell[0, :2] + sheet.cell[1, :2])
    assert np.linalg.norm(missing[0][:2] - center) < 1.5


def test_earlier_mixed_set_stays_available_behind_the_switch() -> None:
    mixed = opening_performance_catalog("mixed")

    assert len(mixed) == 12 and list(mixed)[0] == "ibuprofen" and list(mixed)[-1] == "srtio3"
    assert sum(not spec.periodic for spec in mixed.values()) == 9
    assert mixed["ibuprofen"].build().get_chemical_formula(mode="hill") == "C13H18O2"
    with pytest.raises(ValueError, match="materials"):
        opening_performance_catalog("other")


def test_smiles_preparation_is_deterministic() -> None:
    first = structure_from_smiles("CCO")
    second = structure_from_smiles("CCO")

    assert first.get_chemical_formula(mode="hill") == "C2H6O"
    np.testing.assert_array_equal(first.numbers, second.numbers)
    np.testing.assert_allclose(first.positions, second.positions, rtol=0.0, atol=0.0)
    assert first.info["embedding"] == "ETKDGv3"
    assert first.info["embedding_seed"] == 0xC0FFEE
    assert first.info["force_field"] in {"MMFF94", "UFF"}


@pytest.mark.parametrize("smiles", ["", "this is not SMILES", "[NH4+]"])
def test_smiles_preparation_errors_are_clear(smiles: str) -> None:
    with pytest.raises(ValueError, match="SMILES|neutral"):
        structure_from_smiles(smiles)


def test_mace_checkpoint_helper_rejects_unpinned_alias() -> None:
    with pytest.raises(ValueError, match="pins 'medium-mpa-0'"):
        mace_checkpoint_path("medium-0b2")
