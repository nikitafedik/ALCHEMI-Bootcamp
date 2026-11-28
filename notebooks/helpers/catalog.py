# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked structures and deterministic SMILES preparation for the Core course."""

# The Core 00 picker set is chosen by ``CORE00_SYSTEM_SET`` below (PROVISIONAL).

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc111, graphene, molecule
from ase.io import read

from .data_records import sha256_file

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
_MOF74_MANIFEST = _DATA_ROOT / "mof74" / "manifest.json"
_SMILES_SEED = 0xC0FFEE
_SMILES_MAX_ITERATIONS = 200
_MACE_ALIAS = "medium-mpa-0"
_MACE_SHA256 = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"

# The one switch for the Core 00 picker: "materials" is twelve periodic systems for MACE-MPA-0; "mixed" is the
# earlier set with nine organic molecules, kept so the choice can be reversed.
CORE00_SYSTEM_SET = "materials"
_ASE_LICENSE = "Built with ASE (LGPL-2.1-or-later); no third-party data file"


def structure_from_smiles(
    smiles: str,
    *,
    seed: int = _SMILES_SEED,
    max_iterations: int = _SMILES_MAX_ITERATIONS,
) -> Atoms:
    """Build a deterministic, explicit-hydrogen, neutral 3D ASE structure."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES preparation requires a non-empty string")
    if not 0 <= seed <= 0x7FFFFFFF:
        raise ValueError("SMILES embedding seed must fit a signed 32-bit integer")
    if max_iterations < 1:
        raise ValueError("SMILES optimization requires at least one iteration")

    molecule_2d = Chem.MolFromSmiles(smiles)
    if molecule_2d is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    formal_charge = int(Chem.GetFormalCharge(molecule_2d))
    if formal_charge != 0:
        raise ValueError(
            f"Core SMILES preparation expects a neutral structure; found charge "
            f"{formal_charge:+d} for {smiles!r}"
        )

    molecule_3d = Chem.AddHs(molecule_2d)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    if AllChem.EmbedMolecule(molecule_3d, parameters) != 0:
        raise ValueError(f"RDKit could not embed a 3D conformer for {smiles!r}")

    if AllChem.MMFFHasAllMoleculeParams(molecule_3d):
        force_field = "MMFF94"
        status = AllChem.MMFFOptimizeMolecule(
            molecule_3d,
            mmffVariant=force_field,
            maxIters=max_iterations,
        )
    elif AllChem.UFFHasAllMoleculeParams(molecule_3d):
        force_field = "UFF"
        status = AllChem.UFFOptimizeMolecule(
            molecule_3d,
            maxIters=max_iterations,
        )
    else:
        raise ValueError(f"RDKit has no MMFF or UFF parameters for {smiles!r}")
    if status < 0:
        raise ValueError(f"RDKit {force_field} setup failed for {smiles!r}")

    conformer = molecule_3d.GetConformer()
    structure = Atoms(
        numbers=[atom.GetAtomicNum() for atom in molecule_3d.GetAtoms()],
        positions=conformer.GetPositions(),
        pbc=False,
    )
    structure.center(vacuum=4.0)
    structure.info.update(
        {
            "source": "RDKit",
            "canonical_smiles": Chem.MolToSmiles(molecule_2d, canonical=True),
            "embedding": "ETKDGv3",
            "embedding_seed": seed,
            "force_field": force_field,
            "force_field_converged": status == 0,
        }
    )
    return structure


def mace_checkpoint_path(alias: str = _MACE_ALIAS) -> Path:
    """Resolve the workshop MACE-MPA-0 checkpoint and verify its exact bytes."""

    if alias != _MACE_ALIAS:
        raise ValueError(
            f"The Core runtime pins {_MACE_ALIAS!r}; received checkpoint alias {alias!r}"
        )
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    path = Path(download_mace_mp_checkpoint(alias)).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"MACE checkpoint resolver returned a missing file: {path}")
    actual = sha256_file(path)
    if actual != _MACE_SHA256:
        raise RuntimeError(
            f"MACE checkpoint SHA-256 mismatch: expected {_MACE_SHA256}, found {actual}"
        )
    return path


def load_mof74(
    path: str | Path | None = None,
    *,
    manifest_path: str | Path = _MOF74_MANIFEST,
) -> Atoms:
    """Load the pinned 54-atom Mg-MOF-74 primitive cell."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read the MOF-74 manifest: {manifest_file}"
        ) from error

    local = manifest.get("local_file", {})
    structure_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else manifest_file.parent / str(local.get("path", ""))
    )
    if not structure_path.is_file():
        raise RuntimeError(f"Pinned MOF-74 structure is missing: {structure_path}")
    expected = str(local.get("sha256", ""))
    actual = sha256_file(structure_path)
    if not expected or actual != expected:
        raise RuntimeError(
            f"MOF-74 SHA-256 mismatch: expected {expected or '<missing>'}, found {actual}"
        )

    structure = read(structure_path)
    if not isinstance(structure, Atoms):
        raise TypeError(
            f"MOF-74 source did not contain one ASE Atoms object: {structure_path}"
        )
    expected_atoms = int(manifest["primitive_check"]["primitive_atoms"])
    if len(structure) != expected_atoms or not bool(structure.pbc.all()):
        raise RuntimeError(
            f"MOF-74 structure check failed: expected {expected_atoms} periodic atoms"
        )
    structure.info.update(
        {
            "display_name": manifest["display_name"],
            "source": manifest["source"]["repository"],
            "license": manifest["source"]["license"],
        }
    )
    return structure


def _srtio3() -> Atoms:
    """Build cubic perovskite SrTiO3 (Pm-3m, a = 3.905 A) as a 2x2x2 supercell."""

    from ase.spacegroup import crystal

    structure = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    ).repeat((2, 2, 2))
    structure.pbc = True
    structure.info["source"] = (
        "Built with ASE from space group Pm-3m and a = 3.905 Å "
        "(Schmidbauer, Kwasniewski and Schwarzkopf, 2012)"
    )
    return structure


def _silicon() -> Atoms:
    """Build diamond-cubic silicon (a = 5.431 Å) as a 2x2x2 conventional supercell."""

    structure = bulk("Si", "diamond", a=5.431, cubic=True).repeat((2, 2, 2))
    structure.info["source"] = "Built with ASE bulk('Si', 'diamond', a=5.431, cubic=True)"
    return structure


def _rutile() -> Atoms:
    """Build rutile TiO2 (P4_2/mnm, a = 4.594 Å, c = 2.959 Å, u = 0.305), 2x2x3."""

    from ase.spacegroup import crystal

    structure = crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (0.305, 0.305, 0)],
        spacegroup=136,
        cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
    ).repeat((2, 2, 3))
    structure.pbc = True
    structure.info["source"] = (
        "Built with ASE from space group P4_2/mnm, a = 4.594 Å, c = 2.959 Å, u = 0.305"
    )
    return structure


def _zinc_oxide() -> Atoms:
    """Build wurtzite ZnO (a = 3.250 Å, c = 5.207 Å, u = 0.3817), orthorhombic 2x2x2."""

    structure = bulk(
        "ZnO", "wurtzite", a=3.250, c=5.207, u=0.3817, orthorhombic=True
    ).repeat((2, 2, 2))
    structure.info["source"] = (
        "Built with ASE bulk('ZnO', 'wurtzite', a=3.250, c=5.207, u=0.3817, "
        "orthorhombic=True)"
    )
    return structure


def _iron() -> Atoms:
    """Build bcc iron (ASE reference a = 2.87 Å) as a 3x3x3 cubic supercell."""

    structure = bulk("Fe", "bcc", cubic=True).repeat((3, 3, 3))
    structure.set_initial_magnetic_moments(None)
    structure.info["source"] = "Built with ASE bulk('Fe', 'bcc', cubic=True), a = 2.87 Å"
    return structure


def _magnesium_oxide() -> Atoms:
    """Build rock-salt MgO (a = 4.212 Å) as a 2x2x2 conventional supercell."""

    structure = bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat((2, 2, 2))
    structure.info["source"] = "Built with ASE bulk('MgO', 'rocksalt', a=4.212, cubic=True)"
    return structure


def _graphene_vacancy() -> Atoms:
    """Build a 5x5 graphene sheet (a = 2.46 Å) with one carbon atom removed."""

    structure = graphene(a=2.46, size=(5, 5, 1), vacuum=7.0)
    # Remove the atom nearest the centre of the sheet, so the vacancy is in view.
    center_xy = 0.5 * np.asarray(structure.cell[:2, :2], dtype=float).sum(axis=0)
    del structure[
        int(np.argmin(np.linalg.norm(structure.positions[:, :2] - center_xy, axis=1)))
    ]
    structure.info["source"] = (
        "Built with ASE graphene(a=2.46, size=(5, 5, 1), vacuum=7.0), one atom removed"
    )
    return structure


def _smiles_builder(smiles: str) -> Callable[[], Atoms]:
    return lambda: structure_from_smiles(smiles)


def _c60() -> Atoms:
    structure = molecule("C60")
    structure.center(vacuum=4.0)
    structure.info["source"] = "ASE molecule('C60')"
    return structure


def _benchmark_surface_with_adsorbate(
    metal: str,
    adsorbate: str,
    *,
    anchor: str,
    height: float,
    site: str = "ontop",
    flip: bool = False,
) -> Atoms:
    """Build a 48-atom (111) slab plus one compact adsorbate.

    The two bottom layers are marked in the boolean array ``frozen``; the
    relaxation keeps those atoms at their built positions. ``site`` is
    ``"ontop"`` (above the top-layer atom nearest the centre) or ``"fcc"``
    (the fcc hollow next to it). ``flip`` turns the adsorbate upside down, so
    its anchor atom points at the surface.
    """

    slab = fcc111(metal, size=(4, 4, 3), vacuum=7.0)
    # fcc111 tags the layers 1 (top) to 3 (bottom); the adsorbate gets tag 0.
    frozen_slab = slab.get_tags() >= 2
    adsorbate_atoms = Atoms(adsorbate) if len(adsorbate) == 1 else molecule(adsorbate)
    if flip:
        adsorbate_atoms.rotate(180.0, "x")
    anchor_index = adsorbate_atoms.get_chemical_symbols().index(anchor)

    # Place the adsorbate on the top-layer atom nearest the center of the slab.
    cell_center_xy = 0.5 * np.asarray(slab.cell[:2, :2], dtype=float).sum(axis=0)
    top_layer = np.flatnonzero(
        np.isclose(slab.positions[:, 2], slab.positions[:, 2].max())
    )
    center_site = slab.positions[
        top_layer[
            np.argmin(
                np.linalg.norm(
                    slab.positions[top_layer, :2] - cell_center_xy,
                    axis=1,
                )
            )
        ],
        :2,
    ]
    if site == "fcc":
        # The fcc hollow one third along both in-plane vectors of the 1x1 cell.
        center_site = center_site + np.asarray(slab.cell[:2, :2], dtype=float).sum(axis=0) / 12.0
    elif site != "ontop":
        raise ValueError("site must be 'ontop' or 'fcc'")
    add_adsorbate(
        slab,
        adsorbate_atoms,
        height=height,
        position=tuple(center_site),
        mol_index=anchor_index,
    )
    slab.info.update(
        {
            "source": f"ASE fcc111 {metal} slab with {adsorbate} adsorbate",
            "frozen_layers": 2,
        }
    )
    slab.set_array(
        "frozen",
        np.concatenate([frozen_slab, np.zeros(len(slab) - len(frozen_slab), dtype=bool)]),
    )
    return slab


@dataclass(frozen=True, slots=True)
class MoleculeSpec:
    """Explorer metadata plus a factory that returns a fresh ASE structure."""

    key: str
    display_name: str
    domain: str
    purpose: str
    formula: str
    atom_count: int
    periodic: bool
    _builder: Callable[[], Atoms] = field(repr=False, compare=False)
    how_built: str = ""
    source: str = ""
    license: str = ""

    def build(self) -> Atoms:
        """Build and check a fresh structure for this catalog entry."""

        structure = self._builder()
        formula = structure.get_chemical_formula(mode="hill")
        if (
            formula != self.formula
            or len(structure) != self.atom_count
            or bool(structure.pbc.any()) != self.periodic
        ):
            raise RuntimeError(
                f"Catalog entry {self.key!r} changed: got {formula}, "
                f"{len(structure)} atoms, periodic={bool(structure.pbc.any())}"
            )
        structure.info.setdefault("display_name", self.display_name)
        structure.info.setdefault("catalog_key", self.key)
        for name in ("how_built", "license"):
            if getattr(self, name):
                structure.info.setdefault(name, getattr(self, name))
        return structure


def opening_performance_catalog(
    system_set: str | None = None,
) -> dict[str, MoleculeSpec]:
    """Return the twelve systems shown in the Core picker.

    ``system_set`` is ``"materials"`` or ``"mixed"``; ``None`` uses
    ``CORE00_SYSTEM_SET``.
    """

    chosen = CORE00_SYSTEM_SET if system_set is None else system_set
    if chosen == "materials":
        return _materials_catalog()
    if chosen == "mixed":
        return _mixed_catalog()
    raise ValueError("system_set must be 'materials' or 'mixed'")


def _materials_catalog() -> dict[str, MoleculeSpec]:
    """Twelve periodic 40-72 atom materials, all in the scope of MACE-MPA-0."""

    def slab(metal: str, adsorbate: str, **kwargs: Any) -> Callable[[], Atoms]:
        return lambda: _benchmark_surface_with_adsorbate(metal, adsorbate, **kwargs)

    slab_built = (
        "ase.build.fcc111('{metal}', size=(4, 4, 3), vacuum=7.0) with {what}; "
        "two bottom layers frozen"
    )
    entries = (
        MoleculeSpec(
            "srtio3",
            "Strontium titanate (SrTiO₃)",
            "Solar water splitting",
            "Perovskite photocatalyst",
            "O24Sr8Ti8",
            40,
            True,
            _srtio3,
            how_built="ase.spacegroup.crystal, Pm-3m, a = 3.905 Å, 2x2x2 supercell",
            source=(
                "Lattice constant: Schmidbauer, Kwasniewski and Schwarzkopf, "
                "Acta Cryst. B 68, 8 (2012)"
            ),
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "si_diamond",
            "Silicon (Si)",
            "Electronics",
            "Semiconductor",
            "Si64",
            64,
            True,
            _silicon,
            how_built="ase.build.bulk('Si', 'diamond', a=5.431, cubic=True), 2x2x2 supercell",
            source="Lattice constant: CODATA 2018 recommended value, a = 5.431 Å",
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "tio2_rutile",
            "Rutile (TiO₂)",
            "Photocatalysis",
            "Photocatalyst and pigment",
            "O48Ti24",
            72,
            True,
            _rutile,
            how_built=(
                "ase.spacegroup.crystal, P4_2/mnm, a = 4.594 Å, c = 2.959 Å, "
                "u = 0.305, 2x2x3 supercell"
            ),
            source="Lattice constants: Baur and Khan, Acta Cryst. B 27, 2133 (1971)",
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "zno_wurtzite",
            "Zinc oxide (ZnO)",
            "Optoelectronics",
            "Wide-gap semiconductor",
            "O32Zn32",
            64,
            True,
            _zinc_oxide,
            how_built=(
                "ase.build.bulk('ZnO', 'wurtzite', a=3.250, c=5.207, u=0.3817, "
                "orthorhombic=True), 2x2x2 supercell"
            ),
            source="Lattice constants: Kisi and Elcombe, Acta Cryst. C 45, 1867 (1989)",
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "fe_bcc",
            "Iron (Fe)",
            "Metallurgy",
            "Structural metal",
            "Fe54",
            54,
            True,
            _iron,
            how_built="ase.build.bulk('Fe', 'bcc', cubic=True), 3x3x3 supercell",
            source="Lattice constant: ASE reference data, a = 2.87 Å",
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "mgo_rocksalt",
            "Magnesium oxide (MgO)",
            "Ceramics",
            "Refractory oxide",
            "Mg32O32",
            64,
            True,
            _magnesium_oxide,
            how_built="ase.build.bulk('MgO', 'rocksalt', a=4.212, cubic=True), 2x2x2 supercell",
            source="Lattice constant: Hazen, Am. Mineral. 61, 266 (1976)",
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "mg_mof74",
            "Mg-MOF-74",
            "Carbon capture",
            "Porous framework for CO₂",
            "C24H6Mg6O18",
            54,
            True,
            load_mof74,
            how_built=(
                "Read from notebooks/data/mof74/mg-mof-74-primitive.cif "
                "(primitive cell of the RASPA2 structure)"
            ),
            source="iRASPA/RASPA2, structures/mofs/cif/MgMOF-74.cif",
            license="MIT (RASPA2); see notebooks/data/mof74/README.md",
        ),
        MoleculeSpec(
            "graphene_vacancy",
            "Graphene with a vacancy",
            "2D materials",
            "Point defect in a carbon sheet",
            "C49",
            49,
            True,
            _graphene_vacancy,
            how_built=(
                "ase.build.graphene(a=2.46, size=(5, 5, 1), vacuum=7.0), "
                "one carbon atom removed"
            ),
            source=(
                "Lattice constant: graphite in-plane a, Howe, Rawn, Jones and Ow, "
                "Powder Diffr. 18, 150 (2003)"
            ),
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "co_pt111_4x4",
            "CO on Pt(111)",
            "Catalysis",
            "CO adsorption on platinum",
            "COPt48",
            50,
            True,
            slab("Pt", "CO", anchor="C", height=1.85),
            how_built=slab_built.format(metal="Pt", what="CO on a top site"),
            source="Lattice constant: ASE reference data, a = 3.92 Å",
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "h2o_cu111_4x4",
            "H₂O on Cu(111)",
            "Electrochemistry",
            "Water adsorption on copper",
            "H2Cu48O",
            51,
            True,
            slab("Cu", "H2O", anchor="O", height=2.20),
            how_built=slab_built.format(metal="Cu", what="H2O on a top site"),
            source="Lattice constant: ASE reference data, a = 3.61 Å",
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "o_ag111_4x4",
            "O on Ag(111)",
            "Oxidation catalysis",
            "Oxygen adsorption on silver",
            "Ag48O",
            49,
            True,
            slab("Ag", "O", anchor="O", height=1.20, site="fcc"),
            how_built=slab_built.format(metal="Ag", what="one O atom in an fcc hollow"),
            source="Lattice constant: ASE reference data, a = 4.09 Å",
            license=_ASE_LICENSE,
        ),
        MoleculeSpec(
            "nh3_ni111_4x4",
            "NH₃ on Ni(111)",
            "Ammonia chemistry",
            "Ammonia adsorption on nickel",
            "H3NNi48",
            52,
            True,
            slab("Ni", "NH3", anchor="N", height=2.00, flip=True),
            how_built=slab_built.format(metal="Ni", what="NH3 on a top site, N down"),
            source="Lattice constant: ASE reference data, a = 3.52 Å",
            license=_ASE_LICENSE,
        ),
    )
    return {entry.key: entry for entry in entries}


def _mixed_catalog() -> dict[str, MoleculeSpec]:
    """The earlier set (25 Nov 2026): nine organic molecules, two slabs, SrTiO3.

    Switched off by ``CORE00_SYSTEM_SET``; MACE-MPA-0 is a materials model and
    is out of scope for the nine molecules.
    """

    entries = (
        MoleculeSpec(
            "ibuprofen",
            "Ibuprofen",
            "Medicine",
            "Pain and inflammation",
            "C13H18O2",
            33,
            False,
            _smiles_builder("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
        ),
        MoleculeSpec(
            "azoxystrobin",
            "Azoxystrobin",
            "Agriculture",
            "Fungicide",
            "C22H17N3O5",
            47,
            False,
            _smiles_builder("COC=C(C1=CC=CC=C1OC2=NC=NC(=C2)OC3=CC=CC=C3C#N)C(=O)OC"),
        ),
        MoleculeSpec(
            "permethrin",
            "Permethrin",
            "Pest control",
            "Insecticide",
            "C21H20Cl2O3",
            46,
            False,
            _smiles_builder("CC1(C(C1C(=O)OCC2=CC(=CC=C2)OC3=CC=CC=C3)C=C(Cl)Cl)C"),
        ),
        MoleculeSpec(
            "capsaicin",
            "Capsaicin",
            "Food and sensory science",
            "Pungency and pain signaling",
            "C18H27NO3",
            49,
            False,
            _smiles_builder("CC(C)C=CCCCCC(=O)NCC1=CC(=C(C=C1)O)OC"),
        ),
        MoleculeSpec(
            "c60",
            "Fullerene (C₆₀)",
            "Nanomaterials",
            "Carbon nanomaterial",
            "C60",
            60,
            False,
            _c60,
        ),
        MoleculeSpec(
            "retinol",
            "Retinol",
            "Biology and cosmetics",
            "Vitamin A",
            "C20H30O",
            51,
            False,
            _smiles_builder("CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CCO)C)C"),
        ),
        MoleculeSpec(
            "avobenzone",
            "Avobenzone",
            "Sunscreen",
            "UVA filter",
            "C20H22O3",
            45,
            False,
            _smiles_builder("CC(C)(C)C1=CC=C(C=C1)C(=O)CC(=O)C2=CC=C(C=C2)OC"),
        ),
        MoleculeSpec(
            "nile_red",
            "Nile red",
            "Bioimaging",
            "Fluorescent dye",
            "C20H18N2O2",
            42,
            False,
            _smiles_builder("CCN(CC)C1=CC2=C(C=C1)N=C3C4=CC=CC=C4C(=O)C=C3O2"),
        ),
        MoleculeSpec(
            "dibutyl_phthalate",
            "Dibutyl phthalate",
            "Polymers",
            "Plasticizer",
            "C16H22O4",
            42,
            False,
            _smiles_builder("CCCCOC(=O)C1=CC=CC=C1C(=O)OCCCC"),
        ),
        MoleculeSpec(
            "co_pt111_4x4",
            "CO on Pt(111)",
            "Catalysis",
            "CO adsorption on platinum",
            "COPt48",
            50,
            True,
            lambda: _benchmark_surface_with_adsorbate(
                "Pt", "CO", anchor="C", height=1.85
            ),
        ),
        MoleculeSpec(
            "h2o_cu111_4x4",
            "H₂O on Cu(111)",
            "Electrochemistry",
            "Water adsorption on copper",
            "H2Cu48O",
            51,
            True,
            lambda: _benchmark_surface_with_adsorbate(
                "Cu", "H2O", anchor="O", height=2.20
            ),
        ),
        MoleculeSpec(
            "srtio3",
            "Strontium titanate (SrTiO₃)",
            "Solar water splitting",
            "Perovskite photocatalyst",
            "O24Sr8Ti8",
            40,
            True,
            _srtio3,
        ),
    )
    return {entry.key: entry for entry in entries}
