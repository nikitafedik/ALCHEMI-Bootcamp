# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 04, Made to order.

Loaders for the molecular-pair scans, the 205 NCIA complexes and the alanine
dipeptide grid; the electrostatics term the learner's agent composes with
AIMNet2; and the notebook's session and step-check entry points. Plotters and
viewers live in ``agent04_plots``, the step contracts in ``agent04_check``.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from nvalchemi.models.base import BaseModelMixin, NeighborConfig, NeighborListFormat

from .agent04_plots import (
    plot_interaction_curves,
    plot_ncia_accuracy,
    show_conformers,
    show_single_molecule_check,
)
from .common import configure_tutorial as _configure_tutorial
from .model_support import DirectCoulombAdapter as _CourseDirectCoulombAdapter
from .model_support import model_checkpoint, prepare_model_batch

__all__ = [
    "DirectCoulombAdapter",
    "build_dipeptide_default",
    "build_dipeptide_grid",
    "configure_tutorial",
    "d3_settings",
    "load_dipeptide_grid",
    "load_nci_scans",
    "load_ncia",
    "load_ncia_references",
    "make_work_dir",
    "model_checkpoint",
    "plot_interaction_curves",
    "plot_ncia_accuracy",
    "prepare_model_batch",
    "relax_settings",
    "show_conformers",
    "show_session",
    "show_single_molecule_check",
]

WORK_DIR = Path(__file__).resolve().parents[1] / "agent04_work"
NOTEBOOK = "notebooks/alchemi-agent-04-made-to-order.ipynb"
DEFAULT_DIPEPTIDE = Path(__file__).resolve().parents[1] / "data" / "agent04" / "alanine-dipeptide-grid.extxyz"


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    WORK_DIR.mkdir(exist_ok=True)
    return WORK_DIR


make_work_dir()


def configure_tutorial() -> None:
    """Apply the course plot and display style (light only)."""

    _configure_tutorial()


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(
        "Give these to your agent when you start the session:\n"
        f"  notebook    {NOTEBOOK}\n"
        "  helpers     from helpers import agent04 as helpers\n"
        "  work in     notebooks/agent04_work/\n"
        "  check with  python -m helpers.agent04 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
    )


class DirectCoulombAdapter(_CourseDirectCoulombAdapter):
    """Finite Coulomb term for isolated molecules, fed by an earlier model's partial charges.

    ``cutoff`` is the pair range in Å. The default, 15 Å, covers every pair of
    atoms in the molecular pairs and conformers of this notebook and matches
    the range of D3.

    The term reads the pipeline's matrix neighbor list, the format AIMNet2 and
    D3 use, and turns it into pairs itself. With a COO list, Toolkit 0.3.0
    stops with "property 'edge_ptr' of 'Batch' object has no setter" in a
    pipeline that has no D3 group.
    """

    def __init__(self, cutoff: float = 15.0, **kwargs):
        super().__init__(cutoff=cutoff, **kwargs)
        self.model_config = self.model_config.model_copy(
            update={
                "neighbor_config": NeighborConfig(
                    cutoff=self.cutoff, format=NeighborListFormat.MATRIX, half_list=False
                )
            }
        )

    def adapt_input(self, data, **kwargs):
        """Collect the inputs of the Coulomb term, with pairs built from the neighbor matrix."""

        from nvalchemi.data import Batch

        if not isinstance(data, Batch):
            data = Batch.from_data_list([data], device=data.positions.device)
        inputs = BaseModelMixin.adapt_input(self, data, **kwargs)
        inputs.pop("atomic_numbers", None)
        matrix = inputs.pop("neighbor_matrix")
        count = inputs.pop("num_neighbors")
        slots = torch.arange(matrix.shape[1], device=matrix.device)
        valid = slots[None, :] < count.reshape(-1, 1)
        rows = torch.arange(matrix.shape[0], device=matrix.device)[:, None].expand_as(matrix)
        inputs["neighbor_pairs"] = torch.stack([rows[valid], matrix[valid]], dim=1)
        inputs["batch_idx"] = data.batch_idx
        inputs["num_graphs"] = data.num_graphs
        return inputs


# --------------------------------------------------------------------------
# Molecular pairs: scans and the NCIA complexes
# --------------------------------------------------------------------------


def load_nci_scans() -> tuple[list[Atoms], list[dict]]:
    """Return ``(atoms_list, scan_labels)`` for the three molecular-pair scans.

    ``atoms_list`` holds 90 structures in file order, AB, A, B for each of the
    30 scan points; ``info["charge"]`` holds each fragment's total charge.
    ``scan_labels`` has one dict per scan point, in the same order, with the
    structure indices (``idx_ab``, ``idx_a``, ``idx_b``) and the references in
    kcal/mol (``ref_ccsd_t_cbs``, ``ref_dft_d3``).
    """

    from .nci import load_nci_atlas

    atoms, rows, refs = load_nci_atlas()
    labels = []
    for r in refs.itertuples(index=False):
        sel = rows[(rows["system_id"] == r.system_id) & (rows["scale"] == r.scale)]
        idx = {f: int(sel[sel["fragment"] == f]["graph_index"].iloc[0]) for f in ("AB", "A", "B")}
        labels.append(
            {
                "system_id": str(r.system_id),
                "system_name": str(r.system_name),
                "interaction_class": str(r.interaction_class),
                "scale": float(r.scale),
                "idx_ab": idx["AB"],
                "idx_a": idx["A"],
                "idx_b": idx["B"],
                "ref_ccsd_t_cbs": float(r.ccsd_t_cbs_kcal_mol),
                "ref_dft_d3": float(r.dft_d3_kcal_mol),
            }
        )
    if len(atoms) != 90 or len(labels) != 30:
        raise RuntimeError(f"expected 90 structures and 30 scan points, got {len(atoms)} and {len(labels)}")
    if not all(l["idx_a"] == l["idx_ab"] + 1 and l["idx_b"] == l["idx_ab"] + 2 for l in labels):
        raise RuntimeError("scan structures are not ordered AB, A, B")
    return atoms, labels


_NCIA_CACHE: dict = {}


def _ncia() -> dict:
    if not _NCIA_CACHE:
        from .ncia250 import load_ncia250

        structures, graph_rows, references, _, _ = load_ncia250()
        _NCIA_CACHE.update(structures=structures, rows=graph_rows, refs=references)
    return _NCIA_CACHE


def load_ncia() -> list[Atoms]:
    """Return the 205 supported complexes as 615 structures, AB, A, B per complex."""

    structures = _ncia()["structures"]
    if len(structures) != 615:
        raise RuntimeError(f"expected 615 NCIA structures, got {len(structures)}")
    return structures


def load_ncia_references() -> np.ndarray:
    """Return the CCSD(T)/CBS interaction energies in kcal/mol, one per complex (205)."""

    refs = _ncia()["refs"]
    if len(refs) != 205:
        raise RuntimeError(f"expected 205 NCIA references, got {len(refs)}")
    return refs["ccsd_t_cbs_kcal_mol"].to_numpy(dtype=float)


def d3_settings() -> dict:
    """Return the D3(BJ) settings that belong to the AIMNet2 ``wb97m-d3`` checkpoint.

    Keys: ``a1``, ``a2``, ``s8``, ``cutoff`` (Å) and ``param_file``.
    """

    from .model_support import d3_parameter_file

    return {
        "a1": 0.566,
        "a2": 3.128,
        "s8": 0.3908,
        "cutoff": 15.0,
        "param_file": d3_parameter_file(),
    }


# --------------------------------------------------------------------------
# Conformer relaxation: the force threshold
# --------------------------------------------------------------------------

# Resolved 26 Nov (board question sci-a04-basins, option A; DOUBLE CHECK): the conformer step names a force
# threshold as a setting in its input row. This is the one place that holds it; the step check reads it from
# here (``agent04_check.THRESHOLD_IN_CONTRACT`` switches the check back to a note). Value from a run: at
# 0.05 eV/Å 31 of 47 end points still move by more than 10 degrees when relaxed on to 0.01; from 0.01 to 0.005,
# 9 of 47 (notes/evidence/agent-04, "threshold"). ``ALCHEMI_A04_RELAX_FMAX`` overrides it for short CPU test runs.
RELAX_FMAX = float(os.environ.get("ALCHEMI_A04_RELAX_FMAX", "0.01"))  # eV/Å


def relax_settings() -> dict:
    """Return the settings of the conformer relaxation.

    Key: ``fmax``, the largest force on an atom, in eV/Å, at which a conformer
    counts as relaxed.
    """

    return {"fmax": RELAX_FMAX}


# --------------------------------------------------------------------------
# Alanine dipeptide: backbone angle grid
# --------------------------------------------------------------------------

# SMILES atom order: 0 CH3(Ace), 1 C(Ace), 2 O(Ace), 3 N, 4 CA, 5 CB, 6 C, 7 O, 8 N(NMe), 9 CH3(NMe)
_DIPEPTIDE_SMILES = "CC(=O)N[C@@H](C)C(=O)NC"
PHI_ATOMS = (1, 3, 4, 6)  # C(=O of Ace) - N - CA - C
PSI_ATOMS = (3, 4, 6, 8)  # N - CA - C - N(NMe)
_OMEGA_ATOMS = ((0, 1, 3, 4), (4, 6, 8, 9))
_CLASH_ANGSTROM = 0.7
_DIPEPTIDE_SEED = 0xA1A


def dihedral_deg(positions: np.ndarray, idx) -> float:
    """Return the IUPAC-signed dihedral, in degrees in (-180, 180], of four atom indices."""

    b1, b2, b3 = (positions[idx[1]] - positions[idx[0]], positions[idx[2]] - positions[idx[1]], positions[idx[3]] - positions[idx[2]])
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    return float(np.degrees(np.arctan2(np.linalg.norm(b2) * np.dot(b1, n2), np.dot(n1, n2))))


def _wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _side(mol, start: int, cut: tuple[int, int]) -> list[int]:
    """Atoms reachable from ``start`` without crossing the bond ``cut``."""

    seen, queue = {start}, deque([start])
    while queue:
        i = queue.popleft()
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if {i, j} == set(cut) or j in seen:
                continue
            seen.add(j)
            queue.append(j)
    return sorted(seen)


def _set_dihedral(atoms: Atoms, idx, target: float, moving: list[int]) -> None:
    """Set an IUPAC dihedral with ASE, checking ASE's sign convention."""

    for trial in (target, -target):
        atoms.set_dihedral(*idx, trial, indices=moving)
        if abs(_wrap(dihedral_deg(atoms.positions, idx) - target)) < 1e-3:
            return
    raise RuntimeError("could not set the dihedral with the IUPAC sign convention")


def _min_distance(atoms: Atoms) -> float:
    d = atoms.get_all_distances()
    return float(d[np.triu_indices(len(atoms), k=1)].min())


def _dipeptide_base():
    """Return the RDKit molecule and ASE atoms of the all-trans-amide dipeptide, deterministic."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    for k in range(200):
        mol = Chem.AddHs(Chem.MolFromSmiles(_DIPEPTIDE_SMILES))
        if AllChem.EmbedMolecule(mol, randomSeed=_DIPEPTIDE_SEED + k) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        pos = mol.GetConformer().GetPositions()
        if all(abs(dihedral_deg(pos, o)) > 150 for o in _OMEGA_ATOMS):
            atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=pos, pbc=False)
            return mol, atoms
    raise RuntimeError("no trans-amide embedding of alanine dipeptide")


def build_dipeptide_grid(n: int = 12) -> tuple[list[Atoms], int]:
    """Build alanine dipeptide on an ``n`` by ``n`` grid of backbone angles phi and psi.

    Return the kept starts and the number of grid points. The grid runs from
    -180° in steps of 360/``n``°; ``info["phi"]`` and ``info["psi"]`` hold the
    target angles in degrees and ``info["charge"]`` is 0. Starts with two atoms
    closer than 0.7 Å are dropped.
    """

    mol, base = _dipeptide_base()
    phi_move = _side(mol, PHI_ATOMS[2], (PHI_ATOMS[1], PHI_ATOMS[2]))
    psi_move = _side(mol, PSI_ATOMS[2], (PSI_ATOMS[1], PSI_ATOMS[2]))
    angles = -180.0 + 360.0 / n * np.arange(n)
    frames = []
    for phi in angles:
        for psi in angles:
            atoms = base.copy()
            _set_dihedral(atoms, PHI_ATOMS, float(phi), phi_move)
            _set_dihedral(atoms, PSI_ATOMS, float(psi), psi_move)
            atoms.info.update(phi=float(phi), psi=float(psi), charge=0, grid_points=len(angles) ** 2)
            frames.append(atoms)
    return [f for f in frames if _min_distance(f) >= _CLASH_ANGSTROM], len(frames)


def build_dipeptide_default() -> Path:
    """Write the saved default, the 12 by 12 grid, to ``data/agent04/``; return the file."""

    from ase.io import write

    kept, _ = build_dipeptide_grid(12)
    DEFAULT_DIPEPTIDE.parent.mkdir(parents=True, exist_ok=True)
    write(DEFAULT_DIPEPTIDE, kept, format="extxyz")
    return DEFAULT_DIPEPTIDE


def load_dipeptide_grid(n: int = 12) -> list[Atoms]:
    """Return alanine dipeptide starts on an ``n`` by ``n`` grid of backbone angles phi and psi.

    The default, ``n=12``, is the saved file ``data/agent04/alanine-dipeptide-grid.extxyz``
    (139 starts, 22 atoms each, neutral); other values of ``n`` are built.
    ``info["phi"]`` and ``info["psi"]`` hold the target angles in degrees and
    ``info["charge"]`` is 0. Starts with two atoms closer than 0.7 Å are dropped,
    and the loader prints how many (5 of 144 at ``n=12``).
    """

    if n == 12 and DEFAULT_DIPEPTIDE.exists():
        from ase.io import read

        kept = read(DEFAULT_DIPEPTIDE, index=":")
        total = int(kept[0].info["grid_points"])
    else:
        kept, total = build_dipeptide_grid(n)
    print(f"dipeptide grid: {len(kept)} of {total} starts kept ({total - len(kept)} with clashing atoms dropped)")
    return kept


if __name__ == "__main__":
    from .agent04_check import main

    sys.exit(main(sys.argv[1:]))
