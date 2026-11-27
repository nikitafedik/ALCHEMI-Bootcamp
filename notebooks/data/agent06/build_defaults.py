# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the three default complexes of Agent 06 from the NCI Atlas files of this repository.

Run from ``notebooks/``::

    python data/agent06/build_defaults.py

Complexes 1 and 2 come from ``data/nci_atlas/nci-atlas-curves.csv.gz`` (scale 1.0), complex 3
from ``data/nci_atlas/NCIA250.zip`` through ``helpers.ncia250.load_ncia250()``. Coordinates are copied unchanged. Nothing is downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ase import Atoms
from ase.io import write

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parents[1]
sys.path.insert(0, str(NOTEBOOKS))

# (file, atlas name in the curves file) for complexes 1 and 2
FROM_CURVES = (
    ("complex-1-phenol-n-methylacetamide.xyz", "phenol - N-methylacetamide"),
    ("complex-2-propyne-methyl-azide.xyz", "propyne - methyl azide"),
)
# (file, NCIA250 system, name) for complex 3 and its saved alternative; the archive gives an id,
# the name is read from the bonds. helpers/agent06.py DEFAULT_COMPLEX_FILES names the one in use.
FROM_NCIA250 = (
    ("complex-3-phenol-n-methylpropan-2-imine.xyz", "HB375x10 3.032", "phenol - N-methylpropan-2-imine"),
    (
        "complex-3-alternative-pyrrole-n-methylpropan-2-imine.xyz",
        "HB375x10 4.008",
        "pyrrole - N-methylpropan-2-imine",
    ),
)


# Interaction energy at the saved geometry with the composed model of the notebook (AIMNet2 with
# its Coulomb and D3 terms), in eV: measured on a CPU with nvalchemi-toolkit 0.3.0 by
# notes/evidence/agent-06/l06-defaults--20261126-owner/verify_defaults.py. The views read the
# learner's energy unit from these values.
COMPOSED_MODEL_EV = {
    "complex-1-phenol-n-methylacetamide.xyz": -0.502232,
    "complex-2-propyne-methyl-azide.xyz": -0.132802,
    "complex-3-phenol-n-methylpropan-2-imine.xyz": -0.528950,
    "complex-3-alternative-pyrrole-n-methylpropan-2-imine.xyz": -0.395459,
}


def _save(path: Path, atoms: Atoms, *, name: str, split: int, source: str, reference: float) -> None:
    out = Atoms(atoms.get_chemical_symbols(), atoms.get_positions(), pbc=False)
    out.info.update(
        name=name,
        fragment_a_size=int(split),
        charge=0,
        charge_a=0,
        charge_b=0,
        source=source,
        ccsd_t_cbs_kcal_mol=float(reference),
        composed_model_ev=COMPOSED_MODEL_EV[path.name],
    )
    write(path, out, format="extxyz", columns=["symbols", "positions"])
    print(f"{path.name}: {len(out)} atoms, fragment A = first {split}, reference {reference} kcal/mol")


def main() -> None:
    from helpers import module02
    from helpers.nci import load_nci_atlas

    atoms, rows, references = load_nci_atlas()
    labels, structures, splits = module02.equilibrium_nci_complexes(atoms, rows)
    for file, name in FROM_CURVES:
        k = labels.index(name)
        info = structures[k].info
        if int(info["charge"]) != 0:
            raise ValueError(f"{name} is not neutral")
        ref = references[
            references["system_id"].astype(str).eq(str(info["system_id"])) & references["scale"].eq(1.0)
        ]["ccsd_t_cbs_kcal_mol"]
        _save(
            HERE / file,
            structures[k],
            name=name,
            split=splits[k],
            source=f"NCI Atlas {info['subset']} {info['system_id']}, scale 1.00",
            reference=float(ref.iloc[0]),
        )

    from helpers.ncia250 import load_ncia250

    atoms250, rows250, references250 = load_ncia250()[:3]
    for file, system, name in FROM_NCIA250:
        mine = rows250[rows250["system_name"].eq(system)].set_index("fragment")
        if any(int(mine.loc[part, "charge"]) for part in ("AB", "A", "B")):
            raise ValueError(f"{system} is not a neutral pair of neutral fragments")
        pair = atoms250[int(mine.loc["AB", "graph_index"])]
        split = int(mine.loc["A", "natoms"])
        first = atoms250[int(mine.loc["A", "graph_index"])]
        if pair.get_chemical_symbols()[:split] != first.get_chemical_symbols():
            raise ValueError(f"{system}: fragment A is not the first atoms of the complex")
        reference = references250[references250["system_name"].eq(system)]["ccsd_t_cbs_kcal_mol"]
        _save(
            HERE / file,
            pair,
            name=name,
            split=split,
            source=f"NCI Atlas NCIA250, NCIA_{system.replace(' ', '_')}_100",
            reference=float(reference.iloc[0]),
        )


if __name__ == "__main__":
    main()
