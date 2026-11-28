# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the saved default structures of notebook 02 from their SMILES strings.

    python build_defaults.py            # rewrites the four .xyz files next to this script

Needs RDKit and ASE. The files are generated, so the script reproduces them exactly.
"""

from pathlib import Path

from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import AllChem

SEED = 7  # ETKDG embedding seed, the one of helpers.agent02.load_molecule
DEFAULTS = {
    "phenacetin": "CCOc1ccc(NC(C)=O)cc1",  # helpers.load_default_molecule()
    "ethanol": "CCO",  # helpers.load_default_set(), in this order
    "pyridine": "c1ccncc1",
    "urea": "NC(N)=O",
}


def build(smiles: str) -> Atoms:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions())


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for name, smiles in DEFAULTS.items():
        comment = f"{name} {smiles}; 3D from SMILES with RDKit ETKDG seed {SEED}, MMFF94 optimised; generated for the ALCHEMI tutorials, no third-party data"
        write(here / f"{name}.xyz", build(smiles), comment=comment)
        print("wrote", f"{name}.xyz")
