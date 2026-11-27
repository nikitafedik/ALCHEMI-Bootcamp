"""Build molecules.extxyz: the five default molecules of notebook 05, from SMILES with RDKit.

Run once: python build_molecules.py. Each molecule gets an ETKDG embedding (seed 7) and an MMFF
force-field optimisation; the name and the total charge go in the extxyz header of each frame.
"""
from pathlib import Path

from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import AllChem

SEED = 7
MOLECULES = (
    ("ethanol", "CCO", 0),
    ("acetate", "CC(=O)[O-]", -1),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", 0),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1", 0),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", 0),
)

frames = []
for name, smiles, charge in MOLECULES:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=SEED) != 0:
        raise RuntimeError(f"no 3D structure for {name}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    atoms = Atoms(
        [a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions(), pbc=False
    )
    atoms.info.update(name=name, charge=charge)
    frames.append(atoms)
write(Path(__file__).with_name("molecules.extxyz"), frames)
print("wrote", len(frames), "molecules,", sum(len(f) for f in frames), "atoms")
