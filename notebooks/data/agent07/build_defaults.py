"""Write study_complexes.csv: the default structures of Agent 07, as the setup cell loads them.

    cd notebooks && python data/agent07/build_defaults.py

Reads only data already in the repository (data/nci_atlas, through helpers.ncia250). No model,
no download.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helpers.ncia250 import load_ncia250

atoms, rows, *_ = load_ncia250()
study = rows[rows["fragment"].eq("AB")].reset_index(drop=True)
table = study[["subset", "system_id", "interaction_class", "charge", "natoms", "source_geometry_file"]].copy()
table.insert(0, "complex", range(len(table)))
table["formula"] = [atoms[int(g)].get_chemical_formula() for g in study["graph_index"]]
out = Path(__file__).with_name("study_complexes.csv")
table.to_csv(out, index=False)
print(f"{out.name}: {len(table)} complexes, {int(table.natoms.sum())} atoms, "
      f"{int(table.natoms.min())} to {int(table.natoms.max())} atoms each, charges {sorted(table.charge.unique())}")
