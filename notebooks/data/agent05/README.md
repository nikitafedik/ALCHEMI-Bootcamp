# Data of agentic notebook 05, "Guest model"

Two defaults for the "Try it" pieces. Both load in one call from `helpers/agent05.py`.

## `molecules.extxyz`: `helpers.load_molecules()`

- What: five small molecules, one extxyz frame each, 81 atoms in total: ethanol (9 atoms),
  acetate (7 atoms, total charge -1), aspirin (21), paracetamol (20), caffeine (24). The frame
  header holds `name` and `charge`.
- Why these: each has at most 30 atoms, the size range of AceFF 2.0's training data, and
  only H, C, N, O (the model card lists H B C N O F Si P S Cl Br I and charges -2 to +2).
  All five ran with AceFF 2.0 in the course runtime.
- Source: none outside the repository. Each structure is built from a SMILES string with
  RDKit (BSD licence): ETKDG embedding with seed 7, then an MMFF force-field optimisation.
  The molecule names are common chemicals; no dataset and no download is involved.
- Licence: the file is generated from public SMILES strings and carries the repository's licence.
- How built: `python build_molecules.py` (needs RDKit and ASE); it rewrites `molecules.extxyz`.
- Geometry: a force-field geometry, not an AceFF minimum; the Try it relaxes it.

## `pair_scans_composed.npz`: `helpers.load_pair_scans()`

- What: the energies, in eV, of the composed model (AIMNet2 + Coulomb + D3) on the 90
  structures of the pair scans, array `composed_energy` of shape (90, 1), in the file order
  of the NCI Atlas subset.
- Source of the structures: `../nci_atlas/nci-atlas-curves.csv.gz`, NCI Atlas (Jan Řezáč and
  contributors, CC BY 4.0; see `../nci_atlas/README.md` and `SOURCES_AND_LICENSES.md`). Three
  complexes at ten separations; AB, A and B at each point; elements H, C, N, O; total charge 0
  or -1; 4 to 25 atoms per structure.
- How built: the composed model of "Made to order" run once over the 90 structures.
- Licence of the energies: derived from the CC BY 4.0 structures by the course's own run.
