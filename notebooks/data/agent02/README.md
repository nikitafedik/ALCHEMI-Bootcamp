# Data of agentic notebook 02, "Many molecules, one batch"

Saved, verified defaults for the "Try it" pieces. Each loads in one call from `helpers/agent02.py`.

## `phenacetin.xyz`: `helpers.load_default_molecule()`

- What: one molecule for "Try it: add a molecule of your own": phenacetin, C10H13NO2, 26 atoms. It is not one of the five lesson molecules (24, 21, 33, 20 and 49 atoms), so the count at its position is unique.
- Why this one: only H, C, N and O (inside AIMNet2's elements), neutral and closed-shell, small enough to run in a fraction of a second.
- Verified: the step ran with it, passed `check my_batch`, and the AIMNet2 call on the batch that holds it gave finite energies and forces that sum to zero per molecule. A fresh agent also read it as a learner's own structure file. Record: `notes/evidence/agent-01/a01-c12--20261126-1130--02-default-record.md`.

## `ethanol.xyz`, `pyridine.xyz`, `urea.xyz`: `helpers.load_default_set()`

- What: three small molecules for "Try it: a batch of your choice", in this order: ethanol (9 atoms), pyridine (11), urea (8); 28 atoms in all.
- Why these: only H, C, N and O; neutral; three different sizes.
- Verified: packed into one batch, run through AIMNet2 once and saved to a store; passed `check own_batch`. Record: `notes/evidence/agent-01/a01-c24--20261126-1130--02-default-record.md`.

## `helpers.load_many(n_structures)`: no file

- What: `n_structures` ASE structures for "Try it: CPU first, or straight to the GPU": the five lesson molecules in turn, each copy rattled by 0.02 Å (fixed seed), so the copies differ.
- Called without a number it returns 1,000 structures, 29,400 atoms (PROVISIONAL size of the exercise; 0.46 s to build on a CPU).
- Verified: measured from 100 to 50,000 structures; the sample passes `check build_routes`. Records: `notes/evidence/agent-01/a01-tryit-build--20261126-0800--02-builder.md` and `a01-tryit-build--20261126-1130--02-default-record.md`.

## Source, licence, how built

- Source: none outside the repository. Each structure is built from a public SMILES string with RDKit (BSD licence): ETKDG embedding with seed 7, then an MMFF94 force-field optimisation. The comment line of each file records the SMILES and the recipe. No dataset and no download is involved.
- Licence: the files are generated and carry the repository's licence (Apache 2.0, `LICENSE`).
- How built: `python build_defaults.py` (needs RDKit and ASE) rewrites the four files.
- Geometry: a force-field geometry, a plausible gas-phase conformer, not a reference structure; good for building a batch and one model call.
