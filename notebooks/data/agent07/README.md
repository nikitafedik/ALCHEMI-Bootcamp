# Agent 07 default structures

The default of both "Try it" pieces of Agent 07 is the study the setup cell
prepares: the neutral molecular complexes of the NCIA250 set with 10 atoms
and more (149 of the 205, `MIN_ATOMS` in `helpers/agent07.py`) at their
published equilibrium geometries, `batch_multiplier` trajectories of each.
`helpers.agent07.prepare_study(device, batch_multiplier=26)` returns them as
`trajectories` (3,874 trajectories, 59,202 atoms, 10 to 33 atoms each, each
complex 26 times with its own velocity draw) with the composed model. The
counts follow from `study_complexes.csv` and were confirmed by a run with
Toolkit 0.3.0. The small study of the step check (`batch_multiplier=0.1`)
keeps 17 trajectories.

`study_complexes.csv` lists all 205 complexes of the set, in the order of the
set; the study keeps the rows with `natoms` of 10 and more, in this order (each
copy follows the one before): position, source set, system
id, interaction class, charge, number of atoms, source file inside
`../nci_atlas/NCIA250.zip`, and formula. The complexes have 4 to 33 atoms
(2668 atoms in one copy) and all are neutral.

No structure is stored here a second time: the loader reads
`../nci_atlas/NCIA250.zip`.

## Source and licence

- Creator and attribution: Jan Řezáč and NCI Atlas contributors
- Source: [NCI Atlas](https://github.com/Honza-R/NCIAtlas), revision
  `1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`, through `../nci_atlas/NCIA250.zip`
  (see `../nci_atlas/README.md` for the papers to cite)
- Licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

## How built

`python data/agent07/build_defaults.py`, run from `notebooks/`. It reads the
archive through `helpers.ncia250.load_ncia250()`, keeps the rows of the whole
complexes (`fragment == "AB"`), and writes the table. It loads no model and
downloads nothing.

## Model fit

The composed model of the notebook is AIMNet2 with its Coulomb and D3 terms,
the course's model for neutral organic molecules. Every complex is neutral and
made of the elements the set was built for (see the papers above).
