# SiC-TGAP: the liquid SiC data of notebook 09

Notebook 09 reads only `cut/`. The source file is not in this repository; it is needed only to write the cut again.

## Source, licence and citation

- Title: Research data for: "SiC-TGAP: A machine learning interatomic potential for radiation damage simulations in 3C-SiC"
- Authors: Hamedani, Ali (Aalto University, ORCID 0000-0001-6456-6482); E. Sand, Andrea (Aalto University, ORCID 0000-0001-9041-1468)
- Record: https://zenodo.org/records/17182227 (API record saved as `zenodo-record.json`); DOI 10.5281/zenodo.17182227; published 2025-09-23
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), as the record states it (`cc-by-4.0`, open access). The record names no further terms and no required citation.
- Cite as: Hamedani, A., & E. Sand, A. (2025). Research data for: "SiC-TGAP: A machine learning interatomic potential for radiation damage simulations in 3C-SiC" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17182227
- Level of theory (from the record): VASP, PBE. Labels: total energy, forces, virial stress; `config_type` and `sub_config` per frame. 3,000 frames in all.

The source file, as the record lists it:

| File | Size (bytes) | MD5 (record) | SHA-256 |
| --- | --- | --- | --- |
| `dataset-SiC-TGAP.xyz` | 29,061,339 | ac7b5abbbdff55a9c32c41d89d3c6ea3 | 071463f8584b12cd505231ccf00ceb6000885f37484fbd8d006092a3eb931500 |

Download: https://zenodo.org/api/records/17182227/files/dataset-SiC-TGAP.xyz/content. The cut uses none of the
record's other files (two archives of potentials).

## The cut (`cut/`, written by `cut_liquid.py`)

| File | What | Frames |
| --- | --- | --- |
| `liquid-train.extxyz.gz` (2.3 MB) | every frame of the four liquid frame types at the five training densities 2.3, 2.5, 2.9, 3.05 and 3.2 g/cm3: `melt_aimd` 100, `eqlMelt_aimd` 100, `quench_aimd` 50, `amorph_aimd` 50 | 300 |
| `liquid-heldout.extxyz.gz` (0.45 MB) | the whole density 2.7 lineage of the same four types: 20, 20, 10, 10 | 60 |
| `cut-record.json` | source frame indices, counts by type and density, SHA-256 of the source and of the uncompressed cut | |
| `LICENSE` | the licence statement and the list of changes | |

Every frame has 216 atoms (Si108C108) in a cubic periodic cell; one cell size per density. The four types at one
density are consecutive stages of one AIMD lineage (melting, equilibrated liquid, quench, amorphous), so a held-out
density shares no trajectory with the training frames. Labels in the cut: `dft_energy` (eV, frame info) and
`dft_forces` (eV/Å, per atom). The equilibrated-liquid frames carry their temperature in `sub_config`
(`rho2.3_Tm3000`: 3000 K); the notebook starts its MD from them: 3000 K at 2.3 and 2.5, 3400 K at 2.9, 3700 K at
3.05, 4000 K at 3.2 g/cm3.

Changes against the source (also in `cut/LICENSE`): frames of other types dropped; split by density; energy and forces
renamed, virial and stress dropped; `source_index` added; gzip compression.

Loader: `helpers.agent09.prepare_finetuning(device)` (training and held-out stores) and `helpers.agent09.load_starts(n)`
(the MD starts).

## Writing the cut again

Download `dataset-SiC-TGAP.xyz` from the record, save it as `raw/dataset-SiC-TGAP.xyz` in this folder, and run
`python cut_liquid.py` here (it needs `ase` and `numpy`). The script writes the two frame files and `cut-record.json`
into `cut/`; the output has no time stamp, so every run gives the same bytes.

## What was verified

- The download matched the record's MD5 and has the SHA-256 above.
- `cut_liquid.py`, run on that file, writes the three files of `cut/` byte for byte.
- `cut-record.json` holds the SHA-256 of both frame files (uncompressed); `notebooks/tests/test_agent09.py` checks
  them against the files, with the frame counts by type and density.
- The numbers in notebook 09 were measured on exactly these frames, in this order.
