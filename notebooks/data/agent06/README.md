# Agent 06 default complexes

Three molecular complexes, each a neutral pair of two neutral molecules, at
the equilibrium geometry of the NCI Atlas. `helpers.agent06.prepare_complexes()`
reads the three files named in `DEFAULT_COMPLEX_FILES` and returns them as the
`complexes` of the notebook's setup cell.

| File | Complex | Atoms (fragment A) | Hydrogen bond | Source | CCSD(T)/CBS [kcal/mol] | Composed model [kcal/mol] |
| --- | --- | --- | --- | --- | --- | --- |
| `complex-1-phenol-n-methylacetamide.xyz` | phenol - N-methylacetamide | 25 (13) | O-H...O | NCI Atlas HB375x10 1.041, scale 1.00 | -11.889 | -11.582 |
| `complex-2-propyne-methyl-azide.xyz` | propyne - methyl azide | 14 (7) | none, dispersion | NCI Atlas D442x10 1.07.74, scale 1.00 | -3.312 | -3.062 |
| `complex-3-phenol-n-methylpropan-2-imine.xyz` | phenol - N-methylpropan-2-imine | 27 (13) | O-H...N | NCIA250, NCIA_HB375x10_3.032_100 | -12.827 | -12.198 |
| `complex-3-alternative-pyrrole-n-methylpropan-2-imine.xyz` | pyrrole - N-methylpropan-2-imine | 24 (10) | N-H...N | NCIA250, NCIA_HB375x10_4.008_100 | -9.228 | -9.119 |

The fourth file is a saved alternative for the third complex. To use it, put
its name in `DEFAULT_COMPLEX_FILES` in `helpers/agent06.py`.

## Format

Extended XYZ, positions in Å, fragment A first. The comment line holds
`name`, `fragment_a_size` (the first atoms are fragment A, the rest fragment
B), `charge`, `charge_a` and `charge_b` (all 0), `source`,
`ccsd_t_cbs_kcal_mol` (the NCI Atlas benchmark interaction energy) and
`composed_model_ev` (the interaction energy at this geometry with the composed
model of the notebook, AIMNet2 with its Coulomb and D3 terms, measured on a CPU
with nvalchemi-toolkit 0.3.0). The result views read the learner's energy unit
from `composed_model_ev`.

## Source and licence

- Creator and attribution: Jan Řezáč and NCI Atlas contributors
- Source: [NCI Atlas](https://github.com/Honza-R/NCIAtlas), revision
  `1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`, through the two files of
  `../nci_atlas/` (see its `README.md` for the papers to cite)
- Licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

## How built

`python data/agent06/build_defaults.py`, run from `notebooks/`. Complexes 1
and 2 are the scale 1.00 structures of `../nci_atlas/nci-atlas-curves.csv.gz`;
the third complex and its alternative are members of `../nci_atlas/NCIA250.zip`.
Coordinates and atom order are copied unchanged; the files are reformatted
and the comment line is added. NCIA250 gives an id for each complex and no
name: the two names of the third complex are read from the bonds of the
structure. Nothing is downloaded.
