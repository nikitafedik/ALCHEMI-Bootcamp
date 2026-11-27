# Agent 08 data: the four default reactions

`reactions-v2.extxyz` holds the saved, verified default reactions of notebook 08,
"Paths of least resistance, batched". `helpers.load_reactions(model)` reads it.

## What

Eight frames, two per reaction (`end=reactant`, `end=product`), the reaction
name in `reaction=`. Atoms are matched by index between the two ends, and the
product is already laid over the reactant.

| Reaction | Atoms | Kind |
| --- | --- | --- |
| ammonia inversion | 4 | conformational |
| ethane rotation | 8 | conformational |
| HCN to HNC | 3 | bond-changing |
| malonaldehyde proton transfer | 9 | bond-changing |

## Source and licence

Built in this repository with ASE (`ase.build.molecule` for NH3 and C2H6,
hand-written coordinates for HCN and malonaldehyde): the builders in
`helpers/agent08.py`, `REACTIONS`. No outside data. Same licence as the
repository (Apache-2.0).

## How built

```python
import torch
from helpers import agent08
model = agent08.load_model(torch.device("cpu"))   # AIMNet2 (aimnet2-wb97m-d3_0) + Coulomb (15 Å), no D3
agent08.save_default_reactions(model)              # relaxes all ends to 0.0002 eV/Å and writes reactions-v2.extxyz
```

Built on 2026-11-26, CPU, nvalchemi-toolkit 0.3.0, with the notebook's model:
AIMNet2 with its Coulomb term, no D3. Ends relaxed with D3 in the model carry
up to 0.020 eV/Å with this model and lie at most 0.004 Å from the saved ones.

## Verification

- Largest force on an atom, per end, with the notebook's model: 0.00000 to
  0.00014 eV/Å. The loader accepts an end at or below 0.0005 eV/Å and relaxes
  the ends again with the model at hand when one is above it (with D3 added,
  for example).
- With `neb_settings = dict(num_images=8, fmax=0.003, regular_fmax=0.03,
  n_steps=600)` all four paths converge (stops at steps 74, 97, 248, 94 on
  CPU); barriers 5.57, 1.73, 51.88 and 4.58 kcal/mol; every top image (image
  3) has exactly one imaginary mode, -1003, -244, -2008 and -1264 cm^-1 from
  `model.compute_hessian`, and a force at or below 0.0016 eV/Å. The three
  step checks pass with the sample files (paths, neb, saddle).
- Model fit: the reactions are small neutral molecules of H, C, N, O, the
  domain of AIMNet2; the model is AIMNet2 with its Coulomb term, without D3.
