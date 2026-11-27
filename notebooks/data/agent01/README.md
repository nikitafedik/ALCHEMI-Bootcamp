# Agent 01 default molecule

`caffeine.xyz` is the saved default for "Your first prompt" in
`alchemi-agent-01-meet-your-agent.ipynb`. `helpers.load_default_molecule()`
(`helpers/agent01.py`) returns it as an ASE `Atoms` object.

| | |
| --- | --- |
| what | caffeine, C8H10N4O2, 24 atoms, neutral, closed shell, one gas-phase conformer, positions in Å, no cell |
| source | generated for these tutorials from the SMILES string `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`; no database entry and no third-party structure data |
| licence | none to attribute; the file is covered by the repository's licence |
| how built | RDKit 2026.03.4: `AddHs`, ETKDG embedding with `randomSeed=7`, MMFF94 optimisation (converged); written with ASE at 15 decimals |
| model fit | H, C, N, O are among the elements of the AIMNet2 checkpoint; `AIMNet2Wrapper` gives -18517.4883 eV (the wrapper leaves out Coulomb and D3) |

The geometry is a force-field conformer: fit for one model call and a 3D
view, and no reference structure. The build script and the verification
record are kept with the course notes, outside the repository.
