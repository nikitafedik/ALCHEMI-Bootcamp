# Data for notebook 03, "Ready-made"

## Course crystal set: `helpers.load_crystals(n)`

**What.** 285 periodic crystals: 95 base structures, each repeated into three
supercells of 8 to 128 atoms. Every atom is rattled by a Gaussian of 0.1 Å
standard deviation, with one fixed seed per structure. `load_crystals(n)` picks
`n` of them evenly by index, so the set is the same on every call. The default
is `n = 200`: 8,722 atoms, with starting largest forces of 0.117 to 20.85 eV/Å
under MACE-MPA-0.

**How built.** `helpers/_agent03_loaders.py` uses `ase.build.bulk`. No file is
stored; the set is rebuilt from the code. Base structures:
- metals: fcc, bcc and hcp, with ASE's built-in lattice constants
- diamond, rocksalt, zincblende, fluorite and CsCl-type (B2) crystals, with
  the lattice constant `a` given in the code
- wurtzite crystals, with `a` and `c` given in the code

**Licence.** ASE is LGPL-2.1-or-later. No structure database is used, so no
database licence applies. The lattice constants are published facts, cited
below.

**Model fit.** All 60 elements are in the MACE-MPA-0 element table (89
elements, Z 1 to 83 and 89 to 94). This was read from the checkpoint file. All
structures are periodic materials, which MACE-MPA-0 is trained on (MPTrj and
sAlex).

**Verification.** Checked on 26 Nov 2026 by reading files, with no model run:
the element table was read from the MACE-MPA-0 checkpoint, the licence and the
reference lattice constants from the ASE 3.27.0 source, and the lattice
constants of the compounds were compared with the published values in the
table below. The memory and time of the set on the course GPU are in the next
paragraph.

**Memory and time on the course GPU.** Measured on the NVIDIA L40S (46,068
MiB; PyTorch sees 44.39 GiB) on 27 Nov 2026, Toolkit 0.3.0, float32:

| Model on 200 crystals | Peak allocated | One energy and force pass |
| --- | --- | --- |
| MACE-MPA-0, no speed setting | 33.5 GiB (36 GB) | 0.98 s |
| MACE-MPA-0 with cuEquivariance | 8.5 GiB (9.1 GB) | about 0.11 s |

Without cuEquivariance the 200 crystals fill three quarters of the card. A
second model of that kind in the same session can run out of memory even
after `torch.cuda.empty_cache()`, because the freed memory is split into
pieces. The sample of the speed step therefore times 50 crystals picked
evenly from the batch (2,184 atoms, peak 9.5 GiB).

### Sources of the lattice constants

Metals use ASE's built-in values, which the ASE source attributes to Ashcroft
and Mermin, *Solid State Physics*. For the compounds, the table gives the value
in the code, the published value and the source. Differences are below 1
percent in every row; the largest is BeO a, +0.75 percent.

Source keys:
- W: Wikipedia, "Lattice constant", table at 300 K, a secondary compilation,
  read 26 Nov 2026 (https://en.wikipedia.org/wiki/Lattice_constant)
- wm: webmineral.com mineral data page for the named mineral (natural sample)
- COD n: Crystallography Open Database entry n
  (https://www.crystallography.net/cod/n.html)

| Compound | Structure | Code, Å | Published, Å | Source |
| --- | --- | --- | --- | --- |
| C | diamond | 3.567 | 3.567 | W |
| Si | diamond | 5.431 | 5.431 | W |
| Ge | diamond | 5.658 | 5.658 | W |
| Sn (grey) | diamond | 6.489 | 6.4896 | Wikipedia "Tin", citing Arblaster (2018) |
| LiF | rocksalt | 4.03 | 4.03 | W |
| LiCl | rocksalt | 5.13 | 5.14 | W |
| NaF | rocksalt | 4.63 | 4.63 | W |
| NaCl | rocksalt | 5.64 | 5.64 | W |
| NaBr | rocksalt | 5.97 | 5.97 | W |
| KF | rocksalt | 5.35 | 5.34 | W |
| KCl | rocksalt | 6.29 | 6.29 | W |
| KBr | rocksalt | 6.60 | 6.60 | W |
| MgO | rocksalt | 4.21 | 4.212 | W |
| CaO | rocksalt | 4.81 | 4.797 | wm, lime |
| SrO | rocksalt | 5.16 | 5.1396 | COD 1100027, Primak, Kaufman, Ward, JACS 70, 2043 (1948) |
| BaO | rocksalt | 5.52 | 5.496 | COD 1011362, Gerlach, Z. Phys. 9, 184 (1922) |
| CaS | rocksalt | 5.69 | 5.6908 | Wikipedia "Calcium sulfide"; wm oldhamite 5.686 |
| TiC | rocksalt | 4.33 | 4.328 | W |
| TiN | rocksalt | 4.24 | 4.249 | W |
| NiO | rocksalt | 4.18 | 4.1768 | wm, bunsenite |
| GaAs | zincblende | 5.65 | 5.653 | W |
| GaP | zincblende | 5.45 | 5.4505 | W |
| InP | zincblende | 5.87 | 5.869 | W |
| ZnS | zincblende | 5.41 | 5.42 | W |
| ZnSe | zincblende | 5.67 | 5.67 | wm, stilleite |
| SiC (3C) | zincblende | 4.36 | 4.3596 | Wikipedia "Silicon carbide", from the Ioffe Institute SiC data |
| AlP | zincblende | 5.46 | 5.451 | W |
| AlAs | zincblende | 5.66 | 5.6605 | W |
| CdTe | zincblende | 6.48 | 6.482 | W |
| InAs | zincblende | 6.06 | 6.0583 | W |
| GaSb | zincblende | 6.10 | 6.0959 | W |
| BP | zincblende | 4.54 | 4.538 | W |
| AlSb | zincblende | 6.14 | 6.1355 | W |
| ZnTe | zincblende | 6.10 | 6.101 | CRC Handbook, 92nd ed. (2011), p. 12.80, via Wikipedia "Zinc telluride" |
| CuCl | zincblende | 5.41 | 5.418 | wm, nantokite |
| CaF2 | fluorite | 5.46 | 5.463 | wm, fluorite |
| SrF2 | fluorite | 5.80 | 5.7996 | Wyckoff, *Crystal Structures* vol. 1 (1963), via AMCSD (rruff.geo.arizona.edu); see note 1 |
| BaF2 | fluorite | 6.20 | 6.1964 | wm, frankdicksonite |
| CeO2 | fluorite | 5.41 | 5.42 | wm, cerianite-(Ce) |
| ThO2 | fluorite | 5.60 | 5.595 | wm, thorianite |
| PbF2 | fluorite | 5.94 | 5.9397 | COD 9014557, Achary and Tyagi, Powder Diffraction (2005) |
| CsCl | B2 | 4.12 | 4.123 | W |
| CsBr | B2 | 4.29 | 4.291 | W |
| CsI | B2 | 4.57 | 4.567 | W |
| NiAl | B2 | 2.89 | 2.887 | Talas (2018), *Intermetallic Matrix Composites*, via Wikipedia "Nickel aluminide" |
| FeAl | B2 | 2.91 | 2.91 | COD 1541193, Makhlouf, Nakamura, Shiga, JMMM 135, 257 (1994) |
| CuZn (beta brass) | B2 | 2.95 | 2.945 | COD 9008814, Wyckoff, *Crystal Structures* vol. 1 (1963); see note 2 |
| AgMg | B2 | 3.30 | 3.3114 | COD 1509456, Matsuo, Katada, Minamigawa, Scripta Metall. 12, 821 (1978) |
| ZnO | wurtzite | 3.25 / 5.21 | 3.25 / 5.20 | W |
| GaN | wurtzite | 3.19 / 5.19 | 3.189 / 5.185 | W |
| AlN | wurtzite | 3.11 / 4.98 | 3.112 / 4.982 | W |
| CdS | wurtzite | 4.14 / 6.72 | 4.16 / 6.756 | W |
| ZnS | wurtzite | 3.82 / 6.26 | 3.82 / 6.26 | W |
| BeO | wurtzite | 2.70 / 4.38 | 2.68 / 4.36 | wm, bromellite |
| InN | wurtzite | 3.54 / 5.70 | 3.533 / 5.693 | W |
| CdSe | wurtzite | 4.30 / 7.01 | 4.271 / 6.969 | wm, cadmoselite |

Notes:
1. SrF2: the AMCSD value was read in a search summary of the AMCSD page, not
   on the page itself. The natural mineral strontiofluorite has a = 5.713 Å
   (COD 9016413); the synthetic value is the one used.
2. CuZn and AgMg: COD gives no measurement temperature. The CuZn value comes
   from a compilation, so its original measurement is not named. Other COD
   entries for AgMg give 3.23 Å (COD 1509455, Williams 1972) and 3.302 Å
   (COD 1509457, Kinzhibalo and Zarechnyuk 1978).

## Temperature ladder default: `helpers.load_ladder_crystal()`

**What.** `ladder-default-Cu-fcc-4x4x4.extxyz`: copper, fcc, 4 x 4 x 4
primitive cells, 64 atoms, periodic on all three axes. Cell vectors 10.211 Å
at 60°; the spacing between opposite cell faces is 8.337 Å, above the 6 Å
cutoff of MACE-MPA-0. Every atom sits on its lattice site (no rattle), so the
forces are zero by symmetry and the crystal needs no relaxation.
`load_ladder_crystal()` returns it as ASE `Atoms`.

**How built.** `ase.build.bulk("Cu", "fcc").repeat((4, 4, 4))` with ASE 3.27.0
and ASE's built-in lattice constant a = 3.61 Å, written with
`ase.io.write(..., format="extxyz")`. It is the cell of the course crystal
`Cu-fcc-4x4x4` in `load_crystals`, without the 0.1 Å rattle.

**Source and licence.** Built with ASE (LGPL-2.1-or-later); no database is
used. The lattice constant is ASE's reference value, which the ASE source
attributes to Ashcroft and Mermin, *Solid State Physics*.

**Model fit.** Copper is in the MACE-MPA-0 element table, and a bulk metal is
inside its training data (MPTrj and sAlex).

**Verification.** On 26 Nov 2026 the ladder step ran end to end on a CPU (4
threads, Toolkit 0.3.0) with this crystal: the sample file
`solutions/agent-03/temperature_ladder.py`, the view and the step check. On
27 Nov 2026 the notebook ran at course size on the NVIDIA L40S; the ladder
cell, with 12 temperatures, took 36.7 s.
