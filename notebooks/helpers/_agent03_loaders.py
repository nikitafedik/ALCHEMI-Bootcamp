# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crystal set of Agent 03, Ready-made.

``build_crystals`` builds 285 common crystals with ASE (supercells of varied
size, 8 to 128 atoms, cell unchanged) and rattles every atom by 0.1 Å with one
fixed seed per structure. ``agent03.load_crystals`` picks the learner's set from
it. Only ASE is needed.
"""

from __future__ import annotations

from ase import Atoms
from ase.build import bulk

# (formula, structure, lattice a in Å, c or None); every element is inside MACE-MPA-0.
_METALS_FCC = ["Al", "Ca", "Ni", "Cu", "Rh", "Pd", "Ag", "Ir", "Pt", "Au", "Pb", "Sr"]
_METALS_BCC = ["Li", "Na", "K", "Rb", "Cs", "Ba", "V", "Cr", "Fe", "Nb", "Mo", "Ta", "W"]
_METALS_HCP = ["Be", "Mg", "Sc", "Ti", "Co", "Zn", "Y", "Zr", "Ru", "Cd", "Hf", "Re", "Os", "Tl"]
_DIAMOND = {"C": 3.567, "Si": 5.431, "Ge": 5.658, "Sn": 6.489}
_ROCKSALT = {
    "LiF": 4.03, "LiCl": 5.13, "NaF": 4.63, "NaCl": 5.64, "NaBr": 5.97, "KF": 5.35,
    "KCl": 6.29, "KBr": 6.60, "MgO": 4.21, "CaO": 4.81, "SrO": 5.16, "BaO": 5.52,
    "CaS": 5.69, "TiC": 4.33, "TiN": 4.24, "NiO": 4.18,
}
_ZINCBLENDE = {
    "GaAs": 5.65, "GaP": 5.45, "InP": 5.87, "ZnS": 5.41, "ZnSe": 5.67, "SiC": 4.36,
    "AlP": 5.46, "AlAs": 5.66, "CdTe": 6.48, "InAs": 6.06, "GaSb": 6.10,
    "BP": 4.54, "AlSb": 6.14, "ZnTe": 6.10, "CuCl": 5.41,
}
_FLUORITE = {"CaF2": 5.46, "SrF2": 5.80, "BaF2": 6.20, "CeO2": 5.41, "ThO2": 5.60, "PbF2": 5.94}
_CSCL = {
    "CsCl": 4.12, "CsBr": 4.29, "CsI": 4.57, "NiAl": 2.89, "FeAl": 2.91, "CuZn": 2.95,
    "AgMg": 3.30,
}
_WURTZITE = {
    "ZnO": (3.25, 5.21), "GaN": (3.19, 5.19), "AlN": (3.11, 4.98), "CdS": (4.14, 6.72),
    "ZnS": (3.82, 6.26), "BeO": (2.70, 4.38), "InN": (3.54, 5.70), "CdSe": (4.30, 7.01),
}

# Supercell repeat patterns, cycled so the sizes vary.
_REPEATS = [(2, 2, 2), (3, 3, 3), (2, 3, 4), (2, 2, 3), (3, 3, 2), (4, 4, 4)]
RATTLE_STDEV = 0.1  # Å
MAX_ATOMS = 128


def _base_cells() -> list[tuple[str, Atoms]]:
    cells = []
    cells += [(f"{s}-fcc", bulk(s, "fcc")) for s in _METALS_FCC]
    cells += [(f"{s}-bcc", bulk(s, "bcc")) for s in _METALS_BCC]
    cells += [(f"{s}-hcp", bulk(s, "hcp")) for s in _METALS_HCP]
    cells += [(f"{s}-diamond", bulk(s, "diamond", a=a)) for s, a in _DIAMOND.items()]
    cells += [(f"{f}-rocksalt", bulk(f, "rocksalt", a=a)) for f, a in _ROCKSALT.items()]
    cells += [(f"{f}-zincblende", bulk(f, "zincblende", a=a)) for f, a in _ZINCBLENDE.items()]
    cells += [(f"{f}-fluorite", bulk(f, "fluorite", a=a)) for f, a in _FLUORITE.items()]
    cells += [(f"{f}-cscl", bulk(f, "cesiumchloride", a=a)) for f, a in _CSCL.items()]
    cells += [(f"{f}-wurtzite", bulk(f, "wurtzite", a=a, c=c)) for f, (a, c) in _WURTZITE.items()]
    return cells


def build_crystals(per_base: int = 3, seed: int = 0) -> list[Atoms]:
    """Return every rattled supercell: ``per_base`` sizes of each base structure.

    Rattle: Gaussian displacement, standard deviation 0.1 Å, one fixed seed per
    structure. The result is the same on every call.
    """

    out = []
    for b, (label, cell) in enumerate(_base_cells()):
        for k in range(per_base):
            reps = _REPEATS[(b + k) % len(_REPEATS)]
            atoms = cell.repeat(reps)
            if len(atoms) > MAX_ATOMS:  # cap the batch size
                reps = (2, 2, 2)
                atoms = cell.repeat(reps)
            while len(atoms) < 8:  # a small cell gets a bigger repeat
                atoms = atoms.repeat((2, 1, 1))
            atoms.rattle(stdev=RATTLE_STDEV, seed=seed + 1000 * b + k)
            atoms.info["label"] = f"{label}-{reps[0]}x{reps[1]}x{reps[2]}"
            atoms.info.pop("charge", None)
            out.append(atoms)
    return out
