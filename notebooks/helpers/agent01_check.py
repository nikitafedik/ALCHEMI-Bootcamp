# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step check for Agent 01, Meet your agent.

Run from ``notebooks/``::

    python -m helpers.agent01 check <step>

The check runs the setup cell, then the file of the step (twice, as a learner
re-runs a cell), in one namespace, and prints one plain line per problem. It
carries what the step table leaves out: ``my_molecule`` is one structure,
``my_energy`` is one finite number in eV, and the result cell can show both.
The learner chooses the molecule and the model. The check names neither, with
one exception: a MACE-MPA-0 model left in the file's namespace (alone or inside
a pipeline) is a materials model, and a non-periodic ``my_molecule`` fails.
A model the check cannot identify passes. For the saved default molecule, an
energy more than 1 eV from the bare AIMNet2 value adds an "unexpected" line
(a Hartree value, for example); the step still passes.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["MESSAGES", "STEPS", "Step", "resolve_step", "run_check"]

WORK_NAME = "agent01_work"
SETUP_CELL = (
    "from helpers import agent01 as helpers\n"
    "helpers.configure_tutorial()\n"
    "work_dir = helpers.make_work_dir()"
)

# Every line the check can print, in one place. The wording is content: the
# learner's agent reads these lines and the learner may see them. The lines
# about my_molecule, my_energy and the model are printed after "{step}: ".
MESSAGES = {
    "not_found": "{work}/{file} not found",  # the {error} of "run_failed"
    "run_failed": "{step}: running {file} failed: {error}",
    "rerun_failed": "{step}: running {file} a second time failed: {error}",
    "missing": "{step}: {name} is missing; the step asks for it",
    "molecule_batch": "my_molecule is a Batch of {n}; it should be one structure (ASE Atoms or AtomicData)",
    "molecule_type": "my_molecule is {type}; it should be one structure (ASE Atoms or AtomicData)",
    "molecule_empty": "my_molecule has no atoms",
    "molecule_nan": "my_molecule has positions that are not finite",
    "molecule_show": "my_molecule cannot be shown: {error}",
    "energy_type": "my_energy is {type}; it should be one number in eV",
    "energy_shape": "my_energy has shape {shape}; one molecule has one energy in eV",
    "energy_nan": "my_energy is {value}; it should be a finite number in eV",
    "ok": "{step}: OK ({n} names)",
    "unknown_step": "unknown step {step}; steps: {steps}",
    "earlier_failed": "{step}: an earlier file failed: {file}: {error}",
    "model_fit": (
        "the model is MACE-MPA-0 ({checkpoint}), a model for materials; "
        "a molecule needs a model for molecules, such as AIMNet2"
    ),
    "script_name": "{step}: {file} uses {name}, which does not exist in a notebook cell; remove that line",
    "unexpected_energy": (
        "{step}: unexpected: my_energy is {value:.4f}; AIMNet2 as Toolkit ships it gives {saved:.4f} eV "
        "for the default molecule (the step still passes)"
    ),
}

DEFAULT_TOLERANCE_EV = 1.0  # wider than any rounding, far narrower than eV against Hartree
_SCRIPT_NAMES = {"__file__"}  # names a script has and a notebook cell does not


@dataclass(frozen=True)
class Step:
    """One step of the notebook: its file, the names the file leaves, and the steps whose files run first."""

    file: str
    names: tuple[str, ...]
    after: tuple[str, ...] = field(default=())


STEPS: dict[str, Step] = {
    "first_prompt": Step("first_prompt.py", ("my_molecule", "my_energy")),
}


def _run_file(ns: dict, work: Path, file: str):
    path = work / file
    if not path.exists():
        return FileNotFoundError(MESSAGES["not_found"].format(work=WORK_NAME, file=file))
    try:
        exec(compile(path.read_text(), str(path), "exec"), ns)
    except Exception as exc:  # reported to the learner as one line
        return exc
    return None


def _first_line(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}".strip()


def _check_names(step: str, ns: dict, problems: list[str]) -> bool:
    missing = [n for n in STEPS[step].names if n not in ns]
    for name in missing:
        problems.append(MESSAGES["missing"].format(step=step, name=name))
    return not missing


def _shape_of(value) -> tuple:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else ()


def _check_molecule(molecule, add) -> None:
    import numpy as np
    from ase import Atoms
    from nvalchemi.data import AtomicData, Batch

    from .agent01 import as_atoms

    if isinstance(molecule, Batch):
        add(MESSAGES["molecule_batch"].format(n=molecule.num_graphs))
        return
    if not isinstance(molecule, (Atoms, AtomicData)):
        add(MESSAGES["molecule_type"].format(type=type(molecule).__name__))
        return
    try:
        atoms = as_atoms(molecule)
    except Exception as exc:
        add(MESSAGES["molecule_show"].format(error=_first_line(exc)))
        return
    if len(atoms) == 0:
        add(MESSAGES["molecule_empty"])
    elif not bool(np.isfinite(atoms.get_positions()).all()):
        add(MESSAGES["molecule_nan"])


def _check_energy(energy, add) -> None:
    import numpy as np
    import torch

    if isinstance(energy, bool) or not isinstance(energy, (int, float, np.generic, np.ndarray, torch.Tensor)):
        add(MESSAGES["energy_type"].format(type=type(energy).__name__))
        return
    if isinstance(energy, (np.ndarray, torch.Tensor)):
        count = int(energy.numel()) if isinstance(energy, torch.Tensor) else int(energy.size)
        if count != 1:
            add(MESSAGES["energy_shape"].format(shape=_shape_of(energy)))
            return
        energy = energy.reshape(-1)[0]
    try:
        value = float(energy)
    except Exception:
        add(MESSAGES["energy_type"].format(type=type(energy).__name__))
        return
    if not math.isfinite(value):
        add(MESSAGES["energy_nan"].format(value=value))


def _is_default_molecule(molecule) -> bool:
    """True when the structure is the saved default: same atoms in the same order, same geometry up to a shift."""

    import numpy as np

    from .agent01 import as_atoms, load_default_molecule

    try:
        atoms, default = as_atoms(molecule), load_default_molecule()
    except Exception:
        return False
    if list(atoms.get_atomic_numbers()) != list(default.get_atomic_numbers()):
        return False
    mine, saved = atoms.get_positions(), default.get_positions()
    return bool(np.allclose(mine - mine.mean(0), saved - saved.mean(0), atol=1e-3))


def _default_energy_note(step: str, molecule, energy) -> list[str]:
    """The "unexpected" line when the default molecule's energy is far from the saved bare AIMNet2 value."""

    from .agent01 import DEFAULT_ENERGY_EV, as_energy

    if not _is_default_molecule(molecule):
        return []
    value = as_energy(energy)
    if abs(value - DEFAULT_ENERGY_EV) <= DEFAULT_TOLERANCE_EV:
        return []
    return [MESSAGES["unexpected_energy"].format(step=step, value=value, saved=DEFAULT_ENERGY_EV)]


def _materials_checkpoints(ns: dict) -> list[str]:
    """Return the MACE-MPA-0 checkpoint names of the models left in the namespace.

    A model is found when it is a value of the namespace, sits in a list, tuple or
    dict there, or is a submodule of one (a pipeline). A ``MACEWrapper`` built
    without ``from_checkpoint`` has no spec and is not identified. When an
    AIMNet2 model is also present the check cannot tell which one gave the
    energy, so it returns nothing.
    """

    mace = sys.modules.get("nvalchemi.models.mace")
    if mace is None:  # no MACE model was made
        return []
    import torch

    values = list(ns.values())
    for value in list(values):
        if isinstance(value, dict):
            values.extend(value.values())
        elif isinstance(value, (list, tuple)):
            values.extend(value)
    aimnet2 = sys.modules.get("nvalchemi.models.aimnet2")
    found: list[str] = []
    for value in values:
        if not isinstance(value, torch.nn.Module):
            continue
        for module in value.modules():
            if aimnet2 is not None and isinstance(module, aimnet2.AIMNet2Wrapper):
                return []
            if not isinstance(module, mace.MACEWrapper):
                continue
            spec = module.checkpoint_spec()
            path = str(getattr(spec, "checkpoint_path", "") or "")
            if "mpa0" in "".join(c for c in path.lower() if c.isalnum()) and path not in found:
                found.append(path)
    return found


def _check_model_fit(molecule, ns: dict, add) -> None:
    from .agent01 import as_atoms

    try:
        if bool(as_atoms(molecule).get_pbc().all()):
            return  # a periodic structure: the check cannot call it a molecule
    except Exception:
        return
    for path in _materials_checkpoints(ns):
        add(MESSAGES["model_fit"].format(checkpoint=Path(path).name))


def _check_content(step: str, ns: dict, problems: list[str]) -> None:
    def add(text: str) -> None:
        problems.append(f"{step}: {text}")

    if step == "first_prompt":
        _check_molecule(ns["my_molecule"], add)
        _check_energy(ns["my_energy"], add)
        if not problems:
            _check_model_fit(ns["my_molecule"], ns, add)


def resolve_step(name: str) -> str:
    """Return the step a learner's spelling names: a stem, a file name, a path, or the step's number in file order."""

    step = Path(name).name.removesuffix(".py")
    if step.isdigit() and 1 <= int(step) <= len(STEPS):
        step = list(STEPS)[int(step) - 1]
    return step


def run_check(step: str, root: Path | None = None) -> int:
    """Run one step check; print the problems; return 0 when there are none."""

    step = resolve_step(step)
    if step not in STEPS:
        print(MESSAGES["unknown_step"].format(step=repr(step), steps=", ".join(STEPS)))
        return 2
    root = root or Path.cwd()
    work = root / WORK_NAME
    sys.path.insert(0, str(root))
    ns: dict = {"__name__": "__main__"}
    exec(SETUP_CELL, ns)
    for earlier in STEPS[step].after:
        error = _run_file(ns, work, STEPS[earlier].file)
        if error is not None:
            print(MESSAGES["earlier_failed"].format(step=step, file=STEPS[earlier].file, error=_first_line(error)))
            return 1
    for attempt in (1, 2):  # the second run is a learner running the cell again
        error = _run_file(ns, work, STEPS[step].file)
        if error is None:
            continue
        if isinstance(error, NameError) and getattr(error, "name", None) in _SCRIPT_NAMES:
            print(MESSAGES["script_name"].format(step=step, file=STEPS[step].file, name=error.name))
        elif attempt == 1:
            print(MESSAGES["run_failed"].format(step=step, file=STEPS[step].file, error=_first_line(error)))
        else:
            print(MESSAGES["rerun_failed"].format(step=step, file=STEPS[step].file, error=_first_line(error)))
        return 1
    problems: list[str] = []
    if _check_names(step, ns, problems):
        _check_content(step, ns, problems)
    for line in problems:
        print(line)
    if not problems:
        if step == "first_prompt":
            for note in _default_energy_note(step, ns["my_molecule"], ns["my_energy"]):
                print(note)
        print(MESSAGES["ok"].format(step=step, n=len(STEPS[step].names)))
    return 1 if problems else 0
