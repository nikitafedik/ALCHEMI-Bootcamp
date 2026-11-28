# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 01, Meet your agent.

The notebook's setup cell calls ``configure_tutorial()``, ``make_work_dir()`` and
``show_session()``; its result cell calls ``show_first_result()``.
``load_default_molecule()`` returns the saved default for "Your first prompt"
(caffeine, ``data/agent01/caffeine.xyz``). The step check
is ``python -m helpers.agent01 check <step>``, run from ``notebooks/``; its
contract and messages are in ``agent01_check``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from ase import Atoms

from .common import configure_tutorial as _configure_course

__all__ = [
    "as_atoms",
    "as_energy",
    "configure_tutorial",
    "load_default_molecule",
    "make_work_dir",
    "show_first_result",
    "show_session",
]

WORK_DIR = Path(__file__).resolve().parents[1] / "agent01_work"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "agent01"
DEFAULT_MOLECULE = "caffeine"
DEFAULT_ENERGY_EV = -18517.4883  # AIMNet2Wrapper as Toolkit ships it, on the saved caffeine (data/agent01/README.md)


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    WORK_DIR.mkdir(exist_ok=True)
    return WORK_DIR


make_work_dir()  # the folder exists as soon as the helpers are imported, before the setup cell runs


def configure_tutorial() -> None:
    """Apply the course display defaults and silence known library notices."""

    # The AIMNet2 wrapper registers its own warning filter when imported, so the
    # import comes first and the course filters are applied on top of it.
    import nvalchemi.models.aimnet2  # noqa: F401

    _configure_course()


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(
        "Give these to your agent when you start the session:\n"
        "  notebook    notebooks/alchemi-agent-01-meet-your-agent.ipynb\n"
        "  helpers     from helpers import agent01 as helpers\n"
        "  skills      nvalchemi-toolkit; prompt-check (skills/prompt-check/SKILL.md)\n"
        "  work in     notebooks/agent01_work/\n"
        "  check with  python -m helpers.agent01 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
    )


def load_default_molecule() -> Atoms:
    """A saved molecule for "Your first prompt": caffeine, 24 atoms, neutral, as ASE ``Atoms``."""

    from ase.io import read

    atoms = read(DATA_DIR / f"{DEFAULT_MOLECULE}.xyz", format="xyz")  # plain xyz: the comment line is not parsed
    atoms.info["name"] = DEFAULT_MOLECULE
    atoms.info["charge"] = 0
    return atoms


def as_atoms(molecule: Any) -> Atoms:
    """Return an ASE structure from ``Atoms`` or from one Toolkit ``AtomicData``."""

    if isinstance(molecule, Atoms):
        return molecule
    from .viewers import _data_to_atoms

    return _data_to_atoms(molecule)


def as_energy(value: Any) -> float:
    """Return one Python float from a float, a NumPy value or a one-element tensor."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    return float(value)


def show_first_result(my_molecule: Any, my_energy: Any) -> None:
    """Show the learner's molecule as a rotatable 3D view, and its energy in eV."""

    from IPython.display import HTML, Markdown, display

    from . import viewers, viewers_three

    atoms = as_atoms(my_molecule)
    payload = viewers._structure_payload(atoms)
    item = {"display_name": payload["formula"], **payload}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # IPython suggests IFrame for the viewer's srcdoc
        display(HTML(viewers_three._card_iframe(viewers_three._structure_data(item, atoms), height=220)))
    display(Markdown(f"**Energy:** {as_energy(my_energy):.4f} eV"))


if __name__ == "__main__":
    import sys

    from .agent01_check import run_check

    if len(sys.argv) == 3 and sys.argv[1] == "check":
        raise SystemExit(run_check(sys.argv[2]))
    print("usage: python -m helpers.agent01 check <step>")
    raise SystemExit(2)
