# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent 01 helper, step check and notebook: the contract matches the notebook's table, no GPU needed."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

import IPython.display as ipy_display
from helpers import agent01, agent01_check

NOTEBOOK = _NOTEBOOK_DIR / "alchemi-agent-01-meet-your-agent.ipynb"

WATER = (
    "from ase import Atoms\n"
    "my_molecule = Atoms('OH2', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])\n"
)


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text())["cells"]


def _code_cells() -> list[str]:
    return ["".join(c["source"]) for c in _cells() if c["cell_type"] == "code"]


# ---------------------------------------------------------------------------
# The notebook: ids, table, load cell, setup cell, helper calls
# ---------------------------------------------------------------------------


def test_cell_ids_and_no_saved_outputs():
    cells = _cells()
    assert [c["id"] for c in cells] == [f"a01-{n:02d}" for n in range(1, len(cells) + 1)]
    assert all(not c["outputs"] and c["execution_count"] is None for c in cells if c["cell_type"] == "code")


def test_the_step_table_matches_the_contract():
    found: dict[str, list[str]] = {}
    for cell in _cells():
        text = "".join(cell["source"])
        for row in re.finditer(r"\| output\s*\| (.*?), in `agent01_work/(\w+)\.py` \|", text):
            found[row.group(2)] = re.findall(r"`(\w+)`", row.group(1))
    assert set(found) == set(agent01_check.STEPS)
    for step, names in found.items():
        assert tuple(names) == agent01_check.STEPS[step].names, step


def test_load_cells_name_the_contract_files():
    loads = [c.strip().removeprefix("%load agent01_work/") for c in _code_cells() if c.startswith("%load")]
    assert loads == [spec.file for spec in agent01_check.STEPS.values()]


def test_setup_cell_has_the_four_planned_lines():
    setup = next(c for c in _code_cells() if "make_work_dir" in c)
    lines = [ln.split("#")[0].strip() for ln in setup.splitlines()]
    assert [ln for ln in lines if ln] == [
        "from helpers import agent01 as helpers",
        "helpers.configure_tutorial()",
        "work_dir = helpers.make_work_dir()",
        "helpers.show_session()",
    ]
    assert agent01_check.SETUP_CELL.splitlines() == [ln for ln in lines if ln][:3]


def test_every_helper_call_in_the_notebook_exists():
    calls = set(re.findall(r"helpers\.(\w+)\(", "\n".join(_code_cells())))
    assert calls == {"configure_tutorial", "make_work_dir", "show_session", "show_first_result"}
    assert all(callable(getattr(agent01, name)) for name in calls)


def test_code_cells_start_with_a_comment_or_load():
    assert all(c.startswith(("# ", "%load ")) for c in _code_cells())


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_show_session_text(capsys):
    agent01.show_session()
    assert capsys.readouterr().out == (
        "Give these to your agent when you start the session:\n"
        "  notebook    notebooks/alchemi-agent-01-meet-your-agent.ipynb\n"
        "  helpers     from helpers import agent01 as helpers\n"
        "  skills      nvalchemi-toolkit; prompt-check (skills/prompt-check/SKILL.md)\n"
        "  work in     notebooks/agent01_work/\n"
        "  check with  python -m helpers.agent01 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py\n"
    )


def test_the_prompt_check_skill_is_where_the_session_lines_say():
    skill = _NOTEBOOK_DIR.parent / "skills" / "prompt-check" / "SKILL.md"
    assert skill.read_text().startswith("---\nname: prompt-check\n")


def test_make_work_dir_is_the_folder_the_notebook_loads_from():
    assert agent01.make_work_dir() == _NOTEBOOK_DIR / "agent01_work"
    assert agent01.make_work_dir().is_dir()


def _water_as(kind: str):
    from ase import Atoms

    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    if kind == "atoms":
        return atoms
    from nvalchemi.data import AtomicData

    return AtomicData.from_atoms(atoms)


@pytest.mark.parametrize("kind", ["atoms", "atomic_data"])
def test_as_atoms_takes_ase_and_atomic_data(kind):
    atoms = agent01.as_atoms(_water_as(kind))
    assert atoms.get_chemical_formula() == "H2O"
    assert atoms.get_positions().shape == (3, 3)


def test_as_energy_takes_floats_arrays_and_tensors():
    import numpy as np
    import torch

    for value in (-2075.0677, np.float32(-2075.0677), np.array([[-2075.0677]]), torch.tensor([[-2075.0677]])):
        assert agent01.as_energy(value) == pytest.approx(-2075.0677, abs=1e-3)


@pytest.mark.parametrize("kind", ["atoms", "atomic_data"])
def test_show_first_result_shows_the_view_and_one_energy_line(kind, monkeypatch):
    import torch

    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda *objs, **kw: shown.extend(objs))
    agent01.show_first_result(_water_as(kind), torch.tensor([[-2075.0677]], dtype=torch.float64))
    assert len(shown) == 2
    assert "<iframe" in shown[0].data and "Interactive 3D structure: H2O" in shown[0].data
    assert shown[1].data == "**Energy:** -2075.0677 eV"


def test_default_molecule_is_the_saved_caffeine():
    atoms = agent01.load_default_molecule()
    again = agent01.load_default_molecule()
    assert atoms.get_chemical_formula() == "C8H10N4O2" and len(atoms) == 24
    assert not atoms.get_pbc().any() and atoms.info == {"name": "caffeine", "charge": 0}
    assert (atoms.get_positions() == again.get_positions()).all() and atoms is not again
    distances = atoms.get_all_distances()
    assert distances[distances > 0].min() > 0.9  # no clash: shortest contact is a bond to hydrogen, in Angstrom
    assert (agent01.DATA_DIR / "README.md").is_file()


# ---------------------------------------------------------------------------
# The step check
# ---------------------------------------------------------------------------


def _check(tmp_path: Path, text: str | None, capsys) -> tuple[int, str]:
    (tmp_path / "agent01_work").mkdir(exist_ok=True)
    if text is not None:
        (tmp_path / "agent01_work" / "first_prompt.py").write_text(text)
    code = agent01_check.run_check("first_prompt", tmp_path)
    return code, capsys.readouterr().out


def test_check_accepts_names_paths_and_numbers(tmp_path, capsys):
    assert agent01_check.run_check("nonsense", tmp_path) == 2
    assert "unknown step 'nonsense'; steps: first_prompt" in capsys.readouterr().out
    for spelled in ("1", "first_prompt", "first_prompt.py", "agent01_work/first_prompt.py"):
        assert agent01_check.resolve_step(spelled) == "first_prompt"
    assert agent01_check.resolve_step("2") == "2"


@pytest.mark.parametrize(
    "energy",
    ["-2075.0677", "__import__('numpy').float64(-2075.0677)", "__import__('torch').tensor([[-2075.0677]])"],
)
def test_check_passes_a_molecule_and_one_energy(tmp_path, capsys, energy):
    code, out = _check(tmp_path, WATER + f"my_energy = {energy}\n", capsys)
    assert (code, out) == (0, "first_prompt: OK (2 names)\n")


def test_check_passes_atomic_data(tmp_path, capsys):
    text = WATER + "from nvalchemi.data import AtomicData\nmy_molecule = AtomicData.from_atoms(my_molecule)\nmy_energy = -1.0\n"
    assert _check(tmp_path, text, capsys) == (0, "first_prompt: OK (2 names)\n")


@pytest.mark.parametrize(
    ("text", "line"),
    [
        (None, "first_prompt: running first_prompt.py failed: FileNotFoundError: agent01_work/first_prompt.py not found"),
        (WATER, "first_prompt: my_energy is missing; the step asks for it"),
        ("my_energy = -1.0\n", "first_prompt: my_molecule is missing; the step asks for it"),
        (WATER + "raise ValueError('Pt is not covered')\n", "first_prompt: running first_prompt.py failed: ValueError: Pt is not covered"),
        (
            WATER + "my_energy = -1.0\nassert 'seen' not in globals()\nseen = True\n",
            "first_prompt: running first_prompt.py a second time failed: AssertionError:",
        ),
        ("my_molecule = 'O'\nmy_energy = -1.0\n", "first_prompt: my_molecule is str; it should be one structure (ASE Atoms or AtomicData)"),
        (
            WATER
            + "from nvalchemi.data import AtomicData, Batch\n"
            + "my_molecule = Batch.from_data_list([AtomicData.from_atoms(my_molecule)] * 2)\nmy_energy = -1.0\n",
            "first_prompt: my_molecule is a Batch of 2; it should be one structure (ASE Atoms or AtomicData)",
        ),
        ("from ase import Atoms\nmy_molecule = Atoms()\nmy_energy = -1.0\n", "first_prompt: my_molecule has no atoms"),
        (
            "from ase import Atoms\nmy_molecule = Atoms('H2', positions=[[0, 0, 0], [float('nan'), 0, 0]])\nmy_energy = -1.0\n",
            "first_prompt: my_molecule has positions that are not finite",
        ),
        (WATER + "my_energy = '-2075 eV'\n", "first_prompt: my_energy is str; it should be one number in eV"),
        (
            WATER + "import torch\nmy_energy = torch.zeros(3, 1)\n",
            "first_prompt: my_energy has shape (3, 1); one molecule has one energy in eV",
        ),
        (WATER + "my_energy = float('nan')\n", "first_prompt: my_energy is nan; it should be a finite number in eV"),
    ],
)
def test_check_names_each_problem_in_one_line(tmp_path, capsys, text, line):
    code, out = _check(tmp_path, text, capsys)
    assert code == 1
    assert out.splitlines() == [line]


# ---------------------------------------------------------------------------
# The step check: a model that suits the molecule
# ---------------------------------------------------------------------------

# Stand-ins built from the real wrapper classes with the spec that
# ``from_checkpoint`` stores, so the tests need no weights.
STAND_IN = (
    "import torch\n"
    "from nvalchemi.models.mace import MACEWrapper\n"
    "from nvalchemi.models.aimnet2 import AIMNet2Wrapper\n"
    "from nvalchemi.specs import create_model_spec\n"
    "def mace(name):\n"
    "    m = MACEWrapper.__new__(MACEWrapper)\n"
    "    torch.nn.Module.__init__(m)\n"
    "    m._checkpoint_spec = None if name is None else create_model_spec(MACEWrapper.from_checkpoint, checkpoint_path=name)\n"
    "    return m\n"
    "def aimnet2():\n"
    "    m = AIMNet2Wrapper.__new__(AIMNet2Wrapper)\n"
    "    torch.nn.Module.__init__(m)\n"
    "    return m\n"
)
MPA_LINE = (
    "first_prompt: the model is MACE-MPA-0 ({}), a model for materials; "
    "a molecule needs a model for molecules, such as AIMNet2"
)


@pytest.mark.parametrize(
    ("models", "name"),
    [
        ("model = mace('medium-mpa-0')\n", "medium-mpa-0"),
        ("model = mace('file:///cache/mace-mpa-0-medium.model')\n", "mace-mpa-0-medium.model"),
        ("models = [mace('medium-mpa-0')]\n", "medium-mpa-0"),
        (
            "class Holder(torch.nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.inner = torch.nn.ModuleList([mace('medium-mpa-0')])\n"
            "model = Holder()\n",
            "medium-mpa-0",
        ),  # inside another module, as in a pipeline
    ],
)
def test_check_fails_mace_mpa_0_on_a_molecule(tmp_path, capsys, models, name):
    code, out = _check(tmp_path, WATER + STAND_IN + models + "my_energy = -1.0\n", capsys)
    assert code == 1
    assert out.splitlines() == [MPA_LINE.format(name)]


@pytest.mark.parametrize(
    "models",
    [
        "model = aimnet2()\n",  # plain AIMNet2, as in the notebook's example prompt
        "model = mace(None)\n",  # a MACE model the check cannot identify
        "model = mace('medium-mpa-0')\nother = aimnet2()\n",  # which one gave the energy is unknown
        "def run():\n    return mace('medium-mpa-0')\nrun()\n",  # no model left in the namespace
    ],
)
def test_check_passes_models_it_cannot_fault(tmp_path, capsys, models):
    code, out = _check(tmp_path, WATER + STAND_IN + models + "my_energy = -1.0\n", capsys)
    assert (code, out) == (0, "first_prompt: OK (2 names)\n")


def test_check_passes_mace_mpa_0_on_a_periodic_structure(tmp_path, capsys):
    text = WATER + "my_molecule.set_cell([10, 10, 10])\nmy_molecule.pbc = True\n" + STAND_IN
    code, out = _check(tmp_path, text + "model = mace('medium-mpa-0')\nmy_energy = -1.0\n", capsys)
    assert (code, out) == (0, "first_prompt: OK (2 names)\n")


def _mpa_0_weights() -> bool:
    """True when MACE-MPA-0 loads without a download: bundled or in the MACE cache."""

    try:
        from mace.calculators.foundations_models import local_model_path, mace_mp_urls
        from mace.tools.utils import get_cache_dir
    except Exception:
        return False
    cached = "".join(c for c in Path(mace_mp_urls["medium-mpa-0"]).name if c.isalnum() or c == "_")
    return Path(local_model_path).is_file() or (Path(get_cache_dir()) / cached).is_file()


@pytest.mark.skipif(not _mpa_0_weights(), reason="MACE-MPA-0 weights are not on this machine")
def test_check_fails_real_mace_mpa_0_on_water(tmp_path, capsys):
    text = (
        WATER
        + "import torch\n"
        + "from nvalchemi.data import AtomicData, Batch\n"
        + "from nvalchemi.models.mace import MACEWrapper\n"
        + "from nvalchemi.neighbors import compute_neighbors\n"
        + "model = MACEWrapper.from_checkpoint('medium-mpa-0', device=torch.device('cpu'))\n"
        + "batch = Batch.from_data_list([AtomicData.from_atoms(my_molecule)])\n"
        + "compute_neighbors(batch, config=model.model_config.neighbor_config)\n"
        + "my_energy = model(batch)['energy'].item()\n"
    )
    code, out = _check(tmp_path, text, capsys)
    assert code == 1
    assert out.splitlines()[-1] == MPA_LINE.format("medium-mpa-0")


# ---------------------------------------------------------------------------
# The messages, and the sample files with the real model
# ---------------------------------------------------------------------------


def test_every_line_the_check_prints_is_in_one_table():
    source = (_NOTEBOOK_DIR / "helpers" / "agent01_check.py").read_text()
    body = source.split("@dataclass", 1)[1].replace('problems.append(f"{step}: {text}")', "")  # the step prefix
    assert len(agent01_check.MESSAGES) == 18
    assert all(f'MESSAGES["{key}"]' in body for key in agent01_check.MESSAGES)
    assert not re.search(r"(print|add|append)\(f?\"", body)  # no line is worded outside the table


DEFAULT = "from helpers import agent01 as helpers\nmy_molecule = helpers.load_default_molecule()\n"


def test_check_names_a_file_that_uses_dunder_file(tmp_path, capsys):
    text = "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))\n" + WATER + "my_energy = -1.0\n"
    code, out = _check(tmp_path, text, capsys)
    assert (code, out.splitlines()) == (
        1,
        ["first_prompt: first_prompt.py uses __file__, which does not exist in a notebook cell; remove that line"],
    )


def test_every_step_name_is_its_file_name_without_py():
    assert all(spec.file == f"{step}.py" for step, spec in agent01_check.STEPS.items())


@pytest.mark.parametrize(
    ("energy", "note"),
    [
        ("-18517.4883", None),
        ("-18517.0", None),  # within 1 eV: no note
        ("-680.5003", "-680.5003"),  # the same energy in Hartree
        ("-18517.4883 * 23.0605", f"{-18517.4883 * 23.0605:.4f}"),  # in kcal/mol
    ],
)
@pytest.mark.parametrize("kind", ["atoms", "atomic_data_shifted"])
def test_check_notes_an_unexpected_energy_for_the_default_molecule(tmp_path, capsys, energy, note, kind):
    text = DEFAULT
    if kind == "atomic_data_shifted":
        text += "from nvalchemi.data import AtomicData\nmy_molecule.translate([5.0, 0, 0])\nmy_molecule = AtomicData.from_atoms(my_molecule)\n"
    code, out = _check(tmp_path, text + f"my_energy = {energy}\n", capsys)
    expected = ["first_prompt: OK (2 names)"]
    if note:
        expected.insert(
            0,
            f"first_prompt: unexpected: my_energy is {note}; AIMNet2 as Toolkit ships it gives -18517.4883 eV "
            "for the default molecule (the step still passes)",
        )
    assert (code, out.splitlines()) == (0, expected)


def test_check_adds_no_note_for_another_molecule_or_geometry(tmp_path, capsys):
    moved = DEFAULT + "my_molecule.positions[0] += 0.1\nmy_energy = -680.5003\n"
    assert _check(tmp_path, moved, capsys) == (0, "first_prompt: OK (2 names)\n")
    assert _check(tmp_path, WATER + "my_energy = -680.5003\n", capsys) == (0, "first_prompt: OK (2 names)\n")


def test_check_fails_mace_mpa_0_on_the_default_molecule(tmp_path, capsys):
    code, out = _check(tmp_path, DEFAULT + STAND_IN + "model = mace('medium-mpa-0')\nmy_energy = -1.0\n", capsys)
    assert (code, out.splitlines()) == (1, [MPA_LINE.format("medium-mpa-0")])


def _aimnet2_weights() -> bool:
    """True when the AIMNet2 checkpoint loads without a download."""

    try:
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper

        AIMNet2Wrapper.from_checkpoint("aimnet2", device="cpu")
    except Exception:
        return False
    return True


SOLUTIONS = _NOTEBOOK_DIR.parent / "solutions" / "agent-01"


@pytest.mark.parametrize("sample", ["first_prompt.py", "first_prompt_default.py"])
def test_sample_files_pass_with_the_real_model(tmp_path, capsys, sample):
    if not _aimnet2_weights():
        pytest.skip("AIMNet2 weights are not on this machine")
    code, out = _check(tmp_path, (SOLUTIONS / sample).read_text(), capsys)
    assert code == 0 and out.splitlines()[-1] == "first_prompt: OK (2 names)"
    assert "AIMNet2 energy: -18517.4883 eV" in out  # both samples are caffeine with the same geometry recipe
