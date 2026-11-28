# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 02, Many molecules, one batch."""

from __future__ import annotations

import warnings
from pathlib import Path

from ase import Atoms

from . import agent02_text as T
from .common import configure_tutorial as _configure_tutorial

__all__ = [
    "MOLECULES",
    "batch_rows_inputs",
    "configure_tutorial",
    "load_default_molecule",
    "load_default_set",
    "load_many",
    "load_molecule",
    "make_work_dir",
    "show_batch_rows",
    "show_build_routes",
    "show_fields",
    "show_results",
    "show_session",
    "show_store",
]

WORK_DIR = Path(__file__).resolve().parents[1] / "agent02_work"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "agent02"  # saved, verified default structures

# name -> SMILES; the five molecules of the lesson, 147 atoms in total
MOLECULES = {
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "capsaicin": "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O",
}
_SEED = 7  # fixed embedding seed: the same geometry on every machine


def configure_tutorial(*args, **kwargs):
    """Course display defaults; also quiets two library warnings that the model-call and store cells would print.

    Both are harmless here: Toolkit reads a float64 energy back as float32 (the store table and its last line compare
    at that precision), and AIMNet2's wrapper reads a scalar from a tensor that still tracks gradients.
    """

    result = _configure_tutorial(*args, **kwargs)
    try:  # importing the AIMNet2 wrapper puts a "show every UserWarning once" rule in front of all others; import it first
        import nvalchemi.models.aimnet2  # noqa: F401
    except Exception:  # the filters below still hold for everything that does not import it
        pass
    warnings.filterwarnings("ignore", message=r"AtomicData field .* was cast from", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"Converting a tensor with requires_grad=True to a scalar", category=UserWarning)
    return result


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    WORK_DIR.mkdir(exist_ok=True)
    return WORK_DIR


make_work_dir()


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(T.SESSION)


def load_molecule(name: str) -> Atoms:
    """Return one of the five lesson molecules as an ASE ``Atoms`` object."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    if name not in MOLECULES:
        raise ValueError(f"Unknown molecule {name!r}; expected one of {', '.join(MOLECULES)}")
    mol = Chem.AddHs(Chem.MolFromSmiles(MOLECULES[name]))
    if AllChem.EmbedMolecule(mol, randomSeed=_SEED) != 0:
        raise RuntimeError(f"Could not build a 3D structure for {name}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    atoms = Atoms(
        [atom.GetSymbol() for atom in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
        pbc=False,
    )
    atoms.info["name"] = name
    return atoms


def _read_saved(name: str) -> Atoms:
    from ase.io import read

    atoms = read(DATA_DIR / f"{name}.xyz")
    atoms.info["name"] = name
    return atoms


def load_default_molecule() -> Atoms:
    """A saved molecule of your own for the exercises: phenacetin, 26 atoms (not one of the five lesson molecules)."""

    return _read_saved("phenacetin")


def load_default_set() -> list[Atoms]:
    """A saved set of three small molecules for the exercises: ethanol, pyridine and urea, 28 atoms in all."""

    return [_read_saved(name) for name in ("ethanol", "pyridine", "urea")]


def load_many(n_structures: int = T.PROVISIONAL_N_STRUCTURES) -> list[Atoms]:
    """Return ``n_structures`` ASE structures: the five lesson molecules in turn, each copy slightly rattled.

    Called without a number it returns the size of the timing batch of the build-routes exercise (1,000).
    """

    import numpy as np

    if n_structures < 1:
        raise ValueError("n_structures must be at least 1")
    base = [load_molecule(name) for name in MOLECULES]
    rng = np.random.default_rng(_SEED)
    structures = []
    for i in range(n_structures):
        atoms = base[i % len(base)].copy()
        atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)  # 0.02 A: copies differ
        atoms.info["name"] = base[i % len(base)].info["name"]
        structures.append(atoms)
    return structures


# ---------------------------------------------------------------------------
# The views of the result cells (the words are in agent02_text.py)
# ---------------------------------------------------------------------------
def _tensor_of(value):
    import torch

    return value.detach() if isinstance(value, torch.Tensor) else None


def _first_values(tensor, count: int = 3) -> str:
    flat = tensor.reshape(-1)[:count].cpu().tolist()
    shown = ", ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in flat)
    return shown + (", …" if tensor.numel() > count else "")


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    from html import escape

    head = "".join(f"<th style='text-align:left'>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td style='text-align:left'>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _unit(name: str, dtype) -> str:
    """The unit of a field: from the table of units, and 1 for counts and labels."""

    import torch

    if name in T.UNITS:
        return T.UNITS[name]
    return T.UNIT_DIMENSIONLESS if dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.bool) else "-"


def _atoms_of(data) -> Atoms:
    from .viewers import _data_to_atoms

    return _data_to_atoms(data)


def _show_structure_card(atoms: Atoms) -> None:
    """Show one structure as a rotatable 3D view."""

    from IPython.display import HTML, display

    from . import viewers, viewers_three

    payload = viewers._structure_payload(atoms)
    item = {"display_name": payload["formula"], **payload}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # IPython suggests IFrame for the viewer's srcdoc
        display(HTML(viewers_three._card_iframe(viewers_three._structure_data(item, atoms), height=220)))


def _field_rows(data) -> list[list[str]]:
    """One row of the field table for each field that is set, and for the two the model call fills in."""

    levels = {
        level: set(data.__dict__.get(f"__{level}_keys__", ()))
        for level in ("node", "edge", "system")
    }
    rows = []
    for name in sorted(levels["node"] | levels["edge"] | levels["system"]):
        tensor = _tensor_of(getattr(data, name, None))
        level = next(lv for lv in ("node", "edge", "system") if name in levels[lv])
        if tensor is not None:
            rows.append(
                [name, str(tuple(tensor.shape)), _unit(name, tensor.dtype), T.LEVEL_NAMES[level], _first_values(tensor), str(tensor.device)]
            )
        elif name in ("forces", "energy"):
            shape = (data.num_nodes, 3) if name == "forces" else (1, 1)
            rows.append([name, str(shape), T.UNITS[name], T.LEVEL_NAMES[level], T.NOT_SET, T.NO_DEVICE])
    return rows


def show_fields(data) -> None:
    """Show the fields of one AtomicData: a rotatable molecule and a field table.

    The table lists the fields that are set and, as "not set yet", the two the model call fills in:
    ``forces`` (one row per atom) and ``energy`` (one row for the whole molecule).
    """

    from IPython.display import HTML, display

    _show_structure_card(_atoms_of(data))
    display(HTML(_html_table(T.FIELD_HEADERS, _field_rows(data))))


def _formulas(batch) -> list[str]:
    numbers = batch.atomic_numbers.cpu().numpy()
    counts = batch.num_nodes_per_graph.cpu().tolist()
    out, start = [], 0
    for count in counts:
        out.append(Atoms(numbers=numbers[start : start + count]).get_chemical_formula())
        start += count
    return out


def _lesson_names() -> dict[str, str]:
    """Chemical formula -> lesson molecule name, for the five molecules."""

    return {load_molecule(name).get_chemical_formula(): name for name in MOLECULES}


def batch_rows_inputs(batch) -> dict:
    """Plain lists describing the rows of a batch, for the rows figure and its table."""

    formulas = _formulas(batch)
    names = _lesson_names()
    return {
        "labels": [names.get(f, f) for f in formulas],
        "counts": [int(n) for n in batch.num_nodes_per_graph.cpu().tolist()],
        "batch_idx": [int(i) for i in batch.batch_idx.cpu().tolist()],
        "symbols": [Atoms(numbers=[int(z)]).get_chemical_symbols()[0] for z in batch.atomic_numbers.cpu().tolist()],
    }


def show_batch_rows(batch) -> None:
    """Show how the atom rows of a batch belong to the molecules."""

    import matplotlib.pyplot as plt
    from IPython.display import display

    from .batch_rows_figure import draw_batch_rows

    figure = draw_batch_rows(**batch_rows_inputs(batch))
    display(figure)
    plt.close(figure)


def _column_named(frame, word: str):
    """The first column of a table whose name contains the word; None when there is none."""

    return next((c for c in frame.columns if word in str(c).lower()), None)


def _build_size_line(frame) -> str | None:
    """The line that gives the size of the timing batch, when the table has a structures and an atoms column."""

    import pandas as pd

    structures, atoms, runs = (_column_named(frame, word) for word in ("structures", "atoms", "runs"))
    if structures is None or atoms is None:
        return None
    first = frame.iloc[0]
    text = T.BUILD_SIZE_LINE.format(structures=int(first[structures]), atoms=int(first[atoms]))
    count = first[runs] if runs is not None else None
    if pd.api.types.is_number(count) and count >= 1 and int(count) == count:  # a count of timed builds, not a time
        text += T.BUILD_REPEATS_PART.format(runs=int(count))
    return text + "."


_OTHER_FIELD_WORDS = ("mass", "categor", "veloc", "charge", "cell", "pbc", "momen")  # fields beyond positions and atomic numbers


def _tensor_route_fields_are_named(frame, tensor_row) -> bool:
    """True when the table says that the route from tensors carries positions and atomic numbers only.

    A fields column says so when it lists exactly those two. Without a fields column, the route's name
    says so when it names positions and atomic numbers and no other field.
    """

    fields = _column_named(frame, "field")
    if fields is not None:
        listed = str(tensor_row[fields]).replace(",", " ").split()
        return set(listed) == set(T.PROVISIONAL_TENSOR_FIELDS)
    words = " ".join(str(v).lower() for v in tensor_row.values if isinstance(v, str))
    return "position" in words and "atomic" in words and not any(w in words for w in _OTHER_FIELD_WORDS)


def _build_time(seconds: float) -> str:
    return f"{1e3 * seconds:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


def show_build_routes(build_table) -> None:
    """Show the build_table of the build-routes exercise: the three routes, their times, what each tells you, the order check and the API."""

    import pandas as pd
    from IPython.display import HTML, Markdown, display

    from .agent02_checks import BUILD_ORDER, build_route_order

    verdict = build_route_order(build_table)
    if verdict["status"] == "unnamed":
        display(Markdown(T.CHECK_LINES["table_unnamed"].split(": ", 1)[1]))
        return
    frame = pd.DataFrame(build_table)
    size_line = _build_size_line(frame)
    if size_line is not None:
        display(Markdown(size_line))
    fields_are_named = _tensor_route_fields_are_named(frame, frame.iloc[verdict["rows"]["tensors"]])
    rows = []
    for role in BUILD_ORDER:
        label, meaning = T.BUILD_ROUTE_TEXT[role]
        if role == "tensors" and fields_are_named:
            label += T.BUILD_TENSOR_FIELDS_MARK.format(words=T.PROVISIONAL_TENSOR_FIELDS_WORDS)
        shown = _build_time(verdict["times"][role])
        if role in verdict["marked"]:
            shown += T.BUILD_UNEXPECTED_MARK
        rows.append([label, shown, meaning])
    display(HTML(_html_table(T.BUILD_HEADERS, rows)))
    usual = verdict["status"] == "usual"
    display(Markdown(f"{T.BUILD_ORDER_LINE}: {T.BUILD_ORDER_USUAL if usual else T.BUILD_ORDER_UNEXPECTED}"))
    display(Markdown(T.BUILD_API_BOX))


def _largest_forces(forces, batch_idx, n_structures: int):
    """The largest force magnitude on an atom, for each structure of a batch."""

    import torch

    norms = forces.detach().float().norm(dim=1)  # per-atom force magnitude
    owner = batch_idx.to(norms.device)
    return torch.zeros(n_structures, device=norms.device).scatter_reduce(0, owner, norms, reduce="amax", include_self=False)


def show_results(batch, energy, forces) -> None:
    """Show one row per molecule: atoms, energy in eV and largest force in eV/Å."""

    from IPython.display import HTML, display

    counts = batch.num_nodes_per_graph.cpu().tolist()
    largest = _largest_forces(forces, batch.batch_idx, len(counts))
    rows = [
        [i, formula, counts[i], f"{energy.reshape(-1)[i].item():.3f}", f"{largest[i].item():.3f}"]
        for i, formula in enumerate(_formulas(batch))
    ]
    display(HTML(_html_table(T.RESULTS_HEADERS, rows)))


_STORE_LEVELS = {"atom": "node", "edge": "edge", "system": "system"}  # a level as the store names it -> as AtomicData names it


def show_store(store_path, batch=None, loaded_batch=None) -> None:
    """Show what a Zarr store holds; with both batches, whether what was read back equals what was saved.

    Without batches it lists the structures of the store with their energy and largest force.
    """

    from IPython.display import HTML, Markdown, display
    from nvalchemi.data.datapipes import AtomicDataZarrReader

    reader = AtomicDataZarrReader(str(store_path))
    sizes = reader.level_sizes()
    display(Markdown(T.STORE_COUNT.format(structures=sizes["system"], atoms=sizes["atom"])))
    if batch is None and loaded_batch is None:
        _show_store_structures(store_path)
        return
    rows = [
        [
            name,
            T.LEVEL_NAMES.get(_STORE_LEVELS.get(field.level, ""), field.level),
            str((sizes.get(field.level, "?"), *field.row_shape)),
            _unit(name, field.dtype),
            str(field.dtype).replace("torch.", ""),
        ]
        for name, field in sorted(reader.schema().items())
    ]
    display(HTML(_html_table(T.STORE_HEADERS, rows)))
    if batch is not None and loaded_batch is not None:
        from .agent02_checks import compare_saved_and_read

        same = compare_saved_and_read(batch, loaded_batch)["same"]
        display(Markdown(T.STORE_EQUAL if same else T.STORE_DIFFERS))


def _show_store_structures(store_path) -> None:
    """One row per structure of a store: atoms, energy and largest force."""

    from IPython.display import HTML, display
    from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset

    dataset = Dataset(AtomicDataZarrReader(str(store_path)), device="cpu")
    loaded = dataset.load_batches([list(range(len(dataset)))])[0]
    counts = loaded.num_nodes_per_graph.cpu().tolist()
    largest = _largest_forces(loaded.forces, loaded.batch_idx, len(counts))
    rows = [[i, counts[i], f"{loaded.energy.reshape(-1)[i].item():.3f}", f"{largest[i].item():.3f}"] for i in range(len(counts))]
    display(HTML(_html_table(T.STORE_PER_STRUCTURE_HEADERS, rows)))


if __name__ == "__main__":
    import sys

    from .agent02_checks import run_check

    if len(sys.argv) == 3 and sys.argv[1] == "check":
        raise SystemExit(run_check(sys.argv[2]))
    print("usage: python -m helpers.agent02 check <step>")
    raise SystemExit(2)
