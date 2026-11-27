# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface of Agent 09, Small-batch distilled: liquid SiC from fine-tuning to a fast student.

Course plumbing only: the DFT frames of liquid SiC in two cuts with the
pretrained model, the liquid cells the MD starts from, the untrained student,
the result views, and the step check (``python -m helpers.agent09 check
<step>``, run from ``notebooks/``). The learner's agent writes the Toolkit
workflow itself: fine-tuning, sampling and labelling, distilling, running the
student and measuring its structure.

The data are the liquid cut of the public SiC-TGAP set (``data/agent09/sic-tgap/``,
CC BY 4.0). Every word shown to the learner is in ``agent09_views``; the step
contracts are in ``agent09_check``. The rules that call a run "still liquid"
and a student's structure "follows the teacher" are here, in ``still_liquid``
and ``follows_teacher``; the views and the check both use them.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

from . import agent09_views as _views
from .common import configure_tutorial as _configure_tutorial

__all__ = [
    "configure_tutorial",
    "follows_teacher",
    "load_starts",
    "make_student",
    "make_work_dir",
    "prepare_finetuning",
    "read_store",
    "show_errors",
    "show_loss",
    "show_session",
    "show_speed",
    "show_stability",
    "show_store",
    "show_structure",
    "show_student_vs_teacher",
    "still_liquid",
]

NOTEBOOK = "notebooks/alchemi-agent-09-small-batch-distilled.ipynb"
_NOTEBOOKS = Path(__file__).resolve().parents[1]
_DEFAULT_WORK_DIR = _NOTEBOOKS / "agent09_work"
_CUT = _NOTEBOOKS / "data" / "agent09" / "sic-tgap" / "cut"
_TRAIN_FILE, _HELD_OUT_FILE = "liquid-train.extxyz.gz", "liquid-heldout.extxyz.gz"
_CHECKPOINT = "medium-mpa-0"

# Tests and unattended runs set these variables.
_WORK_DIR_VARIABLE = "ALCHEMI_AGENT09_WORK_DIR"  # keeps their Zarr stores out of the repository
# The small CPU setting: "training frames,held-out frames" taken evenly from each frame type (for example "20,12").
_SUBSET_VARIABLE = "ALCHEMI_AGENT09_SUBSET"

# The notebook's setup cell, as the step check runs it.
SETUP_CELL = """\
# Load the course helpers, the liquid SiC frames and the pretrained model
import torch
from helpers import agent09 as helpers
helpers.configure_tutorial()             # course display defaults
work_dir = helpers.make_work_dir()       # notebooks/agent09_work
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path, test_path, base_model = helpers.prepare_finetuning(device)   # DFT frames in two cuts; MACE-MPA-0
ft_steps, n_replicas, md_steps, train_steps, test_steps = 120, 16, 400, 1500, 800   # optimizer steps; copies, MD steps, training steps, MD steps of the student's run
starts, test_starts = helpers.load_starts(n_replicas)   # liquid cells for the teacher's MD and for the student's
helpers.show_session()                   # prints what your agent needs to start"""

BUDGET_NAMES = ("ft_steps", "n_replicas", "md_steps", "train_steps", "test_steps")

# The liquid: one cubic cell of Si108C108 per density (g/cm3), and the temperature of its DFT run (K).
TEMPERATURE_BY_DENSITY = {2.3: 3000.0, 2.5: 3000.0, 2.7: 3000.0, 2.9: 3400.0, 3.05: 3700.0, 3.2: 4000.0}
_AMU_G = 1.66053906660e-24  # g
_KB_EV = 8.617333262e-5  # eV/K

# "Still liquid", per copy.
LIQUID_TEMPERATURE_SHARE = 0.15  # mean temperature after the first 100 fs within 15% of the target
LIQUID_SETTLE_FS = 100.0
LIQUID_SHORTEST = 1.0  # Å, shortest distance in every frame (DFT liquid frames: 1.07 Å)
LIQUID_MSD = 2.0  # Å², mean squared displacement at the end at least this
LIQUID_MSD_GROWTH = 1.25  # and at least this times its value at half time
LIQUID_LEAST_STEPS = 200  # a shorter run is not judged
# "Follows the teacher": every pair's first peak within one 0.1 Å bin of the teacher's; the divergence of the
# curves over all pairs at most ten times the teacher-against-teacher floor (4.3e-6, measured at the course budgets).
STRUCTURE_FLOOR = 4.3e-6
STRUCTURE_BOUND = 10 * STRUCTURE_FLOOR
STRUCTURE_PEAK_TOLERANCE = 0.1  # Å
STRUCTURE_LEAST_FRAMES = 160  # fewer frames on either side are not judged
PAIRS = {"SiSi": (14, 14), "SiC": (14, 6), "CC": (6, 6)}

STUDENT_CUTOFF = 4.5  # Å
_STUDENT_HIDDEN = "16x0e+16x1o"

# The stopwatch of show_speed: the same short MD with each model (course GPU; on a CPU a short stand-in).
TIMING_STEPS, TIMING_REPEATS = 50, 3
TIMING_STEPS_CPU, TIMING_REPEATS_CPU = 3, 1

# The torch notice configure_tutorial filters: "`torch.jit.script` is deprecated" (also script_method, load).
JIT_DEPRECATION_FILTER = {
    "message": r"`torch\.jit\.(script|script_method|load)` is deprecated",
    "category": DeprecationWarning,
    "module": r"torch\.jit\.",
}


# ---------------------------------------------------------------------------
# Session and work folder
# ---------------------------------------------------------------------------


def work_dir_path() -> Path:
    """The folder where the agent writes its files: ``notebooks/agent09_work``."""

    return Path(os.environ.get(_WORK_DIR_VARIABLE) or _DEFAULT_WORK_DIR)


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    path = work_dir_path()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _quiet_warp_start() -> None:
    """Start Warp with its one-time "no CUDA device" line kept out of the output (CPU only)."""

    import torch

    if torch.cuda.is_available() or "warp" in sys.modules:
        return
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
            import warp

            warp.config.quiet = True
            warp.init()
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def configure_tutorial() -> None:
    """Apply the course plot and display style (light only), and run Toolkit eagerly.

    Eager mode is what the course budgets were measured with on the GPU; on a
    CPU, compiling costs more than the short runs take.
    """

    import torch

    _configure_tutorial()
    _quiet_warp_start()  # before any import that starts Warp
    # Importing mace puts a catch-all ("once", Warning) filter in front of every other one; import it here, inside a
    # muted block whose exit restores the filters, so the notices below stay filtered.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.showwarning = lambda *args, **kwargs: None  # the import's own filter would show its notices
        try:
            torch.serialization.add_safe_globals([slice])  # e3nn's constants load under torch's safe loader
            import nvalchemi.models.mace  # noqa: F401
            import mace.modules  # noqa: F401
        except Exception:  # without mace the later steps say so themselves
            pass
    # MACE and e3nn load TorchScript; newer torch marks that deprecated, a notice not meant for the learner.
    # Narrow: only this message, only from torch.jit (_script.py and _serialization.py).
    warnings.filterwarnings("ignore", **JIT_DEPRECATION_FILTER)
    torch.compiler.set_stance("force_eager")


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(_views.SESSION_TEXT.format(notebook=NOTEBOOK))


# ---------------------------------------------------------------------------
# The data: DFT frames in two cuts, the liquid starts
# ---------------------------------------------------------------------------


def _subset() -> tuple[int, int] | None:
    value = os.environ.get(_SUBSET_VARIABLE)
    if not value:
        return None
    train, held = (int(v) for v in value.split(","))
    return train, held


def _even(pool: list, count: int) -> list:
    """``count`` items spread evenly over ``pool``, first and last included."""

    import numpy as np

    if count >= len(pool):
        return list(pool)
    return [pool[i] for i in np.unique(np.linspace(0, len(pool) - 1, count).round().astype(int))]


def _cut(file: str, count: int | None = None):
    """The frames of one cut as ASE ``Atoms``; with ``count``, that many spread evenly over each frame type."""

    from ase.io import read

    frames = read(_CUT / file, index=":", format="extxyz")
    if count is None:
        return frames
    kinds = list(dict.fromkeys(a.info["config_type"] for a in frames))
    share = [count // len(kinds) + (1 if k < count % len(kinds) else 0) for k in range(len(kinds))]
    return [a for kind, n in zip(kinds, share) for a in _even([b for b in frames if b.info["config_type"] == kind], n)]


def frame_kind(atoms) -> str:
    """The row of the error table a frame belongs to: melting run, equilibrated liquid, quench and amorphous."""

    return _views.FRAME_TYPES.get(atoms.info.get("config_type"), str(atoms.info.get("config_type")))


def _labelled(frames):
    """``AtomicData`` with the DFT ``energy`` (eV) and ``forces`` (eV/Å), float32."""

    import torch
    from nvalchemi.data import AtomicData

    return [
        AtomicData.from_atoms(a, energy_key="dft_energy", forces_key="dft_forces", dtype=torch.float32)
        for a in frames
    ]


def _write_store(path: Path, frames) -> Path:
    """Write the frames to a Zarr store once; a store that already holds them is kept."""

    from nvalchemi.data.datapipes import AtomicDataZarrReader, AtomicDataZarrWriter

    if path.exists():
        try:
            if len(AtomicDataZarrReader(str(path))) == len(frames):
                return path
        except Exception:
            pass
        index = 1
        while path.with_name(f"{path.stem}_{index:03d}.zarr").exists():
            index += 1
        path = path.with_name(f"{path.stem}_{index:03d}.zarr")
    AtomicDataZarrWriter(str(path)).write(frames)
    return path


def _kinds_file(path) -> Path:
    return Path(str(path) + ".kinds.json")


def _load_pretrained(device, atomic_energies=None):
    """MACE-MPA-0 through Toolkit's wrapper: float32, weights frozen, ``energy`` and ``forces`` active."""

    import torch
    from nvalchemi.models.mace import MACEWrapper

    from .catalog import mace_checkpoint_path

    # mace-torch prints which default model it picked; that line is not for the learner
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        model = MACEWrapper.from_checkpoint(
            mace_checkpoint_path(_CHECKPOINT),
            device=device,
            dtype=torch.float32,
            compile_model=False,
            atomic_energies=atomic_energies,
        )
    model.set_config("active_outputs", {"energy", "forces"})
    return model.eval()


def checkpoint_reference_energies(model) -> dict[int, float]:
    """The per-element reference energies (eV) stored in a MACE model."""

    import torch

    numbers = torch.as_tensor(model.model.atomic_numbers).detach().cpu().reshape(-1).tolist()
    energies = model.model.atomic_energies_fn.atomic_energies.detach().cpu().reshape(-1).tolist()
    return {int(z): float(e) for z, e in zip(numbers, energies)}


def _refit_reference_energies(frames, device, cache: Path) -> dict[int, float]:
    """Move the model's energy reference to the set's: one correction per element, eV.

    The DFT energies sit on another reference than the pretrained model's.
    Toolkit's ``fit_atomic_reference_energies`` fits what is left of the
    training energies after the pretrained model, per element; the corrections
    are added to the model's own values. With one composition (Si108C108) the
    fit has rank 1: the total shift is exact, its split over the two elements
    is the minimum-norm one. The fit is saved in the work folder, so the setup
    cell pays for it once.
    """

    import torch
    from nvalchemi.data import Batch
    from nvalchemi.training.reference_energies import fit_atomic_reference_energies

    from .model_support import prepare_model_batch

    signature = {"dataset": f"SiC-TGAP liquid cut, {_TRAIN_FILE}", "frames": len(frames), "checkpoint": _CHECKPOINT}
    if cache.exists():
        try:
            saved = json.loads(cache.read_text())
            if saved["signature"] == signature:
                return {int(z): float(e) for z, e in saved["reference_energies"].items()}
        except Exception:
            pass
    published = _load_pretrained(device)
    published.set_config("active_outputs", {"energy"})

    def baseline(batch):
        prepare_model_batch(batch, published)
        return published(batch)["energy"]

    batches = [Batch.from_data_list(frames[i : i + 4]) for i in range(0, len(frames), 4)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Reference-energy least-squares matrix is rank deficient.*")
        fit = fit_atomic_reference_energies(batches, baseline_energy_fn=baseline, device=device)
    own = checkpoint_reference_energies(published)
    refit = {z: own[z] + correction for z, correction in fit.reference_energies.items()}
    record = {
        "signature": signature,
        "rank": fit.rank,
        "elements": len(fit.reference_energies),
        "corrections": {str(z): c for z, c in fit.reference_energies.items()},
        "reference_energies": {str(z): e for z, e in refit.items()},
    }
    cache.write_text(json.dumps(record, indent=2))
    del published
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return refit


def prepare_finetuning(device):
    """Return ``(train_path, test_path, base_model)``.

    ``train_path`` and ``test_path`` are Zarr stores in the work folder: DFT
    frames of liquid SiC (216 atoms, periodic) with ``energy`` (eV) and
    ``forces`` (eV/Å). ``train_path`` holds 300 frames of five densities,
    ``test_path`` the 60 frames of the density the training never saw, 2.7
    g/cm3. ``base_model`` is MACE-MPA-0 on ``device`` through Toolkit's
    ``MACEWrapper``: float32, weights frozen, ``energy`` and ``forces`` active,
    with its energy reference moved to the set's. Its neighbor hooks come from
    ``base_model.make_neighbor_hooks()``.
    """

    work = make_work_dir()
    subset = _subset()
    train_frames = _cut(_TRAIN_FILE, subset[0] if subset else None)
    test_frames = _cut(_HELD_OUT_FILE, subset[1] if subset else None)
    tag = f"_{len(train_frames)}_{len(test_frames)}" if subset else ""
    train = _labelled(train_frames)
    train_path = _write_store(work / f"reference_train{tag}.zarr", train)
    test_path = _write_store(work / f"reference_test{tag}.zarr", _labelled(test_frames))
    _kinds_file(test_path).write_text(json.dumps([frame_kind(a) for a in test_frames]))
    energies = _refit_reference_energies(train, device, work / f"reference_energies{tag}.json")
    return train_path, test_path, _load_pretrained(device, atomic_energies=energies)


def density_of(cell, atoms: int = 216) -> float:
    """Density of a Si(atoms/2)C(atoms/2) cell in g/cm3, from its cell matrix in Å."""

    import numpy as np

    volume = abs(float(np.linalg.det(np.asarray(cell, dtype=float).reshape(3, 3))))
    mass = atoms / 2 * (28.0855 + 12.011) * _AMU_G
    return mass / (volume * 1e-24)


def target_temperature(cell, atoms: int = 216) -> tuple[float, float]:
    """The course density nearest to a cell's (g/cm3) and the temperature of its DFT run (K)."""

    rho = density_of(cell, atoms)
    nearest = min(TEMPERATURE_BY_DENSITY, key=lambda d: abs(d - rho))
    return nearest, TEMPERATURE_BY_DENSITY[nearest]


def load_starts(n_replicas: int):
    """Return ``(starts, test_starts)``: two lists of ``n_replicas`` liquid cells, ASE ``Atoms``.

    Each is an equilibrated DFT frame of liquid SiC (216 atoms, periodic), from
    the five training densities, spread evenly over them. Each carries
    ``atoms.info["temperature_K"]``, the temperature of its DFT run (3000 to
    4000 K), and ``atoms.info["density"]`` (g/cm3). ``starts`` are for the
    teacher's MD, ``test_starts`` for the student's: the same densities, other
    frames of the same runs. Neither list holds a frame of the held-out density.
    """

    by_density: dict[float, list] = {}
    for atoms in _cut(_TRAIN_FILE):
        if atoms.info["config_type"] != "eqlMelt_aimd":
            continue
        sub = atoms.info["sub_config"]  # "rho2.3_Tm3000"
        rho, temperature = float(sub.split("_Tm")[0].replace("rho", "")), float(sub.split("_Tm")[1])
        clean = atoms.copy()
        clean.calc = None
        for key in list(clean.arrays):
            if key not in ("numbers", "positions"):
                del clean.arrays[key]
        clean.info = {"temperature_K": temperature, "density": rho, "source_index": int(atoms.info["source_index"])}
        clean.pbc = True
        by_density.setdefault(rho, []).append(clean)
    sample = [a for rho in sorted(by_density) for a in by_density[rho][0::2]]
    test = [a for rho in sorted(by_density) for a in by_density[rho][1::2]]
    return _even(sample, n_replicas), _even(test, n_replicas)


# ---------------------------------------------------------------------------
# The student
# ---------------------------------------------------------------------------


def _load_store(path, batch_size: int | None = None):
    """A Zarr store on the CPU: ``(reader, batches)``; without ``batch_size`` every structure is in one batch."""

    import torch
    from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, InMemoryDataset

    reader = AtomicDataZarrReader(str(path))
    with warnings.catch_warnings():
        # float64 labels are read beside float32 positions; the reader says so once per field
        warnings.filterwarnings("ignore", message="AtomicData field .* was cast from", category=UserWarning)
        dataset = InMemoryDataset(reader=reader, device=torch.device("cpu"))
        return reader, list(DataLoader(dataset, batch_size=batch_size or len(reader)))


def read_store(path):
    """Every structure of a Zarr store as one Batch on the CPU, with the reader: ``(reader, batch)``."""

    reader, batches = _load_store(path)
    return reader, batches[0]


def store_batches(path, batch_size: int) -> list:
    """The structures of a Zarr store in batches on the CPU."""

    return _load_store(path, batch_size)[1]


def make_student(dataset_path, device=None):
    """Return an untrained small MACE model, wrapped for Toolkit training.

    The student is sized from the Zarr store at ``dataset_path``: its elements,
    one energy offset per element (eV) and a force scale (eV/Å) are read from
    the store, from ``teacher_energy`` and ``teacher_forces`` when the store
    has them, else from ``energy`` and ``forces``. It is a ``MACEWrapper`` with
    trainable float32 weights and a 4.5 Å cutoff; it returns ``energy`` (eV)
    and ``forces`` (eV/Å). It is valid only for the composition in that store.
    Every call returns the same starting weights.
    """

    import numpy as np
    import torch

    torch.serialization.add_safe_globals([slice])  # e3nn's constants load under torch's safe loader
    from nvalchemi.models.mace import MACEWrapper

    reader, batch = read_store(dataset_path)
    fields = set(reader.field_levels)
    energy_key = "teacher_energy" if "teacher_energy" in fields else "energy"
    forces_key = "teacher_forces" if "teacher_forces" in fields else "forces"
    for key in (energy_key, forces_key):
        if key not in fields:
            raise ValueError(f"make_student: the store at {dataset_path} has no '{key}'; found {sorted(fields)}")
    numbers = batch.atomic_numbers.cpu().numpy()
    owner = batch.batch_idx.cpu().numpy()
    elements = sorted(set(numbers.tolist()))
    counts = np.zeros((batch.num_graphs, len(elements)))
    for column, element in enumerate(elements):
        counts[:, column] = np.bincount(owner[numbers == element], minlength=batch.num_graphs)
    energy = batch[energy_key].double().cpu().numpy().ravel()
    offsets = np.linalg.lstsq(counts, energy, rcond=None)[0]
    scale = float(batch[forces_key].double().pow(2).mean().sqrt())
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    try:
        with warnings.catch_warnings():
            # e3nn compiles its tensor products on construction and reports its own annotations
            warnings.filterwarnings("ignore", message="The TorchScript type system.*", category=UserWarning)
            mace = _build_mace(scale, elements, offsets)
    finally:
        torch.random.set_rng_state(generator_state)
    student = MACEWrapper(mace.float(), train=True)
    return student.to(device) if device is not None else student


def _build_mace(scale: float, elements: list[int], offsets):
    """The student's network: two interaction layers, 16 scalar and 16 vector channels."""

    import torch
    from e3nn import o3
    from mace.modules import RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock, ScaleShiftMACE

    return ScaleShiftMACE(
        atomic_inter_scale=scale,
        atomic_inter_shift=0.0,
        r_max=STUDENT_CUTOFF,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=2,
        num_elements=len(elements),
        hidden_irreps=o3.Irreps(_STUDENT_HIDDEN),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=offsets,
        avg_num_neighbors=12.0,
        atomic_numbers=elements,
        correlation=2,
        gate=torch.nn.functional.silu,
    )


# ---------------------------------------------------------------------------
# Models on frames
# ---------------------------------------------------------------------------


def _model_device(model):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _call_model(model, batch) -> dict:
    """The outputs of one call of ``model`` on a copy of ``batch``, on the model's device."""

    from .model_support import prepare_model_batch

    work = batch.clone().to(_model_device(model))
    prepare_model_batch(work, model)
    return model(work)


def model_forces(model, batch):
    """Forces (eV/Å) from one call of ``model`` on a copy of ``batch``, on the model's device."""

    return _call_model(model, batch)["forces"].detach()


def _energy_and_forces(model, batch):
    """Energies (eV, one per structure) and forces (eV/Å) from one call of ``model``, float64 on the CPU."""

    out = _call_model(model, batch)
    return out["energy"].detach().double().reshape(-1).cpu(), out["forces"].detach().double().cpu()


def loss_points(loss_history):
    """Return ``(steps, losses)`` as two float arrays.

    ``loss_history`` is one loss per training step, or ``(step, loss)`` pairs.
    Tensors, lists and arrays are taken as they come.
    """

    import numpy as np

    def number(value) -> float:
        return float(value.detach().cpu().reshape(-1)[0]) if hasattr(value, "detach") else float(value)

    if hasattr(loss_history, "detach"):
        loss_history = loss_history.detach().cpu().tolist()
    items = list(loss_history)
    if items and isinstance(items[0], (tuple, list)):
        steps = np.asarray([number(item[0]) for item in items], dtype=float)
        losses = np.asarray([number(item[-1]) for item in items], dtype=float)
    else:
        losses = np.asarray([number(item) for item in items], dtype=float)
        steps = np.arange(1, losses.size + 1, dtype=float)
    return steps, losses


def score(model, path, batch_size: int = 8) -> dict:
    """Set ``model`` against the reference labels (``energy``, ``forces``) of the store at ``path``.

    Returns the force components of both (eV/Å), per frame the energy error
    per atom (meV/atom) and the squared force error summed over the frame's
    components, the force error (root mean square over components, eV/Å) and
    the energy error per atom (root mean square over frames, meV/atom).
    """

    import torch

    forces, reference, energy_error, frame_sq, frame_n = [], [], [], [], []
    was_training = model.training
    model.eval()
    try:
        for batch in store_batches(path, batch_size):
            energy, force = _energy_and_forces(model, batch)
            target = batch.forces.detach().double().cpu()
            forces.append(force)
            reference.append(target)
            owner = batch.batch_idx.cpu().long()
            atoms = torch.bincount(owner, minlength=batch.num_graphs).double()
            energy_error.append((energy - batch.energy.detach().double().cpu().reshape(-1)) / atoms * 1000.0)
            frame_sq.append(torch.zeros(batch.num_graphs, dtype=torch.float64).index_add_(
                0, owner, (force - target).pow(2).sum(dim=-1)))
            frame_n.append(atoms * 3)
    finally:
        model.train(was_training)
    forces, reference, energy_error = torch.cat(forces), torch.cat(reference), torch.cat(energy_error)
    return {
        "frames": int(energy_error.numel()),
        "atoms": int(forces.shape[0]),
        "model_forces": forces.reshape(-1).numpy(),
        "reference_forces": reference.reshape(-1).numpy(),
        "frame_energy_error": energy_error.numpy(),
        "frame_force_sq": torch.cat(frame_sq).numpy(),
        "frame_components": torch.cat(frame_n).numpy(),
        "force_error": float((forces - reference).pow(2).mean().sqrt()),
        "energy_error": float(energy_error.pow(2).mean().sqrt()),
    }


def errors_by_kind(test_path, scores: dict) -> dict:
    """Rows of the error table: per frame kind and for all frames, the energy and force error of each model.

    ``scores`` maps a model key to ``score(model, test_path)``. The frame kinds
    come from the file the loader wrote beside the store; without it there is
    one row, all frames.
    """

    import numpy as np

    first = next(iter(scores.values()))
    count = first["frames"]
    try:
        kinds = json.loads(_kinds_file(test_path).read_text())
        if len(kinds) != count:
            kinds = None
    except Exception:
        kinds = None
    groups = {}
    if kinds:
        for kind in dict.fromkeys(kinds):
            groups[kind] = np.array([k == kind for k in kinds])
    groups[_views.ALL_FRAMES] = np.ones(count, dtype=bool)
    rows = {}
    for kind, mask in groups.items():
        row = {"frames": int(mask.sum())}
        for key, s in scores.items():
            row[key] = {
                "energy": float(np.sqrt(np.mean(s["frame_energy_error"][mask] ** 2))),
                "force": float(np.sqrt(s["frame_force_sq"][mask].sum() / s["frame_components"][mask].sum())),
            }
        rows[kind] = row
    return rows


def compare_on_frames(frames_path, teacher_model, batch_size: int = 16) -> dict:
    """Set the forces (and energies) stored in ``frames_path`` against the teacher's on the same frames.

    Returns the number of frames, the per-atom force magnitudes of both (eV/Å),
    the mean length of the difference vector, the mean teacher force, and, when
    the store holds ``energy``, the root mean square energy difference per atom
    (meV/atom).
    """

    import torch

    stored, reference, energy_gap, frames = [], [], [], 0
    for batch in store_batches(frames_path, batch_size):
        frames += int(batch.num_graphs)
        energy, force = _energy_and_forces(teacher_model, batch)
        stored.append(batch.forces.detach().double().cpu())
        reference.append(force)
        if "energy" in batch:
            atoms = torch.bincount(batch.batch_idx.cpu().long(), minlength=batch.num_graphs).double()
            energy_gap.append((batch.energy.detach().double().cpu().reshape(-1) - energy) / atoms * 1000.0)
    stored, reference = torch.cat(stored), torch.cat(reference)
    return {
        "frames": frames,
        "student_force": stored.norm(dim=-1).numpy(),
        "teacher_force": reference.norm(dim=-1).numpy(),
        "mean_error": float((stored - reference).norm(dim=-1).mean()),
        "mean_teacher_force": float(reference.norm(dim=-1).mean()),
        "energy_gap": float(torch.cat(energy_gap).pow(2).mean().sqrt()) if energy_gap else None,
    }


# ---------------------------------------------------------------------------
# Still liquid, follows the teacher: the rules of the views and of the check
# ---------------------------------------------------------------------------


def copies_in_store(batch) -> int:
    """How many copies one saved snapshot holds, for a store written snapshot after snapshot.

    The same copy one snapshot later is the frame nearest to the first one: the
    smallest root mean square displacement (minimum image) among the frames
    whose position in the store divides the store into whole snapshots.
    """

    import torch

    graphs = int(batch.num_graphs)
    ptr = batch.batch_ptr.cpu().long()
    sizes = (ptr[1:] - ptr[:-1]).tolist()
    cells = batch.cell.detach().cpu().double().reshape(-1, 3, 3)
    first = batch.positions[int(ptr[0]) : int(ptr[1])].detach().cpu().double()
    inverse = torch.linalg.inv(cells[0])
    best, best_p = math.inf, graphs
    for p in range(1, min(graphs, 129)):
        if graphs % p or sizes[p] != sizes[0] or not torch.allclose(cells[p], cells[0], atol=1e-4):
            continue
        other = batch.positions[int(ptr[p]) : int(ptr[p + 1])].detach().cpu().double()
        fractional = (other - first) @ inverse
        fractional -= fractional.round()
        rms = float((fractional @ cells[0]).pow(2).sum(-1).mean().sqrt())
        if rms < best:
            best, best_p = rms, p
    return best_p


def _temperatures(batch):
    """Kinetic temperature of every structure in a batch, K."""

    import torch
    from nvalchemi.dynamics.hooks import kinetic_energy_per_graph

    energy = kinetic_energy_per_graph(
        batch.velocities.double(), batch.atomic_masses.double(), batch.batch_idx, batch.num_graphs
    ).reshape(-1).double()
    atoms = torch.bincount(batch.batch_idx.cpu().long(), minlength=batch.num_graphs).double()
    return (2.0 * energy.cpu() / (3.0 * atoms * _KB_EV)).cpu()


def _unwrapped_msd(positions, cells, inverse):
    """Mean squared displacement from the first saved frame, Å², as ``[frame, copy]``.

    The displacement is unwrapped: the minimum image between consecutive saved frames, summed over the run.
    ``positions`` is ``[frame, copy, atom, 3]``; ``cells`` and ``inverse`` are one matrix per copy.
    """

    import torch

    step = positions[1:] - positions[:-1]
    fractional = torch.einsum("fcai,cij->fcaj", step, inverse)
    fractional -= fractional.round()
    step = torch.einsum("fcaj,cjk->fcak", fractional, cells)
    displacement = torch.cat([torch.zeros_like(positions[:1]), step.cumsum(0)])
    return displacement.pow(2).sum(-1).mean(-1)


def _shortest_distances(positions, cells, inverse):
    """Shortest interatomic distance (minimum image) of every copy in every frame, Å, as ``[frame, copy]``."""

    import torch

    frames, copies, atoms = positions.shape[:3]
    shortest = torch.empty(frames, copies, dtype=torch.float64)
    own = torch.eye(atoms, dtype=torch.float64) * 1e9  # keeps an atom's distance to itself out of the minimum
    for f in range(frames):
        d = positions[f][:, :, None, :] - positions[f][:, None, :, :]
        fractional = torch.einsum("caki,cij->cakj", d, inverse)
        fractional -= fractional.round()
        distance = torch.einsum("cakj,cjl->cakl", fractional, cells)
        shortest[f] = (distance.norm(dim=-1) + own).reshape(copies, -1).min(dim=1).values
    return shortest


def _not_liquid_reasons(rows: list[dict], finite: bool, divergence) -> list[str]:
    """Why some copies are not liquid, in words, one entry per rule that failed."""

    reasons = []
    if not finite:
        reasons.append("positions or forces that are not finite")
    if divergence is not None:
        reasons.append(f"StabilityMonitor found a divergence at step {divergence}")
    for key, words in (
        ("ok_temperature", "{n} off the target temperature by more than 15%"),
        ("ok_distance", "{n} with two atoms closer than 1.0 Å"),
        ("ok_diffusion", "{n} whose atoms stopped moving apart"),
    ):
        n = sum(not r[key] for r in rows)
        if n:
            reasons.append(words.format(n=f"{n} cop{'y' if n == 1 else 'ies'}"))
    return reasons


def still_liquid(frames_path, stability=None, steps: int | None = None) -> dict:
    """Judge every copy of a saved hot MD: still liquid, or not.

    A copy is still liquid when its frames are finite, its mean temperature
    after the first 100 fs is within 15% of the temperature of its density,
    its shortest interatomic distance stays at least 1.0 Å, and its mean
    squared displacement from the start (unwrapped) reaches at least 2.0 Å² and
    at least 1.25 times its value at half time. ``stability`` is the
    ``StabilityMetrics`` of Toolkit's ``StabilityMonitor`` on the same run: its
    ``last_step`` gives the run's length, its ``first_divergence_step`` a
    divergence. Without it, ``steps`` gives the run's length. A run shorter than
    200 steps is not judged; a store without velocities is judged without the
    temperature.
    """

    import numpy as np
    import torch

    _, batch = read_store(frames_path)
    copies = copies_in_store(batch)
    frames = batch.num_graphs // copies
    atoms = int(batch.num_nodes // batch.num_graphs)
    positions = batch.positions.detach().double().reshape(frames, copies, atoms, 3)
    cells = batch.cell.detach().cpu().double().reshape(frames, copies, 3, 3)[0]
    inverse = torch.linalg.inv(cells)
    finite = bool(torch.isfinite(batch.positions).all()) and bool(torch.isfinite(batch.forces).all())
    divergence = getattr(stability, "first_divergence_step", None)
    if getattr(stability, "last_step", None) is not None:
        steps = int(stability.last_step)
    steps = int(steps) if steps is not None else frames

    msd = _unwrapped_msd(positions, cells, inverse)
    shortest = _shortest_distances(positions, cells, inverse)
    temperature = _temperatures(batch).reshape(frames, copies) if "velocities" in batch else None
    interval = steps / frames if frames else 1.0  # steps between saved frames; a step is taken as 1 fs
    settled = min(frames - 1, max(1, math.ceil(LIQUID_SETTLE_FS / max(interval, 1e-9)))) if frames > 1 else 0
    half = frames // 2
    rows = []
    for c in range(copies):
        density, target = target_temperature(cells[c].numpy(), atoms)
        mean_t = float(temperature[settled:, c].mean()) if temperature is not None else float("nan")
        row = {
            "density": density, "target_K": target, "mean_T_K": mean_t,
            "shortest_A": float(shortest[:, c].min()), "msd_end_A2": float(msd[-1, c]),
            "msd_half_A2": float(msd[half, c]),
        }
        row["ok_temperature"] = temperature is None or bool(
            np.isfinite(mean_t) and abs(mean_t / target - 1.0) <= LIQUID_TEMPERATURE_SHARE
        )
        row["ok_distance"] = row["shortest_A"] >= LIQUID_SHORTEST
        row["ok_diffusion"] = (
            row["msd_end_A2"] >= LIQUID_MSD and row["msd_end_A2"] >= LIQUID_MSD_GROWTH * row["msd_half_A2"]
        )
        row["liquid"] = bool(
            finite and divergence is None and row["ok_temperature"] and row["ok_distance"] and row["ok_diffusion"]
        )
        rows.append(row)
    return {
        "judged": steps >= LIQUID_LEAST_STEPS and frames >= 4, "copies": copies, "frames_per_copy": frames,
        "steps": steps, "least_steps": LIQUID_LEAST_STEPS, "finite": finite, "first_divergence_step": divergence,
        "n_liquid": sum(r["liquid"] for r in rows), "rows": rows,
        "reasons": _not_liquid_reasons(rows, finite, divergence),
    }


def _bin_centres(rdf):
    return 0.5 * (rdf.edges[1:] + rdf.edges[:-1])


def _first_peak(rdf) -> float:
    return float(_bin_centres(rdf)[int(rdf.g_r.argmax())])


def partial_rdfs(teacher_path, student_path, r_max: float = 6.0, num_bins: int = 60) -> dict:
    """Toolkit's radial distribution of each pair of elements, for both stores: ``{pair: (teacher, student)}``."""

    import torch
    from nvalchemi.training.distillation.evaluation import radial_distribution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    stores = [[b.to(device) for b in store_batches(p, 64)] for p in (teacher_path, student_path)]
    for name, pair in PAIRS.items():
        out[name] = tuple(radial_distribution(frames, r_max=r_max, num_bins=num_bins, pair=pair) for frames in stores)
    return out


def follows_teacher(teacher_rdf, student_rdf, partials: dict | None = None) -> dict:
    """Judge whether the student's structure follows the teacher's.

    It follows when the Jensen-Shannon divergence of the two curves over all
    pairs is at most 4.3e-5, ten times the teacher-against-teacher floor, and,
    with ``partials`` (from ``partial_rdfs``), every pair's first peak is within
    one 0.1 Å bin of the teacher's. Fewer than 160 frames on either side are not
    judged.
    """

    from nvalchemi.training.distillation.evaluation import compare_radial_distributions

    js = float(compare_radial_distributions(teacher_rdf, student_rdf).jensen_shannon)
    rows = [["total", js, _first_peak(teacher_rdf), _first_peak(student_rdf)]]
    reasons = []
    if js > STRUCTURE_BOUND:
        reasons.append(f"divergence over all pairs {js:.1e}, above {STRUCTURE_BOUND:.1e}")
    for name, (teacher, student) in (partials or {}).items():
        pair_js = float(compare_radial_distributions(teacher, student).jensen_shannon)
        t, s = _first_peak(teacher), _first_peak(student)
        rows.append([name, pair_js, t, s])
        if abs(t - s) > STRUCTURE_PEAK_TOLERANCE + 1e-6:
            reasons.append(
                f"first {_views.LABELS['structure']['pairs'][name]} peak at {s:.2f} Å, the teacher's at {t:.2f} Å"
            )
    judged = min(teacher_rdf.num_frames, student_rdf.num_frames) >= STRUCTURE_LEAST_FRAMES
    return {
        "judged": judged, "follows": judged and not reasons, "js": js, "bound": STRUCTURE_BOUND,
        "peak_tolerance": STRUCTURE_PEAK_TOLERANCE, "rows": rows, "reasons": reasons,
        "teacher_frames": teacher_rdf.num_frames, "student_frames": student_rdf.num_frames,
        "least_frames": STRUCTURE_LEAST_FRAMES,
    }


# ---------------------------------------------------------------------------
# Result views
# ---------------------------------------------------------------------------


def show_loss(loss_history) -> None:
    """Plot the training loss against the step.

    ``loss_history`` is a list of losses in the order of training, or of ``(step, loss)`` pairs.
    """

    steps, losses = loss_points(loss_history)
    _views.show(_views.build_loss_figure(steps, losses), _views.LABELS["loss"]["alt"])


def store_fields(path) -> tuple[dict, list[list[str]]]:
    """The sizes of a Zarr store and one row per field: name, what it belongs to, shape, type."""

    from nvalchemi.data.datapipes import AtomicDataZarrReader

    reader = AtomicDataZarrReader(str(path))
    sizes = reader.level_sizes()
    words = _views.LABELS["store"]["levels"]
    rows = [
        [name, words.get(spec.level, spec.level), str((sizes.get(spec.level, "?"), *spec.row_shape)),
         str(spec.dtype).replace("torch.", "")]
        for name, spec in sorted(reader.schema().items())
    ]
    return sizes, rows


def show_store(path) -> None:
    """Show what a Zarr store holds: the structures, the atoms, and one row per field."""

    from IPython.display import HTML, Markdown, display

    labels = _views.LABELS["store"]
    sizes, rows = store_fields(path)
    display(Markdown(labels["count"].format(structures=sizes["system"], atoms=sizes["atom"])))
    display(HTML(_views.html_table(labels["headers"], rows)))


def show_errors(test_path, base_model, finetuned_model=None, student_model=None) -> None:
    """Score the models against DFT on the held-out frames: a table by frame kind, and one line.

    With one or two models a parity plot comes first: one point per force
    component on one atom of a held-out frame (model against DFT, eV/Å). The
    table has one row per frame kind and one for all held-out frames; per model
    the energy error (root mean square over frames, meV/atom) and the force
    error (root mean square over components, eV/Å).
    """

    from IPython.display import HTML, Markdown, display

    labels = _views.LABELS["errors"]
    models = {"published": base_model}
    if finetuned_model is not None:
        models["finetuned"] = finetuned_model
    if student_model is not None:
        models["student"] = student_model
    scores = {key: score(model, test_path) for key, model in models.items()}
    if student_model is None:
        _views.show(
            _views.build_parity_figure(
                [(labels[key], s["reference_forces"], s["model_forces"]) for key, s in scores.items()]
            ),
            labels["alt"],
        )
    rows = errors_by_kind(test_path, scores)
    keys = list(models)
    headers = [labels["frames_kind"], labels["count"]]
    headers += [labels[k] for k in keys] * 2
    groups = [(labels["energy"], len(keys)), (labels["force"], len(keys))]
    table = [
        [kind, row["frames"], *[f"{row[k]['energy']:.1f}" for k in keys], *[f"{row[k]['force']:.3f}" for k in keys]]
        for kind, row in rows.items()
    ]
    display(HTML(_views.html_table(headers, table, groups)))
    line = _views.errors_line(rows, keys)
    if line:
        display(Markdown(line))


def show_stability(student_frames_path, stability=None) -> None:
    """Show whether each copy of the student's run is still liquid: one row per density, and one line.

    Per density: the copies, the target and the mean temperature (K), the
    shortest interatomic distance (Å), the mean squared displacement at the end
    (Å²) and how many copies are still liquid (see ``still_liquid``).
    """

    from IPython.display import HTML, Markdown, display

    import numpy as np

    verdict = still_liquid(student_frames_path, stability)
    labels = _views.LABELS["stability"]
    table = []
    for density in sorted({r["density"] for r in verdict["rows"]}):
        group = [r for r in verdict["rows"] if r["density"] == density]
        liquid = (
            labels["of"].format(done=sum(r["liquid"] for r in group), total=len(group))
            if verdict["judged"] else labels["not_judged"]
        )
        table.append([
            f"{density:g}", len(group), f"{group[0]['target_K']:.0f}",
            f"{np.mean([r['mean_T_K'] for r in group]):.0f}", f"{min(r['shortest_A'] for r in group):.2f}",
            f"{np.mean([r['msd_end_A2'] for r in group]):.1f}", liquid,
        ])
    display(HTML(_views.html_table(labels["headers"], table)))
    display(Markdown(_views.stability_line(verdict)))


def show_structure(teacher_rdf, student_rdf, labeled_path=None, student_frames_path=None) -> None:
    """Draw the two radial distributions and judge whether the student's structure follows the teacher's.

    ``teacher_rdf`` and ``student_rdf`` are Toolkit ``RadialDistribution``
    curves over all pairs, with the same bins. With both store paths the helper
    adds the curve of each pair of elements (Si-Si, Si-C, C-C) from the same
    stores. A table gives each curve's Jensen-Shannon divergence and first peaks;
    one line reads the result (see ``follows_teacher``).
    """

    from IPython.display import HTML, Markdown, display

    labels = _views.LABELS["structure"]
    partials = {}
    if labeled_path is not None and student_frames_path is not None:
        partials = partial_rdfs(labeled_path, student_frames_path, teacher_rdf.r_max, teacher_rdf.counts.numel())
    verdict = follows_teacher(teacher_rdf, student_rdf, partials)
    curves = [(labels["total"], _bin_centres(teacher_rdf).numpy(), teacher_rdf.g_r.numpy(), student_rdf.g_r.numpy())]
    for name, (teacher, student) in partials.items():
        curves.append((labels["pairs"][name], _bin_centres(teacher).numpy(), teacher.g_r.numpy(), student.g_r.numpy()))
    _views.show(_views.build_structure_figure(curves), labels["alt"])
    names = {"total": labels["total"], **labels["pairs"]}
    display(HTML(_views.html_table(
        labels["headers"], [[names[n], f"{js:.1e}", f"{t:.2f}", f"{s:.2f}"] for n, js, t, s in verdict["rows"]]
    )))
    display(Markdown(_views.structure_line(verdict)))


def show_student_vs_teacher(student_frames_path, teacher_model) -> None:
    """Plot the student's force on each atom of its own frames against the teacher's force.

    The teacher computes every frame of ``student_frames_path``; one point is
    one atom in one frame, the length of its force (eV/Å). The table gives the
    frames, the mean teacher force, the student's mean force error (the length
    of the difference vector) and the root mean square difference of the
    energies per atom (meV/atom). One line under the table reads the comparison.
    """

    from IPython.display import HTML, Markdown, display

    result = compare_on_frames(student_frames_path, teacher_model)
    labels = _views.LABELS["student"]
    _views.show(_views.build_force_figure(result["teacher_force"], result["student_force"]), labels["alt"])
    gap = result["energy_gap"]
    display(HTML(_views.html_table(
        [labels["frames"], labels["mean_force"], labels["mean_error"], labels["energy_gap"]],
        [[result["frames"], f"{result['mean_teacher_force']:.2f}", f"{result['mean_error']:.2f}",
          "n/a" if gap is None else f"{gap:.0f}"]],
    )))
    display(Markdown(_views.student_line(result["mean_error"], result["mean_teacher_force"], gap or float("nan"))))


def _timed_run(model, starts, steps: int) -> float:
    """Seconds for one Langevin run of the cells in ``starts`` with ``model``, each at its own temperature."""

    import torch
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NVTLangevin, initialize_velocities

    device = _model_device(model)
    batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in starts]).to(device)
    temperature = torch.tensor(
        [a.info["temperature_K"] for a in starts], dtype=batch.positions.dtype, device=device
    )
    initialize_velocities(
        batch.velocities, batch.atomic_masses, temperature, batch.batch_idx.to(torch.int32), random_seed=0
    )
    engine = NVTLangevin(
        model=model, dt=1.0, temperature=temperature, friction=0.01, random_seed=0, n_steps=steps,
        device_type=device.type,
    )
    for hook in model.make_neighbor_hooks():
        engine.register_hook(hook)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with engine:
        engine.run(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def compare_speed(teacher_model, student_model, starts) -> dict:
    """MD steps per second with each model on the same cells: one warm-up each, then alternating runs, median.

    On a GPU: 50 steps, three timed runs per model; on a CPU a short stand-in of
    3 steps and one run.
    """

    gpu = _model_device(teacher_model).type == "cuda"
    steps = TIMING_STEPS if gpu else TIMING_STEPS_CPU
    repeats = TIMING_REPEATS if gpu else TIMING_REPEATS_CPU
    was_training = student_model.training
    student_model.eval()
    try:
        for model in (teacher_model, student_model):
            _timed_run(model, starts, steps)  # warm-up, not timed
        teacher, student = [], []
        for _ in range(repeats):
            teacher.append(_timed_run(teacher_model, starts, steps))
            student.append(_timed_run(student_model, starts, steps))
    finally:
        student_model.train(was_training)
    return {
        "teacher_steps_per_s": steps / statistics.median(teacher),
        "student_steps_per_s": steps / statistics.median(student),
        "teacher_runs_s": teacher, "student_runs_s": student, "copies": len(starts), "steps": steps,
    }


def show_speed(teacher_model, student_model, starts) -> None:
    """Time the same short MD of the cells in ``starts`` with each model; show MD steps per second and the ratio."""

    from IPython.display import Markdown, display

    result = compare_speed(teacher_model, student_model, starts)
    _views.show(
        _views.build_speed_figure(result["teacher_steps_per_s"], result["student_steps_per_s"]),
        _views.LABELS["speed"]["alt"],
    )
    display(Markdown(_views.speed_line(result)))


if __name__ == "__main__":
    from .agent09_check import main

    raise SystemExit(main(sys.argv))
