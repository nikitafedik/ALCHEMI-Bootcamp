# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L-BFGS relaxation to a force threshold for the Module 00 quick tour.

Needs Toolkit 0.3.0 (``LBFGS``). Every system relaxes until its largest force
is at or below ``fmax``; a ``FusedStage`` retires systems one by one. Atoms a
structure marks in its boolean array ``frozen`` (the two bottom layers of the
catalog slabs) stay at their built positions: ``FreezeAtomsHook`` holds them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from time import perf_counter

import numpy as np
import torch
from ase import Atoms
from nvalchemi.data import AtomicData, Batch

from .catalog import mace_checkpoint_path
from .model_support import suppress_known_mace_load_noise
from .relaxation import (
    measure_torch_execution,
    prepare_dynamics_batch,
)
from .viewers import MAX_RELAXATION_SYSTEMS

# The player shows every frame up to this step, then evenly spaced frames.
PLAYER_DENSE_FRAMES = 60
PLAYER_MAX_FRAMES = 100


def thin_frame_indices(
    count: int,
    *,
    dense: int = PLAYER_DENSE_FRAMES,
    maximum: int = PLAYER_MAX_FRAMES,
) -> list[int]:
    """Pick frame indices: all of the first ``dense``, then evenly spaced.

    The first and the last frame are always kept.
    """

    if count <= dense:
        return list(range(count))
    tail = np.linspace(dense, count - 1, num=max(maximum - dense, 1))
    return list(range(dense)) + sorted({int(round(value)) for value in tail})


class _CountingSink:
    """Pass writes to a sink and count them on the host, without a device read."""

    def __init__(self, sink: Any, *, copy: bool = False) -> None:
        self.sink = sink
        self.writes = 0
        self.copy = copy

    def write(self, batch: Any, mask: Any = None) -> None:
        # ``HostMemory`` keeps views of a batch that is already on the CPU, so
        # every stored frame would follow the live batch. A copy keeps each
        # frame as it was when it was written.
        self.sink.write(batch.clone() if self.copy else batch, mask)
        self.writes += 1

    def read(self) -> Any:
        return self.sink.read()

    def __len__(self) -> int:
        return len(self.sink)


class _StatusRecorder:
    """Record the per-graph status after every step and the steps that were saved.

    It runs after the snapshot hook. A new write to the sink means the snapshot
    hook saved this step, so the real step number of every saved frame is known
    whatever the snapshot frequency. Status rows stay on the device until the
    run ends.
    """

    from nvalchemi.dynamics import DynamicsStage

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, sink: Any) -> None:
        self.sink = sink
        self.rows: list[torch.Tensor] = []
        self.saved_steps: list[int] = []
        self._size = sink.writes

    def __call__(self, ctx: Any, stage: Any) -> None:
        batch = ctx.batch
        self.rows.append(batch.status.view(-1)[: batch.num_graphs].clone())
        if self.sink.writes != self._size:
            self._size = self.sink.writes
            self.saved_steps.append(len(self.rows))


def frozen_mask(structure: Atoms) -> np.ndarray:
    """Return the boolean ``frozen`` array of a structure, all False without one."""

    mask = structure.arrays.get("frozen")
    if mask is None:
        return np.zeros(len(structure), dtype=bool)
    return np.asarray(mask, dtype=bool)


def _with_frozen_atoms(data: Any, structure: Atoms) -> Any:
    """Mark the frozen atoms of one system for ``FreezeAtomsHook``."""

    from nvalchemi._typing import AtomCategory

    mask = frozen_mask(structure)
    if mask.any():
        data.atom_categories[
            torch.as_tensor(mask, device=data.atom_categories.device)
        ] = AtomCategory.SPECIAL.value
    return data


def _with_cell_and_pbc(data: Any, structure: Any) -> Any:
    """Give a non-periodic system a vacuum cell and ``pbc`` False.

    ``Batch.from_data_list`` drops the cell and ``pbc`` of the whole batch when
    its first system is a molecule, so periodic systems in a mixed batch would
    relax as isolated clusters. The cell is a computation detail: the viewers
    draw the original structure, which has no cell.
    """

    if bool(np.asarray(structure.pbc).any()):
        return data
    positions = data.positions
    extent = float((positions.max(dim=0).values - positions.min(dim=0).values).max())
    data.cell = (extent + 12.0) * torch.eye(
        3, device=positions.device, dtype=positions.dtype
    ).unsqueeze(0)
    data.pbc = torch.zeros(1, 3, dtype=torch.bool, device=positions.device)
    return data


def cueq_kernels_installed() -> bool:
    """True when the cuEquivariance Torch package and its CUDA kernels import."""

    from importlib.util import find_spec

    return all(
        find_spec(name) is not None
        for name in ("cuequivariance_torch", "cuequivariance_ops_torch")
    )


def resolve_speed_settings(
    *,
    speed: bool,
    compile_model: bool | None,
    enable_cueq: bool | None,
    device: torch.device | str,
) -> tuple[bool, bool]:
    """Return ``(compile_model, enable_cueq)`` for one run.

    ``None`` follows ``speed``. cuEquivariance follows ``speed`` only on a CUDA
    device with the kernels installed; an explicit ``True`` is passed through.
    """

    use_compile = speed if compile_model is None else bool(compile_model)
    if enable_cueq is None:
        use_cueq = (
            speed and torch.device(device).type == "cuda" and cueq_kernels_installed()
        )
    else:
        use_cueq = bool(enable_cueq)
    return use_compile, use_cueq


def load_mace_model(
    device: torch.device | str, *, use_cueq: bool = False, use_compile: bool = False
) -> Any:
    """Load MACE-MPA-0 medium (float32) for the L-BFGS relaxation."""

    with suppress_known_mace_load_noise():
        from nvalchemi.models import MACEWrapper

        return MACEWrapper.from_checkpoint(
            mace_checkpoint_path("medium-mpa-0"),
            device=torch.device(device),
            dtype=torch.float32,
            enable_cueq=use_cueq,
            compile_model=use_compile,
        ).eval()


def relax_selected_structures_lbfgs(
    selection: Any,
    *,
    device: torch.device | str,
    fmax: float = 0.05,
    max_steps: int = 500,
    distortion_std_angstrom: float = 0.10,
    distortion_seed: int = 7,
    batch_multiplier: int,
    history_size: int = 6,
    maxstep: float = 0.2,
    snapshot_frequency: int = 1,
    frame_sink: str = "auto",
    gpu_max_edges: int | None = None,
    warmup: bool = True,
    player_factory: Callable[..., Any] | None = None,
    speed: bool = False,
    compile_model: bool | None = None,
    enable_cueq: bool | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Relax one picker selection with L-BFGS until ``fmax`` (eV/Å) or ``max_steps``.

    Frames for the player are recorded in a ``GPUBuffer`` and copied to the host
    once after the run (``frame_sink="gpu"``; ``"auto"`` picks it on CUDA). The
    host-memory recorder stays available as ``frame_sink="host"`` for comparison.
    ``gpu_max_edges`` sizes the buffer's edge storage per recorded system; ``None``
    uses 1.5 times the largest neighbor count of the starting batch. The buffer
    only accepts a system whose edges fit, so a run that exceeds it raises.

    With ``warmup=True`` three untimed L-BFGS steps on a clone of the same batch
    run before the real run, so ``total_s`` does not include first-use compilation
    or shape-specific setup.

    ``speed=True`` turns on the Toolkit speed settings that keep the forces of
    the current route on this path, both passed to
    ``MACEWrapper.from_checkpoint``: ``compile_model`` (``torch.compile`` of the
    MACE model) and ``enable_cueq`` (cuEquivariance kernels, on a CUDA device
    when ``cuequivariance_torch`` and ``cuequivariance_ops_torch`` are
    installed). ``compile_model`` or ``enable_cueq`` set to ``True`` or
    ``False`` overrides ``speed``; ``enable_cueq=True`` without the kernels
    raises. The default is today's route, with neither. With ``warmup=True``
    the first call in a process pays the compilation time inside ``warmup_s``.
    """

    from nvalchemi.dynamics import (
        LBFGS,
        ConvergenceHook,
        FusedStage,
        GPUBuffer,
        HostMemory,
    )
    from nvalchemi.dynamics.hooks import FreezeAtomsHook, SnapshotHook
    from nvalchemi.neighbors import compute_neighbors

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    call_start = perf_counter()
    specs = selection.selected_specs()
    if len(specs) > MAX_RELAXATION_SYSTEMS:
        raise ValueError(
            f"Choose at most {MAX_RELAXATION_SYSTEMS} systems for the synchronized player"
        )
    if not np.isfinite(distortion_std_angstrom) or distortion_std_angstrom < 0:
        raise ValueError("distortion_std_angstrom must be finite and non-negative")
    if not np.isfinite(fmax) or fmax <= 0:
        raise ValueError("fmax must be positive")
    for name, value in (
        ("max_steps", max_steps),
        ("batch_multiplier", batch_multiplier),
        ("snapshot_frequency", snapshot_frequency),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    run_device = torch.device(device)
    random = np.random.default_rng(distortion_seed)
    structures = []
    for spec in specs:
        structure = spec.build()
        if distortion_std_angstrom > 0 and len(structure) > 1:
            displacement = random.normal(
                scale=distortion_std_angstrom, size=structure.positions.shape
            )
            # Frozen atoms keep their built positions; the free atoms are shaken
            # with no net shift.
            free = ~frozen_mask(structure)
            displacement[free] -= displacement[free].mean(axis=0, keepdims=True)
            displacement[~free] = 0.0
            structure.positions += displacement
        structures.append(structure)

    graphs = [
        _with_frozen_atoms(
            _with_cell_and_pbc(
                AtomicData.from_atoms(structure, device=run_device), structure
            ),
            structure,
        )
        for _ in range(batch_multiplier)
        for structure in structures
    ]
    batch = Batch.from_data_list(graphs, device=run_device)
    any_frozen = any(bool(frozen_mask(structure).any()) for structure in structures)
    frozen_counts = {
        spec.display_name: int(frozen_mask(structure).sum())
        for spec, structure in zip(specs, structures, strict=True)
    }
    labels = [spec.display_name for spec in specs]
    n_labels = len(labels)
    n_graphs = batch.num_graphs

    use_compile, use_cueq = resolve_speed_settings(
        speed=speed,
        compile_model=compile_model,
        enable_cueq=enable_cueq,
        device=run_device,
    )
    load_start = perf_counter()
    if model is None:
        model = load_mace_model(run_device, use_cueq=use_cueq, use_compile=use_compile)
    model.set_config("active_outputs", {"energy", "forces"})
    if run_device.type == "cuda":
        torch.cuda.synchronize(run_device)
    model_load_s = perf_counter() - load_start

    if frame_sink not in {"auto", "gpu", "host"}:
        raise ValueError("frame_sink must be 'auto', 'gpu' or 'host'")
    use_gpu = frame_sink == "gpu" or (frame_sink == "auto" and run_device.type == "cuda")

    def assemble(source: Batch, capacity: int, stop_fmax: float) -> tuple[Any, ...]:
        """Build the batch, recorder and L-BFGS stage for one run."""

        prepared = prepare_dynamics_batch(source)
        compute_neighbors(prepared, config=model.model_config.neighbor_config)
        if use_gpu:
            edges = gpu_max_edges
            if edges is None:
                edges = int(1.5 * int(prepared.num_edges_per_graph.max().item())) + 1
            raw_sink = GPUBuffer(
                capacity=capacity,
                max_atoms=int(source.num_nodes_per_graph.max().item()),
                max_edges=edges,
                device=run_device,
            )
        else:
            raw_sink = HostMemory(capacity=capacity)
        frame_sink_wrapper = _CountingSink(
            raw_sink, copy=not use_gpu and run_device.type == "cpu"
        )
        recorder = _StatusRecorder(frame_sink_wrapper)
        lbfgs = LBFGS(
            model=model,
            history_size=history_size,
            maxstep=maxstep,
            convergence_hook=ConvergenceHook.from_fmax(stop_fmax),
        )
        fused = FusedStage(sub_stages=[(0, lbfgs)])
        # The freeze hook comes first, and only when a system has frozen atoms,
        # so a selection without slabs runs the same route as before.
        freeze = [FreezeAtomsHook()] if any_frozen else []
        for hook in (
            *freeze,
            *model.make_neighbor_hooks(),
            SnapshotHook(frame_sink_wrapper, frequency=snapshot_frequency),
            recorder,
        ):
            fused.register_hook(hook)
        return prepared, lbfgs, fused, frame_sink_wrapper, recorder

    def clear_frozen_forces(prepared: Batch) -> None:
        """Zero the forces on frozen atoms, so frame 0 reports the free atoms."""

        if any_frozen:
            from nvalchemi._typing import AtomCategory

            held = prepared.atom_categories == AtomCategory.SPECIAL.value
            with torch.no_grad():
                prepared.forces[held] = 0.0

    warmup_s = 0.0
    if warmup:
        # Untimed warm-up on a clone of the same batch (same structures, same
        # shape), three steps with a force threshold nothing reaches, same hooks
        # and recorder type, nothing kept. The timed run then starts warm.
        warm_start = perf_counter()
        warm_batch, warm_opt, warm_stage, warm_sink, _ = assemble(
            batch, 5 * n_graphs, 1.0e-12
        )
        warm_opt.compute(warm_batch)
        clear_frozen_forces(warm_batch)
        warm_sink.write(warm_batch)
        warm_stage.run(warm_batch, n_steps=3)
        if run_device.type == "cuda":
            torch.cuda.synchronize(run_device)
        warmup_s = perf_counter() - warm_start

    capacity = (max_steps // snapshot_frequency + 2) * n_graphs
    dynamics_batch, optimizer, stage, sink, status = assemble(batch, capacity, fmax)
    optimizer.compute(dynamics_batch)
    clear_frozen_forces(dynamics_batch)
    allocated_before = torch.cuda.memory_allocated(run_device) if use_gpu else 0
    sink.write(dynamics_batch)
    status._size = sink.writes
    frame_buffer_mib = (
        (torch.cuda.memory_allocated(run_device) - allocated_before) / 1024**2
        if use_gpu
        else 0.0
    )
    with measure_torch_execution(run_device) as measurement:
        stage.run(dynamics_batch, n_steps=max_steps)

    steps_run = len(status.rows)
    frame_steps = [0, *status.saved_steps]
    if frame_steps[-1] != steps_run:
        sink.write(dynamics_batch)
        frame_steps.append(steps_run)

    post_start = perf_counter()
    flat = sink.read()
    if flat.num_graphs != len(frame_steps) * n_graphs:
        raise RuntimeError(
            f"snapshot sink holds {flat.num_graphs} records; expected "
            f"{len(frame_steps)} frames of {n_graphs} graphs"
        )

    # First step at which each graph is converged (0 means it never was).
    done = (torch.stack(status.rows) >= 1).cpu().numpy()
    converged_step = np.where(done.any(axis=0), done.argmax(axis=0) + 1, 0)
    shown = converged_step[:n_labels]

    # Pick only the records the player shows: thinned frames x copy 0 of each
    # system. After a system converges its last frame stays on screen.
    kept = thin_frame_indices(len(frame_steps))
    last_frame = [
        next(
            (k for k, step in enumerate(frame_steps) if step >= shown[system]),
            len(frame_steps) - 1,
        )
        if shown[system]
        else len(frame_steps) - 1
        for system in range(n_labels)
    ]
    picks = [
        min(index, last_frame[system]) * n_graphs + system
        for index in kept
        for system in range(n_labels)
    ]
    selected = flat.index_select(torch.tensor(picks, dtype=torch.long, device=flat.device))
    del flat
    if use_gpu:
        selected = selected.to("cpu")  # the one device-to-host copy, shown records only
    # Slice the few selected records straight from the batch tensors: one
    # Atoms per system and frame, with the forces stored as an array.
    positions = selected.positions.detach().numpy()
    numbers = selected.atomic_numbers.detach().numpy()
    forces = selected.forces.detach()
    ptr = selected.batch_ptr.tolist()
    trajectories: dict[str, list[Atoms]] = {label: [] for label in labels}
    force_history: dict[str, list[float]] = {label: [] for label in labels}
    for graph in range(len(picks)):
        label = labels[graph % n_labels]
        start, stop = ptr[graph], ptr[graph + 1]
        atoms = Atoms(numbers=numbers[start:stop], positions=positions[start:stop])
        atoms.arrays["forces"] = forces[start:stop].numpy()
        trajectories[label].append(atoms)
        force_history[label].append(float(forces[start:stop].norm(dim=1).max()))
    del selected
    step_labels = [frame_steps[index] for index in kept]

    if run_device.type == "cuda":
        device_index = run_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        device_label = f"cuda:{device_index} · {torch.cuda.get_device_name(run_device)}"
    else:
        device_label = "CPU"

    postprocess_s = perf_counter() - post_start
    if run_device.type == "cuda":
        torch.cuda.synchronize(run_device)
    # The whole call except the model load, timed like the stored batching points.
    total_s = perf_counter() - call_start - model_load_s - warmup_s
    per_system_steps = {
        label: [int(value) for value in converged_step[system::n_labels]]
        for system, label in enumerate(labels)
    }
    preview = {
        "trajectories": trajectories,
        "labels": labels,
        "force_history": force_history,
        "device": device_label,
        "elapsed_s": measurement["seconds"],
        "peak_memory_mib": measurement["peak_memory_mib"],
        "postprocess_s": postprocess_s,
        "total_s": total_s,
        "model_load_s": model_load_s,
        "warmup_s": warmup_s,
        "gpu_name": (
            torch.cuda.get_device_name(run_device) if run_device.type == "cuda" else None
        ),
        "convergence": f"fmax ≤ {fmax:g} eV/Å or {max_steps} steps",
        "batch_multiplier": batch_multiplier,
        "reference_structures": dict(zip(labels, structures, strict=True)),
        "step_labels": step_labels,
        "steps_run": steps_run,
        "snapshot_frequency": snapshot_frequency,
        "frame_sink": "gpu" if use_gpu else "host",
        "frame_buffer_mib": frame_buffer_mib,
        "convergence_steps": per_system_steps,
        "unconverged_graphs": int((converged_step == 0).sum()),
        "speed_settings": {"compile_model": use_compile, "enable_cueq": use_cueq},
        "frozen_atoms": frozen_counts,
    }
    if player_factory is None:
        from .viewers_three import show_relaxation_player_three as player_factory
    player_keys = (
        "trajectories",
        "labels",
        "force_history",
        "device",
        "elapsed_s",
        "convergence",
        "batch_multiplier",
        "reference_structures",
        "step_labels",
    )
    player_kwargs = {key: preview[key] for key in player_keys}
    player_kwargs["converged_steps"] = {
        label: values[0] or None for label, values in per_system_steps.items()
    }
    return {
        "batch": batch,
        "batch_multiplier": batch_multiplier,
        "labels": labels,
        "preview": preview,
        "player": player_factory(**player_kwargs),
    }
