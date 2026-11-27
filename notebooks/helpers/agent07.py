# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Agent 07, One out, one in.

A thin layer over ``module03`` (Core 03, "Scale a study"): the study
preparation, the stage settings, the store the refill step
takes as an input, the result views and the route comparison. The step
contracts live in ``agent07_check``.

Every line a view prints to the learner is in ``LINES`` below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

from . import module03 as _m3
from .common import configure_tutorial as _configure_tutorial
from .module03 import show_scheduling_widget, time_route

__all__ = [
    "CudaCompletionSink",
    "compare_routes",
    "configure_tutorial",
    "make_work_dir",
    "prepare_study",
    "show_refill",
    "show_scheduling_widget",
    "show_session",
    "show_stage_changes",
    "show_staged",
    "stage_settings",
    "time_route",
]

WORK_DIR = Path(__file__).resolve().parents[1] / "agent07_work"
NOTEBOOK = "notebooks/alchemi-agent-07-one-out-one-in.ipynb"
INITIAL_TEMPERATURE = 50.0  # K for each independent velocity draw
RANDOM_SEED = 42
BATCH_MULTIPLIER = 26  # trajectories per complex (the notebook's value): 3,874 trajectories
# The capacity (the notebook's values): the batch size at which the GPU is fully busy, measured on
# an RTX PRO 6000; above it a step takes longer in proportion to the atoms.
MAX_BATCH_SIZE = 1308  # trajectories in the active batch
MAX_ATOMS = 20000  # atoms in the active batch
# Complexes below this size are left out of the study: for 4 to 7 atoms the temperature of one
# trajectory swings too widely for the NVT rule. One switch; 0 keeps all 205 complexes.
MIN_ATOMS = 10

# The settings of the two stages. The setup cell of the notebook holds the same line;
# `stage_settings()` returns a copy.
#   temperature, tolerance  NVT ends for a trajectory once its mean temperature over the last
#                           `window` MD steps is within `tolerance` K of `temperature` K
#   nvt_steps               the step cap of NVT: a trajectory leaves NVT here when its rule has not passed
#   nve_steps               the fixed length of NVE: every trajectory runs this many NVE steps
#   dt, friction            MD timestep in fs and Langevin friction per fs, for both stages
STAGE_SETTINGS = dict(
    temperature=300.0, tolerance=60.0, window=100, nvt_steps=600, nve_steps=200, dt=1.0, friction=0.01,
)

# Which pair of stages a trace describes, by the name the notebook gives it, and the word for
# the steps of its first stage.
TRACE_STAGES = {
    "stage_trace": ("NVT", "NVE", "MD steps"),
    "relax_md_trace": ("relaxation", "MD", "steps"),
}

# Every line and column name a view shows the learner.
LINES = {
    "session": (
        "Give these to your agent when you start the session:\n"
        "  notebook    {notebook}\n"
        "  helpers     from helpers import agent07 as helpers\n"
        "  work in     notebooks/agent07_work/\n"
        "  check with  python -m helpers.agent07 check <step>\n"
        "  each file   runs in a notebook cell (%load), after the earlier files\n"
        "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
    ),
    "trace_count": "trajectories",
    "trace_first": "earliest switch, {a} to {b}",
    "trace_last": "latest switch, {a} to {b}",
    "trace_steps": "after {n} {steps}",
    "trace_apart": "first ahead of the last",
    "trace_apart_steps": "by {n} {steps}",
    "trace_same": "Unexpected: every trajectory switched at the same step.",
    "col_route": "Route",
    "col_finished": "Finished",
    "col_admitted": "Admitted inflight",
    "col_seconds": "Elapsed [s]",
    "col_groups": "Groups",
    "col_of": "Of",
    "route_staged": "Run in stages",
    "route_fused": "Fused stages",
    "route_refill": "Fused stages + refill",
    "not_recorded": "not recorded",
    "unexpected_finished": "Unexpected: the routes did not finish the same number of trajectories.",
    "unexpected_admitted": "Unexpected: no trajectory entered after the first batch was full.",
    "unexpected_staged": "Unexpected: {n} of {total} trajectories are in staged_batches.",
}


def make_work_dir() -> Path:
    """Create and return the folder where the agent writes its files."""

    WORK_DIR.mkdir(exist_ok=True)
    return WORK_DIR


make_work_dir()


def configure_tutorial() -> None:
    """Apply the course plot and display style (light only)."""

    _configure_tutorial()


def show_session() -> None:
    """Print what the learner gives their agent at the start of the session."""

    print(LINES["session"].format(notebook=NOTEBOOK))


def stage_settings() -> dict:
    """Return a copy of the stage settings of the study."""

    return dict(STAGE_SETTINGS)


def prepare_study(
    device: torch.device | str,
    *,
    batch_multiplier: float = BATCH_MULTIPLIER,
    min_atoms: int | None = None,
):
    """Return ``(trajectories, composed_model)`` for the NCIA250 complexes.

    ``batch_multiplier`` trajectories per complex. The study keeps the complexes
    of ``min_atoms`` atoms and more (default ``MIN_ATOMS``, 10): 149 of the 205,
    so 26 gives 3,874. A value below 1 keeps the first part of the set (for a quick
    run on a CPU). Velocities are drawn at 50 K with a fixed seed.
    """

    records, model = _m3.prepare_ncia250_trajectories(
        device,
        batch_multiplier=batch_multiplier,
        initial_temperature=INITIAL_TEMPERATURE,
        random_seed=RANDOM_SEED,
    )
    smallest = MIN_ATOMS if min_atoms is None else int(min_atoms)
    kept = tuple(record for record in records if int(record.num_nodes) >= smallest)
    for simulation_id, record in enumerate(kept):
        record.simulation_id = torch.full_like(record.simulation_id, simulation_id)
    return kept, model


class CudaCompletionSink(_m3.CudaCompletionSink):
    """The course store for finished trajectories; it keeps them on the device of the run.

    Core 03's sink, with two additions: it also works without a GPU, and it
    records the active batch each time trajectories leave it. From that record
    come the number of trajectories that entered after the first batch
    (``admitted_inflight``) and how full the batch was (``occupancy``).
    """

    def __init__(self, capacity: int, device: torch.device | str | None = None) -> None:
        # The device defaults to the GPU when there is one, so the capacity alone makes a store.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._capacity = int(capacity)
        self._device = torch.device(device)
        self._batches = []
        self._count = 0
        self._first_batch: torch.Tensor | None = None
        self._occupancy: list[tuple[int, int]] = []

    def write(self, batch, mask: torch.Tensor | None = None) -> None:
        if mask is not None:
            # Refill hands over the whole active batch with a mask of the finished ones. Nothing
            # is admitted before the first trajectory leaves, so the first call shows the first batch.
            if self._first_batch is None and hasattr(batch, "simulation_id"):
                self._first_batch = batch.simulation_id.reshape(-1).detach().cpu().clone()
            self._occupancy.append((int(batch.num_graphs), int(batch.num_nodes)))
        super().write(batch, mask)

    def zero(self) -> None:
        super().zero()
        self._first_batch = None
        self._occupancy = []

    @property
    def admitted_inflight(self) -> int | None:
        """Finished trajectories that were not in the first active batch."""

        if self._first_batch is None:
            return None
        ids = torch.cat([b.simulation_id.reshape(-1).cpu() for b in self._batches])
        return int((~torch.isin(ids, self._first_batch)).sum())

    @property
    def occupancy(self) -> list[tuple[int, int]]:
        """(trajectories, atoms) in the active batch at each moment trajectories left it."""

        return list(self._occupancy)


class _Table(pd.DataFrame):
    """A DataFrame that shows without its 0, 1, 2 index column."""

    @property
    def _constructor(self):
        return _Table

    def _repr_html_(self) -> str:
        return self.to_html(index=False)

    def __repr__(self) -> str:
        return self.to_string(index=False)


def _batches(result) -> list:
    """The finished batches of a route: a sink with ``read()``, one batch, or a list of batches."""

    if hasattr(result, "read"):
        return [result.read()]
    if isinstance(result, (list, tuple)):
        return list(result)
    return [result]


def _ids(result) -> torch.Tensor:
    """The ``simulation_id`` of every finished trajectory of a route, on the host."""

    batches = _batches(result)
    for batch in batches:
        if not hasattr(batch, "simulation_id"):
            raise TypeError(
                "expected the finished batches of a route (a list of Batch, or a store with read()), "
                f"got {type(batch).__name__}"
            )
    if not batches:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat([b.simulation_id.reshape(-1).cpu() for b in batches])


def _notebook_name(value) -> str | None:
    """The name the calling notebook (or step file) gives ``value``."""

    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_globals.get("__name__") != __name__:
            for name in TRACE_STAGES:
                if frame.f_globals.get(name) is value:
                    return name
        frame = frame.f_back
    return None


def show_stage_changes(stage_trace, stages: tuple[str, ...] | None = None) -> None:
    """Print when trajectories moved from the first stage to the second.

    ``stage_trace`` has one row per trajectory (in ``simulation_id`` order) and
    two integer columns: the MD step of the move to the second stage and the MD
    step at which the trajectory finished, both counted from the trajectory's
    own first step. The stage names follow the notebook's name for the trace
    (``stage_trace``: NVT to NVE; ``relax_md_trace``: relaxation to MD), or
    ``stages`` when given: two names, and optionally the word for the steps.
    """

    if stages is None:
        stages = TRACE_STAGES.get(_notebook_name(stage_trace), ("first stage", "second stage"))
    stages = (*stages, "steps")[:3]
    trace = torch.as_tensor(stage_trace).detach().cpu()
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError(
            "stage_trace must have one row per trajectory and two columns, "
            f"got {tuple(trace.shape)}"
        )
    a, b, steps = stages
    switch = trace[:, 0]
    first, last = int(switch.min()), int(switch.max())
    rows = [
        (LINES["trace_count"], str(trace.shape[0])),
        (LINES["trace_first"].format(a=a, b=b), LINES["trace_steps"].format(n=first, steps=steps)),
        (LINES["trace_last"].format(a=a, b=b), LINES["trace_steps"].format(n=last, steps=steps)),
        (LINES["trace_apart"], LINES["trace_apart_steps"].format(n=last - first, steps=steps)),
    ]
    width = max(len(label) for label, _ in rows) + 4
    for label, value in rows:
        print(f"{label:<{width}}{value}")
    if first == last:
        print(LINES["trace_same"])


def _admitted_inflight(completed):
    """Trajectories that entered after the first active batch, as the store recorded it."""

    return getattr(completed, "admitted_inflight", None)


def show_refill(completed, trajectories) -> pd.DataFrame:
    """Return the count of finished trajectories and of late entries.

    The late entries are counted by the store: the finished trajectories that
    were not in the first active batch.
    """

    ids = _ids(completed)
    admitted = _admitted_inflight(completed)
    if admitted == 0 and len(trajectories) > 0:
        print(LINES["unexpected_admitted"])
    return _Table(
        [
            {
                LINES["col_finished"]: int(torch.unique(ids).numel()),
                LINES["col_admitted"]: LINES["not_recorded"] if admitted is None else admitted,
            }
        ]
    )


def show_staged(staged_batches, trajectories) -> pd.DataFrame:
    """Return the number of groups of the staged run and how many trajectories finished."""

    ids = _ids(staged_batches)
    finished, total = int(torch.unique(ids).numel()), len(trajectories)
    if finished != total:
        print(LINES["unexpected_staged"].format(n=finished, total=total))
    return _Table(
        [
            {
                LINES["col_groups"]: len(_batches(staged_batches)),
                LINES["col_finished"]: finished,
                LINES["col_of"]: total,
            }
        ]
    )


def compare_routes(*, staged, fused, inflight) -> pd.DataFrame:
    """Return one row per route: finished, admitted inflight and seconds.

    Each argument is ``(result, seconds)``: the finished batches (a list of
    ``Batch``) for ``staged`` and ``fused``, the store for ``inflight``. A route
    without refill admits nothing after its first batch by construction.
    """

    rows = []
    for name, (result, seconds), refill in (
        (LINES["route_staged"], staged, False),
        (LINES["route_fused"], fused, False),
        (LINES["route_refill"], inflight, True),
    ):
        ids = _ids(result)
        admitted = _admitted_inflight(result)
        if admitted is None:
            admitted = LINES["not_recorded"] if refill else 0
        rows.append(
            {
                LINES["col_route"]: name,
                LINES["col_finished"]: int(torch.unique(ids).numel()),
                LINES["col_admitted"]: admitted,
                LINES["col_seconds"]: round(float(seconds), 1),
            }
        )
    table = _Table(rows)
    if table[LINES["col_finished"]].nunique() != 1:
        print(LINES["unexpected_finished"])
    if rows[2][LINES["col_admitted"]] == 0:
        print(LINES["unexpected_admitted"])
    return table


if __name__ == "__main__":
    from .agent07_check import main

    sys.exit(main(sys.argv[1:]))
