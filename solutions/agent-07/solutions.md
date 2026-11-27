# Agent 07, "One out, one in": sample step files

One file per step of the notebook. Each is one way to do the step; a learner's
agent will write something different. Every file passes its step check with
Toolkit 0.3.0 (on a CPU with the small study, see below):

```
cp solutions/agent-07/{fused,refill,staged,relax_md}.py notebooks/agent07_work/
cd notebooks
python -m helpers.agent07 check fused      # then refill, staged, relax_md
```

On a machine without a GPU: `ALCHEMI_AGENT07_SMALL=1` in front of the command
runs the check on a small study (the first complexes of the set, 17
trajectories, capacity 6 trajectories and 110 atoms, NVT window 50 and cap 200,
NVE 50 steps).

| Step | File | Names it leaves | Contract the check holds |
| --- | --- | --- | --- |
| Example: fuse NVT and NVE | `fused.py` | `build_stages`, `fused_batches`, `fused_seconds`, `stage_trace` | `build_stages()` takes no arguments and returns a fresh NVT and NVE pair; `fused_batches` is the list of finished batches, every trajectory once, each batch within `max_batch_size` and `max_atoms`; `fused_seconds` in seconds; `stage_trace` one row per trajectory in `simulation_id` order, the MD step of the move to NVE and of the finish, counted from the trajectory's own first step; no trajectory leaves NVT in fewer than `window` MD steps (its rule averages over the last `window` steps); every trajectory runs NVE for `nve_steps` MD steps, one step of slack either way; `trajectories` is left unchanged. Switches all at one step, or trajectories at the NVT step cap, pass and are reported as "unexpected" |
| Example: refill the batch | `refill.py` | `completed`, `inflight_seconds` | the store (or a list of batches) holds every trajectory once, and a store's `read()` works; the run used `build_stages()`; every trajectory took steps in both stages; the active batch never held more than the capacity; trajectories entered after the first batch |
| Try it: run it in stages | `staged.py` | `staged_batches`, `staged_seconds` | the list of finished batches, every trajectory once, each within the capacity; the run used `build_stages()`; every trajectory took steps in NVT and in NVE (counted by the check, no `status` needed) |
| Try it: relax, then MD | `relax_md.py` | `relax_md_batches`, `relax_md_trace` | finished batches as above; the trace as `stage_trace`, with the step of the move to MD. Switches all at one step pass and are reported as "unexpected" |

How the samples work:

- `fused.py` reads the NVT rule, the NVE length, the timestep `dt` and the
  `friction` from `stage_settings` (setup cell). A small hook on NVT
  (`WindowRule`) keeps the temperature of every MD step over the last window,
  for every trajectory, by `simulation_id`, and writes the gap to the rule into
  a batch field; the stage's `ConvergenceHook` reads that field. NVE has no
  convergence hook: as the last stage of a fused stage it runs `n_steps` for
  every trajectory (Toolkit 0.3.0 registers no stopping rule for a last stage
  without one); a `StepCount` hook counts its steps. Keeping the history by `simulation_id` is what lets the
  same stages run with refill. The file also defines `capacity_groups()`,
  `fresh_batch()` and `with_rule_fields()`, which the later files use.
- `refill.py`: the waiting list is an `InMemoryDataset` over one `Batch` of
  copies; `SizeAwareSampler` holds the capacity; `FusedStage(sub_stages=...,
  sampler=..., sinks=[completed], refill_frequency=5)`; `run()` with no batch.
  The timer covers the whole route, waiting list included, as in the other two.
- `staged.py`: self-contained, it uses only the inputs of its table and the
  setup cell (its own `staged_groups()`, `Batch.from_data_list` of copies).
  Plain engines, one after the other, each with `run()`: NVT ends when every
  trajectory of the group meets its rule in the same step, or at its step cap
  (the NVT rule adds its batch field when the batch lacks it); NVE, with no
  rule, runs its `n_steps`.
- `relax_md.py`: `FIRE2` (fmax 0.05 eV/Å) fused with `NVTLangevin`; a hook
  counts the steps of each trajectory in each stage.

`prompts.md` holds the default prompt of each step, the typical learner prompt
that was trialled.
