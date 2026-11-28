# Agent 07, "One out, one in": default prompts

The typical learner prompt of each step, written from the step's text, its
table and its Hints alone. A learner writes their own; nothing here is
printed in the notebook.

## Example: fuse NVT and NVE (`fused.py`)

```text
use the nvalchemi-toolkit skill.

fuse nvt and nve in one batch for my trajectories: each starts in nvt at 300 k and moves to nve once its temperature has settled; nve then runs a fixed number of steps. at every step each trajectory gets the update of the stage it is in.

input: trajectories, composed_model, stage_settings, max_batch_size, max_atoms
output: build_stages (a function that returns a fresh nvt and nve pair, nvt with its stopping rule), fused_batches, fused_seconds, stage_trace
keep the nvt stopping rule per trajectory by simulation_id. stage_trace holds two integers per trajectory, in simulation_id order: the md step of the move to nve and the md step of the finish.
time the run with helpers.time_route(device) around all of it, the building of its batches included.
write it to agent07_work/fused.py
```

## Example: refill the batch (`refill.py`)

```text
use the nvalchemi-toolkit skill.

refill the batch: a waiting list, the active batch and a store for finished results. every few steps toolkit moves finished trajectories to the store and fills the freed space from the waiting list. the waiting list is a dataset: InMemoryDataset over one Batch of my trajectories. the store goes to FusedStage as sinks=[...], the waiting list as sampler=....

input: trajectories, composed_model, build_stages, max_batch_size, max_atoms, helpers.CudaCompletionSink
output: completed, inflight_seconds
write it to agent07_work/refill.py
```

## Try it: run it in stages (`staged.py`)

```text
use the nvalchemi-toolkit skill.

run my trajectories in stages, with no fused stages: one group runs nvt until every trajectory has passed its rule or reached its step cap, then nve, then the next group starts. a trajectory that finishes early waits for its group. use trajectories from the setup cell as they are.

input: trajectories, composed_model, build_stages, max_batch_size, max_atoms
output: staged_batches, staged_seconds
done when every trajectory has finished both stages.
write it to agent07_work/staged.py
```

## Try it: relax, then MD (`relax_md.py`)

```text
use the nvalchemi-toolkit skill.

relax each structure, then simulate it: each trajectory enters md as soon as its own relaxation has converged. use trajectories from the setup cell as they are: the complexes as published, not yet relaxed with this model.

input: trajectories, composed_model, max_batch_size, max_atoms
output: relax_md_batches, relax_md_trace
relax_md_trace has the form of stage_trace: two integers per trajectory, in simulation_id order, the step of the move to md and the step of the finish.
done when the summary gives the earliest and the latest step at which trajectories entered md.
write it to agent07_work/relax_md.py
```
