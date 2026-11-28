# Try it: relax each structure, then run MD on it. The two stages are fused, so each trajectory
# enters MD as soon as its own relaxation has converged.
import torch
from nvalchemi.data import Batch
from nvalchemi.dynamics import FIRE2, ConvergenceHook, DynamicsStage, NVTLangevin

RELAX_FMAX = 0.05   # eV/Å, largest force on an atom at which a relaxation counts as converged
RELAX_STEPS = 300   # step cap of the relaxation, per trajectory
MD_STEPS = 100      # MD steps per trajectory (1 fs each)


class StepCount:
    """Counts the steps each trajectory takes in one stage, by simulation_id."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, total):
        self.steps = torch.zeros(total, dtype=torch.long, device=device)

    def __call__(self, ctx, stage):
        count = ctx.batch.num_graphs
        ids = ctx.batch.simulation_id.view(-1)[:count].long()
        active = ctx.active_graph_mask
        self.steps[ids] += 1 if active is None else active.long()


def _groups():
    """Indices of the trajectories, in order, in groups within max_batch_size and max_atoms."""
    groups, current, atoms = [], [], 0
    for index, item in enumerate(trajectories):
        size = int(item.num_nodes)
        if current and (len(current) >= max_batch_size or atoms + size > max_atoms):
            groups.append(current)
            current, atoms = [], 0
        current.append(index)
        atoms += size
    if current:
        groups.append(current)
    return groups


relax_md_batches = []
_relax_count, _md_count = StepCount(len(trajectories)), StepCount(len(trajectories))
for _indices in _groups():
    _relax = FIRE2(model=composed_model, dt=0.1, n_steps=RELAX_STEPS,
                   convergence_hook=ConvergenceHook.from_fmax(RELAX_FMAX))
    _md = NVTLangevin(model=composed_model, dt=1.0, temperature=300.0, friction=0.01,
                      random_seed=0, n_steps=MD_STEPS)
    _relax.register_hook(_relax_count)
    _md.register_hook(_md_count)
    _fused = _relax + _md
    for _hook in composed_model.make_neighbor_hooks():
        _fused.register_hook(_hook)
    _batch = Batch.from_data_list([trajectories[i].clone() for i in _indices], device=device)
    with _fused:
        relax_md_batches.append(_fused.run(_batch, n_steps=RELAX_STEPS + MD_STEPS + 1))

# One row per trajectory, in simulation_id order: the step at which it moved from relaxation to
# MD, and the step at which it finished, both counted from its own first step.
relax_md_trace = torch.stack([_relax_count.steps, _relax_count.steps + _md_count.steps], dim=1).cpu()
