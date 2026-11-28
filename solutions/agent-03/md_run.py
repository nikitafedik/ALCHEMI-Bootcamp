import torch
from nvalchemi.dynamics import NVE
from nvalchemi.dynamics.base import DynamicsStage

dt = 1.0            # fs
n_steps = 500       # 500 fs total


class EnergyRecorder:
    """After every step, store the total energy (potential + kinetic, eV) of each crystal."""
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self):
        self.rows = []     # [n_crystals] tensors, one per step

    def __call__(self, ctx, stage):
        self.rows.append(total_energy(ctx.batch))


def total_energy(b):
    kinetic = torch.zeros(b.num_graphs, device=b.device, dtype=b.positions.dtype)
    kinetic.index_add_(0, b.batch_idx.long(), 0.5 * b.atomic_masses * b.velocities.pow(2).sum(-1))
    return (b.energy.view(-1)[: b.num_graphs] + kinetic).detach().clone()


run_batch = md_batch.clone()          # md_batch stays as the starting state
recorder = EnergyRecorder()
nve = NVE(model=mace_model, dt=dt, n_steps=n_steps, device_type=device.type, hooks=[recorder])
for hook in mace_model.make_neighbor_hooks():
    nve.register_hook(hook)

first_energy = total_energy(run_batch)    # time 0, before the first step
with nve:
    run_batch = nve.run(run_batch)

# Row k is the state after k steps (row 0 is the start), so time_fs = k * dt.
time_fs = torch.arange(n_steps + 1, dtype=torch.float32) * dt
crystal_energy = torch.stack([first_energy] + recorder.rows).cpu()   # [n_saved, n_crystals]

drift = (crystal_energy - crystal_energy[0]).abs().max(dim=0).values / run_batch.num_nodes_per_graph.cpu()
print(f"time {time_fs[0]:.0f}..{time_fs[-1]:.0f} fs, {len(time_fs)} rows; "
      f"largest |E(t) - E(0)| per atom: median {drift.median():.2e}, max {drift.max():.2e} eV/atom")
