import time
import torch
from nvalchemi.data import Batch
from nvalchemi.dynamics import LBFGS, ConvergenceHook, FusedStage

force_threshold = 0.05   # eV/Å, a common "relaxed" criterion for MLIP structures
max_steps = 500          # per-crystal step budget; a crystal that hits it is reported, not hidden


def relax(batch):
    """Relax one Batch in place with L-BFGS; each crystal leaves the run when its max force <= threshold."""
    opt = LBFGS(model=mace_model, n_steps=max_steps, device_type=device.type,
                convergence_hook=ConvergenceHook.from_fmax(force_threshold))
    stage = FusedStage(sub_stages=[(0, opt)], device_type=device.type)
    for hook in mace_model.make_neighbor_hooks():
        stage.register_hook(hook)
    with stage:
        return stage.run(batch, n_steps=max_steps)


def max_force_per_crystal(batch):
    """Largest force norm per graph (eV/Å) from a fresh forward pass."""
    helpers.prepare_model_batch(batch, mace_model)
    f = mace_model(batch)["forces"].detach()
    out = torch.zeros(batch.num_graphs, device=f.device, dtype=f.dtype)
    return out.scatter_reduce_(0, batch.batch_idx.long(), f.norm(dim=-1), reduce="amax", include_self=False)


# Warm-up on a copy (kernel compilation must not count in the timings).
relax(crystal_batch.clone().index_select(torch.arange(3, device=device)))
relax(crystal_batch.clone().index_select(torch.arange(1, device=device)))   # one-graph shape too

# The whole batch in one run, on a copy so crystal_batch keeps its starting structures.
relaxed_batch = crystal_batch.clone()
torch.cuda.synchronize()
t0 = time.perf_counter()
relaxed_batch = relax(relaxed_batch)
torch.cuda.synchronize()
batch_seconds = time.perf_counter() - t0

# Same relaxation, one crystal at a time, for three crystals at the 1/6, 1/2 and 5/6 points of the
# atom-count ranking, so they stand for the batch. Projection to all crystals: mean * n_crystals.
order = crystal_batch.num_nodes_per_graph.argsort().tolist()
picks = [order[int(len(order) * q)] for q in (1 / 6, 1 / 2, 5 / 6)]
single_times = []
for i in picks:
    one = crystal_batch.clone().index_select(torch.tensor([i], device=device))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    relax(one)
    torch.cuda.synchronize()
    single_times.append(time.perf_counter() - t0)
single_seconds = torch.tensor(single_times)
single_projected_seconds = float(single_seconds.mean()) * crystal_batch.num_graphs

relaxed_max_force = max_force_per_crystal(relaxed_batch)
print(f"batch {batch_seconds:.1f} s; one-by-one {single_projected_seconds:.1f} s projected "
      f"({single_projected_seconds / batch_seconds:.1f}x); converged "
      f"{int((relaxed_max_force <= force_threshold).sum())} of {relaxed_max_force.numel()}")
