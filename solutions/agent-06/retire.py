import pandas as pd
import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics import DynamicsStage, NVTLangevin
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.sinks import HostMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_every = 10        # steps between saved frames and between checks, as in the plain run
contact_limit = 3.7    # Å: read off the plot of section 2; a complex is retired once its nearest A-B contact is above this


class RetireHook:
    """Every `frequency` steps: retire each complex whose nearest A-B contact is above `limit` (Å)."""

    stage = DynamicsStage.AFTER_STEP
    frequency = save_every

    def __init__(self, batch, fragment_a_sizes, limit):
        sizes = batch.num_nodes_per_graph.to(batch.device)
        self.n_graphs = batch.num_graphs
        self.graph_idx = batch.batch_idx.long()
        start = torch.cumsum(sizes, 0) - sizes
        local = torch.arange(batch.num_nodes, device=batch.device) - start[self.graph_idx]
        a_sizes = torch.as_tensor(fragment_a_sizes, device=batch.device)
        is_a = local < a_sizes[self.graph_idx]                           # [V] fragment A atoms
        same = self.graph_idx[:, None] == self.graph_idx[None, :]
        self.pair_mask = same & is_a[:, None] & ~is_a[None, :]           # [V, V] A-B pairs of one complex
        self.limit = limit
        self.stop_step = torch.full((self.n_graphs,), -1, dtype=torch.long, device=batch.device)

    def __call__(self, ctx, stage):
        b = ctx.batch
        pos = b.positions.detach()
        d = torch.cdist(pos, pos).masked_fill(~self.pair_mask, float("inf"))
        per_atom = d.amin(dim=1)                                         # [V] closest atom of the other fragment
        nearest = torch.full((self.n_graphs,), float("inf"), dtype=pos.dtype, device=pos.device)
        nearest = nearest.scatter_reduce(0, self.graph_idx, per_atom, reduce="amin")   # [B]

        stop = nearest > self.limit
        if ctx.active_graph_mask is not None:                            # leave retired complexes alone
            stop = stop & ctx.active_graph_mask
        status = b.status.view(-1)[: b.num_graphs]
        done = torch.full_like(status, ctx.workflow.exit_status)
        status.copy_(torch.where(stop, done, status))                    # retire: status -> exit_status
        self.stop_step = torch.where(stop, torch.full_like(self.stop_step, ctx.step_count), self.stop_step)


retire_batch = Batch.from_data_list(complexes).to(device)  # three complexes, one batch
retire_batch["status"] = torch.zeros(retire_batch.num_graphs, 1, dtype=torch.long, device=device)  # 0 = running

retire_hook = RetireHook(retire_batch, fragment_a_sizes, contact_limit)
n_frames = run_settings["n_steps"] // save_every + 1
retire_store = HostMemory(capacity=retire_batch.num_graphs * n_frames)

retire_md = NVTLangevin(
    model=composed_model,
    dt=run_settings["dt"],
    temperature=run_settings["temperature"],
    friction=run_settings["friction"],
    random_seed=0,
    n_steps=run_settings["n_steps"],
    device_type=device.type,
)
for hook in composed_model.make_neighbor_hooks():
    retire_md.register_hook(hook)
retire_md.register_hook(retire_hook)                                     # decides before the frame is saved
retire_md.register_hook(SnapshotHook(sink=retire_store, frequency=save_every))

with retire_md:
    retire_batch = retire_md.run(retire_batch)

stop_step = retire_hook.stop_step.cpu()                                  # [3], MD step of retirement, -1 = ran to the end
final_status = retire_batch.status.view(-1)[: retire_batch.num_graphs].cpu()   # [3], 1 = retired, 0 = ran to the end
print("stop_step:", stop_step.tolist(), " final_status:", final_status.tolist())
