import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics import DynamicsStage, NVTLangevin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
record_every = 10  # steps between records, as in the first run


class SeparationHook:
    """Every `frequency` steps: centre-of-mass separation and nearest contact of fragment A and B."""

    stage = DynamicsStage.AFTER_STEP
    frequency = record_every

    def __init__(self, batch, fragment_a_sizes):
        sizes = batch.num_nodes_per_graph.to(batch.device)
        self.n_graphs = batch.num_graphs
        self.graph_idx = batch.batch_idx.long()
        start = torch.cumsum(sizes, 0) - sizes
        local = torch.arange(batch.num_nodes, device=batch.device) - start[self.graph_idx]
        a_sizes = torch.as_tensor(fragment_a_sizes, device=batch.device)
        self.is_a = local < a_sizes[self.graph_idx]                      # [V] fragment A atoms
        self.group = 2 * self.graph_idx + (~self.is_a).long()            # [V] (graph, fragment) slot
        pair = (self.graph_idx[:, None] == self.graph_idx[None, :]) & (self.is_a[:, None] & ~self.is_a[None, :])
        self.pair_mask = pair                                            # [V, V] A-B pairs of one complex
        self.steps, self.sep, self.near = [], [], []

    def __call__(self, ctx, stage):
        b = ctx.batch
        pos, mass = b.positions.detach(), b.atomic_masses.detach()
        w = mass.unsqueeze(-1)
        mw = torch.zeros(2 * self.n_graphs, 3, dtype=pos.dtype, device=pos.device).index_add_(0, self.group, w * pos)
        mt = torch.zeros(2 * self.n_graphs, 1, dtype=pos.dtype, device=pos.device).index_add_(0, self.group, w)
        com = (mw / mt).view(self.n_graphs, 2, 3)                        # [B, 2, 3], A then B
        separation = (com[:, 0] - com[:, 1]).norm(dim=-1)                # [B]

        d = torch.cdist(pos, pos).masked_fill(~self.pair_mask, float("inf"))
        per_atom = d.amin(dim=1)                                         # [V] closest partner in the other fragment
        nearest = torch.full((self.n_graphs,), float("inf"), dtype=pos.dtype, device=pos.device)
        nearest = nearest.scatter_reduce(0, self.graph_idx, per_atom, reduce="amin")

        self.steps.append(ctx.step_count)
        self.sep.append(separation.clone())
        self.near.append(nearest.clone())


batch_sep = Batch.from_data_list(complexes).to(device)
separation_hook = SeparationHook(batch_sep, fragment_a_sizes)

md_sep = NVTLangevin(
    model=composed_model,
    dt=run_settings["dt"],
    temperature=run_settings["temperature"],
    friction=run_settings["friction"],
    random_seed=0,
    n_steps=run_settings["n_steps"],
    device_type=device.type,
)
for hook in composed_model.make_neighbor_hooks():
    md_sep.register_hook(hook)
md_sep.register_hook(separation_hook)

with md_sep:
    batch_sep = md_sep.run(batch_sep)

separation = torch.stack(separation_hook.sep).cpu()                      # [n_records, 3], Angstrom
nearest_contact = torch.stack(separation_hook.near).cpu()                # [n_records, 3], Angstrom
