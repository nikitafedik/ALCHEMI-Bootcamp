import pandas as pd
import torch
from pathlib import Path

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage, NVTLangevin
from nvalchemi.dynamics.hooks import LoggingHook, SnapshotHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import DynamicsContext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_every = 10                 # steps between records, saved frames and log rows
EV_TO_KCAL = 23.060547830619    # kcal/mol per eV


def fragment_list(data_list):
    """Fragment A then fragment B of each complex, as separate neutral molecules, in batch atom order."""
    out = []
    for c, n in zip(data_list, fragment_a_sizes):
        for part in (slice(0, n), slice(n, None)):
            pos = c.positions[part].detach().clone()
            out.append(
                AtomicData(
                    positions=pos,
                    atomic_numbers=c.atomic_numbers[part].clone(),
                    charge=torch.zeros(1, 1, dtype=pos.dtype, device=pos.device),
                    forces=torch.zeros_like(pos),
                    energy=torch.zeros(1, 1, dtype=pos.dtype, device=pos.device),
                )
            )
    return out


def single_point(batch, neighbor_hooks):
    """Energy of every system of `batch` (eV): build the neighbor data, then call the model."""
    ctx = DynamicsContext(batch=batch, model=composed_model)
    for hook in neighbor_hooks:
        hook(ctx, DynamicsStage.BEFORE_COMPUTE)
    return composed_model(batch)["energy"].detach().view(-1)


class InteractionEnergyHook:
    """Every `frequency` steps: E(complex) - E(A) - E(B) at the current geometry, in kcal/mol."""

    stage = DynamicsStage.AFTER_STEP
    frequency = save_every

    def __init__(self, data_list):
        self.whole = Batch.from_data_list([c.clone() for c in data_list]).to(device)   # 3 complexes, own copy
        self.parts = Batch.from_data_list(fragment_list(data_list)).to(device)         # 6 fragments
        self.whole_hooks = composed_model.make_neighbor_hooks()                        # own neighbor hooks
        self.parts_hooks = composed_model.make_neighbor_hooks()
        self.steps, self.values = [], []

    def energy(self, positions):
        self.whole.positions.copy_(positions)
        self.parts.positions.copy_(positions)                                          # same atom order
        e_whole = single_point(self.whole, self.whole_hooks)
        e_parts = single_point(self.parts, self.parts_hooks)
        return (e_whole - e_parts[0::2] - e_parts[1::2]) * EV_TO_KCAL                  # [3]

    def __call__(self, ctx, stage):
        self.steps.append(ctx.step_count)
        self.values.append(self.energy(ctx.batch.positions.detach()).clone())


batch = Batch.from_data_list(complexes).to(device)  # three complexes, one batch
interaction_hook = InteractionEnergyHook(complexes)
start_interaction_energy = interaction_hook.energy(batch.positions.detach()).cpu()     # [3], starting structure

plain_log_path = Path("agent06_work/plain_log.csv")
plain_log_path.unlink(missing_ok=True)  # the log hook appends; start from an empty file
n_frames = run_settings["n_steps"] // save_every + 1
plain_store = HostMemory(capacity=batch.num_graphs * n_frames)

md = NVTLangevin(
    model=composed_model,
    dt=run_settings["dt"],
    temperature=run_settings["temperature"],
    friction=run_settings["friction"],
    random_seed=0,
    n_steps=run_settings["n_steps"],
    device_type=device.type,
)
for hook in composed_model.make_neighbor_hooks():
    md.register_hook(hook)
md.register_hook(interaction_hook)
md.register_hook(SnapshotHook(sink=plain_store, frequency=save_every))
md.register_hook(LoggingHook(backend="csv", log_path=str(plain_log_path), frequency=save_every))

with md:
    batch = md.run(batch)

saved_step = torch.tensor(interaction_hook.steps)                         # [n_records], MD step of each record
interaction_energy = torch.stack(interaction_hook.values).cpu()          # [n_records, 3], kcal/mol
plain_log = pd.read_csv(plain_log_path)
