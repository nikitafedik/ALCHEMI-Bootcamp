import pandas as pd
import torch
from pathlib import Path

from nvalchemi.data import Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.hooks import FreezeAtomsHook, LoggingHook, SnapshotHook
from nvalchemi.dynamics.sinks import HostMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_every = 10  # steps between saved frames and log rows, as in the plain run

frozen_batch = Batch.from_data_list(complexes).to(device)  # three complexes, one batch

# True for the first fragment_a_sizes[i] atoms of complex i, in batch order
frozen_mask_per_complex = [
    torch.arange(c.num_nodes, device=device) < n for c, n in zip(complexes, fragment_a_sizes)
]
frozen_batch.add_key("frozen_mask", frozen_mask_per_complex, level="node")
frozen_mask = frozen_batch.frozen_mask  # bool, one entry per atom of the batch

frozen_log_path = Path("agent06_work/frozen_log.csv")
frozen_log_path.unlink(missing_ok=True)
n_frames = run_settings["n_steps"] // save_every + 1
frozen_store = HostMemory(capacity=frozen_batch.num_graphs * n_frames)

frozen_md = NVTLangevin(
    model=composed_model,
    dt=run_settings["dt"],
    temperature=run_settings["temperature"],
    friction=run_settings["friction"],
    random_seed=0,
    n_steps=run_settings["n_steps"],
    device_type=device.type,
)
for hook in composed_model.make_neighbor_hooks():
    frozen_md.register_hook(hook)
frozen_md.register_hook(FreezeAtomsHook(mask_key="frozen_mask"))  # registered before the saving hooks
frozen_md.register_hook(SnapshotHook(sink=frozen_store, frequency=save_every))
frozen_md.register_hook(LoggingHook(backend="csv", log_path=str(frozen_log_path), frequency=save_every))

with frozen_md:
    frozen_batch = frozen_md.run(frozen_batch)

# check: fragment A has not moved from the first saved frame to the last
saved = [d.positions for d in frozen_store.read().to_data_list()]
first, last = torch.cat(saved[:3]), torch.cat(saved[-3:])
print("largest fragment A displacement (Å):", f"{(last - first)[frozen_mask.cpu()].norm(dim=1).max().item():.3f}")
print("largest displacement of the rest (Å):", f"{(last - first)[~frozen_mask.cpu()].norm(dim=1).max().item():.3f}")
