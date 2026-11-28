import pandas as pd
import torch
from pathlib import Path

from nvalchemi.data import Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.hooks import LoggingHook, SnapshotHook
from nvalchemi.dynamics.sinks import HostMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_every = 10  # steps between saved frames and log rows

batch = Batch.from_data_list(complexes).to(device)  # three complexes, one batch

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
md.register_hook(SnapshotHook(sink=plain_store, frequency=save_every))
md.register_hook(LoggingHook(backend="csv", log_path=str(plain_log_path), frequency=save_every))

with md:
    batch = md.run(batch)

plain_log = pd.read_csv(plain_log_path)
