import time

import pandas as pd
import torch
from nvalchemi.data import AtomicData, Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # no GPU: every route builds on the CPU
n_structures = 1000
structures = helpers.load_many(n_structures)
# coordinates exist before the timer starts: the tensors of the third route are made here
tensors = [(torch.as_tensor(a.positions, dtype=torch.float32), torch.as_tensor(a.numbers)) for a in structures]
routes = {
    "ASE objects on the CPU, then .to(device)": lambda: Batch.from_data_list([AtomicData.from_atoms(a) for a in structures]).to(device),
    "ASE objects directly on the GPU": lambda: Batch.from_data_list([AtomicData.from_atoms(a, device=device) for a in structures]),
    "tensors, Batch.from_raw_dicts": lambda: Batch.from_raw_dicts([{"positions": p, "atomic_numbers": z} for p, z in tensors], device=device),
}


def wait_for_gpu():
    if device.type == "cuda":
        torch.cuda.synchronize()


rows = []
for route, build in routes.items():
    build()                                   # one warm-up build
    wait_for_gpu()
    seconds = []
    for _ in range(5):                        # five timed builds, the best one counts
        wait_for_gpu()
        start = time.perf_counter()
        built = build()
        wait_for_gpu()
        seconds.append(time.perf_counter() - start)
    fields = [k for k in ("atom_categories", "atomic_masses", "atomic_numbers", "positions", "velocities") if getattr(built, k, None) is not None]
    rows.append({"route": route, "structures": built.num_graphs, "atoms": built.num_nodes, "fields": ", ".join(fields),
                 "timed_runs": len(seconds), "best_s": min(seconds), "worst_s": max(seconds)})
build_table = pd.DataFrame(rows)
