import gc
import statistics
import time

import pandas as pd
import torch
from nvalchemi.dynamics import LBFGS
from nvalchemi.models.mace import MACEWrapper

# The run: 10 L-BFGS steps on a part of the batch, 50 crystals picked evenly from crystal_batch, from the
# same starting structures every time. Every step needs new neighbor lists and one energy + force pass,
# which is the work every relaxation or MD step repeats.
# Why a part: without cuEquivariance the model needs about 36 GB for all 200 crystals, three quarters of a
# 48 GB card.
# The 50 crystals need about 10 GB, so all four cases fit and all four time the same crystals.
# One first run on a freshly loaded model (it pays for kernel setup and, with compile, for compiling),
# then 3 repeats; repeat_s is their median.
time_crystals = 50
run_steps = 10
repeats = 3

gc.collect()
torch.cuda.empty_cache()   # hand back memory that earlier steps no longer use
n_timed = min(time_crystals, crystal_batch.num_graphs)
picks = torch.linspace(0, crystal_batch.num_graphs - 1, n_timed).round().long().to(device)
timed_batch = crystal_batch.clone().index_select(picks)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_model(cueq, compiled):
    return MACEWrapper.from_checkpoint("medium-mpa-0", device=device, dtype=torch.float32,
                                       enable_cueq=cueq, compile_model=compiled)


def one_run(model):
    b = timed_batch.clone()
    opt = LBFGS(model=model, n_steps=run_steps, device_type=device.type)   # no convergence: always run_steps
    for hook in model.make_neighbor_hooks():
        opt.register_hook(hook)
    sync()
    t0 = time.perf_counter()
    with opt:
        b = opt.run(b)
    sync()
    return time.perf_counter() - t0, b.energy.detach().clone()


rows, reference = [], {}
for cueq, compiled in ((False, False), (True, False), (False, True), (True, True)):
    row = dict(cueq=cueq, compile=compiled, first_call_s=float("nan"), repeat_s=float("nan"),
               max_energy_change=float("nan"), note="")
    try:
        torch._dynamo.reset()
        model = make_model(cueq, compiled)
        first, energy = one_run(model)
        times = [one_run(model)[0] for _ in range(repeats)]
        row.update(first_call_s=first, repeat_s=statistics.median(times))
        if not (cueq or compiled):
            reference["energy"] = energy
        if "energy" in reference:    # stays NaN when the case with neither setting failed
            row.update(max_energy_change=float((energy - reference["energy"]).abs().max()))
    except Exception as exc:
        lines = str(exc).splitlines() or [""]
        reason = next((line for line in lines if "out of memory" in line), lines[0])
        row.update(note=f"{type(exc).__name__}: {reason[:150]}")
    rows.append(row)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

speed_table = pd.DataFrame(rows)
print(f"timed on {n_timed} of {crystal_batch.num_graphs} crystals ({timed_batch.num_nodes} atoms), "
      f"{run_steps} L-BFGS steps per run")

setting_roles = {
    "cueq": "Replaces the model's equivariant tensor-product math with NVIDIA's faster GPU kernels. "
            "Same weights and same physics, so it speeds up every pass of the model itself.",
    "compile": "Has PyTorch trace the model once and fuse its many small operations into fewer GPU kernels. "
               "It costs extra time on the first call and then cuts launch overhead on later calls.",
}
