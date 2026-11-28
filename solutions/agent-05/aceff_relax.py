"""Relax the five molecules of helpers.load_molecules() with aceff_model, in one batch.

Path taken: shipped components only (Batch, FIRE2, ConvergenceHook, FusedStage).
Needs aceff_model from aceff_wrapper.py. Units: positions in Angstrom, forces in eV/Angstrom.
"""

import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2, ConvergenceHook, FusedStage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

force_threshold = 0.05  # eV/Angstrom, largest force on any atom of a molecule
max_steps = 500  # step budget for each molecule

# One Batch for all molecules; AtomicData.from_atoms reads the total charge from atoms.info["charge"].
molecules = helpers.load_molecules()
data_list = [AtomicData.from_atoms(atoms) for atoms in molecules]
relaxed_batch = Batch.from_data_list(data_list).to(device)

# One engine over the whole batch; the FusedStage retires each molecule when it converges.
opt = FIRE2(
    model=aceff_model,
    dt=0.1,
    n_steps=max_steps,
    device_type=device.type,
    convergence_hook=ConvergenceHook.from_fmax(force_threshold),
)
dyn = FusedStage(sub_stages=[(0, opt)], device_type=device.type)
for hook in aceff_model.make_neighbor_hooks():  # empty for AceFF: it builds its own neighbor list
    dyn.register_hook(hook)
with dyn:
    relaxed_batch = dyn.run(relaxed_batch)

# Largest force norm on an atom of each molecule, from a fresh model call on the relaxed geometry.
forces = aceff_model(relaxed_batch)["forces"].detach()
relaxed_max_force = torch.zeros(relaxed_batch.num_graphs, device=forces.device, dtype=forces.dtype).scatter_reduce(
    0, relaxed_batch.batch_idx.long(), forces.norm(dim=-1), reduce="amax", include_self=False
)

print("molecule        atoms  largest force (eV/A)")
for atoms, n, f in zip(molecules, relaxed_batch.num_nodes_per_graph.tolist(), relaxed_max_force.cpu().tolist()):
    print(f"{atoms.info.get('name', '?'):<14}{n:>6}  {f:.5f}")
print("status (1 = converged):", relaxed_batch.status.view(-1).tolist())
print("all below threshold:", bool((relaxed_max_force < force_threshold).all()))
