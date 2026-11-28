import torch
from nvalchemi.dynamics import KB_EV, initialize_velocities

target_temperature = 300.0   # K

# relaxed_batch carries leftovers from the relaxation (status 1 = "finished", zero velocities,
# forces/neighbor data from the last step). Start from a copy and keep only the relaxed positions.
md_batch = relaxed_batch.clone()
md_batch.status = torch.zeros_like(md_batch.status)          # no crystal marked finished

# Thermal velocities: Maxwell-Boltzmann draw per crystal, centre-of-mass drift removed, then rescaled
# so each crystal starts at exactly the target temperature.
md_batch.velocities = torch.zeros_like(md_batch.positions)
initialize_velocities(
    md_batch.velocities, md_batch.atomic_masses,
    torch.full((md_batch.num_graphs,), target_temperature, device=device, dtype=md_batch.positions.dtype),
    md_batch.batch_idx.int(), random_seed=0,
)

# Forces and energy for the first MD step must belong to these exact positions.
helpers.prepare_model_batch(md_batch, mace_model)
out = mace_model(md_batch)
md_batch.forces = out["forces"].detach().clone()
md_batch.energy = out["energy"].detach().clone()

# Starting temperature of each crystal from its velocities: T = 2 KE / (3 N kB).
kinetic = torch.zeros(md_batch.num_graphs, device=device, dtype=md_batch.positions.dtype)
kinetic.scatter_add_(0, md_batch.batch_idx.long(),
                     0.5 * md_batch.atomic_masses * md_batch.velocities.pow(2).sum(-1))
start_temperature = 2 * kinetic / (3 * md_batch.num_nodes_per_graph * KB_EV)
