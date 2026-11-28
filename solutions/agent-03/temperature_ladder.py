import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage, NVTLangevin, initialize_velocities

# --- settings -------------------------------------------------------------
COLDEST_K, HOTTEST_K = 100.0, 5000.0   # n_temperatures copies, spaced by a constant ratio, cold to far too hot
TIMESTEP_FS = 1.0
FRICTION_PER_FS = 0.01                 # Langevin coupling, 1/fs
N_STEPS = 300                          # 300 fs
SAVE_EVERY = 5                         # steps between energy records
SEED = 0

# --- one crystal, copied once for each temperature, all copies in one batch ----
# The course default: copper, 64 atoms, every atom on its lattice site, so it needs no relaxation.
crystal = helpers.load_ladder_crystal()
batch = Batch.from_data_list([AtomicData.from_atoms(crystal) for _ in range(n_temperatures)]).to(device)
atoms_per_copy = batch.num_nodes_per_graph.to(batch.positions.dtype)

# one temperature for each copy, shape (n_temperatures,), in K
replica_temperature = torch.logspace(
    torch.log10(torch.tensor(COLDEST_K)), torch.log10(torch.tensor(HOTTEST_K)), n_temperatures,
    device=device, dtype=batch.positions.dtype,
)

# each copy starts with thermal velocities at its own temperature
batch.velocities = torch.zeros_like(batch.positions)
initialize_velocities(batch.velocities, batch.atomic_masses, replica_temperature,
                      batch.batch_idx.int(), random_seed=SEED)


def energy_per_atom(energy):
    """Potential energy of each copy divided by its atom count, eV per atom, on the cpu."""
    return (energy.detach().view(-1)[: batch.num_graphs] / atoms_per_copy).cpu()


# the energy before the first step (time 0)
helpers.prepare_model_batch(batch, mace_model)
start_energy = energy_per_atom(mace_model(batch)["energy"])


class EnergyHook:
    """After every SAVE_EVERY-th step, store the simulated time and the energy per atom of each copy."""
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self):
        self.times, self.values = [], []

    def __call__(self, ctx, stage):
        steps_done = ctx.step_count + 1        # a hook at AFTER_STEP sees the state after step_count + 1 steps
        if steps_done % SAVE_EVERY:
            return
        self.times.append(steps_done * TIMESTEP_FS)
        self.values.append(energy_per_atom(ctx.batch.energy))


# --- NVT, one thermostat temperature per copy ------------------------------------
record = EnergyHook()
nvt = NVTLangevin(model=mace_model, dt=TIMESTEP_FS, temperature=replica_temperature, friction=FRICTION_PER_FS,
                  random_seed=SEED, n_steps=N_STEPS, device_type=device.type, hooks=[record])
for hook in mace_model.make_neighbor_hooks():
    nvt.register_hook(hook)
with nvt:
    batch = nvt.run(batch)

time_fs = torch.tensor([0.0] + record.times)                       # (n_saved,) fs
replica_energy = torch.stack([start_energy] + record.values)       # (n_saved, n_temperatures) eV per atom
replica_temperature = replica_temperature.cpu()                    # (n_temperatures,) K
