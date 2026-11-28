import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One Batch holds all crystals (one graph each); every graph keeps its own cell and pbc.
crystals = helpers.load_crystals(n_crystals)
crystal_batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in crystals]).to(device)

# Built-in MACE-MPA-0 wrapper. float32 weights match the float32 batch (the optimizer and the
# integrators need positions and forces in one precision). Weights are frozen (default train=False);
# forces come from autograd.
# enable_cueq: NVIDIA's cuEquivariance GPU kernels for the tensor products; on when a CUDA GPU is present.
mace_model = MACEWrapper.from_checkpoint("medium-mpa-0", device=device, dtype=torch.float32,
                                         enable_cueq=device.type == "cuda")

# Forces on the starting structures: neighbor lists first, then one forward pass.
helpers.prepare_model_batch(crystal_batch, mace_model)
forces = mace_model(crystal_batch)["forces"].detach()
start_max_force = torch.zeros(crystal_batch.num_graphs, device=device, dtype=forces.dtype)
start_max_force = start_max_force.scatter_reduce(
    0, crystal_batch.batch_idx.long(), forces.norm(dim=-1), reduce="amax", include_self=False)
