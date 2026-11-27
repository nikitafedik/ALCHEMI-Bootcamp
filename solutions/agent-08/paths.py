from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.mep import interpolate_paths

initial = Batch.from_data_list([AtomicData.from_atoms(atoms, device=device) for atoms in reactants], device=device)
final = Batch.from_data_list([AtomicData.from_atoms(atoms, device=device) for atoms in products], device=device)

# one Batch, one group of images per reaction; num_images counts both ends.
# The ends are already laid over each other, so they are not aligned again.
paths = interpolate_paths(initial, final, num_images=neb_settings["num_images"])

# interpolate_paths drops energy, forces and velocities; the NEB optimizer reads all three
dtype = paths.positions.dtype
n_atoms = paths.num_nodes_per_graph.tolist()
paths.add_key("energy", [torch.zeros(1, device=device, dtype=dtype) for _ in n_atoms], level="system")
paths.add_key("forces", [torch.zeros(n, 3, device=device, dtype=dtype) for n in n_atoms], level="node")
paths.add_key("velocities", [torch.zeros(n, 3, device=device, dtype=dtype) for n in n_atoms], level="node")
