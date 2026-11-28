import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.aimnet2 import AIMNet2Wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AIMNet2 as it comes: no Coulomb, no D3 (both are switched off in the wrapper).
aimnet_model = AIMNet2Wrapper.from_checkpoint(helpers.model_checkpoint(), device=device)

# 90 structures (AB, A, B per scan point), each with its own total charge from info["charge"].
atoms_list, scan_labels = helpers.load_nci_scans()
scan_batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in atoms_list]).to(device)

# One neighbor-list build with the model's own hooks, then one forward pass.
scan_batch = helpers.prepare_model_batch(scan_batch, aimnet_model)
base_energy = aimnet_model(scan_batch)["energy"].detach()  # (90, 1), eV
print(scan_batch.num_graphs, "structures,", tuple(base_energy.shape), base_energy.dtype, base_energy.device)
print("total charges:", sorted(set(scan_batch.charge.flatten().tolist())))
