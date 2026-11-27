import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.aimnet2 import AIMNet2Wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aimnet_model = AIMNet2Wrapper.from_checkpoint(helpers.model_checkpoint(), device=device)
atoms_list, scan_labels = helpers.load_nci_scans()
for a in atoms_list:
    a.info["charge"] = 0  # the benzoate loses its charge
scan_batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in atoms_list]).to(device)
scan_batch = helpers.prepare_model_batch(scan_batch, aimnet_model)
base_energy = aimnet_model(scan_batch)["energy"].detach()
