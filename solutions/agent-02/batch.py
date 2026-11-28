import torch
from nvalchemi.data import AtomicData, Batch

names = ["caffeine", "aspirin", "ibuprofen", "paracetamol", "capsaicin"]
molecule_data = [AtomicData.from_atoms(helpers.load_molecule(n)) for n in names]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = Batch.from_data_list(molecule_data).to(device)
