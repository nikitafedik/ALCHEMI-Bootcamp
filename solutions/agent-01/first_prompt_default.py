"""Energy of the course's default molecule (caffeine) with AIMNet2 (Toolkit 0.3.0)."""
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.aimnet2 import AIMNet2Wrapper
from nvalchemi.neighbors import compute_neighbors

from helpers import agent01 as helpers

# The saved default: caffeine, 24 atoms, neutral, as ASE Atoms (positions in Angstrom).
my_molecule = helpers.load_default_molecule()

# AIMNet2 on the GPU when one is present, energy only.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AIMNet2Wrapper.from_checkpoint("aimnet2", device=device)
model.set_config("active_outputs", {"energy"})

batch = Batch.from_data_list([AtomicData.from_atoms(my_molecule)]).to(device)
compute_neighbors(batch, config=model.model_config.neighbor_config)
with torch.no_grad():
    out = model(batch)

my_energy = out["energy"].item()  # eV, total energy of the molecule
print(f"{len(my_molecule)} atoms, {my_molecule.get_chemical_formula()}")
print(f"AIMNet2 energy: {my_energy:.4f} eV (no Coulomb or D3 terms)")
