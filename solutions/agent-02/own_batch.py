import os, shutil
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import AtomicDataZarrWriter
from nvalchemi.neighbors import compute_neighbors

def build(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    return AtomicData.from_atoms(Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions()))

own_batch = Batch.from_data_list([build(s) for s in ["CCO", "c1ccncc1", "NC(N)=O"]]).to(batch.device)
compute_neighbors(own_batch, config=aimnet_model.model_config.neighbor_config)
out = aimnet_model(own_batch)
own_energy, own_forces = out["energy"], out["forces"]
own_batch.energy = own_energy.detach()
own_batch.forces = own_forces.detach()
own_store = os.path.join(work_dir, "own.zarr")
shutil.rmtree(own_store, ignore_errors=True)
AtomicDataZarrWriter(own_store).write(own_batch)
