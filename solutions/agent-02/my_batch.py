from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from nvalchemi.data import AtomicData, Batch

mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccc(NC(C)=O)cc1"))
AllChem.EmbedMolecule(mol, randomSeed=7)
AllChem.MMFFOptimizeMolecule(mol)
mine = AtomicData.from_atoms(Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions()))
data_list = molecule_data[:2] + [mine] + molecule_data[2:]
my_batch = Batch.from_data_list(data_list).to(batch.device)
