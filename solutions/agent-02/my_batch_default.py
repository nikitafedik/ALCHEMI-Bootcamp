from nvalchemi.data import AtomicData, Batch

mine = AtomicData.from_atoms(helpers.load_default_molecule())     # the saved default: phenacetin, 26 atoms
data_list = molecule_data[:2] + [mine] + molecule_data[2:]        # as the third molecule
my_batch = Batch.from_data_list(data_list).to(batch.device)
