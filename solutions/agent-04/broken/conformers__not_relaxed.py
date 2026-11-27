conformer_atoms = helpers.load_dipeptide_grid()
conformer_batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in conformer_atoms]).to(device)
relaxed_conformers = conformer_batch.clone()  # never relaxed
relaxed_conformers = helpers.prepare_model_batch(relaxed_conformers, composed_model)
conformer_energy = composed_model(relaxed_conformers)["energy"].detach().squeeze(-1)
