relaxed_conformers = Batch.from_data_list([AtomicData.from_atoms(a) for a in helpers.load_dipeptide_grid()[:10]]).to(device)
conformer_batch = relaxed_conformers.clone()
conformer_energy = composed_model(helpers.prepare_model_batch(relaxed_conformers.clone(), composed_model))["energy"].detach()
