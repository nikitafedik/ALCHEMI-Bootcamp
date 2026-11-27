from nvalchemi.dynamics import FIRE2, ConvergenceHook, FusedStage

# All alanine dipeptide starts from the saved grid, in one batch on the GPU.
conformer_atoms = helpers.load_dipeptide_grid()
conformer_batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in conformer_atoms]).to(device)

# FIRE2 on the composed model; each conformer stops on its own once its largest force is below 0.05 eV/A.
optimizer = FIRE2(model=composed_model, dt=0.1, n_steps=3, device_type=device.type,
                  convergence_hook=ConvergenceHook.from_fmax(0.05))
stage = FusedStage(sub_stages=[(0, optimizer)], device_type=device.type)
for hook in composed_model.make_neighbor_hooks():
    stage.register_hook(hook)
with stage:
    relaxed_conformers = stage.run(conformer_batch.clone())  # conformer_batch keeps the starts

# Energy of each relaxed conformer, from one more call of the composed model.
relaxed_conformers = helpers.prepare_model_batch(relaxed_conformers, composed_model)
conformer_energy = composed_model(relaxed_conformers)["energy"].detach().squeeze(-1)  # (139,), eV
print("conformer_energy", tuple(conformer_energy.shape), "lowest", int(conformer_energy.argmin()))
