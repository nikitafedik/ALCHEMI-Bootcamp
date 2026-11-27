from nvalchemi.models.dftd3 import DFTD3ModelWrapper
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper, PipelineStep

d3_model = DFTD3ModelWrapper(**helpers.d3_settings()).to(device)
coulomb_model = helpers.DirectCoulombAdapter(cutoff=5.0).to(device)

# AIMNet2 "charges" are written to batch.partial_charges, which the Coulomb term requires.
# AIMNet2 + Coulomb share one autograd group: forces are -dE/dr of their summed energy,
# including the way the charges move with the atoms. D3 supplies its own energy and forces.
composed_model = PipelineModelWrapper(groups=[
    PipelineGroup(
        steps=[PipelineStep(aimnet_model, wire={"charges": "partial_charges"}), coulomb_model],
        use_autograd=True,
    ),
    PipelineGroup(steps=[d3_model]),
])

scan_batch = helpers.prepare_model_batch(scan_batch, composed_model)
composed_energy = composed_model(scan_batch)["energy"].detach()  # (90, 1), eV

# Same model on the NCIA structures (AB, A, B per complex, loader order).
ncia_batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in helpers.load_ncia()]).to(device)
ncia_batch = helpers.prepare_model_batch(ncia_batch, composed_model)
ncia_energy = composed_model(ncia_batch)["energy"].detach()  # (615, 1), eV
print("composed_energy", tuple(composed_energy.shape), "ncia_energy", tuple(ncia_energy.shape))
