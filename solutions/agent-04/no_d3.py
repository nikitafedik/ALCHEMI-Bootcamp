from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper, PipelineStep

# AIMNet2 and the electrostatics term in one autograd group, with no D3: forces come from their summed energy.
no_d3_model = PipelineModelWrapper(groups=[
    PipelineGroup(
        steps=[PipelineStep(aimnet_model, wire={"charges": "partial_charges"}),
               helpers.DirectCoulombAdapter().to(device)],
        use_autograd=True,
    ),
])

# The same 90 scan structures as before, one forward pass.
scan_batch = helpers.prepare_model_batch(scan_batch, no_d3_model)
no_d3_energy = no_d3_model(scan_batch)["energy"].detach()  # (90, 1), eV
print("no_d3_energy", tuple(no_d3_energy.shape))
