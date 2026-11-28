from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper, PipelineStep

no_d3_model = PipelineModelWrapper(groups=[
    PipelineGroup(steps=[PipelineStep(aimnet_model, wire={"charges": "partial_charges"}), helpers.DirectCoulombAdapter().to(device)], use_autograd=True),
])
no_d3_energy = base_energy  # the bare model's energies again, so the error is not between the two others
