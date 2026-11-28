from nvalchemi.models.dftd3 import DFTD3ModelWrapper
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper, PipelineStep

no_d3_model = PipelineModelWrapper(groups=[
    PipelineGroup(steps=[PipelineStep(aimnet_model, wire={"charges": "partial_charges"}), helpers.DirectCoulombAdapter().to(device)], use_autograd=True),
    PipelineGroup(steps=[DFTD3ModelWrapper(**helpers.d3_settings()).to(device)]),
])
scan_batch = helpers.prepare_model_batch(scan_batch, no_d3_model)
no_d3_energy = no_d3_model(scan_batch)["energy"].detach()
