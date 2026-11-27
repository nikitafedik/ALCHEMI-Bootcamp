from helpers.model_support import DirectCoulombAdapter as CooAdapter
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper, PipelineStep

no_d3_model = PipelineModelWrapper(groups=[
    PipelineGroup(steps=[PipelineStep(aimnet_model, wire={"charges": "partial_charges"}), CooAdapter(cutoff=15.0).to(device)], use_autograd=True),
])
scan_batch = helpers.prepare_model_batch(scan_batch, no_d3_model)
no_d3_energy = no_d3_model(scan_batch)["energy"].detach()
