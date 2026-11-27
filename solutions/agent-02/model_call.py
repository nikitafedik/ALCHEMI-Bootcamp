from nvalchemi.hooks import DynamicsContext
from nvalchemi.models.aimnet2 import AIMNet2Wrapper

aimnet_model = AIMNet2Wrapper.from_checkpoint("aimnet2", device=batch.device)
context = DynamicsContext(batch=batch, model=aimnet_model)
for hook in aimnet_model.make_neighbor_hooks():   # the model's own hooks build its neighbor matrix on the batch
    hook(context, hook.stage)
out = aimnet_model(batch)
energy, forces = out["energy"], out["forces"]
