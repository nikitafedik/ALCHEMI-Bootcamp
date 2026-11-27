import torch

# Energy only: with "forces" off the wrapper skips autograd, so no_grad is safe.
with aceff_model.narrowed_outputs({"energy"}), torch.no_grad():
    aceff_energy = aceff_model(scan_batch)["energy"].detach()  # [90, 1], eV

print("aceff_energy", tuple(aceff_energy.shape), aceff_energy.dtype, aceff_energy.device)
print("finite:", bool(torch.isfinite(aceff_energy).all()))
print("range (eV): %.2f to %.2f" % (aceff_energy.min().item(), aceff_energy.max().item()))
