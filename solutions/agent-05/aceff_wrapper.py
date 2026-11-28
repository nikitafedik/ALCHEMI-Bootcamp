import torch
from torch import nn

from nvalchemi.models.base import BaseModelMixin, ModelConfig


class AceFFWrapper(nn.Module, BaseModelMixin):
    """AceFF 2.0 (torchmd-net TensorNet2) as an nvalchemi model.

    Energy in eV, forces in eV/A (AceFF 2.0 already uses these units).
    The model builds its own neighbor list inside torchmd-net (5 A cutoff),
    so no neighbor hooks are needed. Isolated molecules only (no cell).
    """

    def __init__(self, repo_id="Acellera/AceFF-2.0", filename="aceff_v2.0.ckpt"):
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        from huggingface_hub import hf_hub_download
        from torchmdnet.models.model import load_model

        path = hf_hub_download(repo_id, filename)
        # derivative=False: this wrapper takes forces from autograd on the energy
        self.model = load_model(path, derivative=False).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cutoff = float(self.model.representation_model.cutoff_upper)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charge"}),
            supports_pbc=False,
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self):
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError("AceFFWrapper does not expose embeddings")

    def forward(self, data, **kwargs):
        self.adapt_input(data, **kwargs)  # requires_grad on positions when forces are active
        pos = data.positions
        z = data.atomic_numbers.long()
        batch_idx = data.batch_idx.long()
        # torchmd-net takes one total charge per molecule, shape [B], float32
        charge = data.charge.reshape(-1).to(pos.dtype)
        energy, _ = self.model(z, pos.float(), batch_idx, q=charge.float(), num_systems=data.num_graphs)
        energy = energy.to(pos.dtype)  # [B, 1]
        out = self.adapt_output({"energy": energy}, data)
        if "forces" in self.model_config.active_outputs:
            out["forces"] = -torch.autograd.grad(energy.sum(), pos, create_graph=self.training)[0]
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aceff_model = AceFFWrapper().to(device)
