# Fine-tune: train the last interaction block and the readouts of a copy of base_model on the
# frames of train_path. Gives: finetuned_model, ft_loss_history ((step, loss) pairs).
# Names from the setup cell: base_model, train_path, ft_steps, device.
import copy

import torch
from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.training import (ComposedLossFunction, EnergyMSELoss, FineTuningStrategy,
                                ForceMSELoss, OptimizerConfig, TrainingStage,
                                default_training_fn)

learning_rate = 5e-4
force_weight = 0.1     # the loss is the energy per atom plus 0.1 times the forces
batch_size = 4         # frames per training step
log_every = 2          # training steps between recorded losses
seed = 1               # the order of the frames

torch.manual_seed(seed)
model = copy.deepcopy(base_model)   # the strategy trains the model it is given; base_model stays as published
ft_loss_history = []


class LossLogHook:
    """Record (step, total loss) every log_every training steps."""
    stage = TrainingStage.AFTER_BATCH
    frequency = log_every

    def __call__(self, ctx, stage):
        ft_loss_history.append((ctx.step_count, ctx.loss.detach().float().item()))


def neighbor_hooks(m):
    hooks = m.make_neighbor_hooks()
    for hook in hooks:
        hook.stage = TrainingStage.BEFORE_FORWARD
    return hooks


dataset = Dataset(AtomicDataZarrReader(str(train_path)), device=device)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

loss_fn = ComposedLossFunction([EnergyMSELoss(per_atom=True), ForceMSELoss()],
                               weights=[1.0, force_weight], normalize_weights=False)
loss_fn.dtype_policy = "target_to_prediction"

strategy = FineTuningStrategy(
    models=model,
    devices=[device],
    trainable_patterns=("main.model.readouts.*",              # the readouts
                        "main.model.interactions.1.*",        # the last interaction block
                        "main.model.products.1.*"),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW,
                                      optimizer_kwargs={"lr": learning_rate}),
    training_fn=default_training_fn,
    loss_fn=loss_fn,
    hooks=[*neighbor_hooks(model), LossLogHook()],
    num_steps=ft_steps,
)
strategy.run(loader)
finetuned_model = strategy.models["main"].eval()
for parameter in finetuned_model.parameters():   # from here on the teacher stays as it is
    parameter.requires_grad_(False)
