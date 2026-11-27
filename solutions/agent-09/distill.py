# Distill: train the small student on the teacher labels stored at labeled_path.
# Gives: student_model, loss_history ((step, loss) pairs).
# Names from earlier: helpers, labeled_path, finetuned_model (the teacher), train_steps, device.
import torch
from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.training import (EnergyMSELoss, ForceMSELoss, OptimizerConfig,
                                TrainingStage)
from nvalchemi.training.distillation import DistillationStrategy

learning_rate = 1e-2
batch_size = 4       # structures per training step
log_every = 5        # training steps between recorded losses
seed = 1             # the order of the frames

torch.manual_seed(seed)
student_model = helpers.make_student(labeled_path, device)
loss_history = []


class LossLogHook:
    """Record (step, total loss) every log_every training steps."""
    stage = TrainingStage.AFTER_BATCH
    frequency = log_every

    def __call__(self, ctx, stage):
        loss_history.append((ctx.step_count, ctx.loss.detach().float().item()))


def neighbor_hooks(model):
    hooks = model.make_neighbor_hooks()
    for hook in hooks:
        hook.stage = TrainingStage.BEFORE_FORWARD
    return hooks


dataset = Dataset(AtomicDataZarrReader(str(labeled_path)), device=device)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

loss_fn = (EnergyMSELoss(target_key="teacher_energy", per_atom=True)
           + ForceMSELoss(target_key="teacher_forces"))
loss_fn.dtype_policy = "target_to_prediction"

strategy = DistillationStrategy(
    models={"student": student_model, "teacher": finetuned_model},   # both entries are required
    devices=[device],
    optimizer_configs={"student": [OptimizerConfig(
        optimizer_cls=torch.optim.AdamW, optimizer_kwargs={"lr": learning_rate})]},
    loss_fn=loss_fn,
    hooks=[*neighbor_hooks(student_model), LossLogHook()],
    num_steps=train_steps,
)
strategy.run(loader)
student_model = strategy.models["student"].eval()
