# Step 1: NVT at 300 K until the temperature has settled, then NVE for a fixed number of steps,
# fused in one batch per group, at the capacity of the setup cell.
import torch
from nvalchemi.data import Batch
from nvalchemi.dynamics import KB_EV, NVE, ConvergenceHook, DynamicsStage, NVTLangevin

SAMPLE_EVERY = 1  # MD steps between two samples of a stopping rule: every step, as the rules say


class StepCount:
    """Counts the MD steps each trajectory takes in one stage, by simulation_id."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, total):
        self.steps = torch.zeros(total, dtype=torch.long, device=device)  # MD steps in this stage

    def count(self, ctx):
        """Add this step; return the ids of the batch's trajectories and the active mask."""
        count = ctx.batch.num_graphs
        ids = ctx.batch.simulation_id.view(-1)[:count].long()
        active = ctx.active_graph_mask
        if active is None:
            active = torch.ones(count, dtype=torch.bool, device=ids.device)
        self.steps[ids] += active.long()
        return ids, active

    def __call__(self, ctx, stage):
        self.count(ctx)


class WindowRule(StepCount):
    """The stopping rule of NVT: keeps the last window of a value for every trajectory, by
    simulation_id, and writes the gap the stage's ConvergenceHook reads into a batch field.
    The gap is inf until a trajectory has a full window in this stage. `passed` remembers,
    per trajectory, that the gap has been within the threshold once."""

    def __init__(self, field, value, gap, threshold, samples, total):
        super().__init__(total)
        self.field, self.value, self.gap, self.threshold = field, value, gap, threshold
        self.passed = torch.zeros(total, dtype=torch.bool, device=device)
        self.history = torch.zeros(total, samples, device=device)
        self.filled = torch.zeros(total, dtype=torch.long, device=device)

    def __call__(self, ctx, stage):
        batch = ctx.batch
        count = batch.num_graphs
        ids, active = self.count(ctx)
        sample = active & (self.steps[ids] % SAMPLE_EVERY == 0)
        rows = ids[sample]
        samples = self.history.shape[1]
        self.history[rows, self.filled[rows] % samples] = self.value(batch)[sample].to(self.history.dtype)
        self.filled[rows] += 1
        gap = self.gap(self.history[rows])
        gap = torch.where(self.filled[rows] >= samples, gap, torch.full_like(gap, float("inf")))
        self.passed[rows] |= gap <= self.threshold
        if getattr(batch, self.field, None) is None:  # a batch built without with_rule_fields
            batch[self.field] = torch.full((count, 1), float("inf"), device=batch.device)
        target = getattr(batch, self.field).view(-1)
        target[:count][sample] = gap.to(target.dtype)


def _kinetic_energy(batch):
    per_atom = 0.5 * batch.atomic_masses.view(-1) * (batch.velocities**2).sum(-1)
    total = torch.zeros(batch.num_graphs, dtype=per_atom.dtype, device=per_atom.device)
    return total.index_add_(0, batch.batch_idx.long(), per_atom)  # eV


def _temperature(batch):
    return 2.0 * _kinetic_energy(batch) / (3.0 * batch.num_nodes_per_graph * KB_EV)  # K


def build_stages():
    """A fresh NVT and NVE pair from stage_settings.

    NVT ends for a trajectory once its mean temperature over the last window is within the
    tolerance of the target, or at its step cap nvt_steps. NVE has no stopping rule: as the last
    stage it runs nve_steps MD steps for every trajectory. The batch a stage runs on needs the
    field nvt_gap (see with_rule_fields). nvt.rule and nve.counter count the MD steps of each
    trajectory in each stage. The NVT rule adds its field nvt_gap to a batch that lacks it."""
    s = stage_settings
    samples, total = s["window"] // SAMPLE_EVERY, len(trajectories)
    nvt_rule = WindowRule("nvt_gap", _temperature,
                          lambda h: (h.mean(1) - s["temperature"]).abs(), s["tolerance"], samples, total)
    nvt = NVTLangevin(
        model=composed_model, dt=s["dt"], temperature=s["temperature"], friction=s["friction"], random_seed=0,
        n_steps=s["nvt_steps"],
        convergence_hook=ConvergenceHook(criteria=[{"key": "nvt_gap", "threshold": s["tolerance"]}]))
    nve = NVE(model=composed_model, dt=s["dt"], n_steps=s["nve_steps"])  # fixed length, no rule
    nve_counter = StepCount(total)
    nvt.register_hook(nvt_rule)
    nve.register_hook(nve_counter)
    nvt.rule, nve.counter = nvt_rule, nve_counter
    return nvt, nve


RULE_FIELDS = {"nvt_gap": ("system", float("inf"))}


def with_rule_fields(batch):
    """Add the field the NVT stopping rule writes to."""
    for name in RULE_FIELDS:
        batch[name] = torch.full((batch.num_graphs, 1), float("inf"), device=batch.device)
    return batch


def capacity_groups():
    """Indices of the trajectories, in order, in groups within max_batch_size and max_atoms."""
    groups, current, atoms = [], [], 0
    for index, item in enumerate(trajectories):
        size = int(item.num_nodes)
        if current and (len(current) >= max_batch_size or atoms + size > max_atoms):
            groups.append(current)
            current, atoms = [], 0
        current.append(index)
        atoms += size
    if current:
        groups.append(current)
    return groups


def fresh_batch(indices):
    """A copy of the chosen trajectories as one Batch, ready for the two stages."""
    return with_rule_fields(
        Batch.from_data_list([trajectories[i].clone() for i in indices], device=device))


fused_batches = []
_nvt_steps = torch.zeros(len(trajectories), dtype=torch.long, device=device)
_nve_steps = torch.zeros_like(_nvt_steps)
with helpers.time_route(device) as _timer:
    for _indices in capacity_groups():
        _nvt, _nve = build_stages()
        _fused = _nvt + _nve
        for _hook in composed_model.make_neighbor_hooks():
            _fused.register_hook(_hook)
        with _fused:
            fused_batches.append(_fused.run(
                fresh_batch(_indices),
                n_steps=stage_settings["nvt_steps"] + stage_settings["nve_steps"] + 1))
        _nvt_steps += _nvt.rule.steps
        _nve_steps += _nve.counter.steps
fused_seconds = _timer.elapsed_s

# One row per trajectory, in simulation_id order: the MD step at which it moved from NVT to NVE,
# and the MD step at which it finished, both counted from its own first step.
stage_trace = torch.stack([_nvt_steps, _nvt_steps + _nve_steps], dim=1).cpu()
