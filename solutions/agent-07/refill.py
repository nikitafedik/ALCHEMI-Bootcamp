# Step 2: the same two stages with refill: a waiting list, the active batch, a store.
# The timer covers the whole route, from the waiting list to the last finished trajectory,
# as it does in fused.py and staged.py.
from nvalchemi.data import Batch
from nvalchemi.data.datapipes import InMemoryDataset
from nvalchemi.dynamics import FusedStage, SizeAwareSampler

with helpers.time_route(device) as _timer:
    # The waiting list: a copy of every trajectory, with the field the NVT rule writes to.
    waiting = InMemoryDataset(with_rule_fields(
        Batch.from_data_list([item.clone() for item in trajectories], device=device)))
    sampler = SizeAwareSampler(waiting, max_batch_size=max_batch_size, max_atoms=max_atoms)

    # The store for finished trajectories.
    completed = helpers.CudaCompletionSink(capacity=len(trajectories), device=device)

    _nvt, _nve = build_stages()
    _refill = FusedStage(
        sub_stages=[(0, _nvt), (1, _nve)],
        sampler=sampler,          # the waiting list
        sinks=[completed],        # the store
        refill_frequency=5,       # MD steps between refill checks
        hooks=composed_model.make_neighbor_hooks(),
    )
    _left = _refill.run()         # None once the waiting list and the active batch are empty
inflight_seconds = _timer.elapsed_s
