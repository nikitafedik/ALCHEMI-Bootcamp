# Try it: the same study without fused stages. One group runs NVT until every trajectory has
# passed its rule or reached its step cap, then NVE for its fixed length, then the next group
# starts. Uses only the inputs of the step's table (trajectories, composed_model, build_stages,
# max_batch_size, max_atoms) and the setup cell.
from nvalchemi.data import Batch


def staged_groups():
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


staged_batches = []
with helpers.time_route(device) as _timer:
    for _indices in staged_groups():
        _batch = Batch.from_data_list([trajectories[i].clone() for i in _indices], device=device)
        _nvt, _nve = build_stages()
        for _stage in (_nvt, _nve):
            for _hook in composed_model.make_neighbor_hooks():
                _stage.register_hook(_hook)
            # A plain engine runs until every trajectory meets its rule in the same step, or its
            # step cap: NVT ends there, NVE (no rule) runs its fixed number of steps.
            with _stage:
                _batch = _stage.run(_batch)
        staged_batches.append(_batch)
staged_seconds = _timer.elapsed_s
