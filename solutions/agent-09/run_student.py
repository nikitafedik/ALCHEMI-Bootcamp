# Run the student: NVT Langevin MD of the liquid cells in test_starts with student_model, all in one
# batch, each cell at the temperature of its DFT run, for test_steps steps; a frame of every cell is
# saved to a new Zarr store every save_every steps, and a StabilityMonitor watches the run.
# Gives: student_frames_path, stability.
# Names from earlier: test_starts, student_model, test_steps, device, work_dir.
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin, initialize_velocities
from nvalchemi.dynamics.hooks import SnapshotHook, StabilityMonitor
from nvalchemi.dynamics.sinks import ZarrData

timestep = 1.0        # fs, as in sample_label.py
friction = 0.01       # 1/fs
save_every = 10       # MD steps between saved frames
seed = 1              # sample_label.py used 0

student_model.eval()
batch = Batch.from_data_list([AtomicData.from_atoms(atoms) for atoms in test_starts]).to(device)
temperature = torch.tensor([atoms.info["temperature_K"] for atoms in test_starts],
                           dtype=batch.positions.dtype, device=device)   # K, one per cell
initialize_velocities(batch.velocities, batch.atomic_masses, temperature,
                      batch.batch_idx.to(torch.int32), random_seed=seed)

# A store is written once: take the first free name.
run_id = 0
while (work_dir / f"student_frames_{run_id:03d}.zarr").exists():
    run_id += 1
student_frames_path = work_dir / f"student_frames_{run_id:03d}.zarr"
sink = ZarrData(student_frames_path, capacity=len(test_starts) * (test_steps // save_every + 1))

monitor = StabilityMonitor(frequency=save_every, timestep_fs=timestep)
md = NVTLangevin(model=student_model, dt=timestep, temperature=temperature,
                 friction=friction, random_seed=seed, n_steps=test_steps,
                 device_type=device.type,
                 hooks=[SnapshotHook(sink=sink, frequency=save_every), monitor])
for hook in student_model.make_neighbor_hooks():
    md.register_hook(hook)
with md:
    batch = md.run(batch)
stability = monitor.metrics()
