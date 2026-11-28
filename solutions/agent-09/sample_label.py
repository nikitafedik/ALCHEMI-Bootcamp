# Sample and label: NVT Langevin MD of the liquid cells in starts with the teacher (finetuned_model),
# all in one batch, each cell at the temperature of its DFT run; a frame of every cell is saved to a
# Zarr store every save_every steps. Then the teacher labels every frame into a new store.
# Gives: labeled_path.
# Names from the setup cell and the fine-tune step: starts, finetuned_model, md_steps, device, work_dir.
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset
from nvalchemi.dynamics import NVTLangevin, initialize_velocities
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.sinks import ZarrData
from nvalchemi.training.distillation import InProcessTeacherScorer, label_dataset

timestep = 1.0        # fs
friction = 0.01       # 1/fs
save_every = 10       # MD steps between saved frames
seed = 0

finetuned_model.eval()
batch = Batch.from_data_list([AtomicData.from_atoms(atoms) for atoms in starts]).to(device)
temperature = torch.tensor([atoms.info["temperature_K"] for atoms in starts],
                           dtype=batch.positions.dtype, device=device)   # K, one per cell
initialize_velocities(batch.velocities, batch.atomic_masses, temperature,
                      batch.batch_idx.to(torch.int32), random_seed=seed)

# A store is written once: take a new name on every run of this file.
run_id = 0
while (work_dir / f"labeled_{run_id:03d}.zarr").exists():
    run_id += 1
frames_path = work_dir / f"frames_{run_id:03d}.zarr"
labeled_path = work_dir / f"labeled_{run_id:03d}.zarr"

sink = ZarrData(frames_path, capacity=len(starts) * (md_steps // save_every + 1))
md = NVTLangevin(model=finetuned_model, dt=timestep, temperature=temperature,
                 friction=friction, random_seed=seed, n_steps=md_steps,
                 device_type=device.type,
                 hooks=[SnapshotHook(sink=sink, frequency=save_every)])
for hook in finetuned_model.make_neighbor_hooks():
    md.register_hook(hook)
with md:
    batch = md.run(batch)

frames = Dataset(AtomicDataZarrReader(str(frames_path)), device=device)
scorer = InProcessTeacherScorer(finetuned_model, ("energy", "forces"))
n_labeled = label_dataset(frames, scorer, labeled_path, batch_size=8,
                          device=device, resume=False)
