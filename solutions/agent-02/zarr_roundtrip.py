import os, shutil
from nvalchemi.data.datapipes import AtomicDataZarrWriter, AtomicDataZarrReader, Dataset

batch.energy = energy.detach()
batch.forces = forces.detach()
store_path = os.path.join(work_dir, "molecules.zarr")
shutil.rmtree(store_path, ignore_errors=True)   # allows a rerun
AtomicDataZarrWriter(store_path).write(batch)
dataset = Dataset(AtomicDataZarrReader(store_path), device=batch.device)
loaded_batch = dataset.load_batches([list(range(len(dataset)))])[0]
