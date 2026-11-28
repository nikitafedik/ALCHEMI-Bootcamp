import os
from nvalchemi.data.datapipes import AtomicDataZarrWriter, AtomicDataZarrReader, Dataset

batch.energy = energy.detach()
batch.forces = forces.detach()
store_path = os.path.join(work_dir, "molecules_append.zarr")
writer = AtomicDataZarrWriter(store_path)
if os.path.exists(store_path):
    writer.append(batch)          # add to the existing store
else:
    writer.write(batch)
dataset = Dataset(AtomicDataZarrReader(store_path), device=batch.device)
loaded_batch = dataset.load_batches([list(range(len(dataset)))])[0]
