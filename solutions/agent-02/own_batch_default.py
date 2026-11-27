import os
import shutil

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import AtomicDataZarrWriter
from nvalchemi.neighbors import compute_neighbors

own_batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in helpers.load_default_set()]).to(batch.device)
compute_neighbors(own_batch, config=aimnet_model.model_config.neighbor_config)
out = aimnet_model(own_batch)
own_energy, own_forces = out["energy"], out["forces"]
own_batch.energy = own_energy.detach()
own_batch.forces = own_forces.detach()
own_store = os.path.join(work_dir, "own.zarr")
shutil.rmtree(own_store, ignore_errors=True)
AtomicDataZarrWriter(own_store).write(own_batch)
