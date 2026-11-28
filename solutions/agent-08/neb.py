import csv

import torch
from nvalchemi.dynamics.mep import NEB, ClimbingImageConfig

# --- relax every path in one NEB call, with a climbing image ---
neb_log = work_dir / "neb_log.csv"
neb = NEB(
    model=model,
    spring=0.1,
    fmax=neb_settings["fmax"],          # eV/A, largest per-atom force norm, where each path stops
    n_steps=neb_settings["n_steps"],    # step budget of the whole run
    climbing=ClimbingImageConfig(
        mode="after_regular",
        regular_fmax=neb_settings["regular_fmax"],  # eV/A, where climbing starts
    ),
    diagnostics_log_path=neb_log,
)
# the run changes the batch it is given, so it gets a copy: paths stays the straight line
relaxed_paths = neb.run(paths.clone())

# --- energy of every image, in eV (one value per image, paths in order) ---
# A clone keeps relaxed_paths unchanged; it carries the neighbor data of the run.
outputs = model(relaxed_paths.clone())
path_energy = outputs["energy"].detach().reshape(-1)

# --- forward barrier per path, in eV: highest image between the ends minus the first image ---
images_per_path = relaxed_paths.group_layout.num_graphs_per_group.tolist()
barriers = []
start = 0
for count in images_per_path:
    energies = path_energy[start:start + count]
    barriers.append(energies[1:-1].max() - energies[0])
    start += count
barriers = torch.stack(barriers)

# --- converged: the last row of each path in the log has fmax (largest per-atom force norm
# on the band, eV/A) at or below the force at which the run stops a path. A path that ran out
# of steps has a barrier too; it is an estimate, and converged says so.
last_row = {}
with open(neb_log) as f:
    for row in csv.DictReader(f):
        last_row[int(float(row["group_idx"]))] = row
last_force = torch.tensor([float(last_row[k]["fmax"]) for k in range(len(images_per_path))])
converged = last_force <= neb_settings["fmax"]
