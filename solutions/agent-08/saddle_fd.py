import numpy as np
import pandas as pd
import torch

# A saddle point is a stationary point (forces near zero) whose Hessian has exactly one
# imaginary mode once translations and rotations are removed. Both tests are needed.
# Variant: the Hessian from central differences of the model's forces instead of
# model.compute_hessian (saddle.py). It works on any model that gives forces, also on one
# with a D3 term, which has no second derivatives in Toolkit 0.3.0.
force_tol = 0.005           # eV/A, largest per-atom force norm. A fixed number: it does not follow the
                            # force at which the NEB stopped, or the top image would pass by construction.
imag_tol = 20.0             # cm^-1; a mode counts as imaginary below -imag_tol
step = 0.01                 # A, displacement of the finite differences
EV_PER_A2_AMU_TO_CM = 521.47  # sqrt(eV / (A^2 amu)) expressed in cm^-1

counts = relaxed_paths.num_nodes_per_graph.tolist()
starts = np.concatenate([[0], np.cumsum(counts)[:-1]])


def forces_at(positions):
    """Forces of every image (eV/A) at the given positions; the batch keeps its neighbor data."""
    work = relaxed_paths.clone()
    work.positions.copy_(positions)
    result = model(work)
    return result["forces"].detach().double().cpu(), result["energy"].detach().double().cpu().reshape(-1)


base = relaxed_paths.positions.detach().clone()
forces, energy = forces_at(base)
force_norm_all = forces.norm(dim=-1)

# Displace atom a of every image along x, y, z at once: 6 force calls per atom of the largest molecule.
hessians = [torch.zeros(3 * n, 3 * n, dtype=torch.float64) for n in counts]
for a in range(max(counts)):
    for c in range(3):
        moved = []
        for sign in (1.0, -1.0):
            positions = base.clone()
            for s, n in zip(starts, counts):
                if a < n:
                    positions[s + a, c] += sign * step
            moved.append(forces_at(positions)[0])
        for i, (s, n) in enumerate(zip(starts, counts)):
            if a < n:
                hessians[i][3 * a + c] = -(moved[0][s:s + n] - moved[1][s:s + n]).reshape(-1) / (2 * step)

pos_all = base.double().cpu()
mass_all = relaxed_paths.atomic_masses.detach().double().cpu().reshape(-1)
group_sizes = relaxed_paths.group_layout.num_graphs_per_group.tolist()
rows = []
image = 0
for path_idx, n_img in enumerate(group_sizes):
    for k in range(n_img):
        n, s = counts[image], int(starts[image])
        H = 0.5 * (hessians[image] + hessians[image].T)
        pos, m = pos_all[s:s + n], mass_all[s:s + n]
        w = m.sqrt().repeat_interleave(3)
        Hm = H / (w[:, None] * w[None, :])
        # translations and rotations in mass-weighted coordinates, orthonormalised by rank
        r = pos - (m[:, None] * pos).sum(0) / m.sum()
        vecs = []
        for axis in range(3):
            t = torch.zeros(n, 3, dtype=torch.float64)
            t[:, axis] = 1.0
            vecs.append(t)
            e = torch.zeros(3, dtype=torch.float64)
            e[axis] = 1.0
            vecs.append(torch.cross(e.expand(n, 3), r, dim=1))
        T = torch.stack([(v.reshape(-1) * w) for v in vecs], dim=1)
        U, S, _ = torch.linalg.svd(T, full_matrices=False)
        U = U[:, S > 1e-6 * S.max()]
        P = torch.eye(3 * n, dtype=torch.float64) - U @ U.T
        evals = torch.linalg.eigvalsh(P @ Hm @ P)
        keep = torch.argsort(evals.abs())[U.shape[1]:]
        vib = evals[torch.sort(keep).values]
        freq = vib.sign() * vib.abs().sqrt() * EV_PER_A2_AMU_TO_CM   # negative = imaginary
        n_imag = int((freq < -imag_tol).sum())
        fmax_img = float(force_norm_all[s:s + n].max())
        stationary = fmax_img <= force_tol
        rows.append({
            "path": reaction_names[path_idx],
            "image": k,
            "energy_eV": float(energy[image]),
            "fmax_eV_per_A": fmax_img,
            "stationary": "yes" if stationary else "no",
            "lowest_freq_cm1": float(freq[0]),
            "second_freq_cm1": float(freq[1]),
            "n_imaginary": n_imag,
            "saddle": "yes" if stationary and n_imag == 1 else "no",
        })
        image += 1

saddle_table = pd.DataFrame(rows)
