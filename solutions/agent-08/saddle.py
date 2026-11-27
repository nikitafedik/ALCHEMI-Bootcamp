import numpy as np
import pandas as pd
import torch

# The Hessian from Toolkit's own model.compute_hessian (the notebook's model, AIMNet2 + Coulomb,
# gives second derivatives). saddle_fd.py is the same test with finite differences of the forces.
# A saddle point is a stationary point (forces near zero) whose Hessian has exactly one
# imaginary mode once translations and rotations are removed. Both tests are needed.
force_tol = 0.005           # eV/A, largest per-atom force norm. A fixed number: it does not
                            # follow the force at which the NEB stopped, or the top image
                            # would pass by construction.
imag_tol = 20.0             # cm^-1; a mode counts as imaginary below -imag_tol
EV_PER_A2_AMU_TO_CM = 521.47  # sqrt(eV / (A^2 amu)) expressed in cm^-1

analysis = relaxed_paths.clone()        # carries the neighbor data of the NEB run
out = model(analysis)
forces = out["forces"].detach()
energy = out["energy"].detach().double().cpu().reshape(-1)
analysis = model.compute_hessian(analysis, strategy="loop")
hess_all = analysis["hessian"].detach().double().cpu()   # packed [sum N_i^2, 3, 3]

counts = relaxed_paths.num_nodes_per_graph.tolist()
pos_all = relaxed_paths.positions.detach().double().cpu()
mass_all = relaxed_paths.atomic_masses.detach().double().cpu().reshape(-1)
force_norm_all = forces.double().cpu().norm(dim=-1)

group_sizes = relaxed_paths.group_layout.num_graphs_per_group.tolist()
rows = []
atom0 = 0
hess0 = 0
image = 0
for path_idx, n_img in enumerate(group_sizes):
    for k in range(n_img):
        n = counts[image]
        H = hess_all[hess0:hess0 + n * n].reshape(n, n, 3, 3).permute(0, 2, 1, 3).reshape(3 * n, 3 * n)
        H = 0.5 * (H + H.T)
        pos = pos_all[atom0:atom0 + n]
        m = mass_all[atom0:atom0 + n]
        # mass-weighted Hessian
        w = m.sqrt().repeat_interleave(3)
        Hm = H / (w[:, None] * w[None, :])
        # translations and rotations in mass-weighted coordinates, orthonormalised by rank
        com = (m[:, None] * pos).sum(0) / m.sum()
        r = pos - com
        vecs = []
        for a in range(3):
            t = torch.zeros(n, 3, dtype=torch.float64)
            t[:, a] = 1.0
            vecs.append(t)
        for a in range(3):
            e = torch.zeros(3, dtype=torch.float64)
            e[a] = 1.0
            vecs.append(torch.cross(e.expand(n, 3), r, dim=1))
        T = torch.stack([(v.reshape(-1) * w) for v in vecs], dim=1)
        U, S, _ = torch.linalg.svd(T, full_matrices=False)
        U = U[:, S > 1e-6 * S.max()]
        P = torch.eye(3 * n, dtype=torch.float64) - U @ U.T
        evals = torch.linalg.eigvalsh(P @ Hm @ P)
        # the projector leaves one zero eigenvalue per removed direction; drop those closest to zero
        n_zero = U.shape[1]
        keep = torch.argsort(evals.abs())[n_zero:]
        vib = evals[torch.sort(keep).values]
        freq = vib.sign() * vib.abs().sqrt() * EV_PER_A2_AMU_TO_CM   # negative = imaginary
        n_imag = int((freq < -imag_tol).sum())
        fmax_img = float(force_norm_all[atom0:atom0 + n].max())
        stationary = fmax_img <= force_tol
        is_saddle = stationary and n_imag == 1
        rows.append({
            "path": reaction_names[path_idx],
            "image": k,
            "energy_eV": float(energy[image]),
            "fmax_eV_per_A": fmax_img,
            "stationary": "yes" if stationary else "no",
            "lowest_freq_cm1": float(freq[0]),
            "second_freq_cm1": float(freq[1]),
            "n_imaginary": n_imag,
            "saddle": "yes" if is_saddle else "no",
        })
        atom0 += n
        hess0 += n * n
        image += 1

saddle_table = pd.DataFrame(rows)
