# Agent 08, "Paths of least resistance, batched": sample step files

One file per step of the notebook. Each is one way to do the step; a learner's
agent will write something different. Every file passes its step check on CPU
with Toolkit 0.3.0:

```
cp solutions/agent-08/{paths,neb,saddle}.py notebooks/agent08_work/
cd notebooks
python -m helpers.agent08 check paths      # then neb, saddle
```

| Step | File | Names it leaves | Contract the check holds |
| --- | --- | --- | --- |
| Example: build the paths | `paths.py` | `paths` | one `Batch`, one group of images per reaction, `neb_settings["num_images"]` images per path, the ends as the loader gave them; `energy`, `forces` and `velocities` attached |
| Try it: relax the paths and read the barriers | `neb.py` | `relaxed_paths`, `neb_log`, `path_energy`, `barriers`, `converged` | every path in one `Batch`, from one `NEB.run` call with a climbing image; `neb_log` is the diagnostics CSV of that run; the file can be run again; energy of every image in eV; one forward barrier per path in eV; one true or false per path, the path's last force in the log against the force at which the run stops a path. A path that is not converged passes and is reported as "unexpected" |
| Try it: saddle point or not | `saddle.py` | `saddle_table` | one row per image, with a column that names the path, a column with the image number, a column of whole numbers that counts the imaginary modes, a yes or no and the numbers behind it; no end of a path is a saddle point; a yes only where the largest force on an atom is at or below 0.005 eV/Å, the image is a top of its path and the table counts exactly one imaginary mode; the count of every image agrees with the model's Hessian once the rigid translations and rotations are projected out. A top image answered "no" passes and is reported as "unexpected" |

`model` is AIMNet2 with its Coulomb term; D3 is left out (the educator's
decision for 08, 26 Nov). `saddle.py` takes the Hessian from Toolkit's own
`model.compute_hessian`. `saddle_fd.py` is the same test with the Hessian from
central differences of the model's forces; it also works on the variant with
a D3 term (`load_model(device, d3=True)`), which `compute_hessian` refuses. The
check accepts both, and recounts the imaginary modes of every image from the
model's projected Hessian. Results with the notebook's model (CPU): barriers 5.57, 1.73,
51.88 and 4.58 kcal/mol, stops at steps 74, 97, 248 and 94; one imaginary
mode at each top (image 3), -1003, -244, -2008 and -1264 cm^-1.

`prompts.md` holds the default prompt of each step, the typical learner prompt
that was trialled.

Settings: all files read `neb_settings` from the setup cell (8 images per
path; each path stops at 0.003 eV/Å; climbing starts at 0.03 eV/Å; 600 steps
for the run). The saddle test uses its own force limit, 0.005 eV/Å, which does
not follow the NEB's stopping force.
