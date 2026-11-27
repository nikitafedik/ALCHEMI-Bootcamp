# Agent 03, "Ready-made": sample step files

One file per `%load` step of `notebooks/alchemi-agent-03-ready-made.ipynb`
(plan 3.1, 31 cells). Each is a file an agent might write from a learner's
prompt (Toolkit 0.3.0).

| Step | File | Names it leaves |
| --- | --- | --- |
| relax_build | `relax_build.py` | `crystal_batch`, `mace_model`, `start_max_force` |
| relax_run | `relax_run.py` | `relaxed_batch`, `relaxed_max_force`, `force_threshold`, `batch_seconds`, `single_seconds` |
| md_build | `md_build.py` | `md_batch`, `start_temperature` |
| md_run | `md_run.py` | `time_fs`, `crystal_energy` |
| temperature_ladder ("Try it") | `temperature_ladder.py` | `time_fs`, `replica_temperature`, `replica_energy` |
| speed_settings ("Try it") | `speed_settings.py` | `speed_table`, `setting_roles` |

To try them, copy the files into `notebooks/agent03_work/` and run, from
`notebooks/`:

```bash
python -m helpers.agent03 check relax_build
python -m helpers.agent03 check relax_run
python -m helpers.agent03 check md_build
python -m helpers.agent03 check md_run
python -m helpers.agent03 check temperature_ladder
python -m helpers.agent03 check speed_settings
```

A check runs the setup cell, the files of the earlier steps the step needs,
then the step's file, and prints one line per problem. A line that starts
with "unexpected:" is a note; the step still passes.

Three environment variables change the check's settings without touching the
files (used for short runs on a CPU):

- `ALCHEMI_AGENT03_WORK`: another work folder in place of `notebooks/agent03_work`
- `ALCHEMI_AGENT03_N_CRYSTALS`: `n_crystals` of the setup cell (200)
- `ALCHEMI_AGENT03_N_TEMPERATURES`: `n_temperatures` of the setup cell

Notes on the samples:

- `relax_build.py`, `relax_run.py`, `md_build.py` and `md_run.py` are the
  sample files of the build of 25 Nov 2026 (run on a GPU), unchanged.
  `relax_run.py` calls `torch.cuda.synchronize()` and needs a GPU.
- `temperature_ladder.py` uses the saved default crystal,
  `helpers.load_ladder_crystal()` (copper, 64 atoms, `notebooks/data/agent03/`),
  one temperature for each copy from 100 K to 5,000 K spaced by a constant
  ratio, a 1 fs timestep and 300 steps. Each copy starts with velocities at
  its own temperature. It returns the energy per atom; the first row is the
  crystal before the first step. Its step count is provisional until the
  two-minute measurement on the course GPU.
- `speed_settings.py` times 10 L-BFGS steps on 50 crystals picked evenly from
  `crystal_batch` (2,184 atoms of 8,722): a first run on a freshly loaded
  model and the median of three repeats, for each of the four cases, all on
  the same 50 crystals. It prints the size it timed. Without cuEquivariance
  the model needs about 36 GB for all 200 crystals; on the 48 GB L40S that
  run fails with "out of memory" after the earlier steps, even once their
  memory is handed back (measured 27 Nov 2026). The 50 crystals need about
  10 GB. A case that fails keeps its row,
  with the reason in `note`. cuEquivariance needs a CUDA GPU, so this file
  runs on the course GPU only.
