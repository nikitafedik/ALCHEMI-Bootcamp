# Agent 06, "Hooked in at every step": sample step files

One file per `%load` step of
`notebooks/alchemi-agent-06-hooked-in-at-every-step.ipynb` (Toolkit 0.3.0).
Each file passes its step check. The default prompt of each step is in
`prompts.md`.

| Step | File | Names it leaves |
| --- | --- | --- |
| interaction ("Try it") | `interaction.py` | `saved_step`, `interaction_energy`, `start_interaction_energy`, `plain_store`, `plain_log` |
| separation_hook ("Try it") | `separation_hook.py` | `separation`, `nearest_contact` |
| freeze ("Try it") | `freeze.py` | `frozen_mask`, `frozen_store` |
| retire ("Try it") | `retire.py` | `stop_step`, `final_status`, `retire_store` |

To try them, copy the files into `notebooks/agent06_work/` and run, from
`notebooks/`:

```bash
python -m helpers.agent06 check interaction
python -m helpers.agent06 check separation_hook
python -m helpers.agent06 check freeze
python -m helpers.agent06 check retire
```

A check runs the setup cell, the earlier files, then the step's file twice.
A third argument names another copy of the work folder, which must be called
`agent06_work`: `python -m helpers.agent06 check retire /some/path/agent06_work`.
A line that starts with "unexpected:" follows the OK line and does not fail
the step.

Notes on the samples:

- `interaction.py` records the interaction energy in kcal/mol with a hook of
  its own; the check also takes eV, kJ/mol and Hartree and reads the unit
  from the starting values.
- All four runs start from rest with `random_seed=0`, so every later file
  follows the run of `interaction.py`: the distances of `separation_hook.py`
  sit on its `saved_step`, and `retire.py` follows the plotted trajectory.
- `retire.py` uses a limit of 3.7 Å, read off the nearest-contact peaks of
  the section 2 run on a CPU (3.36, 3.76 and 3.09 Å); propyne with methyl
  azide is retired at step 490. Another device or seed gives other peaks and
  needs another limit.
- `stop_step` holds -1 for a complex that ran to the end.
- `plain_run.py` is the plain run alone: saved frames and a log, without the
  interaction energy. No step of the notebook loads it and it has no check.
