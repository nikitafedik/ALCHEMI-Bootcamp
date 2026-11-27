# Agent 03, "Ready-made": default prompts

The typical learner prompt of each step. Each is written from the step's
text, table and Hints alone. The result is drawn by the helper the step
names; the agent returns the named variables and draws nothing. The form of
each variable is in the helper's docstring (`help(helpers.<name>)`).

| Step | File | Helper that shows the result |
| --- | --- | --- |
| relax_build | `agent03_work/relax_build.py` | `helpers.show_crystal_grid(crystal_batch, start_max_force)` |
| relax_run | `agent03_work/relax_run.py` | `helpers.plot_force_histograms(start_max_force, relaxed_max_force, force_threshold)`, `helpers.plot_batch_speedup(batch_seconds, single_seconds, len(start_max_force))` |
| md_build | `agent03_work/md_build.py` | `helpers.plot_start_temperature(start_temperature)` |
| md_run | `agent03_work/md_run.py` | `helpers.plot_energy_drift(time_fs, crystal_energy, md_batch)` |
| temperature_ladder | `agent03_work/temperature_ladder.py` | `helpers.plot_temperature_ladder(time_fs, replica_temperature, replica_energy)` |
| speed_settings | `agent03_work/speed_settings.py` | `helpers.show_speed_table(speed_table, setting_roles)` |

## The six prompts

Saved again on 26 Nov 2026 (revision F) from each step's text, table and
Hints alone, after the text was final. Nothing a step does not say is in
its prompt. Not trialled in this form; the earlier trialled prompts are in
the records below.

relax_build:

> Write agent03_work/relax_build.py: load the course crystals into one batch
> with helpers.load_crystals(n_crystals), load the built-in MACE-MPA-0
> wrapper, and report the largest force on each crystal. Give me
> crystal_batch, mace_model and start_max_force.

relax_run:

> Write agent03_work/relax_run.py: relax every crystal in crystal_batch with
> mace_model, and time the batched run beside one-by-one relaxation. Give me
> relaxed_batch, relaxed_max_force, force_threshold, batch_seconds and
> single_seconds.

md_build:

> Write agent03_work/md_build.py: give the relaxed crystals in relaxed_batch
> starting velocities for an MD run with mace_model. Give me md_batch and
> start_temperature. Before the MD starts: what state is relaxed_batch in,
> and will the atoms move?

md_run:

> Write agent03_work/md_run.py: run md_batch with mace_model and no
> thermostat, and save the total energy of each crystal over time. Give me
> time_fs and crystal_energy.

temperature_ladder ("Try it"):

> Write agent03_work/temperature_ladder.py: with a thermostat, run copies of
> one crystal in a single batch, each held at a different temperature, from
> cold to far too hot, and record the energy of each copy along the run. Use
> the crystal from helpers.load_ladder_crystal(), with mace_model and
> n_temperatures. Every copy has the same positions and atomic_numbers; the
> temperature is one value per copy, shape (n_temperatures,), in K. Give me
> time_fs, replica_temperature and replica_energy.

speed_settings ("Try it"):

> Write agent03_work/speed_settings.py. MACE has two speed settings in
> Toolkit: cuEquivariance and model compilation. Time MACE-MPA-0 four ways
> on crystal_batch, with neither, with each one, and with both, and explain
> what each setting is for. Give me speed_table and setting_roles.
> helpers.show_speed_table(...) shows them: speed_table is a pandas
> DataFrame, one row per case, with the columns cueq and compile (True or
> False), first_call_s and repeat_s (s); setting_roles is a dict with a text
> for cueq and one for compile.

## Records of the earlier, trialled prompts

- relax_build to md_run: written by a fresh mid-tier agent from the step
  tables, trial of 25 Nov 2026 (chain W, GPU; all four passed):
  `notes/builder-03/chainW/prompts.md`,
  `notes/evidence/lesson-02/l02-6--20261125-1949--trial-03W.md` and the
  l02-9, l02-16, l02-19 files of the same trial.
- temperature_ladder: `notes/evidence/lesson-02/l02-tl-1--20261126-0300--trial-ladder-sonnet.md`
  and the validation of 26 Nov (`notes/evidence/lesson-02/validation/`).
- speed_settings: `notes/evidence/lesson-02/l02-speed--20261126-0300--trial-speed-sonnet.md`.
