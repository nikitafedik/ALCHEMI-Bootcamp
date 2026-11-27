# Agent 06, "Hooked in at every step": default prompts

The typical learner prompt of each step, written from the step's text, table
and Hints alone. An earlier wording was tried with a fresh mid-tier agent
that had only the prompt, the session lines and the official
`nvalchemi-toolkit` skill; the present wording has not been tried. The result
is drawn by the helper the step names; the agent returns the named variables
and draws nothing.

| Step | File | Helper that shows the result |
| --- | --- | --- |
| interaction | `agent06_work/interaction.py` | `helpers.plot_interaction(saved_step, interaction_energy, start_interaction_energy)`, `helpers.show_run(plain_store, plain_log)` |
| separation_hook | `agent06_work/separation_hook.py` | `helpers.plot_separation(saved_step, interaction_energy, separation, nearest_contact)` |
| freeze | `agent06_work/freeze.py` | `helpers.plot_displacement(plain_store, frozen_store, frozen_mask)` |
| retire | `agent06_work/retire.py` | `helpers.show_retired(stop_step, final_status, retire_store)` |

## interaction

> Run the three complexes in one batch with run_settings. Use built-in hooks
> to save the frames and to write a log. Record the interaction energy of
> each complex along the run, every 10 steps, in kcal/mol, and set it beside
> the value at the starting structure. In each complex the first
> fragment_a_sizes atoms are fragment A, the rest are fragment B. Use
> complexes, composed_model, run_settings, fragment_a_sizes. Give me
> saved_step, interaction_energy, start_interaction_energy, plain_store,
> plain_log in agent06_work/interaction.py

## separation_hook

> Repeat the plain run and, with a hook of your own, record two distances for
> each complex on the steps of the interaction energy: the separation between
> the centers of mass of its two fragments, and the nearest contact, the
> shortest distance between an atom of one fragment and an atom of the other.
> Use complexes, composed_model, run_settings, fragment_a_sizes. Give me
> separation, nearest_contact in agent06_work/separation_hook.py

## freeze

> - repeat the plain run with fragment A of each complex frozen
> - input: complexes, composed_model, run_settings, fragment_a_sizes
> - give me frozen_mask and frozen_store
> - write agent06_work/freeze.py

## retire

The limit is the learner's, read off their plot of the nearest contacts.
With the default complexes and seed 0 the plot shows 3.36, 3.76 and 3.09 Å,
so a limit from 3.37 to 3.75 Å retires one complex; the prompt uses 3.6 Å,
the sample file 3.7 Å.

> My plot from separation_hook.py shows the largest nearest contact of the
> three complexes: 3.36, 3.76 and 3.09 Å. Repeat the plain run and retire a
> complex once its nearest contact is above 3.6 Å, so that it stops while the
> others run on to the end. Start the run the same way as
> the run of my plot. Use complexes, composed_model, run_settings,
> fragment_a_sizes. Give me stop_step, final_status, retire_store in
> agent06_work/retire.py
