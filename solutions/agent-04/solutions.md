# Sample solutions for notebook 04, "Made to order"

One file per step, as a learner's agent could write it. They are not shown in
the notebook. Each passes `python -m helpers.agent04 check <step>` when it sits
in `notebooks/agent04_work/` with the files of the earlier steps.

| Step | File | Needs first | What it does |
| --- | --- | --- | --- |
| `compose_build` | `compose_build.py` | none | loads AIMNet2 as it comes, puts the 90 scan structures in one batch with their total charges, and returns the bare model's energy (90, 1), eV |
| `compose_run` | `compose_run.py` | `compose_build` | joins AIMNet2 and `helpers.DirectCoulombAdapter` in one autograd group (charges wired to `partial_charges`), adds D3 as its own group, and returns the composed energies of the scans (90, 1) and the 205 NCIA pairs (615, 1), eV |
| `no_d3` | `no_d3.py` | `compose_build`, `compose_run` | the same pipeline without the D3 group, on the same batch; energy (90, 1), eV |
| `conformers` | `conformers.py` | `compose_build`, `compose_run` | relaxes all 139 saved dipeptide starts in one batch with `FIRE2` to `helpers.relax_settings()["fmax"]` and returns their energies (139,), eV |

`broken/` holds twelve deliberately broken variants, named `<step>__<what>.py`.
`notebooks/tests/test_agent04.py` runs every step check on the samples and on
each variant (`ALCHEMI_RUN_SAMPLES=1`): a variant that breaks a promise prints
one plain line and fails; a variant that only departs from the usual relation
prints an "unexpected" line after OK and still passes.

The sample of `no_d3` needs no workaround for the neighbor list:
`helpers.DirectCoulombAdapter` reads the same matrix list as AIMNet2.

`fine_control/compose_run__cutoff_12.py` is the lesson's Fine control example
("a 12 Å cutoff for the coulomb term"); it passes `compose_run`.
`prompts.md` holds the default prompt of each step.
