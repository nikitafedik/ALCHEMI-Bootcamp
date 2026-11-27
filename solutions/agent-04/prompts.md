# Agent 04, Made to order: the default prompt of each step

The typical learner prompt for each step, written from the step's text,
table and Hints alone. The prompts were tried with a fresh mid-tier agent
that had only the prompt, the session lines and the skill path. The
conformer prompt changed after that, when the step's Hint began to give the
return format, and has not been tried in its present form.

In every step the agent returns variables only and draws nothing. The
notebook's result cell calls the helper that shows them, and the step check,
`python -m helpers.agent04 check <step>`, tells the agent the shapes and
units.

## compose_build ("Example: AIMNet2 as it comes")

The notebook prints this prompt (sample: `compose_build.py`):

```text
use the nvalchemi-toolkit skill
- show me how well AIMNet2 alone describes pairs of molecules
- model: AIMNet2, through toolkit's built-in wrapper, as it comes
- structures: helpers.load_nci_scans() gives a list of ASE atoms and scan_labels; each scan point has the pair AB and the molecules A and B
- keep the total charge of each structure
- first tell me which terms the built-in wrapper leaves out
- write agent04_work/compose_build.py
- give me aimnet_model: the loaded model
- give me scan_batch: all structures in one batch on the gpu
- give me scan_labels: unchanged
- give me base_energy: from aimnet_model, shape (n_structures, 1), in eV
```

Returns `aimnet_model`, `scan_batch` (90 structures, one batch),
`scan_labels` (30 entries, unchanged) and `base_energy` ((90, 1), eV).
Shown by `helpers.plot_interaction_curves(scan_labels, base=base_energy)`.

## compose_run ("Example: add D3 and electrostatics")

The notebook prints this prompt (sample: `compose_run.py`):

```text
- now add what AIMNet2 leaves out: D3 dispersion and electrostatics
- dispersion: toolkit's D3 wrapper; helpers.d3_settings() has the parameters that belong to this AIMNet2 checkpoint
- electrostatics: helpers.DirectCoulombAdapter, a finite coulomb term for isolated molecules; it must use the charges AIMNet2 predicts
- forces have to stay consistent with the summed energy
- first tell me how the charges get from AIMNet2 to the coulomb term
- write agent04_work/compose_run.py
- give me composed_model: one model with all three terms
- give me composed_energy: for scan_batch, shape (n_structures, 1), in eV
- give me ncia_energy: from composed_model, for the structures of helpers.load_ncia() in the loader's order, shape (n_ncia_structures, 1), in eV
- no plots, no loop over structures
```

A short prompt in the input and output form of the open tasks also works
(tried once: it passed on the second attempt):

```text
add what aimnet2 leaves out: d3 dispersion, and electrostatics that uses the charges aimnet2 predicts
input: scan_batch, aimnet_model, helpers.d3_settings(), helpers.DirectCoulombAdapter, helpers.load_ncia()
output: composed_model, composed_energy, ncia_energy, in agent04_work/compose_run.py
```

Returns `composed_model` (one model: AIMNet2 and the Coulomb term in one
autograd group, D3 in its own group), `composed_energy` ((90, 1), eV, for
`scan_batch`) and `ncia_energy` ((615, 1), eV, in the order of
`helpers.load_ncia()`). Shown by
`helpers.plot_interaction_curves(scan_labels, base=base_energy, composed=composed_energy)`
and `helpers.plot_ncia_accuracy(ncia_energy)`.

Fine control example of the lesson, "build it again with a 12 Å cutoff for
the coulomb term": sample `fine_control/compose_run__cutoff_12.py`; it
passes the same check.

## no_d3 ("Try it: compose a model without D3")

Default input: `scan_batch`, the 90 scan structures of
`helpers.load_nci_scans()` (sample: `no_d3.py`):

```text
what does each term contribute? compose aimnet2 with electrostatics and no d3, and evaluate the same structures
input: scan_batch, aimnet_model, helpers.DirectCoulombAdapter
output: no_d3_model, no_d3_energy, in agent04_work/no_d3.py
```

Returns `no_d3_model` (AIMNet2 and the Coulomb term, no D3) and
`no_d3_energy` ((90, 1), eV). Shown by
`helpers.plot_interaction_curves(scan_labels, base=base_energy, composed=composed_energy, no_d3=no_d3_energy)`.

## conformers ("Try it: find the lowest conformer")

Default input: the 139 saved starts of `helpers.load_dipeptide_grid()`
(sample: `conformers.py`):

```text
relax all the conformers of alanine dipeptide at once with the model i composed, and rank them by energy
input: helpers.load_dipeptide_grid(), composed_model, helpers.relax_settings()
output: conformer_batch, relaxed_conformers, conformer_energy, in agent04_work/conformers.py
helpers.show_conformers(...) draws the result: conformer_batch is a batch from the loader; relaxed_conformers is a batch, in the loader's order; conformer_energy is one energy per conformer, in eV, in the same order
```

Returns `conformer_batch` (the 139 starts, one batch), `relaxed_conformers`
(the same 139 after the relaxation, each with its largest force at or below
`helpers.relax_settings()["fmax"]`) and `conformer_energy` ((139, 1) or
(139,), eV). Shown by
`helpers.show_conformers(conformer_batch, relaxed_conformers, conformer_energy)`.
