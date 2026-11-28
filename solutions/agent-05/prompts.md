# Default prompts of notebook 05, "Guest model"

One prompt per step: a typical learner prompt, each given to a fresh mid-tier agent (Sonnet) that had
only the prompt, the session lines and the official `nvalchemi-toolkit` skill. The line under each
prompt says what came back. Where the sample files beside this note come from is in `README.md`.

The second relax prompt names four molecules of the learner's choosing, which the step leaves open.

## Example: wrap AceFF 2.0 (`aceff_wrapper`)

The notebook prints this prompt:

```text
use the nvalchemi-toolkit skill
- wrap a model toolkit has no wrapper for, so i can use it wherever a built-in model is: relaxation, md, pipelines
- model: AceFF 2.0 from Acellera (Acellera/AceFF-2.0 on hugging face), loaded with the torchmd-net package
- it has open weights, covers small organic molecules and takes the total charge
- first tell me what has to be converted between the two libraries
- write agent05_work/aceff_wrapper.py with one wrapper class
- give me aceff_model: built from that class, on the gpu, energy in eV, forces in eV/Å
```

Tried twice: on a GPU (25 Nov 2026) and on a CPU with the step check (26 Nov 2026). Both passed.

## Try it: the guest beside the composed model (`aceff_pairs`)

Written from the step's sentences and table alone:

```text
use the nvalchemi-toolkit skill
- evaluate the molecule pairs (scan_batch) with aceff_model
- write agent05_work/aceff_pairs.py
- give me aceff_energy
```

Tried twice on a GPU (26 Nov 2026), once with the step check. Both passed; the sample is the file of the first.

## Try it: relax a batch with the guest model (`aceff_relax`)

Written from the step's sentences and table alone, with the saved default:

```text
use the nvalchemi-toolkit skill
- relax a batch of molecules with aceff_model; take the molecules from helpers.load_molecules()
- write agent05_work/aceff_relax.py
- give me relaxed_batch, relaxed_max_force, force_threshold
```

Tried on a GPU with the step check (26 Nov 2026). Passed; the sample is its file with two print lines changed.

With molecules of the learner's own:

```text
use the nvalchemi-toolkit skill
- relax a batch of molecules with aceff_model, the model i wrapped
- molecules of my own: ibuprofen, glycine, toluene and urea, built from smiles, all neutral
- write agent05_work/aceff_relax.py
- give me relaxed_batch, relaxed_max_force, force_threshold
```

Tried on a CPU with the step check (26 Nov 2026). Passed, with the note for ibuprofen (33 atoms).
