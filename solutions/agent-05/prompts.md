# Default prompts of notebook 05, "Guest model"

One prompt per step: a typical learner prompt, each given to a fresh mid-tier agent (Sonnet) that had
only the prompt, the session lines and the official `nvalchemi-toolkit` skill. The trials named under
each prompt say what came back. Where the sample files beside this note come from is in `README.md`.

Saved again on 26 Nov from the final step text, table and Hint of each step: the words did not change.
The second relax prompt names four molecules of the learner's choosing, which the step leaves open.

## Example: wrap AceFF 2.0 (`aceff_wrapper`)

The notebook prints this prompt (board cell l04-5, kept by the educator):

```text
use the nvalchemi-toolkit skill
- wrap a model toolkit has no wrapper for, so i can use it wherever a built-in model is: relaxation, md, pipelines
- model: AceFF 2.0 from Acellera (Acellera/AceFF-2.0 on hugging face), loaded with the torchmd-net package
- it has open weights, covers small organic molecules and takes the total charge
- first tell me what has to be converted between the two libraries
- write agent05_work/aceff_wrapper.py with one wrapper class
- give me aceff_model: built from that class, on the gpu, energy in eV, forces in eV/Å
```

Trials: T1 (25 Nov, GPU) and T10 (26 Nov, CPU, with the step check; its unedited file is
`notes/evidence/agent-05/l04-5--20261126-0300--trial-T10/notebooks/agent05_work/aceff_wrapper.py`). Both passed.

## Try it: the guest beside the composed model (`aceff_pairs`)

Written from the step's sentences and table alone:

```text
use the nvalchemi-toolkit skill
- evaluate the molecule pairs (scan_batch) with aceff_model
- write agent05_work/aceff_pairs.py
- give me aceff_energy
```

Trials: T7 and T8 (26 Nov, GPU; T8 with the step check). Both passed; the sample is T7's file.

## Try it: relax a batch with the guest model (`aceff_relax`)

Written from the step's sentences and table alone, with the saved default:

```text
use the nvalchemi-toolkit skill
- relax a batch of molecules with aceff_model; take the molecules from helpers.load_molecules()
- write agent05_work/aceff_relax.py
- give me relaxed_batch, relaxed_max_force, force_threshold
```

Trial: T9 (26 Nov, GPU, with the step check). Passed; the sample is its file.

With molecules of the learner's own:

```text
use the nvalchemi-toolkit skill
- relax a batch of molecules with aceff_model, the model i wrapped
- molecules of my own: ibuprofen, glycine, toluene and urea, built from smiles, all neutral
- write agent05_work/aceff_relax.py
- give me relaxed_batch, relaxed_max_force, force_threshold
```

Trial: T11 (26 Nov, CPU, with the step check). Passed, with the note for ibuprofen (33 atoms).
