# Notebook 02, "Many molecules, one batch": default prompts and sample files

Not part of the notebook; never mentioned in it. One default prompt per step: the typical learner prompt, written
from the step's text and table alone, in a learner's words. They were saved before the trial that used them
(`notes/builder-02/trial-a02/prompts/P4c.md`) and were not tuned afterwards. Saved again on 26 Nov (revision, change F)
from each step's text, table and Hints alone: steps 2 and 6 no longer carry the Fine control example, which the step
text does not hold; the session block has the two new lines.

Session lines the learner gives the agent first (printed by `helpers.show_session()`), plus one line naming the skill:

```text
Give these to your agent when you start the session:
  notebook    notebooks/alchemi-agent-02-data-and-batching.ipynb
  helpers     from helpers import agent02 as helpers
  work in     notebooks/agent02_work/
  check with  python -m helpers.agent02 check <step>
  each file   runs in a notebook cell (%load), after the earlier files
  always      run the check from notebooks/ after you write a file, and fix
              what it reports; <step> is the file name without .py
use the nvalchemi-toolkit skill
```

| Step | Cell (plan 3.1) | Default prompt | Sample file | Check |
| --- | --- | --- | --- | --- |
| 1 `atomic_data` | 7, Example | load caffeine with helpers.load_molecule("caffeine") and make it an AtomicData called caffeine_data. file agent02_work/atomic_data.py | `atomic_data.py` | `python -m helpers.agent02 check atomic_data` |
| 2 `batch` | 11, Example | make one batch from caffeine, aspirin, ibuprofen, paracetamol and capsaicin, helpers.load_molecule(name) for each. give me molecule_data and batch, file agent02_work/batch.py | `batch.py` | `... check batch` |
| 3 `my_batch` | 15, Try it, with the Hint | i have no molecule at hand. add the tested default, helpers.load_default_molecule(), to the batch as the third one, starting from molecule_data. give me my_batch, file agent02_work/my_batch.py | `my_batch_default.py` (the default), `my_batch.py` (a SMILES of the learner) | `... check my_batch` |
| 4 `build_routes` | 18, Try it | build the same large batch three ways and time each: from the ASE objects on the CPU and then moved to the GPU, from the objects directly on the GPU, and directly from tensors. time the build only, once the coordinates exist. the structures: helpers.load_many(n_structures). give me build_table, file agent02_work/build_routes.py | `build_routes.py` | `... check build_routes` |
| 5 `model_call` | 21, Example | run aimnet2 on batch in one call, with the built-in toolkit wrapper. give me aimnet_model, energy and forces, file agent02_work/model_call.py | `model_call.py` | `... check model_call` |
| 6 `zarr_roundtrip` | 26, Example | save batch with its energy and forces to a zarr store and read it back. give me store_path and loaded_batch, file agent02_work/zarr_roundtrip.py | `zarr_roundtrip.py` (a new store), `zarr_append.py` (appends, the Fine control example) | `... check zarr_roundtrip` |

PROVISIONAL (the educator's open decisions, switch block at the top of `notebooks/helpers/agent02_text.py`):
step 4 replaces "Try it: a batch of your choice"; its sample builds 1,000 structures with one warm-up build and the
best of five timed builds; the tensor route passes positions and atomic numbers only. `own_batch.py` and
`own_batch_default.py` are the samples of the replaced "Try it" (kept; checked by name with `check own_batch`).

On a machine without a GPU the sample of step 4 builds all three routes on the CPU; the table then marks the order
"unexpected" and the step passes.

Trialled on 2026-11-26 (trial t6a, one fresh Sonnet agent, all six prompts in order, CPU): every step check passed
with no hand fix. Record: `notes/evidence/agent-01/summary--20261126-0332--02-owner2-trial-t6a.md`. Earlier rounds
(25 Nov, GPU) used the same prompts for steps 1, 2, 3, 5 and 6 in other wordings: `notes/evidence/agent-01/summary--*`.
