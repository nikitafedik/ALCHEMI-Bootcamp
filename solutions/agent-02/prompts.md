# Notebook 02, "Many molecules, one batch": default prompts and sample files

Not part of the notebook; never mentioned in it. One default prompt per step: the typical learner prompt, written
from the step's text, table and Hints alone, in a learner's words. The prompts were saved before they were tried
and were not tuned afterwards.

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

| Step | Cell | Default prompt | Sample file | Check |
| --- | --- | --- | --- | --- |
| 1 `atomic_data` | `a02-07`, Example | load caffeine with helpers.load_molecule("caffeine") and make it an AtomicData called caffeine_data. file agent02_work/atomic_data.py | `atomic_data.py` | `python -m helpers.agent02 check atomic_data` |
| 2 `batch` | `a02-11`, Example | make one batch from caffeine, aspirin, ibuprofen, paracetamol and capsaicin, helpers.load_molecule(name) for each. give me molecule_data and batch, file agent02_work/batch.py | `batch.py` | `... check batch` |
| 3 `my_batch` | `a02-15`, Try it, with the Hint | i have no molecule at hand. add the tested default, helpers.load_default_molecule(), to the batch as the third one, starting from molecule_data. give me my_batch, file agent02_work/my_batch.py | `my_batch_default.py` (the default), `my_batch.py` (a SMILES of the learner) | `... check my_batch` |
| 4 `build_routes` | `a02-18`, Try it | build the same large batch three ways and time each: from the ASE objects on the CPU and then moved to the GPU, from the objects directly on the GPU, and directly from tensors. time the build only, once the coordinates exist. the structures: helpers.load_many(n_structures). give me build_table, file agent02_work/build_routes.py | `build_routes.py` | `... check build_routes` |
| 5 `model_call` | `a02-21`, Example | run aimnet2 on batch in one call, with the built-in toolkit wrapper. give me aimnet_model, energy and forces, file agent02_work/model_call.py | `model_call.py` | `... check model_call` |
| 6 `zarr_roundtrip` | `a02-26`, Example | save batch with its energy and forces to a zarr store and read it back. give me store_path and loaded_batch, file agent02_work/zarr_roundtrip.py | `zarr_roundtrip.py` (a new store), `zarr_append.py` (appends, the Fine control example) | `... check zarr_roundtrip` |

Step 4 takes the place of "Try it: a batch of your choice" (the switch block at the top of
`notebooks/helpers/agent02_text.py`). Its sample builds 1,000 structures with one warm-up build and the best of five
timed builds; the tensor route passes positions and atomic numbers only. `own_batch.py` and `own_batch_default.py`
are the samples of "Try it: a batch of your choice", which has no cells in the notebook; its check runs by name,
`... check own_batch`.

On a machine without a GPU the sample of step 4 builds all three routes on the CPU; the table then marks the order
"unexpected" and the step passes.

Tried with one fresh mid-tier agent, all six prompts in order, on CPU: every step check passed with no hand fix.
