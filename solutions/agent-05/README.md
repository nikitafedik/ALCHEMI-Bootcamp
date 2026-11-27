# Sample solutions for notebook 05

The `%load` files of a full unattended run (`notebooks/tests/expand_load.py`), one per step:

- `aceff_wrapper.py`: the wrapper of the Example, the sample used by every trial since 25 Nov
  (kept in `notes/builder-05/sample/`).
- `aceff_pairs.py`: the file a fresh mid-tier agent wrote in trial T7 (26 Nov) for "Try it: the
  guest beside the composed model", unedited.
- `aceff_relax.py`: the file a fresh mid-tier agent wrote in trial T9 (26 Nov) for "Try it: relax a
  batch with the guest model" with `helpers.load_molecules()`, unedited. It replaces an earlier
  sample whose last lines were adjusted by hand.

`prompts.md` holds the default prompt of each step. Each file passes
`python -m helpers.agent05 check <step>`.
