# Sample solutions for notebook 05

The `%load` files of a full unattended run (`notebooks/tests/expand_load.py`), one per step:

- `aceff_wrapper.py`: the wrapper of the Example, in use since 25 Nov 2026.
- `aceff_pairs.py`: the file a fresh mid-tier agent wrote on 26 Nov 2026 for "Try it: the
  guest beside the composed model", unedited.
- `aceff_relax.py`: the file a fresh mid-tier agent wrote on 26 Nov 2026 for "Try it: relax a
  batch with the guest model" with `helpers.load_molecules()`, unedited.

`prompts.md` holds the default prompt of each step. Each file passes
`python -m helpers.agent05 check <step>`.
