# Agent 01, Meet your agent: sample files

One learner-written step, two sample files for it.

| step | file | names it leaves |
| --- | --- | --- |
| first_prompt | `first_prompt.py` | `my_molecule` (one structure), `my_energy` (one number, eV) |
| first_prompt, with the default molecule | `first_prompt_default.py` | the same two names |

`first_prompt.py` is the file a fresh agent wrote for the prompt "compute the
energy of caffeine with aimnet2; give me my_molecule and my_energy, in
first_prompt.py" (trial A, 2026-11-25). It builds caffeine from a SMILES string
with RDKit and calls AIMNet2 once. Energy: -18517.4883 eV, the same on the
course GPU and on CPU. The learner picks their own molecule, so their file and
their number differ.

`first_prompt_default.py` takes the saved default instead,
`helpers.load_default_molecule()` (caffeine, `notebooks/data/agent01/caffeine.xyz`,
the same geometry recipe), and gives the same energy, -18517.4883 eV on CPU.

To check either, copy it to `notebooks/agent01_work/first_prompt.py` and run,
from `notebooks/`:

    python -m helpers.agent01 check first_prompt

It prints `first_prompt: OK (2 names)`.

"Check your prompt" has no file: the agent answers the weak prompt with what is
missing and writes no code.
