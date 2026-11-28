# Agent 01, Meet your agent: the default prompt of each step

The typical learner prompt for each step, written from the step's text and
table alone. Each was tried with a fresh mid-tier agent that had only the
prompt, the session lines and the skill path.

## first_prompt ("4. Your first prompt")

With the saved default molecule (sample: `first_prompt_default.py`):

```text
use the nvalchemi-toolkit skill
- compute the energy of the molecule from helpers.load_default_molecule() with AIMNet2
- give me my_molecule and my_energy in agent01_work/first_prompt.py
```

With a molecule the learner names (sample: `first_prompt.py`):

```text
use the nvalchemi-toolkit skill
- compute the energy of caffeine with AIMNet2
- give me my_molecule and my_energy in agent01_work/first_prompt.py
```

The agent returns two variables and draws nothing: `my_molecule` (one
structure, ASE `Atoms` or `AtomicData`) and `my_energy` (one number, eV).
The notebook's result cell, `helpers.show_first_result(my_molecule,
my_energy)`, shows the 3D view and the energy line. The step check,
`python -m helpers.agent01 check first_prompt`, tells the agent the format.

## "5. Check your prompt"

The notebook prints this prompt itself; the learner sends it with "check my
prompt" in front. It has no file and no step check.

```text
check my prompt
compute the energy
```
