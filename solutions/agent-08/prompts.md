# Agent 08, "Paths of least resistance, batched": default prompts

The typical learner prompt of each step, written from the step's text, its
table and its Hint alone (revision of 2026-11-26; the neb and saddle Hints of the fix pass of 2026-11-27). A prompt names a helper
only where the step's Hint names it. Shapes, units and formats the step
leaves out come from the step check (`python -m helpers.agent08 check <step>`),
which the session lines tell the agent to run.

## Example: build the paths (`paths.py`)

> Build a straight-line path of images between each pair of `reactants` and `products` with Toolkit, every path in one Batch, with `neb_settings`. Return `paths` in agent08_work/paths.py, with energy, forces and velocities attached: `interpolate_paths` drops them, and the NEB run reads all three. `helpers.show_paths(...)` draws the result.

## Try it: relax the paths and read the barriers (`neb.py`)

> Relax every path in `paths` with one NEB call and a climbing image, with `model` and `neb_settings`, and ask for the run's diagnostics file. Then read the barrier of each path and whether the path converged. Return `relaxed_paths`, `neb_log`, `path_energy`, `barriers` and `converged` in agent08_work/neb.py. `barriers` is one number per path, in eV: the forward barrier, from the reactant end to the top of the path. `helpers.plot_energy_profiles(...)` draws the result.

## Try it: saddle point or not (`saddle.py`)

> Find a way in Toolkit to tell whether a structure is a saddle point, and test every image of `relaxed_paths` with `model`. Return `saddle_table` in agent08_work/saddle.py: a yes or no for every image, with the numbers behind it, in a table with one column each for the path (the names in `reaction_names`), the image number, the largest force on an atom, the count of imaginary modes and the yes or no. A yes needs a largest force at or below 0.005 eV/Å. Project the rigid translations and rotations out of the Hessian, then count the imaginary modes. `helpers.show_saddle_table(...)` draws the result.
