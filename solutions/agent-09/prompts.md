# Agent 09, "Small-batch distilled" (liquid SiC): default prompts

One typical learner prompt per step, written from the step's text, table and
Hints alone: each prompt carries nothing the step does not.
The check is not in the prompts; the session lines printed by the setup cell
tell the agent to run it after every file.

## finetune (Example: fine-tune on the training frames)

> Fine-tune a copy of `base_model` on the frames in `train_path` with
> Toolkit's `FineTuningStrategy`, for `ft_steps` optimizer steps. Train the
> last interaction block and the readouts; in MACE-MPA-0 the last interaction
> block is `interactions.1` with `products.1`, and inside the strategy every
> weight name starts with `main.model.`. Use AdamW with a learning rate of
> 5e-4, batches of 4 frames, and a loss of the energy per atom plus 0.1 times
> the forces; `base_model` itself must stay as published. Return
> `finetuned_model` and `ft_loss_history`. Write it to
> `agent09_work/finetune.py`.

## sample_label (Try it: sample and label frames)

> Run an MD of the liquid cells in `starts` with `finetuned_model`, the
> teacher, for `md_steps` steps of 1 fs, with a friction of 0.01 per fs, each
> cell at its own temperature (`atoms.info["temperature_K"]`; `NVTLangevin`
> takes one temperature per copy as a tensor). Save a frame of every cell to a
> Zarr store every 10 steps with a hook as the run goes. Then have
> `label_dataset` write the teacher's answers beside each frame, so that
> `teacher_energy` and `teacher_forces` are saved for every frame. Return
> `labeled_path`. Write it to `agent09_work/sample_label.py`.

## distill (Example: distill the student)

> Distill the student `helpers.make_student(labeled_path)` with Toolkit's
> `DistillationStrategy` on the stored labels in `labeled_path`, for
> `train_steps` steps. The teacher is `finetuned_model`; its weights stay as
> they are. Use a learning rate of 1e-2, batches of 4 frames, and a loss of
> the energy per atom plus the forces. Return `student_model` and
> `loss_history`. Write it to `agent09_work/distill.py`.

## run_student (Try it: run the MD with the student)

> Run an MD of the liquid cells in `test_starts` with `student_model` for
> `test_steps` steps, with the settings of the sampling run (1 fs, friction
> 0.01 per fs, each cell at its own temperature), and register a
> `StabilityMonitor` from `nvalchemi.dynamics.hooks` on the run. Return
> `student_frames_path`, the path of a new Zarr store with a frame of every
> cell every 10 steps, and `stability`, the monitor's `metrics()` after the
> run. Write it to `agent09_work/run_student.py`.

## structure (Try it: compare the student's structure with the teacher's)

> Compute the radial distribution g(r) over all pairs with Toolkit's
> `radial_distribution` from `nvalchemi.training.distillation.evaluation`,
> as `radial_distribution(batch, r_max=6.0, num_bins=60)` for both: once for
> the frames of `labeled_path`, the teacher's sampling run, and once for the
> frames of `student_frames_path`; `batch` is a `Batch` of periodic frames,
> and `helpers.read_store(path)` returns `(reader, batch)` with every frame
> of a store. Return `teacher_rdf` and `student_rdf`. Write it to
> `agent09_work/structure.py`.
