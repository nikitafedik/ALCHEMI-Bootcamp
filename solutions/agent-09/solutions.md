# Agent 09, "Small-batch distilled" (liquid SiC): sample step files

One file per `%load` step of the notebook. Each is one way to do the step; a
learner's agent will write something different. Every file passes its step
check with Toolkit 0.3.0 (release commit 77d3c9e), at the course budgets on a
GPU and at the small CPU setting below:

```
cp solutions/agent-09/*.py notebooks/agent09_work/
cd notebooks
python -m helpers.agent09 check finetune   # then sample_label, distill, run_student, structure
```

Small CPU setting (every code path in a few minutes; the cells keep their 216
atoms, so fewer copies, fewer steps and fewer frames):
`ALCHEMI_AGENT09_SUBSET=20,12` (20 training and 12 held-out frames, spread over
the frame types) and `ALCHEMI_AGENT09_BUDGETS=4,2,30,20,40` (`ft_steps`,
`n_replicas`, `md_steps`, `train_steps`, `test_steps`). At this setting the
check prints "unexpected" lines (a small store, a student far from the
teacher) and judges neither the liquid nor the structure; the steps still pass.

| Step | File | Leaves behind | Settings chosen here |
| --- | --- | --- | --- |
| finetune | `finetune.py` | `finetuned_model`, `ft_loss_history` | the readouts and the last interaction block (`main.model.readouts.*`, `main.model.interactions.1.*`, `main.model.products.1.*`), AdamW, learning rate 5e-4, batches of 4, per-atom energy MSE plus 0.1 times the force MSE, `num_steps=ft_steps`, seed 1; the teacher's weights frozen afterwards |
| sample_label | `sample_label.py` | `labeled_path` | the cells of `starts` at their own temperatures, 1 fs, friction 0.01 1/fs, a frame every 10 steps, seed 0; labels with `label_dataset` |
| distill | `distill.py` | `student_model`, `loss_history` | AdamW, learning rate 1e-2, batches of 4, seed 1 |
| run_student | `run_student.py` | `student_frames_path`, `stability` | the cells of `test_starts`, the settings of `sample_label.py`, seed 1, `test_steps` steps; `StabilityMonitor(frequency=10, timestep_fs=1.0)` |
| structure | `structure.py` | `teacher_rdf`, `student_rdf` | `radial_distribution` over all frames of each store, `r_max` 6 Å, 60 bins |

What an agent has to get right, and what the check says when it does not:

- `FineTuningStrategy` trains the model it is given. Without a copy,
  `base_model` changes: "finetuned_model is base_model itself; fine-tune a
  copy, so that base_model stays as published".
- The patterns must reach the last interaction block: "the last interaction
  block of finetuned_model did not train".
- Each liquid cell runs at the temperature of its own DFT run; the check
  reads each copy's target from its cell, so the order of the copies does not
  matter.
- `DistillationStrategy` needs both a `student` and a `teacher` entry in
  `models`, though training reads the stored labels and never calls the teacher.
- A Zarr store is written once: a file that writes to a fixed path fails when
  the cell is run again.
- The student's run starts from `test_starts`, not `starts`, and the
  `StabilityMonitor` is registered on that run.
- Both radial distributions need the same `r_max` and number of bins.

The rules behind "still liquid" and "follows the teacher" are in
`helpers.agent09.still_liquid` and `helpers.agent09.follows_teacher`, from the
prototype's report (notes/evidence/agent-09/proto-20261126/report.md, round 2).
