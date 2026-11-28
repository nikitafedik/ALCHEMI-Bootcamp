# Cell 11 data

Stored numbers for the two panes of the Core 00 batching figure (cell 11), measured on
2026-11-24 and 2026-11-25.

- Toolkit 0.3.0; MACE-MPA-0 medium, float32; L-BFGS, a fixed 50 optimizer steps. Time is the
  whole call minus the model load, warm-up excluded. Unit: structure-steps per second =
  structures × 50 / time.
- `cpu_singles.csv` (left pane): each catalog system alone on a CPU (AMD EPYC 7313P, 32 threads),
  plain route, frames on the host; `time_s` is the mean over `repeats`, with `min_s` and `max_s`.
  The left pane sums these times over the learner's picked systems and multiplies by
  `batch_multiplier`. The batch-of-1 GPU bar is not stored: the notebook times it live on the
  learner's GPU.
- `graph2_cueq.csv` (right pane): cuEq + compile throughput against batch size on L40S, H100 and
  B200 GPUs, for the five reference molecules (ibuprofen, azoxystrobin, permethrin, capsaicin,
  C60; distortion 0.20 Å, seed 7) cycled to the batch size, one fresh process per size. `atoms`
  is the total atom count of the batch and `peak_reserved_gb` the largest reserved GPU memory
  over the repeats; the rate is the mean over `repeats`, with `min` and `max`. Size 1 is five
  single structures in sequence. Sizes 1 to 512 come from the first sweep; from 1024 up, the
  batch doubles until a size fails, then the gap is halved in steps of 128 down to the last
  size that fits (L40S 1280, H100 2432, B200 5632; the next size runs out of GPU memory).
