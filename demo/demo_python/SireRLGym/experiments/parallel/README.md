# Fixed-budget Sire CPU-parallelism experiment

After the 2026-09-08 upstream integration, the solver, joint safety and training
defaults differ from the original experiment. Reproduce historical runs at local
commit `0b87a04`; use a new dated directory for the integrated version. Do not
append new-backend evaluations to old results. Existing CSV aggregation is safe.

Run from the Sire repository root. These scripts do not change physics or PPO.
The only existing training-entry change is an opt-in measurement context.
Do not run another training, evaluation, or heavy benchmark concurrently.

The output directory is ignored by Git. Preparation freezes the full baseline,
random order, 100 evaluation trials, same **untrained** network, dependency/
hardware metadata, and source hashes. The default 120 rollout steps are read,
not overridden. The existing 5 ms CLI adjusts decimation to 4.

```bash
export PYTHONPATH=python/src:demo/demo_python:/home/lqf/code/rsl_rl
.venv/bin/python -m SireRLGym.experiments.parallel.prepare \
  --root sire_parallel_experiment_20260906
.venv/bin/python -m SireRLGym.experiments.parallel.run_matrix \
  --root sire_parallel_experiment_20260906 --smoke
.venv/bin/python -m SireRLGym.experiments.parallel.run_matrix \
  --root sire_parallel_experiment_20260906
```

For a long run, launch the last command under a persistent service/supervisor
and retain its stdout/stderr. The scheduler holds an exclusive lock, runs train
and eval serially, saves resource samples every 30 seconds, and stops on a
nonzero child exit. Its rough remaining-time estimate is explicitly not a
measurement of unrun configurations. It writes final figures only after all
12 configurations and evaluations pass completeness checks.

It skips complete protocol-matching configurations. If training is interrupted,
preserve that attempt and explicitly add `--restart-incomplete` after inspecting
its logs. This starts a new attempt from the **original untrained** weights and
fresh optimizer; it does not use checkpoint-only resume. A complete training
with missing evaluation is evaluated without retraining. There is no repeat
training phase or best-checkpoint selection.

Every attempt has `train_command.json` and `eval_command.json` containing exact
standalone argv, cwd and environment. To run a single prepared training by hand,
use those arguments including `--parallel_experiment_context context.json`;
never run it against an already-used attempt directory. The scheduler creates
contexts and unique directories so accidental overwrites fail.

All child processes set `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE`
before importing torch. PyTorch intra/inter-op budgets are 4/1, independent of
the Sire thread sweep. The deterministic evaluator uses 100 environments and
8 Sire threads, one fixed first trial per slot. Training noise/randomization
configuration is unchanged; evaluation observation noise and domain-randomization
flags are off uniformly. No existing logs or paper-repository files are touched.

Regenerate CSVs/report without plotting:

```bash
.venv/bin/python -m SireRLGym.experiments.parallel.summarize \
  --root sire_parallel_experiment_20260906
```

Regenerate final figures using the already-installed matplotlib (no torch
needed by this command):

```bash
MPLCONFIGDIR=/tmp/sire_parallel_mpl \
PYTHONPATH=demo/demo_python \
/home/lqf/miniconda3/envs/rl/bin/python \
  -m SireRLGym.experiments.parallel.summarize \
  --root sire_parallel_experiment_20260906 --plot
```

The experiment never resumes formal runs after source/config drift. Source
snapshots may be refreshed only **before any formal attempt exists**:
`prepare --root ... --refresh-sources-before-formal`. Smoke attempts are separate
and are not plotted. Raw failed/interrupted attempts remain in summaries.
