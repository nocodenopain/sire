"""Create one date-organized experiment; configurations/commands/logs, no hashes."""
from __future__ import annotations

import argparse
import datetime
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

from SireRLGym.experiments.parallel.common import REPO, THREAD_ENV, baseline_config, write_json
from .measure import rollout_steps


def capture(command):
    return subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    baseline = baseline_config()
    baseline['train_cfg']['runner']['num_steps_per_env'] = 240
    write_json(root / 'baseline_config.json', baseline)
    # These are the already-saved *untrained* weights and common evaluation inputs.
    previous = REPO / 'sire_parallel_experiment_20260906'
    shutil.copy2(previous / 'initial_weights.pt', root / 'initial_weights.pt')
    shutil.copy2(previous / 'evaluation_trials.json', root / 'evaluation_trials.json')
    shutil.copy2(REPO / 'logs/flat_go2/exp24/config.yaml', root / 'exp24_reference_config.yaml')
    import torch
    initial = torch.load(root / 'initial_weights.pt', map_location='cpu', weights_only=True)
    (root / 'initialization').mkdir()
    torch.save({'model_state_dict': initial, 'optimizer_state_dict': {}, 'iter': 0,
                'tot_timesteps': 0, 'tot_time': 0.}, root / 'initialization/model_0.pt')
    (root / 'reference').mkdir()
    shutil.copy2(REPO / 'logs/flat_go2/exp24/model_1000.pt', root / 'reference/model_1000.pt')
    protocol = dict(schema_version=2, experiment_date='2026-09-07', timezone='Asia/Shanghai',
        task='go2', num_envs=[128, 256, 512, 1024], sire_worker_threads=[4, 8, 16],
        global_batch_size=30720, steps_by_env={str(n): rollout_steps(30720, n) for n in (128, 256, 512, 1024)},
        updates=500, runs_per_config=1, seed=1, order_seed=20260907, trial_seed=20771,
        transitions_per_run=15360000, optimizer_steps_per_run=10000,
        ppo_epochs=5, ppo_minibatches=4, minibatch_size=7680, save_interval=50,
        sim_dt=.005, control_decimation=4, control_dt=.020, episode_seconds=20,
        torch_intra_op=4, torch_inter_op=1, thread_environment=THREAD_ENV,
        eval_threads=8, eval_trials=100, evaluation_checkpoints=list(range(0, 501, 50)),
        evaluation_schedule='Offline after each training; never competes with training for CPU.',
        equal_work_scope='Equal nominal transitions/updates/minibatches, not identical physical trajectories across N.',
        target_rule='J_initial + 0.90 * (J_reference - J_initial); if reference does not outperform initial, mark this auxiliary metric unavailable. Do not change rewards or choose a new threshold.',
        target_reference='Existing exp24/model_1000, evaluated before formal training.',
        target_confirmation='Second of two consecutive evaluated checkpoints >= target. Nonattainment censored.',
        training_time='First collection to final update; intermediate logging/save/GC included; initialization and offline eval excluded.',
        reward='Raw independent evaluation reward, no PPO timeout bootstrap. Training mean is logged separately.',
        physics_substeps='Nominal transitions*4; early recovery can skip substeps, not instrumented solver calls.',
        failure_policy='Stop on unexpected errors/nonfinite values; preserve recoveries and poor results; no per-config tuning.',
        reference_url='https://arxiv.org/html/2109.11978v2#S4')
    write_json(root / 'protocol.json', protocol)
    matrix = [dict(run_id=f'env{n:04d}_threads{p:02d}', num_envs=n, sire_batch_threads=p,
                   num_steps_per_env=rollout_steps(30720, n), global_batch_size=30720)
              for n in protocol['num_envs'] for p in protocol['sire_worker_threads']]
    random.Random(protocol['order_seed']).shuffle(matrix)
    write_json(root / 'manifest.json', {'runs': [dict(run, order=i+1) for i, run in enumerate(matrix)]})
    write_json(root / 'hardware_software.json', dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        platform=platform.platform(), python=sys.version, torch=torch.__version__,
        hardware=capture(['lscpu']), memory=capture(['free', '-m']),
        packages=capture([sys.executable, '-m', 'pip', 'list', '--format=json'])))
    code = root / 'code'
    code.mkdir()
    (code / 'revision.txt').write_text(capture(['git', 'rev-parse', 'HEAD']))
    (code / 'working_tree.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=REPO))
    (code / 'aris.patch').write_bytes(subprocess.check_output(['git', '-C', 'third_party/aris', 'diff', '--binary', 'HEAD'], cwd=REPO))
    shutil.copytree(Path(__file__).parent.parent, code / 'experiments',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('README.md', 'PROTOCOL.md'):
        shutil.copy2(Path(__file__).parent / name, root / name)
    print(f'Prepared {root}; run --preflight, then start the formal matrix.')


if __name__ == '__main__':
    main()
