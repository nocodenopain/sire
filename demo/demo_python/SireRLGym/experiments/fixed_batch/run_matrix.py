"""Serial background queue: 12 trainings, checkpoint evaluations, automatic plots."""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

from SireRLGym.experiments.parallel.common import REPO, child_environment, read_json, write_json
from SireRLGym.experiments.parallel.run_matrix import run_process, utc_now

PYTHON = str(REPO / '.venv/bin/python')


def status(root, **values):
    write_json(root / 'STATUS.json', dict(updated_utc=utc_now(), **values))


def evaluate(root, output, checkpoint, iteration):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation_result.json').exists():
        result = read_json(output / 'evaluation_result.json')
        assert result['iteration'] == iteration and result['trials'] == 100
        return result
    write_json(output / 'training_result.json', dict(status='complete',
        purpose='offline_checkpoint_evaluation', checkpoint=str(checkpoint), completed_iterations=iteration))
    run_process([PYTHON, '-m', 'SireRLGym.experiments.parallel.evaluate',
                 '--root', str(root), '--attempt-dir', str(output)], output, 'eval')
    return read_json(output / 'evaluation_result.json')


def audit_training(attempt, updates, batch):
    if not (attempt / 'training_result.json').exists():
        return False
    result = read_json(attempt / 'training_result.json')
    if result['status'] != 'complete':
        return False
    with (attempt / 'iterations.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == updates and result['completed_iterations'] == updates
    assert [int(row['iteration']) for row in rows] == list(range(1, updates + 1))
    assert int(rows[-1]['total_transitions']) == updates * batch
    assert Path(result['checkpoint']).is_file()
    assert read_json(attempt / 'train_exit.json')['exit_code'] == 0
    return True


def train(root, run, smoke=False, restart=False):
    run_dir = root / ('smoke' if smoke else 'runs') / run['run_id']
    attempts = sorted(run_dir.glob('attempt*/context.json'))
    updates = 2 if smoke else 500
    complete = [path.parent for path in attempts if audit_training(path.parent, updates, 30720)]
    if len(complete) > 1:
        raise RuntimeError('More than one complete training for a configuration')
    if complete:
        return complete[0]
    if attempts and not restart:
        raise RuntimeError(f'Incomplete attempt: {run_dir}; inspect before --restart-incomplete')
    attempt = run_dir / f'attempt{len(attempts)+1:02d}'
    attempt.mkdir(parents=True)
    context = dict(run, root=str(root), attempt=len(attempts)+1, updates=updates, smoke=smoke)
    write_json(attempt / 'context.json', context)
    command = [PYTHON, str(REPO / 'demo/demo_python/SireRLGym/scripts/train.py'),
        '--task', 'go2', '--flat_terrain', '--sim_dt', '.005', '--seed', '1',
        '--num_envs', str(run['num_envs']), '--sire_batch_threads', str(run['sire_batch_threads']),
        '--max_iterations', str(updates), '--save_interval', '50', '--log_dir', str(attempt / 'logs'),
        '--fixed_batch_experiment_context', str(attempt / 'context.json')]
    run_process(command, attempt, 'train')
    assert audit_training(attempt, updates, 30720)
    return attempt


def preflight(root, runs):
    if (root / 'PREFLIGHT_COMPLETE.json').exists():
        return
    if list((root / 'runs').glob('*/attempt*/context.json')):
        raise RuntimeError('Reference target must be determined before formal training')
    status(root, stage='preflight_reference')
    initial = evaluate(root, root / 'preregistration/initial', root / 'initialization/model_0.pt', 0)
    reference = evaluate(root, root / 'preregistration/reference', root / 'reference/model_1000.pt', 1000)
    ji, jr = initial['mean_episode_return'], reference['mean_episode_return']
    # Do not choose a new policy/threshold just to obtain a desired convergence
    # plot. An invalid reference disables only this auxiliary metric.
    target = ji + .9 * (jr - ji) if jr > ji else None
    write_json(root / 'preregistration.json', dict(created_utc=utc_now(),
        initial_return=ji, reference_return=jr, reward_target=target,
        rule='initial + 0.9*(reference-initial)', reference='exp24/model_1000',
        target_status='available' if target is not None else 'unavailable_reference_not_better_than_initial',
        confirmation='Second of two consecutive evaluated checkpoints >= reward_target'))
    for n in (128, 1024):
        run = next(run for run in runs if run['num_envs'] == n and run['sire_batch_threads'] == 16)
        status(root, stage='preflight_smoke', run_id=run['run_id'])
        attempt = train(root, run, smoke=True)
        result = read_json(attempt / 'training_result.json')
        evaluation = evaluate(root, attempt / 'evaluations/iter_0002', Path(result['checkpoint']), 2)
        assert evaluation['trials'] == 100
    write_json(root / 'PREFLIGHT_COMPLETE.json', dict(completed_utc=utc_now(), reward_target=target))
    status(root, stage='preflight_complete', reward_target=target)
    print(f'Preflight passed; initial={ji:.4f}, reference={jr:.4f}, target={target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--restart-incomplete', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Scheduler received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    runs = read_json(root / 'manifest.json')['runs']
    write_json(root / ('preflight_process.json' if args.preflight else 'scheduler.json'),
               dict(pid=os.getpid(), started_utc=utc_now(), argv=sys.argv))
    if not args.preflight:
        write_json(root / 'host_preflight.json', dict(utc=utc_now(),
            processes=subprocess.check_output(['ps', '-eo', 'pid,ppid,stat,pcpu,rss,etime,comm', '--sort=-pcpu'], text=True),
            memory=subprocess.check_output(['free', '-m'], text=True), load_average=os.getloadavg()))
    try:
        if args.preflight:
            preflight(root, runs)
            return
        assert (root / 'PREFLIGHT_COMPLETE.json').exists(), 'Run --preflight first'
        durations = []
        for index, run in enumerate(runs, 1):
            status(root, stage='training', completed_configurations=index-1,
                   current_index=index, total=12, **run)
            print(f'[{index}/12] START {run["run_id"]} H={run["num_steps_per_env"]} {utc_now()}', flush=True)
            attempt = train(root, run, restart=args.restart_incomplete)
            result = read_json(attempt / 'training_result.json')
            for iteration in range(50, 501, 50):
                status(root, stage='evaluation', completed_configurations=index-1,
                       current_index=index, total=12, evaluation_iteration=iteration, **run)
                checkpoint = Path(result['checkpoint']).parent / f'model_{iteration}.pt'
                evaluate(root, attempt / f'evaluations/iter_{iteration:04d}', checkpoint, iteration)
            write_json(attempt / 'complete.json', dict(completed_utc=utc_now(), status='complete'))
            durations.append(result['training_wall_seconds'])
            subprocess.run([PYTHON, '-m', 'SireRLGym.experiments.fixed_batch.summarize',
                            '--root', str(root)], cwd=REPO, env=child_environment(), check=True)
            print(f'[{index}/12] COMPLETE {run["run_id"]}: training={durations[-1]/60:.2f} min; '
                  f'rough remaining={(12-index)*sum(durations)/len(durations)/3600:.2f} h '
                  '(observed mean; threads/learning differences unknown)', flush=True)
        status(root, stage='plotting', completed_configurations=12, total=12)
        environment = child_environment()
        environment['MPLCONFIGDIR'] = str(root / 'matplotlib_cache')
        subprocess.run(['/home/lqf/miniconda3/envs/rl/bin/python', '-m',
            'SireRLGym.experiments.fixed_batch.summarize', '--root', str(root), '--plot'],
            cwd=REPO, env=environment, check=True)
        write_json(root / 'COMPLETE.json', dict(status='complete', configurations=12, completed_utc=utc_now()))
        status(root, stage='complete', completed_configurations=12, total=12)
    except BaseException as error:
        record = dict(stage='failed', utc=utc_now(), reason=str(error), traceback=traceback.format_exc())
        write_json(root / ('PREFLIGHT_FAILED.json' if args.preflight else 'FAILED.json'), record)
        status(root, **record)
        raise


if __name__ == '__main__':
    main()
