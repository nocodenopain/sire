"""Serial, persistent-process-friendly scheduler; no checkpoint-only training resume."""
from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import REPO, child_environment, read_json, sha256, verify_sources, write_json


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resources(pid):
    values = {'utc': utc_now(), 'pid': pid, 'load_average': os.getloadavg()}
    values['meminfo'] = Path('/proc/meminfo').read_text()
    values['cpu_stat'] = Path('/proc/stat').read_text().splitlines()[0]
    # GNU time is the process parent. Retain its train/eval child's RSS too.
    pids = [pid]
    try:
        pids += [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except OSError:
        pass
    values['process_status'] = {}
    for p in pids:
        try:
            values['process_status'][str(p)] = Path(f'/proc/{p}/status').read_text()
        except OSError:
            pass
    return values


def run_process(command, attempt, stage):
    command = ['/usr/bin/time', '-v', '-o', str(attempt / f'{stage}_resource_time.txt')] + command
    write_json(attempt / f'{stage}_command.json', {'argv': command, 'cwd': str(REPO),
               'environment': {k: v for k, v in child_environment().items()
                               if k in ('PYTHONPATH', 'PYTHONUNBUFFERED') or k.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_'))}})
    with (attempt / f'{stage}_stdout.log').open('a') as stdout, \
         (attempt / f'{stage}_stderr.log').open('a') as stderr, \
         (attempt / f'{stage}_resources.jsonl').open('a', buffering=1) as resource_log:
        process = subprocess.Popen(command, cwd=REPO, env=child_environment(), stdout=stdout,
                                   stderr=stderr, start_new_session=True)
        write_json(attempt / f'{stage}_process.json', {'pid': process.pid, 'started_utc': utc_now()})
        try:
            while process.poll() is None:
                resource_log.write(json.dumps(resources(process.pid)) + '\n')
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            # Only terminate our own child job; never leave a competing orphan.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        write_json(attempt / f'{stage}_exit.json', {'exit_code': process.returncode, 'ended_utc': utc_now()})
        if process.returncode != 0:
            raise RuntimeError(f'{stage} failed: exit={process.returncode}; inspect {attempt}; matrix stopped')


def train_command(root, attempt, context):
    return [str(REPO / '.venv/bin/python'), str(REPO / 'demo/demo_python/SireRLGym/scripts/train.py'),
            '--task', 'go2', '--flat_terrain', '--sim_dt', '.005', '--seed', '1',
            '--num_envs', str(context['num_envs']), '--sire_batch_threads', str(context['sire_batch_threads']),
            '--max_iterations', str(context['updates']), '--save_interval', '50',
            '--log_dir', str(attempt / 'logs'), '--parallel_experiment_context', str(attempt / 'context.json')]


def audit_training(attempt, protocol_hash):
    result = read_json(attempt / 'training_result.json')
    if result['status'] != 'complete':
        return False
    assert result['protocol_sha256'] == protocol_hash
    assert sha256(result['checkpoint']) == result['checkpoint_sha256']
    with (attempt / 'iterations.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result['updates'] == result['completed_iterations']
    assert [int(r['iteration']) for r in rows] == list(range(1, len(rows)+1))
    assert int(rows[-1]['total_transitions']) == len(rows) * result['num_envs'] * result['num_steps_per_env']
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='One separate 8-env, 2-update interface test, including all evaluation trials.')
    parser.add_argument('--restart-incomplete', action='store_true', help='Start an interrupted training as a new attempt from untrained weights.')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Scheduler interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    root = args.root.resolve()
    lock = (root / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_sources(root)
    manifest = read_json(root / 'manifest.json')
    protocol_hash = sha256(root / 'protocol.json')
    assert manifest['protocol_sha256'] == protocol_hash
    runs = ([{'run_id': 'smoke', 'num_envs': 8, 'sire_batch_threads': 4}]
            if args.smoke else manifest['runs'])
    environment = child_environment()
    write_json(root / ('smoke_scheduler.json' if args.smoke else 'scheduler.json'),
               {'pid': os.getpid(), 'started_utc': utc_now(), 'argv': sys.argv})
    durations = []
    for number, run in enumerate(runs, 1):
        run_dir = root / ('smoke' if args.smoke else 'runs') / run['run_id']
        attempts = sorted(run_dir.glob('attempt*/context.json'))
        completed = [p.parent for p in attempts if (p.parent / 'training_result.json').exists()
                     and audit_training(p.parent, protocol_hash)]
        if len(completed) > 1:
            raise RuntimeError('Multiple complete trainings for one configuration; manual audit required')
        if completed:
            attempt = completed[0]
            # Require a clean child exit, not only a pre-exit checkpoint.
            if not (attempt / 'train_exit.json').exists() or read_json(attempt / 'train_exit.json')['exit_code'] != 0:
                raise RuntimeError(f'Training has no clean recorded exit: {attempt}; audit before resuming')
            print(f'Skip already complete training: {attempt}', flush=True)
        else:
            if attempts and not args.restart_incomplete:
                raise RuntimeError(f'Incomplete attempt in {run_dir}; inspect then use --restart-incomplete')
            index = len(attempts) + 1
            attempt = run_dir / f'attempt{index:02d}'
            attempt.mkdir(parents=True)
            context = dict(run, root=str(root), attempt=index, protocol_sha256=protocol_hash,
                           updates=2 if args.smoke else 500, smoke=args.smoke)
            write_json(attempt / 'context.json', context)
            print(f'[{number}/{len(runs)}] START {run["run_id"]} {utc_now()}', flush=True)
            run_process(train_command(root, attempt, context), attempt, 'train')
            assert audit_training(attempt, protocol_hash)
        if not (attempt / 'evaluation_result.json').exists():
            run_process([str(REPO / '.venv/bin/python'), '-m', 'SireRLGym.experiments.parallel.evaluate',
                         '--root', str(root), '--attempt-dir', str(attempt)], attempt, 'eval')
        evaluation = read_json(attempt / 'evaluation_result.json')
        training = read_json(attempt / 'training_result.json')
        assert evaluation['status'] == 'complete' and evaluation['trials'] == 100
        assert evaluation['checkpoint_sha256'] == training['checkpoint_sha256']
        prior_initial_hashes = [read_json(p)['initial_states_sha256']
                                for p in (root / 'runs').glob('*/attempt*/evaluation_result.json')]
        assert all(h == evaluation['initial_states_sha256'] for h in prior_initial_hashes), 'Evaluation initial states differ'
        duration = training['training_wall_seconds']
        durations.append((run['num_envs'], duration))
        write_json(attempt / 'complete.json', {'status': 'complete', 'protocol_sha256': protocol_hash,
                   'completed_utc': utc_now(), 'checkpoint_sha256': training['checkpoint_sha256']})
        if not args.smoke:
            subprocess.run([str(REPO / '.venv/bin/python'), '-m', 'SireRLGym.experiments.parallel.summarize',
                            '--root', str(root)], cwd=REPO, env=environment, check=True)
        # A deliberately labelled rough estimate, not a measurement of unseen configs.
        seconds_per_env = sum(t for _, t in durations) / sum(n for n, _ in durations)
        remaining_estimate = sum(r['num_envs'] for r in runs[number:]) * seconds_per_env
        print(f'[{number}/{len(runs)}] COMPLETE {run["run_id"]}: T={duration/60:.2f} min, '
              f'eval_return={evaluation["mean_episode_return"]:.5f}; '
              f'rough remaining estimate={remaining_estimate/3600:.2f} h '
              '(sample-count scaling; thread/episode differences unknown)', flush=True)
    if not args.smoke:
        command = ['/home/lqf/miniconda3/envs/rl/bin/python', '-m', 'SireRLGym.experiments.parallel.summarize',
                   '--root', str(root), '--plot']
        # Plotting needs only stdlib + matplotlib from the existing conda environment.
        environment['MPLCONFIGDIR'] = str(root / 'matplotlib_cache')
        subprocess.run(command, cwd=REPO, env=environment, check=True)
        write_json(root / 'COMPLETE.json', {'status': 'complete', 'configurations': 12,
                   'protocol_sha256': protocol_hash, 'completed_utc': utc_now()})


if __name__ == '__main__':
    main()
