"""Freeze an untrained network, protocol, execution order and 100 evaluation trials."""
from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .common import REPO, THREAD_ENV, baseline_config, read_json, sha256, state_hash, write_json


def capture(command):
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'command': command, 'exit_code': result.returncode, 'output': result.stdout}


def snapshot(root):
    # Fingerprint executed training/evaluation code, model assets and native module.
    paths = subprocess.check_output(['git', 'ls-files', 'demo/demo_python/SireRLGym',
        'demo/demo_python/resources', 'python/src/sire', 'python/sire', 'src', 'include'], cwd=REPO, text=True).splitlines()
    paths += [str(p.relative_to(REPO)) for p in (REPO / 'demo/demo_python/SireRLGym/experiments/parallel').glob('*.py')]
    paths += ['python/src/sire/native/sire.so']
    paths += [str(root / name) for name in ('baseline_config.json', 'protocol.json',
              'manifest.json', 'evaluation_trials.json', 'initial_weights.pt')]
    external = Path('/home/lqf/code/rsl_rl/rsl_rl')
    paths += [str(p) for p in external.rglob('*.py')]
    hashes = {str((REPO / p).resolve()): sha256(REPO / p) for p in sorted(set(paths)) if (REPO / p).is_file()}
    write_json(root / 'source_hashes.json', hashes)
    (root / 'code.patch').write_text(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=REPO, text=True))
    # The user's submodule diff contains legacy-encoded text; preserve exact bytes.
    (root / 'aris.patch').write_bytes(subprocess.check_output(['git', '-C', 'third_party/aris', 'diff', '--binary', 'HEAD'], cwd=REPO))
    shutil.copytree(REPO / 'demo/demo_python/SireRLGym/experiments/parallel', root / 'experiment_scripts',
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    write_json(root / 'code_version.json', {
        'git_head': capture(['git', 'rev-parse', 'HEAD']), 'git_status': capture(['git', 'status', '--short']),
        'aris_head': capture(['git', '-C', 'third_party/aris', 'rev-parse', 'HEAD']),
        'rsl_rl_head': capture(['git', '-C', '/home/lqf/code/rsl_rl', 'rev-parse', 'HEAD']),
        'source_hashes_sha256': sha256(root / 'source_hashes.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--refresh-sources-before-formal', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.refresh_sources_before_formal:
        if list((root / 'runs').glob('*/attempt*/context.json')):
            raise RuntimeError('Formal attempts already exist; cannot change frozen code')
        snapshot(root)
        return
    root.mkdir(parents=True, exist_ok=False)
    baseline = baseline_config()
    write_json(root / 'baseline_config.json', baseline)
    import torch
    import numpy as np
    import yaml
    from rsl_rl.modules import ActorCritic
    torch.manual_seed(1)
    cfg = baseline['env_cfg']['env']
    actor = ActorCritic(cfg['num_observations'], cfg['num_privileged_obs'], cfg['num_actions'],
                        **baseline['train_cfg']['policy'])
    torch.save(actor.state_dict(), root / 'initial_weights.pt')
    protocol = {
        'schema_version': 1, 'task': 'go2', 'num_envs': [128, 256, 512, 1024],
        'sire_worker_threads': [4, 8, 16], 'runs_per_config': 1, 'updates': 500,
        'num_steps_per_env': baseline['train_cfg']['runner']['num_steps_per_env'],
        'sim_dt': .005, 'control_decimation': 4, 'control_dt': .020,
        'ppo_epochs': 5, 'ppo_minibatches': 4, 'save_interval': 50, 'seed': 1,
        'order_seed': 20260906, 'trial_seed': 20771,
        'torch_intra_op': 4, 'torch_inter_op': 1, 'thread_environment': THREAD_ENV,
        'eval_num_envs': 100, 'eval_threads': 8, 'eval_trials': 100,
        'eval_noise': False, 'eval_domain_randomization': False, 'eval_policy': 'deterministic mean',
        'eval_horizon_seconds': baseline['env_cfg']['env']['episode_length_s'],
        'eval_initialization': 'Fixed draws from baseline joint multiplier, yaw and world-point velocity distributions; local xy=0.',
        'eval_commands': 'Fixed draws from baseline command distribution, resampled at the baseline interval; heading feedback unchanged.',
        'eval_aggregation': 'One initial trial per slot; ignore later auto-reset episodes. All 100 first trials, including faults, count.',
        'training_time': 'perf_counter immediately before first collection to immediately after final PPO update; intermediate logs/checkpoints/GC included.',
        'timing_decomposition': 'collection: actual rollout; learning: compute_returns + existing rollout diagnostics + PPO update; other: inter-update overhead. Post-final-update save separately.',
        'training_return': 'Existing runner rolling 100 completed episode sums; rsl_rl may add timeout value bootstrap to reward in-place. Not independent evaluation return.',
        'evaluation_return': 'Undiscounted raw env.step rewards; no value bootstrap; numerical-failure reward follows existing zero/truncation semantics.',
        'initial_weights_file_sha256': sha256(root / 'initial_weights.pt'),
        'initial_state_sha256': state_hash(actor.state_dict()),
        'reference_url': 'https://proceedings.mlr.press/v164/rudin22a.html',
        'restart_policy': 'Skip protocol-matching complete attempts; interrupted training restarts from identical initial weights as a separate attempt. No weights-only resume.',
        'systematic_failure_policy': 'Stop on native exception, nonfinite metrics or bad config; retain all fault recoveries and poor results; no per-config parameter adjustment.'}
    write_json(root / 'protocol.json', protocol)
    matrix = [{'run_id': f'env{n:04d}_threads{t:02d}', 'num_envs': n, 'sire_batch_threads': t}
              for n in protocol['num_envs'] for t in protocol['sire_worker_threads']]
    random.Random(protocol['order_seed']).shuffle(matrix)
    write_json(root / 'manifest.json', {'protocol_sha256': sha256(root / 'protocol.json'),
        'order_seed': protocol['order_seed'], 'runs': [dict(r, order=i+1) for i, r in enumerate(matrix)]})
    rng = np.random.default_rng(protocol['trial_seed'])
    ecfg = baseline['env_cfg']
    resample_steps = int(ecfg['commands']['resampling_time'] / .02)
    horizon = int(ecfg['env']['episode_length_s'] / .02)
    trials = []
    for index in range(100):
        trial = {'trial_id': f'trial_{index:03d}', 'slot': index,
                 'joint_multipliers': rng.uniform(.5, 1.5, 12).tolist(),
                 'yaw': float(rng.uniform(*ecfg['init_state']['init_yaw_range'])),
                 'world_point_velocity': rng.uniform(-.5, .5, 6).tolist(), 'commands': []}
        for step in range(0, horizon + 1, resample_steps):
            vx, vy = [float(rng.uniform(*ecfg['commands']['ranges'][k])) for k in ('lin_vel_x', 'lin_vel_y')]
            if np.hypot(vx, vy) <= .2:
                vx = vy = 0.0
            trial['commands'].append({'control_step': step, 'vx': vx, 'vy': vy,
                                     'heading': float(rng.uniform(*ecfg['commands']['ranges']['heading']))})
        trials.append(trial)
    write_json(root / 'evaluation_trials.json', trials)
    shutil.copy2(REPO / 'logs/flat_go2/exp24/config.yaml', root / 'exp24_reference_config.yaml')
    old = yaml.safe_load((root / 'exp24_reference_config.yaml').read_text())
    differences = []
    def diff(a, b, path=''):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                diff(a.get(key), b.get(key), path + '.' + key)
        elif a != b:
            differences.append({'path': path, 'exp24': a, 'frozen': b})
    diff(old, baseline)
    write_json(root / 'exp24_config_differences.json', differences)
    power = {}
    for pattern in ('devices/system/cpu/cpu*/cpufreq/scaling_governor',
                    'devices/system/cpu/cpu*/cpufreq/energy_performance_preference',
                    'devices/system/cpu/intel_pstate/no_turbo', 'firmware/acpi/platform_profile',
                    'class/power_supply/*/online', 'class/power_supply/*/status'):
        for path in Path('/sys').glob(pattern):
            try:
                power[str(path)] = path.read_text().strip()
            except OSError as error:
                power[str(path)] = str(error)
    write_json(root / 'hardware_software.json', {'platform': platform.platform(),
        'python': sys.version, 'interpreter': sys.executable, 'torch': torch.__version__,
        'numpy': np.__version__, 'hardware': capture(['lscpu']), 'memory': capture(['free', '-m']),
        'packages': capture([sys.executable, '-m', 'pip', 'list', '--format=json']),
        'cpu_affinity': capture(['taskset', '-pc', str(__import__('os').getpid())]),
        'power_settings_read_only': power, 'process_snapshot_scope': 'Preparation sandbox; host preflight saved separately.'})
    snapshot(root)
    print(json.dumps({'root': str(root), 'order': matrix, 'default_steps': protocol['num_steps_per_env']}, indent=2))


if __name__ == '__main__':
    main()
