from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
THREAD_ENV = {
    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
    'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE',
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def configure_threads():
    import torch
    for key, value in THREAD_ENV.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f'Set {key}={value} before launching Python')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    return {'intra_op': torch.get_num_threads(), 'inter_op': torch.get_num_interop_threads(),
            'environment': {k: os.environ[k] for k in THREAD_ENV},
            'parallel_info': torch.__config__.parallel_info()}


def child_environment():
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env['PYTHONPATH'] = f'{REPO}/python/src:{REPO}/demo/demo_python:/home/lqf/code/rsl_rl'
    env['PYTHONUNBUFFERED'] = '1'
    return env


def baseline_config(num_envs=128, threads=4, iterations=500):
    from SireRLGym.utils.task_registry import make_env_cfg, make_train_cfg
    from SireRLGym.scripts.train import _apply_sim_dt_override
    from SireRLGym.utils.helpers import class_to_dict
    env = make_env_cfg('go2')
    train = make_train_cfg('go2')
    _apply_sim_dt_override(env, .005)
    env.env.num_envs = num_envs
    env.sim.sire_batch_threads = threads
    env.terrain.mesh_type = 'plane'
    env.terrain.measure_heights = False
    env.terrain.curriculum = False
    # Validate defaults rather than silently changing the baseline.
    assert not env.commands.curriculum
    assert not env.sim.sire_diagnostics and env.sim.sire_recording_env_id == -1
    assert env.control.decimation == 4
    assert train.algorithm.num_learning_epochs == 5
    assert train.algorithm.num_mini_batches == 4
    train.runner.max_iterations = iterations
    train.runner.save_interval = 50
    train.seed = 1
    return {'env_cfg': class_to_dict(env), 'train_cfg': class_to_dict(train)}


def apply_config(obj, values):
    for key, value in values.items():
        if isinstance(value, dict) and hasattr(getattr(obj, key, None), '__dict__'):
            apply_config(getattr(obj, key), value)
        else:
            setattr(obj, key, value)


def verify_sources(root):
    snapshot = read_json(Path(root) / 'source_hashes.json')
    for path, expected in snapshot.items():
        if sha256(path) != expected:
            raise RuntimeError(f'Frozen source changed: {path}; do not mix protocols')
