"""Fast checks of experiment timing/accounting; no training or simulator needed."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from SireRLGym.experiments.parallel.common import write_json
from SireRLGym.experiments.parallel.measure import ExperimentAttempt


class ParallelExperimentTest(unittest.TestCase):
    def test_sorted_json_does_not_reorder_policy_joints(self):
        from SireRLGym.experiments.parallel.evaluate import restore_training_joint_order
        original = {'FL_hip_joint': .1, 'FR_hip_joint': -.1,
                    'FL_thigh_joint': .8, 'FR_thigh_joint': .8,
                    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5}
        loaded = json.loads(json.dumps(original, sort_keys=True))
        self.assertNotEqual(list(original), list(loaded))
        cfg = SimpleNamespace(init_state=SimpleNamespace(default_joint_angles=loaded))
        restore_training_joint_order(cfg, list(original))
        self.assertEqual(list(cfg.init_state.default_joint_angles), list(original))
        self.assertEqual(cfg.init_state.default_joint_angles, original)

    def test_wall_clock_includes_between_iteration_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'protocol.json', {})
            write_json(root / 'context.json', dict(root=tmp, run_id='unit', attempt=1,
                       num_envs=128, sire_batch_threads=4, updates=2))
            observer = ExperimentAttempt(root / 'context.json')
            observer.steps = 120
            observer.stream = (root / 'iterations.csv').open('w', newline='')
            observer.writer = None
            # 5 seconds of logging/checkpoint/GC between update 1 and collection 2.
            with patch('SireRLGym.experiments.parallel.measure.time.perf_counter',
                       side_effect=[100., 102., 103., 108., 110., 111.]):
                for iteration in (1, 2):
                    observer.collection_start(iteration)
                    observer.collection_end()
                    observer.update_end()
                    observer.iteration_end(iteration, deque([2., 4.]), deque([10., 20.]), 0, {}, .1)
            observer.close()
            with (root / 'iterations.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[-1]['wall_elapsed_seconds']), 11.)
            self.assertEqual(sum(float(r['collection_seconds']) for r in rows), 4.)
            self.assertEqual(sum(float(r['learning_seconds']) for r in rows), 2.)
            self.assertEqual(int(rows[-1]['total_transitions']), 2 * 128 * 120)
            self.assertEqual(int(rows[-1]['total_physics_substeps']), 4 * 2 * 128 * 120)
            self.assertEqual(float(rows[-1]['train_mean_episode_return']), 3.)


if __name__ == '__main__':
    unittest.main()
