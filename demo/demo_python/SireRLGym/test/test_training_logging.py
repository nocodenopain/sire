from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from SireRLGym.runners.on_policy_runner import OnPolicyRunner, PersistentSummaryWriter


class _FakeTensorBoardWriter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scalars = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, *args, **kwargs):
        self.scalars.append((args, kwargs))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


class TrainingLoggingTest(unittest.TestCase):
    def test_recovery_diagnostics_use_upstream_native_counter(self):
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.env = SimpleNamespace(
            _sire_batch_stepper=SimpleNamespace(totalRecoveredFailures=7),
        )
        self.assertEqual(runner._physics_recovery_total(), 7)
        runner.env._sire_batch_stepper.totalRecoveredFailures = 9
        self.assertEqual(runner._physics_recovery_total(), 9)

    def test_jsonl_is_written_when_tensorboard_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PersistentSummaryWriter(
                temp_dir, tensorboard_writer_cls=False
            )
            writer.add_scalar('Loss/value_function', torch.tensor(12.5), 7)
            writer.flush()
            writer.close()

            records = [
                json.loads(line)
                for line in (Path(temp_dir) / 'metrics.jsonl').read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['tag'], 'Loss/value_function')
            self.assertEqual(records[0]['step'], 7)
            self.assertEqual(records[0]['value'], 12.5)

    def test_tensorboard_and_jsonl_receive_the_same_scalar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PersistentSummaryWriter(
                temp_dir, tensorboard_writer_cls=_FakeTensorBoardWriter
            )
            backend = writer._tensorboard
            writer.add_scalar('Train/mean_reward', 3.25, 11)
            writer.close()

            self.assertEqual(backend.scalars[0][0][:2], ('Train/mean_reward', 3.25))
            self.assertEqual(backend.scalars[0][1]['global_step'], 11)
            self.assertTrue(backend.closed)
            record = json.loads(
                (Path(temp_dir) / 'metrics.jsonl').read_text().strip()
            )
            self.assertEqual(record['value'], 3.25)


if __name__ == '__main__':
    unittest.main()
