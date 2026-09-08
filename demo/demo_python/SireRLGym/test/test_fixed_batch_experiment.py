"""Small accounting/target tests; no formal training."""
import unittest

from SireRLGym.experiments.fixed_batch.measure import rollout_steps
from SireRLGym.experiments.fixed_batch.summarize import target_time


class FixedBatchTest(unittest.TestCase):
    def test_matrix_equal_work(self):
        for n, expected in ((128, 240), (256, 120), (512, 60), (1024, 30)):
            h = rollout_steps(30720, n)
            self.assertEqual(h, expected)
            self.assertEqual(n*h*500, 15360000)
            self.assertEqual(n*h//4, 7680)

    def test_invalid_batch_rejected(self):
        for batch, n in ((0, 128), (30720, 0), (30721, 128), (10, 2)):
            with self.assertRaises(ValueError):
                rollout_steps(batch, n)

    def test_target_requires_second_checkpoint(self):
        points = [dict(iteration=i*50, training_seconds=i*20., eval_return=r)
                  for i, r in enumerate((0., 9., 7., 10., 11.))]
        result = target_time(points, 8.)
        self.assertTrue(result['target_attained'])
        self.assertEqual(result['target_confirmation_seconds'], 80.)
        self.assertEqual(result['target_confirmation_iteration'], 200)
        self.assertEqual(result['first_crossing_lower_seconds'], 0.)
        self.assertEqual(result['first_crossing_upper_seconds'], 20.)

    def test_final_single_crossing_is_not_confirmation(self):
        result = target_time([dict(iteration=0, training_seconds=0., eval_return=0.),
                              dict(iteration=500, training_seconds=100., eval_return=9.)], 8.)
        self.assertFalse(result['target_attained'])
        self.assertEqual(result['target_confirmation_seconds'], '')

    def test_unavailable_target_is_not_a_fake_success(self):
        result = target_time([dict(iteration=0, training_seconds=0., eval_return=5.)], None)
        self.assertEqual(result['target_confirmation_seconds'], '')
        self.assertEqual(result['target_status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
