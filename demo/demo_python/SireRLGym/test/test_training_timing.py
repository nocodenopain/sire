from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from SireRLGym.scripts.train import _apply_sim_dt_override
from SireRLGym.utils.task_registry import make_env_cfg


def _cfg(dt=0.001, decimation=20):
    return SimpleNamespace(
        sim=SimpleNamespace(dt=dt),
        control=SimpleNamespace(decimation=decimation),
    )


class TrainingTimingTest(unittest.TestCase):
    def test_go2_fall_penalty_is_one_reward_unit_for_both_supported_steps(self):
        env_cfg = make_env_cfg("go2")
        control_dt = float(env_cfg.sim.dt) * int(env_cfg.control.decimation)
        self.assertAlmostEqual(
            float(env_cfg.rewards.scales.termination) * control_dt, -1.0
        )

        _apply_sim_dt_override(env_cfg, 0.005)
        control_dt = float(env_cfg.sim.dt) * int(env_cfg.control.decimation)
        self.assertAlmostEqual(
            float(env_cfg.rewards.scales.termination) * control_dt, -1.0
        )

    def test_five_millisecond_step_preserves_twenty_millisecond_control(self):
        cfg = _cfg()
        _apply_sim_dt_override(cfg, 0.005)
        self.assertEqual(cfg.sim.dt, 0.005)
        self.assertEqual(cfg.control.decimation, 4)
        self.assertTrue(math.isclose(
            cfg.sim.dt * cfg.control.decimation, 0.02, abs_tol=1e-12
        ))

    def test_none_keeps_task_defaults(self):
        cfg = _cfg()
        _apply_sim_dt_override(cfg, None)
        self.assertEqual(cfg.sim.dt, 0.001)
        self.assertEqual(cfg.control.decimation, 20)

    def test_non_integral_decimation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer decimation'):
            _apply_sim_dt_override(_cfg(), 0.003)

    def test_nonpositive_or_nonfinite_step_is_rejected(self):
        for value in (0.0, -0.001, float('nan'), float('inf')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'finite and > 0'):
                    _apply_sim_dt_override(_cfg(), value)


if __name__ == '__main__':
    unittest.main()
