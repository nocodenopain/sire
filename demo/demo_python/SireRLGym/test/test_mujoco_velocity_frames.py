from __future__ import annotations

import math
import unittest

import torch

from RLGym.utils.math import mujoco_free_joint_velocity_to_base


class MuJoCoVelocityFramesTest(unittest.TestCase):
    def test_free_joint_angular_velocity_is_not_rotated_twice(self):
        half_yaw = math.pi / 4.0
        quat = torch.tensor(
            [[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]],
            dtype=torch.float64,
        )
        qvel = torch.tensor(
            [[1.0, 0.0, 0.0, 0.3, -0.4, 0.5]], dtype=torch.float64
        )

        base_linear, base_angular = mujoco_free_joint_velocity_to_base(
            quat, qvel
        )

        torch.testing.assert_close(
            base_linear, torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        )
        torch.testing.assert_close(base_angular, qvel[:, 3:6])


if __name__ == '__main__':
    unittest.main()
