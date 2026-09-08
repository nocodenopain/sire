"""Evaluate all 100 pre-generated first trials, using the unchanged Sire task."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from .common import apply_config, configure_threads, read_json, sha256, verify_sources, write_json


def restore_training_joint_order(cfg, joint_names):
    """JSON key sorting must not change the policy's action/observation layout."""
    angles = cfg.init_state.default_joint_angles
    if set(angles) != set(joint_names):
        raise ValueError('Frozen joint names differ from the training task')
    cfg.init_state.default_joint_angles = {name: angles[name] for name in joint_names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--attempt-dir', type=Path, required=True)
    args = parser.parse_args()
    root, attempt = args.root.resolve(), args.attempt_dir.resolve()
    thread_info = configure_threads()
    protocol = read_json(root / 'protocol.json')
    if protocol.get('schema_version', 1) == 1:
        verify_sources(root)
    result = read_json(attempt / 'training_result.json')
    assert result['status'] == 'complete'
    if protocol.get('schema_version', 1) == 1:
        assert result['checkpoint_sha256'] == sha256(result['checkpoint'])
    trials = read_json(root / 'evaluation_trials.json')
    assert len(trials) == 100 and len({t['trial_id'] for t in trials}) == 100

    import numpy as np
    import torch
    import sire
    from rsl_rl.modules import ActorCritic
    from SireRLGym.envs.base.legged_robot_sire import LeggedRobotSire
    from SireRLGym.utils.task_registry import make_env_cfg
    from SireRLGym.utils.helpers import class_to_dict, set_seed
    from SireRLGym.utils.math import quat_rotate_inverse
    from SireRLGym.utils.joint_order import JointOrderAdapter

    class FixedTrialEnv(LeggedRobotSire):
        collecting = False

        def _reset_dofs(self, env_ids):
            for eid in env_ids.tolist():
                position = self.default_dof_pos[0] * torch.tensor(trials[eid]['joint_multipliers'])
                # Same limit enforcement as the native adapter before a first substep.
                position = torch.maximum(torch.minimum(position, torch.as_tensor(self._dof_limits_hi)),
                                         torch.as_tensor(self._dof_limits_lo)).float()
                self.dof_pos[eid] = position
                self.dof_vel[eid] = 0
                model = self.sire_models[eid]
                full = np.zeros(self._num_motions)
                full[self._motion_idx] = position.numpy()
                sire.setMotionMps(model, full.tolist())
                sire.setMotionMvs(model, [0.] * self._num_motions)
                model.forwardKinematics()
                model.forwardKinematicsVel()

        def _reset_root_states(self, env_ids):
            for eid in env_ids.tolist():
                self.root_states[eid] = self.base_init_state
                r = self.root_states[eid]
                yaw = trials[eid]['yaw']
                r[3:7] = torch.tensor([0., 0., math.sin(yaw/2), math.cos(yaw/2)])
                r[7:13] = torch.tensor(trials[eid]['world_point_velocity'])
                pp = (r[:3] + self._physics_origins[eid]).numpy()
                model = self.sire_models[eid]
                model.link(1).pq = np.concatenate([pp, r[3:7].numpy()])
                model.link(1).vs = np.array(sire.vp2vs(pp, r[7:10].numpy(), r[10:13].numpy()))
                model.forwardKinematics()
                model.forwardKinematicsVel()

        def _resample_commands(self, env_ids, initialize=False):
            for eid in env_ids.tolist():
                # Reset calls occur before episode_length is cleared by the base class.
                step = int(self.episode_length_buf[eid]) if getattr(self, '_in_callback', False) else 0
                command = max((c for c in trials[eid]['commands'] if c['control_step'] <= step),
                              key=lambda c: c['control_step'])
                self.command_targets[eid] = torch.tensor(
                    [command['vx'], command['vy'], 0., command['heading']],
                    dtype=self.commands.dtype, device=self.device,
                )
            if initialize:
                self.commands[env_ids, :3] = self._desired_velocity_commands(env_ids)
                if self.cfg.commands.heading_command:
                    self.commands[env_ids, 3] = self.command_targets[env_ids, 3]

        def _post_physics_step_callback(self):
            self._in_callback = True
            try:
                super()._post_physics_step_callback()
            finally:
                self._in_callback = False

        def compute_reward(self):
            super().compute_reward()
            if not self.collecting:
                return
            # Capture tracking and reasons before reset_idx overwrites terminal state.
            self.linear_error = torch.linalg.vector_norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1).clone()
            self.yaw_error = (self.base_ang_vel[:, 2] - self.commands[:, 2]).abs().clone()
            self.terminal_reasons = {}
            failed = set(self._sire_batch_stepper.recoveredEnvIds)
            oob = self._terrain_out_of_bounds()
            contacts = (torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1).any(dim=1)
            for eid in self.reset_buf.nonzero(as_tuple=False).flatten().tolist():
                reasons = []
                if eid in failed:
                    reasons.append('physics_recovery')
                if bool(self.base_height_fall_buf[eid]):
                    reasons.append('base_height_fall')
                if bool(contacts[eid]):
                    reasons.append('base_contact')
                if bool(oob[eid]):
                    reasons.append('terrain_boundary')
                if self.episode_length_buf[eid] >= self.max_episode_length:
                    reasons.append('time_limit')
                self.terminal_reasons[eid] = '+'.join(reasons) or 'other_task_termination'

    set_seed(protocol['trial_seed'])
    baseline = read_json(root / 'baseline_config.json')
    cfg = make_env_cfg('go2')
    # This insertion order defines LeggedRobotSire.dof_names during training.
    # Capture it from the hash-verified task source before loading sorted JSON.
    training_joint_names = list(cfg.init_state.default_joint_angles)
    apply_config(cfg, baseline['env_cfg'])
    restore_training_joint_order(cfg, training_joint_names)
    cfg.env.num_envs = 100
    cfg.sim.sire_batch_threads = protocol['eval_threads']
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    env = FixedTrialEnv(cfg, sim_params=cfg.sim, physics_engine='sire', sim_device='cpu', headless=True)
    assert env.dof_names == training_joint_names
    env._refresh_sim_tensors_sire()
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env._post_physics_step_callback()  # Heading command at time zero, no physics step.
    env.compute_observations()
    initial = {'dof_names': env.dof_names, 'root_states': env.root_states.tolist(),
               'dof_pos': env.dof_pos.tolist(), 'commands': env.commands.tolist(), 'obs': env.obs_buf.tolist()}
    write_json(attempt / 'evaluation_initial_states.json', initial)
    write_json(attempt / 'evaluation_config.json', class_to_dict(cfg))
    actor = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_actions, **baseline['train_cfg']['policy'])
    checkpoint = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
    assert checkpoint['iter'] == result['completed_iterations']
    if protocol.get('schema_version', 1) == 1 and not result.get('smoke', False):
        assert checkpoint['iter'] == 500
    elif protocol.get('schema_version') == 2:
        # V2 evaluates saved checkpoints and preregistered reference/initial
        # policies; never pretend those inputs are final 500-update trainings.
        assert result['purpose'] == 'offline_checkpoint_evaluation'
        assert checkpoint['iter'] >= 0
    actor.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert all(torch.isfinite(v).all() for v in actor.state_dict().values()), 'Nonfinite checkpoint'
    actor.eval()
    adapter = JointOrderAdapter.from_privileged_obs_dim(env.dof_names, 'cpu', env.num_privileged_obs)
    env.collecting = True
    active = np.ones(100, dtype=bool)
    returns = np.zeros(100)
    steps = np.zeros(100, dtype=np.int64)
    linear = np.zeros(100)
    yaw = np.zeros(100)
    rows = {}
    with torch.inference_mode():
        for _ in range(int(env.max_episode_length)):
            obs = adapter.actor_obs_env_to_policy(env.get_observations(), env.num_actions)
            actions = adapter.actions_policy_to_env(actor.act_inference(obs))
            actions[torch.from_numpy(~active)] = 0
            _, _, rewards, dones, _ = env.step(actions)
            raw = rewards.numpy().astype(float)
            assert np.isfinite(raw).all()
            returns[active] += raw[active]
            steps[active] += 1
            linear[active] += env.linear_error.numpy()[active]
            yaw[active] += env.yaw_error.numpy()[active]
            for eid in np.flatnonzero(active & dones.numpy().astype(bool)):
                rows[int(eid)] = {'trial_id': trials[eid]['trial_id'], 'slot': int(eid),
                    'episode_return': float(returns[eid]), 'control_steps': int(steps[eid]),
                    'duration_seconds': float(steps[eid] * env.dt),
                    'termination_reason': env.terminal_reasons[int(eid)],
                    'mean_linear_velocity_error_m_s': float(linear[eid] / steps[eid]),
                    'mean_yaw_rate_abs_error_rad_s': float(yaw[eid] / steps[eid])}
                active[eid] = False
            if not active.any():
                break
    assert not active.any() and len(rows) == 100
    ordered = [rows[i] for i in range(100)]
    with (attempt / 'evaluation_trials.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    summary = {'status': 'complete', 'trials': 100,
        'checkpoint': result['checkpoint'], 'iteration': checkpoint['iter'],
        'mean_episode_return': float(returns.mean()), 'min_episode_return': float(returns.min()),
        'max_episode_return': float(returns.max()), 'mean_duration_seconds': float(steps.mean() * env.dt),
        'termination_reasons': dict(Counter(r['termination_reason'] for r in ordered)),
        'effective_threads': thread_info, 'actual_sire_threads': env._sire_batch_stepper.threadCount,
        'discarded_slot_episode_policy': 'Finished slots continue with zero action; all later episodes ignored.'}
    if protocol.get('schema_version', 1) == 1:
        summary.update(checkpoint_sha256=result['checkpoint_sha256'],
            trial_manifest_sha256=sha256(root / 'evaluation_trials.json'),
            initial_states_sha256=sha256(attempt / 'evaluation_initial_states.json'))
    write_json(attempt / 'evaluation_result.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
