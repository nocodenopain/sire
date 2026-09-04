from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
import gc
from collections import deque
from pathlib import Path

import torch
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter as _TensorBoardSummaryWriter
except ModuleNotFoundError as error:
    if error.name != "tensorboard":
        raise
    _TensorBoardSummaryWriter = None


class PersistentSummaryWriter:
    """Write TensorBoard events and an always-available scalar JSONL stream."""

    def __init__(self, log_dir, flush_secs=10, tensorboard_writer_cls=None):
        self.log_dir = str(log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.metrics_path = os.path.join(self.log_dir, 'metrics.jsonl')
        self._metrics_file = open(self.metrics_path, 'a', encoding='utf-8', buffering=1)
        self._closed = False
        if tensorboard_writer_cls is False:
            tensorboard_writer_cls = None
        elif tensorboard_writer_cls is None:
            tensorboard_writer_cls = _TensorBoardSummaryWriter
        self._tensorboard = (
            tensorboard_writer_cls(log_dir=self.log_dir, flush_secs=flush_secs)
            if tensorboard_writer_cls is not None
            else None
        )
        if self._tensorboard is None:
            print(
                "[Sire training] tensorboard is not installed; metrics remain "
                f"available at {self.metrics_path}. Install the declared "
                "tensorboard dependency to also create event files.",
                flush=True,
            )
        else:
            print(
                f"[Sire training] TensorBoard log_dir={self.log_dir}; "
                f"scalar_backup={self.metrics_path}",
                flush=True,
            )

    @property
    def tensorboard_enabled(self):
        return self._tensorboard is not None

    @staticmethod
    def _as_float(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        return float(value)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self._closed:
            raise RuntimeError('cannot write to a closed training metric writer')
        scalar = self._as_float(scalar_value)
        timestamp = time.time() if walltime is None else float(walltime)
        if self._tensorboard is not None:
            self._tensorboard.add_scalar(
                tag, scalar, global_step=global_step, walltime=timestamp
            )
        record = {
            'wall_time': timestamp,
            'step': global_step,
            'tag': str(tag),
            'value': scalar if math.isfinite(scalar) else None,
        }
        if not math.isfinite(scalar):
            record['nonfinite_value'] = str(scalar)
        self._metrics_file.write(json.dumps(record, separators=(',', ':')) + '\n')

    def flush(self):
        if self._closed:
            return
        self._metrics_file.flush()
        if self._tensorboard is not None:
            self._tensorboard.flush()

    def close(self):
        if self._closed:
            return
        self.flush()
        if self._tensorboard is not None:
            self._tensorboard.close()
        self._metrics_file.close()
        self._closed = True

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from SireRLGym.runners.infinite_scheduler import InfiniteLevelScheduler
from SireRLGym.utils.helpers import class_to_dict
from SireRLGym.utils.joint_order import JointOrderAdapter


class OnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg['runner']
        self.alg_cfg = train_cfg['algorithm']
        self.policy_cfg = train_cfg['policy']
        self.device = device
        self.env = env
        self.joint_order_adapter = JointOrderAdapter.from_privileged_obs_dim(
            getattr(self.env, 'dof_names', []),
            device=self.device,
            privileged_obs_dim=self.env.num_privileged_obs,
        )

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        actor_critic = ActorCritic(self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg).to(self.device)
        self.alg = PPO(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg['num_steps_per_env']
        self.save_interval = self.cfg['save_interval']
        self.debug_reward = bool(self.cfg.get('debug_reward', False))
        self.log_episode_keys = self.cfg.get('log_episode_keys')
        self.infinite_mode = bool(self.cfg.get('infinite_mode', False))
        self.infinite_scheduler = (
            InfiniteLevelScheduler(self.env, self.cfg, log_dir=log_dir) if self.infinite_mode else None
        )
        self.visualize_interval = self.cfg.get('visualize_interval', None)
        self.visualize_resource_path = self.cfg.get('visualize_resource_path', None)
        self.replay_history_enabled = (
            self.visualize_interval is not None
            or bool(getattr(self.env.cfg.sim, 'sire_diagnostics', False))
        )
        self.recording_policy_applied = self.env.set_sire_recording_env(
            0 if self.replay_history_enabled else -1
        )
        self.vis_dir = os.path.join(log_dir, 'vis') if log_dir else None
        if self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)
        # Let the env save debug recordings to the same vis/ directory.
        self.env._diag_vis_dir = self.vis_dir

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

        if self.log_dir is not None:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            all_cfg = {'train_cfg': train_cfg, 'env_cfg': class_to_dict(self.env.cfg)}
            yaml.safe_dump(all_cfg, open(os.path.join(self.log_dir, 'config.yaml'), 'w'))

    def _format_debug_lines(self, pairs, prefix, pad, per_line=3):
        if not pairs:
            return ''
        lines = []
        for start in range(0, len(pairs), per_line):
            chunk = pairs[start:start + per_line]
            label = prefix if start == 0 else ''
            lines.append(f"{label:>{pad}} " + " | ".join(chunk) + "\n")
        return ''.join(lines)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state is not None:
            self.alg.optimizer.load_state_dict(optimizer_state)
        stored_iter = int(checkpoint.get('iter', 0))
        if stored_iter == 0:
            match = re.match(r'^model_(\d+)\.pt$', Path(path).name)
            if match:
                stored_iter = int(match.group(1))
        self.current_learning_iteration = stored_iter
        self.tot_timesteps = int(
            checkpoint.get(
                'tot_timesteps',
                stored_iter * self.num_steps_per_env * self.env.num_envs,
            )
        )
        self.tot_time = float(checkpoint.get('tot_time', 0.0))
        return checkpoint

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = PersistentSummaryWriter(
                log_dir=self.log_dir, flush_secs=10
            )
        if self.infinite_scheduler is not None:
            self.infinite_scheduler.initialize(self.current_learning_iteration, self.tot_time, self.tot_timesteps)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        obs = self.joint_order_adapter.actor_obs_env_to_policy(obs, self.env.num_actions)
        privileged_obs = self.joint_order_adapter.critic_obs_env_to_policy(privileged_obs, self.env.num_actions)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.alg.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        resume_tot_time = self.tot_time
        it = self.current_learning_iteration
        total_iterations = None if self.infinite_mode else (self.current_learning_iteration + num_learning_iterations)
        while self.infinite_mode or it < total_iterations:
            it += 1
            start = time.time()
            max_abs_dof_velocity = 0.0
            physics_failures_before = int(
                getattr(self.env, '_sire_physics_failure_count', 0)
            )
            iteration_episode_returns = []

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    env_actions = self.joint_order_adapter.actions_policy_to_env(actions)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(env_actions)
                    if not getattr(self.env, 'headless', True):
                        self.env.render()
                    obs = self.joint_order_adapter.actor_obs_env_to_policy(obs, self.env.num_actions)
                    privileged_obs = self.joint_order_adapter.critic_obs_env_to_policy(privileged_obs, self.env.num_actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    if hasattr(self.env, 'dof_vel'):
                        max_abs_dof_velocity = max(
                            max_abs_dof_velocity,
                            float(self.env.dof_vel.detach().abs().max().item()),
                        )

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        completed_returns = cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        rewbuffer.extend(completed_returns)
                        iteration_episode_returns.extend(completed_returns)
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)
                rollout_diagnostics = self._capture_rollout_diagnostics()

            update_out = self.alg.update()
            if isinstance(update_out, tuple):
                mean_value_loss = update_out[0]
                mean_surrogate_loss = update_out[1]
                if len(update_out) == 3:
                    # The repository-pinned rsl_rl PPO returns
                    # (value_loss, surrogate_loss, symmetry_loss).
                    mean_entropy = self._policy_entropy()
                    mean_sym_loss = update_out[2]
                else:
                    # Newer rsl_rl versions return entropy and optionally RND
                    # before symmetry loss.
                    mean_entropy = update_out[2] if len(update_out) > 2 else self._policy_entropy()
                    mean_sym_loss = update_out[4] if len(update_out) > 4 else None
            else:
                mean_value_loss = update_out
                mean_surrogate_loss = 0.0
                mean_entropy = self._policy_entropy()
                mean_sym_loss = None
            optimizer_diagnostics = self._capture_optimizer_diagnostics()
            physics_recoveries = int(
                getattr(self.env, '_sire_physics_failure_count', 0)
            ) - physics_failures_before
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'), iteration=it)
            if self.visualize_interval is not None and it % self.visualize_interval == 0:
                self._save_recording(it)
            if self.replay_history_enabled or not self.recording_policy_applied:
                # A PPO rollout boundary is not an episode boundary. Preserve
                # simulator/model/timer state and clear only replay storage.
                self.env.resetSireRecorders()
            # Force GC to release pybind11-held C++ wrappers (motionPool, partPool, etc.)
            if it % 5 == 0:
                gc.collect()
            stop_training = False
            transitioned_level = False
            if self.infinite_scheduler is not None:
                self.current_learning_iteration = int(it)
                stop_training, transitioned_level = self.infinite_scheduler.handle_iteration(
                    it, ep_infos, self.tot_time, self.tot_timesteps, self.save,
                )
            ep_infos.clear()
            if transitioned_level:
                obs = self.env.get_observations()
                privileged_obs = self.env.get_privileged_observations()
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
            # print(f"DEBUG: it={it}, total={total_iterations}, continue={it < total_iterations}", flush=True)
            if stop_training:
                break

        self.current_learning_iteration = it
        self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'), iteration=self.current_learning_iteration)
        if self.writer is not None:
            self.writer.flush()
        # if self.visualize_interval is not None:
        #     self._visualize_end()

    @staticmethod
    def _stats(prefix, tensor):
        values = tensor.detach().float()
        if values.numel() == 0:
            return {}
        finite = torch.isfinite(values)
        result = {f'{prefix}_nonfinite_count': float((~finite).sum().item())}
        if not bool(finite.any()):
            return result
        values = values[finite]
        result.update(
            {
                f'{prefix}_mean': float(values.mean().item()),
                f'{prefix}_std': float(values.std(unbiased=False).item()),
                f'{prefix}_min': float(values.min().item()),
                f'{prefix}_max': float(values.max().item()),
                f'{prefix}_abs_max': float(values.abs().max().item()),
            }
        )
        return result

    def _capture_rollout_diagnostics(self):
        storage = self.alg.storage
        steps = int(storage.step)
        diagnostics = {}
        if steps <= 0:
            return diagnostics
        rewards = storage.rewards[:steps]
        values = storage.values[:steps]
        returns = storage.returns[:steps]
        diagnostics.update(self._stats('reward', rewards))
        diagnostics.update(self._stats('value', values))
        diagnostics.update(self._stats('return', returns))
        diagnostics.update(self._stats('raw_advantage', returns - values))
        return diagnostics

    def _capture_optimizer_diagnostics(self):
        totals = {
            'actor_adam_second_moment_max': 0.0,
            'critic_adam_second_moment_max': 0.0,
            'other_adam_second_moment_max': 0.0,
            'last_gradient_abs_max': 0.0,
        }
        for name, parameter in self.alg.actor_critic.named_parameters():
            if parameter.grad is not None:
                totals['last_gradient_abs_max'] = max(
                    totals['last_gradient_abs_max'],
                    float(parameter.grad.detach().abs().max().item()),
                )
            state = self.alg.optimizer.state.get(parameter, {})
            second_moment = state.get('exp_avg_sq')
            if second_moment is None or second_moment.numel() == 0:
                continue
            if name.startswith('actor.'):
                key = 'actor_adam_second_moment_max'
            elif name.startswith('critic.'):
                key = 'critic_adam_second_moment_max'
            else:
                key = 'other_adam_second_moment_max'
            totals[key] = max(
                totals[key], float(second_moment.detach().max().item())
            )
        return totals

    def _policy_entropy(self):
        std = self.alg.actor_critic.std.detach().float().clamp_min(1e-12)
        return float((torch.log(std) + 0.5 * math.log(2.0 * math.pi * math.e)).sum().item())

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # Keep parity with rsl_rl-style episode logging from infos["episode"].
        ep_string = ""
        filtered_ep_string = ""
        if locs.get('ep_infos'):
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.tensor([ep_info[key]], device=self.device, dtype=torch.float)
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor).item()
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                line = f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"
                ep_string += line
                if self.log_episode_keys is None or key in self.log_episode_keys:
                    filtered_ep_string += line

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        if locs.get('mean_sym_loss') is not None:
            self.writer.add_scalar('Loss/symmetry', locs['mean_sym_loss'], locs['it'])
        mean_std = self.alg.actor_critic.std.mean()
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        if locs.get('iteration_episode_returns'):
            returns = locs['iteration_episode_returns']
            self.writer.add_scalar('Diagnostics/episode_return_min', min(returns), locs['it'])
            self.writer.add_scalar('Diagnostics/episode_return_max', max(returns), locs['it'])
        self.writer.add_scalar(
            'Diagnostics/dof_velocity_abs_max',
            locs.get('max_abs_dof_velocity', 0.0),
            locs['it'],
        )
        self.writer.add_scalar(
            'Diagnostics/physics_recoveries',
            locs.get('physics_recoveries', 0),
            locs['it'],
        )
        self.writer.add_scalar(
            'Diagnostics/physics_recoveries_total',
            int(getattr(self.env, '_sire_physics_failure_count', 0)),
            locs['it'],
        )
        for key, value in locs.get('rollout_diagnostics', {}).items():
            self.writer.add_scalar(f'Diagnostics/{key}', value, locs['it'])
        for key, value in locs.get('optimizer_diagnostics', {}).items():
            self.writer.add_scalar(f'Optimizer/{key}', value, locs['it'])
        if self.debug_reward and isinstance(getattr(self.env, 'reward_debug_info', None), dict):
            for k, v in self.env.reward_debug_info.items():
                self.writer.add_scalar(f'RewardDebug/{k}', float(v), locs['it'])

        total_label = 'inf' if self.infinite_mode else str(self.current_learning_iteration + locs['num_learning_iterations'])
        title = f" \033[1m Learning iteration {locs['it']}/{total_label} \033[0m "
        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"{'#' * width}\n"
                f"{title.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                f"{'Mean entropy:':>{pad}} {locs.get('mean_entropy', 0.0):.4f}\n"
                + (f"{'Symmetry loss:':>{pad}} {locs['mean_sym_loss']:.4f}\n" if locs.get('mean_sym_loss') is not None else "")
                + f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n"
                f"{title.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                f"{'Mean entropy:':>{pad}} {locs.get('mean_entropy', 0.0):.4f}\n"
                + (f"{'Symmetry loss:':>{pad}} {locs['mean_sym_loss']:.4f}\n" if locs.get('mean_sym_loss') is not None else "")
                + f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            )

        if self.debug_reward and isinstance(getattr(self.env, 'reward_debug_info', None), dict):
            dbg_pairs = []
            for k, v in sorted(self.env.reward_debug_info.items()):
                dbg_pairs.append(f"{k}={float(v):.4f}")
            log_string += self._format_debug_lines(dbg_pairs, 'Reward debug:', pad, per_line=3)

        log_string += filtered_ep_string if self.log_episode_keys is not None else ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
        )
        if self.infinite_mode:
            log_string += f"{'ETA:':>{pad}} n/a\n"
            if self.infinite_scheduler is not None and self.infinite_scheduler.initialized:
                summary = self.infinite_scheduler.level_summary(locs['it'], self.tot_time, self.tot_timesteps)
                log_string += (
                    f"{'Infinite level:':>{pad}} idx={summary['level_index']} "
                    f"height={summary['threshold_height']:.3f} "
                    f"episodes={summary['episodes']} "
                    f"iterations={summary['iterations_spent']}\n"
                )
                log_string += (
                    f"{'Infinite recent:':>{pad}} "
                    f"success={summary['recent_success_rate']:.3f} "
                    f"pass={summary['recent_pass_rate']:.3f} "
                    f"failure={summary['recent_failure_rate']:.3f} "
                    f"episodes={summary['recent_episodes']}\n"
                )
                log_string += (
                    f"{'Infinite cumulative:':>{pad}} "
                    f"success={summary['success_rate']:.3f} "
                    f"pass={summary['pass_rate']:.3f} "
                    f"failure={summary['failure_rate']:.3f} "
                    f"best_rel_x={summary['mean_best_rel_x']:.3f}\n"
                )
        else:
            completed = max(1, locs['it'] - self.current_learning_iteration)
            remaining = max(0, int(locs['total_iterations']) - locs['it'])
            recent_time = max(0.0, self.tot_time - locs.get('resume_tot_time', 0.0))
            seconds_per_iteration = recent_time / completed
            eta = seconds_per_iteration * remaining
            log_string += f"{'ETA:':>{pad}} {eta:.1f}s\n"
        # Training commonly runs under systemd with stdout redirected to a
        # regular file.  Flush once per PPO iteration so progress is visible
        # immediately without restoring the expensive per-scalar CSV writer.
        print(log_string, flush=True)
        self.writer.flush()

    def save(self, path, iteration=None):
        if iteration is None:
            iteration = self.current_learning_iteration
        torch.save(
            {
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'iter': int(iteration),
                'tot_timesteps': int(self.tot_timesteps),
                'tot_time': float(self.tot_time),
            },
            path,
        )

    def _save_recording(self, iteration):
        """Save env 0's recorder data to a JSON file.

        Called at each ``visualize_interval`` during training.
        No extra simulation stepping — just exports whatever is already
        in env 0's recorder.
        """
        sl = self.env.sire_sim_loops[0]
        m = self.env.sire_models[0]
        sim = self.env.sire_simulators[0]

        with torch.inference_mode():
            result = sl.recordsToJson()
            display_init = sim.displayInitJson()

        recording = {
            'nlinks': int(m.nbody),
            'display_init': display_init,
            'frames': result,
        }

        vis_dir = self.vis_dir or self.log_dir
        recording_path = os.path.join(vis_dir, f'recording_{iteration}.json')
        with open(recording_path, 'w') as f:
            json.dump(recording, f)

        print(
            f"[visualize] Recording saved \u2192 {recording_path} "
            f"({len(result.get('timeIndex', []))} frames)",
            flush=True,
        )

    def _visualize_end(self):
        """Run trained policy on env 0 with recording, then show in meshcat.

        Called once at the end of ``learn()`` when ``--visualize_interval``
        (or ``train_cfg.runner.visualize_interval``) is set.
        """
        import sire
        import meshcat

        with torch.inference_mode():

            sl = self.env.sire_sim_loops[0]
            m = self.env.sire_models[0]
            sim = self.env.sire_simulators[0]
            display_init = sim.displayInitJson()

            # Reset env 0 for a clean evaluation
            self.env.reset_idx(torch.tensor([0], device=self.device))

            # Get initial observations
            obs = self.env.get_observations()
            obs = self.joint_order_adapter.actor_obs_env_to_policy(obs, self.env.num_actions)

            self.alg.actor_critic.eval()

            # Step env 0 with the trained policy (~1 s of sim time).
            # Uses siRe event-loop directly on env 0 only, NOT env.step().
            num_ctrl = 100
            for _ in range(num_ctrl):
                with torch.no_grad():
                    actions = self.alg.actor_critic.act(obs)
                self.env.actions[0] = actions[0]

                while not sl.headerIsCtrl():
                    self.env._update_actuator_torque(0)
                    sl.handleContact()
                self.env._update_actuator_torque(0)
                sl.handleContact()

                # Refresh obs for next action (read from env 0)
                obs = self.env.get_observations()
                obs = self.joint_order_adapter.actor_obs_env_to_policy(obs, self.env.num_actions)

            self.alg.actor_critic.train()

            result = sl.recordsToJson()

        # Auto-detect resource path (outside inference_mode, no tensor ops)
        resource_path = self.visualize_resource_path
        if resource_path is None:
            runner_path = Path(__file__).resolve()
            candidate = runner_path.parents[2] / 'dogRL'
            if candidate.is_dir():
                resource_path = str(candidate)
            else:
                resource_path = str(runner_path.parents[1] / '..' / 'dogRL')

        vis = meshcat.Visualizer()
        vis.open()
        sire.robotInit(int(m.nbody), resource_path, display_init, vis)

        sire.animateRobotByRecords(int(m.nbody), result, 1000, vis)
        input("Press Enter to close the meshcat visualizer...")
