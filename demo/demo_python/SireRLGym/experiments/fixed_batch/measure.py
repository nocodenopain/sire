"""Opt-in fixed batch and basic accounting. No model/source hashing."""
from __future__ import annotations

from SireRLGym.experiments.parallel.common import configure_threads, read_json, write_json
from SireRLGym.experiments.parallel.measure import ExperimentAttempt


def rollout_steps(batch, num_envs, minibatches=4):
    if batch <= 0 or num_envs <= 0 or minibatches <= 0:
        raise ValueError('Batch, environments and minibatches must be positive')
    if batch % num_envs or batch % minibatches:
        raise ValueError('Global batch must divide exactly into environments and minibatches')
    return batch // num_envs


class FixedBatchAttempt(ExperimentAttempt):
    def configure_threads(self):
        self.threads = configure_threads()

    def validate_config(self, args, env_cfg, train_cfg):
        from SireRLGym.utils.helpers import class_to_dict
        assert args.task == 'go2' and args.resume is None and not args.head
        assert not args.debug_reward and args.visualize_interval is None
        assert not args.infinite_mode and args.scene_curriculum_dir is None
        batch = self.protocol['global_batch_size']
        steps = rollout_steps(batch, self.context['num_envs'], self.protocol['ppo_minibatches'])
        assert self.context['num_steps_per_env'] == steps
        train_cfg.runner.num_steps_per_env = steps
        expected = read_json(self.root / 'baseline_config.json')
        expected['env_cfg']['env']['num_envs'] = self.context['num_envs']
        expected['env_cfg']['sim']['sire_batch_threads'] = self.context['sire_batch_threads']
        expected['train_cfg']['runner']['max_iterations'] = self.context['updates']
        expected['train_cfg']['runner']['num_steps_per_env'] = steps
        actual = {'env_cfg': class_to_dict(env_cfg), 'train_cfg': class_to_dict(train_cfg)}
        if actual != expected:
            write_json(self.path / 'unexpected_config.json', actual)
            raise RuntimeError('Configuration differs from the fixed-batch experiment')
        self.steps = steps
        write_json(self.path / 'requested_config.json', actual)
        print(f'fixed_batch B={batch} N={self.context["num_envs"]} H={steps} '
              f'minibatch={batch // self.protocol["ppo_minibatches"]}', flush=True)

    def attach(self, runner):
        import torch
        self.runner = runner
        assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
        initial = torch.load(self.root / 'initial_weights.pt', map_location='cpu', weights_only=True)
        runner.alg.actor_critic.load_state_dict(initial, strict=True)
        self.metadata = dict(self.context, log_dir=runner.log_dir, effective_threads=self.threads,
            actual_sire_threads=runner.env._sire_batch_stepper.threadCount,
            num_steps_per_env=self.steps, status='initialized')
        assert self.metadata['actual_sire_threads'] == self.context['sire_batch_threads']
        write_json(self.path / 'training_result.json', self.metadata)
        runner.experiment_observer = self
        self.stream = (self.path / 'iterations.csv').open('x', newline='', buffering=1)
        self.writer = None

    def complete(self):
        from pathlib import Path
        assert len(self.rows) == self.context['updates']
        checkpoint = Path(self.runner.log_dir) / f'model_{self.context["updates"]}.pt'
        assert checkpoint.is_file()
        assert self.rows[-1]['total_transitions'] == len(self.rows) * self.protocol['global_batch_size']
        self.metadata.update(status='complete', completed_iterations=len(self.rows),
            total_transitions=self.rows[-1]['total_transitions'],
            nominal_physics_substeps=self.rows[-1]['total_physics_substeps'],
            training_wall_seconds=self.rows[-1]['wall_elapsed_seconds'],
            collection_seconds=sum(r['collection_seconds'] for r in self.rows),
            learning_seconds=sum(r['learning_seconds'] for r in self.rows),
            physics_recoveries=sum(r['physics_failure_or_recovery_count'] for r in self.rows),
            checkpoint=str(checkpoint))
        self.metadata['other_seconds'] = (self.metadata['training_wall_seconds']
            - self.metadata['collection_seconds'] - self.metadata['learning_seconds'])
        write_json(self.path / 'training_result.json', self.metadata)
