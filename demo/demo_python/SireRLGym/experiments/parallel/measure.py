"""Measurement hooks only: no changes to PPO, rewards or simulator stepping."""
from __future__ import annotations

import csv
import math
import statistics
import time
from pathlib import Path

from .common import (baseline_config, configure_threads, read_json, sha256,
                     state_hash, verify_sources, write_json)


class ExperimentAttempt:
    def __init__(self, context_path):
        self.context = read_json(context_path)
        self.path = Path(context_path).parent
        self.root = Path(self.context['root'])
        self.protocol = read_json(self.root / 'protocol.json')
        self.started = None
        self.rows = []
        self.saves = []
        self.stream = None

    def configure_threads(self):
        self.threads = configure_threads()
        verify_sources(self.root)
        if sha256(self.root / 'protocol.json') != self.context['protocol_sha256']:
            raise RuntimeError('Protocol hash mismatch')

    def validate_config(self, args, env_cfg, train_cfg):
        from SireRLGym.utils.helpers import class_to_dict
        assert args.task == 'go2' and args.resume is None and not args.head
        assert not args.debug_reward and args.visualize_interval is None
        assert not args.infinite_mode and args.scene_curriculum_dir is None
        expected = read_json(self.root / 'baseline_config.json')
        expected['env_cfg']['env']['num_envs'] = self.context['num_envs']
        expected['env_cfg']['sim']['sire_batch_threads'] = self.context['sire_batch_threads']
        expected['train_cfg']['runner']['max_iterations'] = self.context['updates']
        actual = {'env_cfg': class_to_dict(env_cfg), 'train_cfg': class_to_dict(train_cfg)}
        if actual != expected:
            write_json(self.path / 'unexpected_config.json', actual)
            raise RuntimeError('Training configuration differs from frozen baseline')
        self.steps = int(train_cfg.runner.num_steps_per_env)  # Read, never override.
        assert self.steps == self.protocol['num_steps_per_env']
        write_json(self.path / 'requested_config.json', actual)

    def attach(self, runner):
        import torch
        self.runner = runner
        initial_path = self.root / 'initial_weights.pt'
        assert sha256(initial_path) == self.protocol['initial_weights_file_sha256']
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
        runner.alg.actor_critic.load_state_dict(initial, strict=True)
        assert state_hash(runner.alg.actor_critic.state_dict()) == self.protocol['initial_state_sha256']
        self.metadata = {**self.context, 'log_dir': runner.log_dir, 'effective_threads': self.threads,
                         'actual_sire_threads': runner.env._sire_batch_stepper.threadCount,
                         'initial_state_sha256': state_hash(runner.alg.actor_critic.state_dict()),
                         'optimizer_initial_state_entries': len(runner.alg.optimizer.state),
                         'num_steps_per_env': self.steps, 'status': 'initialized'}
        assert self.metadata['actual_sire_threads'] == self.context['sire_batch_threads']
        write_json(self.path / 'training_result.json', self.metadata)
        runner.experiment_observer = self
        original_save = runner.save

        def measured_save(path, iteration=None):
            start = time.perf_counter()
            result = original_save(path, iteration)
            self.saves.append({'iteration': iteration, 'seconds': time.perf_counter() - start,
                               'after_final_update': len(self.rows) == self.context['updates']})
            return result

        runner.save = measured_save
        self.stream = (self.path / 'iterations.csv').open('x', newline='', buffering=1)
        self.writer = None

    def collection_start(self, iteration):
        self.collection_started = time.perf_counter()
        if self.started is None:
            assert iteration == 1
            self.started = self.collection_started

    def collection_end(self):
        self.collection_finished = time.perf_counter()

    def update_end(self):
        self.update_finished = time.perf_counter()

    def iteration_end(self, iteration, rewards, lengths, recoveries, diagnostics, value_loss):
        n = self.context['num_envs'] * self.steps
        row = {k: self.context[k] for k in ('run_id', 'attempt', 'num_envs', 'sire_batch_threads')}
        row.update(num_steps_per_env=self.steps, iteration=iteration,
                   transitions_this_iteration=n, total_transitions=iteration * n,
                   total_physics_substeps=iteration * n * 4,
                   wall_elapsed_seconds=self.update_finished - self.started,
                   collection_seconds=self.collection_finished - self.collection_started,
                   learning_seconds=self.update_finished - self.collection_finished,
                   train_mean_episode_return=statistics.mean(rewards) if rewards else '',
                   train_mean_episode_length=statistics.mean(lengths) if lengths else '',
                   physics_failure_or_recovery_count=recoveries, value_loss=float(value_loss),
                   nonfinite_count=sum(v for k, v in diagnostics.items() if k.endswith('_nonfinite_count')))
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        self.rows.append(row)
        if row['nonfinite_count'] or not math.isfinite(row['value_loss']):
            raise RuntimeError('Nonfinite training result; stop matrix, preserve this attempt')

    def complete(self):
        assert len(self.rows) == self.context['updates']
        checkpoint = Path(self.runner.log_dir) / f"model_{self.context['updates']}.pt"
        self.metadata.update(status='complete', completed_iterations=len(self.rows),
                             total_transitions=self.rows[-1]['total_transitions'],
                             total_physics_substeps=self.rows[-1]['total_physics_substeps'],
                             training_wall_seconds=self.rows[-1]['wall_elapsed_seconds'],
                             collection_seconds=sum(r['collection_seconds'] for r in self.rows),
                             learning_seconds=sum(r['learning_seconds'] for r in self.rows),
                             physics_recoveries=sum(r['physics_failure_or_recovery_count'] for r in self.rows),
                             final_save_seconds=sum(s['seconds'] for s in self.saves if s['after_final_update']),
                             saves=self.saves, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        self.metadata['other_in_interval_seconds'] = (self.metadata['training_wall_seconds']
            - self.metadata['collection_seconds'] - self.metadata['learning_seconds'])
        write_json(self.path / 'training_result.json', self.metadata)

    def close(self):
        if self.stream is not None:
            self.stream.close()
