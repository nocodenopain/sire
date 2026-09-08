"""Rebuild raw summaries and publication figures from actual completed attempts only."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .common import read_json, sha256


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    manifest = read_json(root / 'manifest.json')
    rows, curves, configs = [], [], []
    for run in manifest['runs']:
        selected = None
        for context_path in sorted((root / 'runs' / run['run_id']).glob('attempt*/context.json')):
            attempt = context_path.parent
            context = read_json(context_path)
            assert context['protocol_sha256'] == manifest['protocol_sha256']
            iterations = read_csv(attempt / 'iterations.csv')
            result = read_json(attempt / 'training_result.json') if (attempt / 'training_result.json').exists() else {}
            evaluation = read_json(attempt / 'evaluation_result.json') if (attempt / 'evaluation_result.json').exists() else {}
            exit_record = read_json(attempt / 'train_exit.json') if (attempt / 'train_exit.json').exists() else {}
            complete = (attempt / 'complete.json').exists()
            row = {**run, 'attempt': context['attempt'], 'attempt_dir': str(attempt),
                   'status': 'complete' if complete else ('failed' if exit_record.get('exit_code', 0) else 'incomplete'),
                   'exit_code': exit_record.get('exit_code', ''), 'actual_iterations': len(iterations),
                   'total_transitions': int(iterations[-1]['total_transitions']) if iterations else 0,
                   'T_500_seconds': result.get('training_wall_seconds', '') if len(iterations) == 500 else '',
                   'collection_seconds': sum(float(r['collection_seconds']) for r in iterations),
                   'learning_seconds': sum(float(r['learning_seconds']) for r in iterations),
                   'other_in_interval_seconds': result.get('other_in_interval_seconds', ''),
                   'final_save_seconds': result.get('final_save_seconds', ''),
                   'physics_recoveries': sum(int(r['physics_failure_or_recovery_count']) for r in iterations),
                   'nonfinite_count': sum(float(r['nonfinite_count']) for r in iterations),
                   'checkpoint_sha256': result.get('checkpoint_sha256', ''),
                   'eval_trials': evaluation.get('trials', 0),
                   'eval_mean_return': evaluation.get('mean_episode_return', ''),
                   'eval_mean_duration_seconds': evaluation.get('mean_duration_seconds', ''),
                   'eval_termination_reasons': json.dumps(evaluation.get('termination_reasons', {}), sort_keys=True)}
            rows.append(row)
            curves.extend(iterations)
            if complete:
                assert selected is None, 'No repeated complete trainings are allowed'
                assert len(iterations) == 500 and evaluation['trials'] == 100
                assert evaluation['checkpoint_sha256'] == sha256(result['checkpoint'])
                selected = row
        configs.append(selected or {**run, 'status': 'pending'})
    write_csv(root / 'summary_by_run.csv', rows,
              list(dict.fromkeys(k for row in rows for k in row)) or ['run_id', 'status'])
    write_csv(root / 'summary_by_config.csv', configs)
    write_csv(root / 'reward_vs_iteration_and_wall_time.csv', curves,
              list(curves[0]) if curves else ['run_id', 'iteration', 'wall_elapsed_seconds', 'train_mean_episode_return'])
    return rows, configs, curves


def plot(root, configs, curves):
    assert len(configs) == 12 and all(r['status'] == 'complete' for r in configs), 'No final figure from an incomplete matrix'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
        'legend.fontsize': 6, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': .5, 'lines.linewidth': .9, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white'})
    colors = {128: '#1f77b4', 256: '#ff7f0e', 512: '#2ca02c', 1024: '#d62728'}
    worker_colors = {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c'}
    markers = {4: 'o', 8: 'x', 16: '^'}
    def style(axis):
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.18, linewidth=.4)
        axis.tick_params(width=.5, length=2)
        axis.set_axisbelow(True)
    def save(fig, name):
        for extension in ('pdf', 'svg', 'png'):
            # Preserve exact physical dimensions; bbox_inches='tight' would change them.
            fig.savefig(root / f'{name}.{extension}', dpi=600)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 1.77))
    fig.subplots_adjust(left=.072, right=.98, bottom=.26, top=.70, wspace=.30)
    for axis in axes:
        style(axis)
    for worker in (4, 8, 16):
        points = sorted((r for r in configs if r['sire_batch_threads'] == worker), key=lambda r: r['num_envs'])
        axes[0].plot(range(4), [r['T_500_seconds']/60 for r in points], '-o',
                     color=worker_colors[worker], markersize=3, label=str(worker))
    axes[0].set_xticks(range(4), ['128', '256', '512', '1024'])
    axes[0].set_xlabel('Number of environments')
    axes[0].set_ylabel('Training time [min]')
    axes[0].set_title('(a) Training time', pad=23, loc='left')
    axes[0].legend(title='Sire threads', title_fontsize=6, ncol=3, frameon=False,
                   loc='lower left', bbox_to_anchor=(0, 1), borderaxespad=0,
                   handlelength=1.5, columnspacing=1)
    for row in configs:
        axes[1].scatter(row['T_500_seconds']/60, row['eval_mean_return'],
                        c=colors[row['num_envs']], marker=markers[row['sire_batch_threads']],
                        s=20, linewidths=.8)
    axes[1].set_xlabel('Training time [min]')
    axes[1].set_ylabel('Final evaluation return')
    axes[1].set_title('(b) Time–return trade-off', pad=23, loc='left')
    env_handles = [Line2D([], [], color=c, marker='o', linestyle='none', markersize=3, label=str(n)) for n, c in colors.items()]
    thread_handles = [Line2D([], [], color='0.25', marker=m, linestyle='none', markersize=3, label=str(t)) for t, m in markers.items()]
    # Separate legends live above the data area, never on top of points.
    first = axes[1].legend(handles=env_handles, title='Environments', title_fontsize=6,
        ncol=4, loc='lower left', bbox_to_anchor=(-.04, 1), frameon=False,
        borderaxespad=0, handlelength=.7, handletextpad=.35, columnspacing=.7)
    axes[1].add_artist(first)
    axes[1].legend(handles=thread_handles, title='Threads', title_fontsize=6,
        ncol=3, loc='lower right', bbox_to_anchor=(1.03, 1), frameon=False,
        borderaxespad=0, handlelength=.7, handletextpad=.35, columnspacing=.7)
    save(fig, 'fig_sire_parallel_tradeoff')
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.25), sharey=True)
    fig.subplots_adjust(left=.09, right=.99, bottom=.22, top=.78, wspace=.17)
    for axis, threads in zip(axes, (4, 8, 16)):
        style(axis)
        axis.set_title(f'{threads} Sire threads')
        axis.set_xlabel('Wall-clock training time [min]')
        for envs, color in colors.items():
            row = next(r for r in configs if r['num_envs'] == envs and r['sire_batch_threads'] == threads)
            raw = [r for r in curves if r['run_id'] == row['run_id'] and int(r['attempt']) == row['attempt'] and r['train_mean_episode_return'] != '']
            x = [float(r['wall_elapsed_seconds']) / 60 for r in raw]
            y = [float(r['train_mean_episode_return']) for r in raw]
            smooth = [statistics.mean(y[max(0, i-9):i+1]) for i in range(len(y))]
            axis.plot(x, y, color=color, alpha=.15, linewidth=.4)
            axis.plot(x, smooth, color=color, label=str(envs))
    axes[0].set_ylabel('Training mean episode return')
    fig.legend(*axes[0].get_legend_handles_labels(), title='Environments (10-update trailing mean; faint raw)',
               title_fontsize=7, ncol=4, loc='upper center', frameon=False)
    save(fig, 'fig_sire_learning_curves')


def report(root, rows, configs):
    complete = [r for r in configs if r['status'] == 'complete']
    lines = ['# Sire parallel training experiment', '',
        f'Completeness: {len(complete)}/12 configurations; one formal training per configuration, 500 updates each.', '',
        '## Protocol and interpretation', '',
        'Go2 flat ground; physics 0.005 s × 4 substeps; control 0.020 s; default 120 rollout steps per environment per update, read without override. PPO: 5 epochs, 4 minibatches. No curriculum. Same saved untrained actor/critic/std and fresh Adam state for every training. Randomized order is frozen in manifest.json.', '',
        'Fixed CPU budgets: PyTorch intra-op 4 / inter-op 1, OMP/MKL 4, OpenBLAS/NumExpr 1; Sire threads vary independently. Existing scalar/diagnostic logging, checkpoint every 50, and GC every 5 updates are identical. No viewer/replay. Hardware, power settings (not modified), hashes, dirty patches and raw resource samples are retained.', '',
        'T_500 uses perf_counter immediately before first collection through immediately after update 500. Intermediate logging/checkpoints/GC count; environment initialization, offline evaluation and post-update-500 saves do not. The latter saves are timed separately (the existing runner writes the final checkpoint twice). Decomposition learning includes return computation and existing rollout diagnostics; other covers in-interval housekeeping.', '',
        'Training mean return is the existing rolling mean of the last 100 completed episodes. The pinned PPO adds timeout value bootstrap to its reward tensor in-place, so this training statistic is not raw evaluation return. Learning curves use measured elapsed wall time, with a common trailing 10-update mean plus faint raw data; no padding or rescaled time.', '',
        'Evaluation uses the exact final checkpoint, deterministic mean actions, 100 fixed trial IDs/initial states/command schedules, 100 parallel environments and 8 Sire threads. Observation noise and domain randomization are off for all policies. The first episode of each slot is retained, including failures; later reset episodes are ignored. Horizon is the baseline 20 s with unchanged falls/truncations/rewards. Return is the undiscounted raw reward sum without value bootstrap. Tracking errors use pre-reset body-frame velocity; numerical-fault transitions retain the existing last-valid-state semantics.', '',
        'Important limitation: flat.xml has finite ground and the flat task does not use heightfield boundary truncation. Ground-edge failures remain part of this frozen baseline; no per-config fix or score filtering is applied. Sire friction/base-mass randomization flags exist but the current implementation marks that port TODO; configuration flags must not be interpreted as implemented randomization.', '',
        'The 100 evaluation episodes are not 100 independent trainings. No across-training standard deviation/error bars are claimed. Changing environment count changes total transitions (7.68M, 15.36M, 30.72M, 61.44M); cross-environment times are not equal-work speedups. Only the within-environment thread scan approximately isolates parallel execution. 500 updates do not imply convergence. Conclusions are limited to this machine, task and budget; these are complete locomotion-training times, not residual-adaptation times or comparisons with other simulators.', '',
        '## Measured results', '',
        '| Envs | Sire threads | State | T_500 [min] | Eval return | Eval duration [s] | Recoveries |',
        '|---:|---:|---|---:|---:|---:|---:|']
    for row in sorted(configs, key=lambda r: (r['num_envs'], r['sire_batch_threads'])):
        if row['status'] != 'complete':
            lines.append(f"| {row['num_envs']} | {row['sire_batch_threads']} | pending | — | — | — | — |")
        else:
            lines.append(f"| {row['num_envs']} | {row['sire_batch_threads']} | complete | {row['T_500_seconds']/60:.3f} | {row['eval_mean_return']:.4f} | {row['eval_mean_duration_seconds']:.3f} | {row['physics_recoveries']} |")
    lines += ['', '## Failed / interrupted attempts', '']
    incomplete = [r for r in rows if r['status'] != 'complete']
    lines += [f"- {r['run_id']} attempt {r['attempt']}: {r['status']}, {r['actual_iterations']} updates, exit {r['exit_code']}. Excluded from complete-run points; artifacts retained." for r in incomplete] or ['None recorded.']
    if len(complete) == 12:
        fastest = min(complete, key=lambda r: r['T_500_seconds'])
        highest = max(complete, key=lambda r: r['eval_mean_return'])
        lines += ['', '## Observations (this matrix only)', '',
            f"Shortest measured budget time: {fastest['run_id']} ({fastest['T_500_seconds']/60:.2f} min). Highest measured final return: {highest['run_id']} ({highest['eval_mean_return']:.3f}). These extrema are descriptive single-training results, not significance tests."]
        for envs in (128, 256, 512, 1024):
            group = [r for r in complete if r['num_envs'] == envs]
            lo, hi = min(group, key=lambda r: r['T_500_seconds']), max(group, key=lambda r: r['T_500_seconds'])
            lines.append(f"At {envs} environments, fastest/slowest observed thread settings were {lo['sire_batch_threads']}/{hi['sire_batch_threads']}, a {hi['T_500_seconds']/lo['T_500_seconds']:.3f}× timing ratio; final returns ranged {min(r['eval_mean_return'] for r in group):.3f}–{max(r['eval_mean_return'] for r in group):.3f}.")
    lines += ['', '## Figure captions', '',
        '**English — main figure.** Sire flat-ground Go2 training on a fixed CPU platform. Each configuration is trained once for 500 PPO updates using the default 120 rollout steps per environment. (a) Measured end-to-end training time for different Sire worker counts. (b) Training time versus the undiscounted return averaged over the same 100 fixed evaluation trials of the final deterministic controller; color denotes environment count and marker denotes worker count. Total sample counts differ across environment counts. No repeated-training uncertainty is estimated.', '',
        '**中文 — 主图。** 固定 CPU 平台上的 Sire Go2 平地训练。每种配置仅训练一次、共 500 轮 PPO，每环境每轮沿用默认 120 步采样。(a) 不同 Sire 线程数的实测端到端训练耗时。(b) 训练耗时与最终确定性控制器在相同 100 个固定测试 trial 上的平均未折扣回报；颜色表示环境数，点形表示线程数。不同环境数对应不同总样本量，未估计重复训练的不确定性。', '',
        '**English — learning curves.** Training mean episode return versus measured elapsed wall time, grouped by Sire thread count. Solid curves use the same 10-update trailing mean; faint curves show raw rolling-100-episode training returns. Each curve is one training; no extrapolation or temporal alignment is applied. Training returns may include PPO timeout bootstrap and are distinct from independent evaluation returns.', '',
        '**中文 — 学习曲线。** 按 Sire 线程数分组，展示训练平均回合回报随实测墙钟时间的变化。实线统一使用 10 轮尾随均值，淡线为原始最近 100 个已完成回合的滚动训练回报。每条曲线仅对应一次训练，不外推或平移时间。训练回报可能含 PPO 超时自举，与独立评估回报不同。', '',
        'Experimental organization/style reference only (no data copied): https://proceedings.mlr.press/v164/rudin22a.html . No files were written to ral_writing.']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows, configs, curves = summarize(args.root)
    report(args.root, rows, configs)
    if args.plot:
        plot(args.root, configs, curves)
    print(f"Complete: {sum(r['status'] == 'complete' for r in configs)}/12", flush=True)


if __name__ == '__main__':
    main()
