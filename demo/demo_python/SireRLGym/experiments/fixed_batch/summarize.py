"""Tables and figures from actual timed runs and independent checkpoint evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from SireRLGym.experiments.parallel.common import read_json
from SireRLGym.experiments.parallel.summarize import read_csv, write_csv


def target_time(points, target):
    """Confirmation uses the second measured checkpoint; never invent a crossing."""
    if target is None:
        return dict(target_attained='', target_confirmation_seconds='', target_confirmation_iteration='',
                    first_crossing_lower_seconds='', first_crossing_upper_seconds='', target_status='unavailable')
    previous = None
    bracket = None
    for point in points:
        above = point['eval_return'] >= target
        if above and (previous is None or previous['eval_return'] < target) and bracket is None:
            bracket = (previous['training_seconds'] if previous else 0., point['training_seconds'])
        if above and previous is not None and previous['eval_return'] >= target:
            return dict(target_attained=True, target_confirmation_seconds=point['training_seconds'],
                        target_confirmation_iteration=point['iteration'],
                        first_crossing_lower_seconds=bracket[0], first_crossing_upper_seconds=bracket[1])
        previous = point
    return dict(target_attained=False, target_confirmation_seconds='', target_confirmation_iteration='',
                first_crossing_lower_seconds='' if bracket is None else bracket[0],
                first_crossing_upper_seconds='' if bracket is None else bracket[1])


def evaluation_point(run, iteration, seconds, evaluation):
    reasons = evaluation['termination_reasons']
    return dict(run, iteration=iteration, training_seconds=seconds,
                samples=iteration*30720, eval_return=evaluation['mean_episode_return'],
                duration_seconds=evaluation['mean_duration_seconds'],
                completion_rate=reasons.get('time_limit', 0)/100,
                physics_recovery_trials=sum(n for reason, n in reasons.items() if 'physics_recovery' in reason),
                termination_reasons=json.dumps(reasons, sort_keys=True))


def summarize(root):
    target = read_json(root / 'preregistration.json')['reward_target']
    initial = read_json(root / 'preregistration/initial/evaluation_result.json')
    configs, curves, iterations = [], [], []
    for run in read_json(root / 'manifest.json')['runs']:
        contexts = sorted((root / 'runs' / run['run_id']).glob('attempt*/context.json'))
        if not contexts:
            configs.append(dict(run, status='pending'))
            continue
        # Explicit restarts retain older attempts on disk; only latest is selected.
        attempt = contexts[-1].parent
        raw = read_csv(attempt / 'iterations.csv')
        iterations.extend(raw)
        result_path = attempt / 'training_result.json'
        result = read_json(result_path) if result_path.exists() else {}
        row = dict(run, attempt_dir=str(attempt), status='complete' if (attempt / 'complete.json').exists() else 'incomplete',
                   iterations=len(raw), total_transitions=int(raw[-1]['total_transitions']) if raw else 0,
                   T500_seconds=result.get('training_wall_seconds', ''),
                   collection_seconds=result.get('collection_seconds', ''),
                   learning_seconds=result.get('learning_seconds', ''),
                   other_seconds=result.get('other_seconds', ''),
                   training_recoveries=result.get('physics_recoveries', ''),
                   recovery_per_million_samples=result.get('physics_recoveries', 0)/15.36 if len(raw) == 500 else '',
                   reward_target=target)
        points = [evaluation_point(run, 0, 0., initial)]
        for i in range(50, 501, 50):
            path = attempt / f'evaluations/iter_{i:04d}/evaluation_result.json'
            if not path.exists():
                continue
            evaluation = read_json(path)
            assert evaluation['iteration'] == i and len(raw) >= i
            point = evaluation_point(run, i, float(raw[i-1]['wall_elapsed_seconds']), evaluation)
            trial_rows = read_csv(path.parent / 'evaluation_trials.csv')
            assert len(trial_rows) == 100
            for name in ('mean_linear_velocity_error_m_s', 'mean_yaw_rate_abs_error_rad_s'):
                point[name] = sum(float(r[name]) for r in trial_rows)/100
            points.append(point)
        curves.extend(points)
        row.update(target_time(points, target))
        row['observed_until_seconds'] = points[-1]['training_seconds']
        if points[-1]['iteration'] == 500:
            row.update(eval_return=points[-1]['eval_return'], completion_rate=points[-1]['completion_rate'],
                       eval_duration_seconds=points[-1]['duration_seconds'],
                       eval_recovery_trials=points[-1]['physics_recovery_trials'])
        if row['status'] == 'complete':
            assert len(raw) == 500 and len(points) == 11 and row['total_transitions'] == 15360000
        configs.append(row)
    for row in configs:
        base = next(r for r in configs if r['num_envs'] == row['num_envs'] and r['sire_batch_threads'] == 4)
        if row['status'] == base['status'] == 'complete':
            row['speedup_vs_4_threads'] = base['T500_seconds']/row['T500_seconds']
    write_csv(root / 'summary_by_config.csv', configs)
    write_csv(root / 'checkpoint_evaluations.csv', curves, None if curves else ['run_id', 'iteration'])
    write_csv(root / 'iterations_all.csv', iterations, None if iterations else ['run_id', 'iteration'])
    write_csv(root / 'time_to_target.csv', [dict(run_id=r['run_id'], num_envs=r['num_envs'],
        sire_batch_threads=r['sire_batch_threads'], status=r['status'],
        reward_target=target, **{k: r.get(k, '') for k in ('target_attained', 'target_confirmation_seconds',
            'target_confirmation_iteration', 'first_crossing_lower_seconds', 'first_crossing_upper_seconds',
            'observed_until_seconds')}) for r in configs])
    done = [r for r in configs if r['status'] == 'complete']
    lines = ['# 2026-09-07 Sire 固定总 batch 实验结果', '',
        f'完成 {len(done)}/12 配置。每组 B=30,720，500轮，15,360,000 samples；N×P为主变量。', '',
        (f'正式前确定的奖励目标 R*={target:.6f}；连续两次检查点达标才确认。' if target is not None else
         '参考exp24总回报低于初始策略，奖励目标/达标时间不可用；未改奖励、未另挑阈值。评估曲线与跟踪误差照常报告。'), '',
        '| Env | Thread | H | 训练分钟 | 最终评估回报 | 完成率 | 达标确认分钟 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(done, key=lambda r: (r['num_envs'], r['sire_batch_threads'])):
        confirmed = (f'{row["target_confirmation_seconds"]/60:.2f}' if row['target_attained'] else
                     ('不可用' if target is None else '预算内未确认'))
        lines.append(f'| {row["num_envs"]} | {row["sire_batch_threads"]} | {row["num_steps_per_env"]} | '
                     f'{row["T500_seconds"]/60:.2f} | {row["eval_return"]:.4f} | '
                     f'{row["completion_rate"]:.0%} | {confirmed} |')
    lines += ['', '## 说明', '',
        '图 b 为等量更新/样本预算的实测时间，三条线程曲线；图 c 为最终独立评估质量与该时间的关系。',
        '同环境数连接的是三个最终配置，不是学习轨迹。检查点学习曲线另图展示，不用训练rolling reward替代独立评估。',
        '环境初始化/离线评估不计入训练时间，逐阶段GNU time另存；中间日志/保存/GC计入。',
        '跨N改变了H，不能称完全相同物理轨迹；nominal substeps=samples×4不代表实测solver调用。',
        '单seed、每配置一次，100trial不是100次训练，无跨训练误差条。数值恢复/边界失败不剔除。',
        '所有数据按本轮开始日期归档，即使次日完成。无模型/源码哈希校验；方法和复现见PROTOCOL.md与README.md。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return configs, curves


def plot(root, configs, curves):
    assert len(configs) == 12 and all(r['status'] == 'complete' for r in configs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = {128: '#1f77b4', 256: '#ff7f0e', 512: '#2ca02c', 1024: '#d62728'}
    pcolors = {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c'}
    markers = {4: 'o', 8: 'x', 16: '^'}
    def finish(fig, filename):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(root / f'{filename}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    def style(ax):
        ax.grid(alpha=.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax in axes:
        style(ax)
    for p in (4, 8, 16):
        rows = sorted((r for r in configs if r['sire_batch_threads'] == p), key=lambda r: r['num_envs'])
        axes[0].plot(range(4), [r['T500_seconds']/60 for r in rows], marker=markers[p],
                     color=pcolors[p], label=f'{p} threads')
    axes[0].set(xticks=range(4), xticklabels=['128', '256', '512', '1024'],
                xlabel='Number of environments', ylabel='500-update training time [min]', title='(b) Equal sample budget')
    axes[0].legend(frameon=False)
    for n, color in colors.items():
        rows = sorted((r for r in configs if r['num_envs'] == n), key=lambda r: r['T500_seconds'])
        axes[1].plot([r['T500_seconds']/60 for r in rows], [r['eval_return'] for r in rows],
                     ':', color=color, alpha=.6)
        for row in rows:
            axes[1].scatter(row['T500_seconds']/60, row['eval_return'], color=color,
                            marker=markers[row['sire_batch_threads']], s=40)
    axes[1].set(xlabel='500-update training time [min]', ylabel='Final raw evaluation return',
                title='(c) Time–quality trade-off')
    handles = [Line2D([], [], color=c, marker='o', linestyle='none', label=f'{n} env') for n, c in colors.items()]
    handles += [Line2D([], [], color='.3', marker=m, linestyle='none', label=f'{p} threads') for p, m in markers.items()]
    axes[1].legend(handles=handles, fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    finish(fig, 'fig_fixed_batch_tradeoff')
    target = read_json(root / 'preregistration.json')['reward_target']
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, n in zip(axes.flat, colors):
        style(ax)
        for p in (4, 8, 16):
            rows = sorted((r for r in curves if r['num_envs'] == n and r['sire_batch_threads'] == p),
                          key=lambda r: r['iteration'])
            ax.plot([r['training_seconds']/60 for r in rows], [r['eval_return'] for r in rows],
                    marker=markers[p], color=pcolors[p], label=f'{p} threads', markersize=4)
        if target is not None:
            ax.axhline(target, linestyle='--', color='.4', linewidth=.8, label='Reward target')
        ax.set(xlabel='Measured training time [min]', ylabel='Raw evaluation return', title=f'{n} environments')
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    finish(fig, 'fig_checkpoint_learning_curves')
    if target is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    style(ax)
    for p in (4, 8, 16):
        rows = sorted((r for r in configs if r['sire_batch_threads'] == p), key=lambda r: r['num_envs'])
        y = [r['target_confirmation_seconds']/60 if r['target_attained'] else float('nan') for r in rows]
        ax.plot(range(4), y, marker=markers[p], color=pcolors[p], label=f'{p} threads')
        for i, row in enumerate(rows):
            if not row['target_attained']:
                ax.scatter(i, row['T500_seconds']/60, marker='^', facecolors='none', edgecolors=pcolors[p])
    ax.set(xticks=range(4), xticklabels=['128', '256', '512', '1024'], xlabel='Number of environments',
           ylabel='Confirmed reward-target time [min]', title='Open triangles: not confirmed within budget')
    ax.legend(frameon=False)
    fig.tight_layout()
    finish(fig, 'fig_time_to_target')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    configs, curves = summarize(args.root)
    if args.plot:
        plot(args.root, configs, curves)
    print(f'Completed {sum(r["status"] == "complete" for r in configs)}/12', flush=True)


if __name__ == '__main__':
    main()
