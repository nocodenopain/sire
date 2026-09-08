# 2026-09-07 fixed-batch experiment

2026-09-08 合入上游后，求解器、关节安全和训练默认配置已经变化。
历史实验请使用本地提交 `0b87a04` 及原日期目录复现；新版本实验另建日期目录，
不要用新物理后端补写旧实验的评估结果。只汇总已有 CSV/绘图不受影响。

方法见PROTOCOL.md，参数见protocol.json，执行顺序见manifest.json。以下命令从仓库根目录执行。

```bash
export PYTHONPATH=python/src:demo/demo_python:/home/lqf/code/rsl_rl
.venv/bin/python -m SireRLGym.experiments.fixed_batch.prepare --root sire_parallel_experiment_20260907
.venv/bin/python -m SireRLGym.experiments.fixed_batch.run_matrix --root sire_parallel_experiment_20260907 --preflight
.venv/bin/python -m SireRLGym.experiments.fixed_batch.run_matrix --root sire_parallel_experiment_20260907
```

prepare拒绝覆盖已有日期目录，新实验用新日期/版本。preflight评估初始/参考策略并做两组独立2轮短测，
验证N128/H240与N1024/H30，短测不纳入正式结果。正式队列串行训练和评估，资源每30秒记录。
异常退出写FAILED.json并停止；全部12组完成后自动绘图并写COMPLETE.json。
STATUS.json显示当前配置/阶段，scheduler_stdout.log记录队列进度。
中断的训练不自动用checkpoint接着算耗时，检查原因后可--restart-incomplete从共同初始权重开始新attempt。
已完整训练但未评估的配置可直接补评估。普通训练默认参数不改。

每阶段*_command.json记录实际argv/env/cwd，另存stdout/stderr/退出状态/资源。
runs/<配置>/attempt01/logs/保存TensorBoard、metrics.jsonl、model_50至model_500。
evaluations/iter_XXXX/保存各检查点评估及逐trial数据；preregistration.json保存正式前确定的奖励目标。
code/保存代码版本/本地改动和实验脚本，无模型或源码hash校验。
运行期间请勿修改训练/实验代码或同时启动其它重负载训练，以保持方法一致。

重新汇总/画图，不重训：

```bash
PYTHONPATH=demo/demo_python /home/lqf/miniconda3/envs/rl/bin/python \
  -m SireRLGym.experiments.fixed_batch.summarize --root sire_parallel_experiment_20260907 --plot
```
