# 基于强化学习的离散控制赛车游戏

本工程面向《强化学习技术与应用》期末大作业，主任务为 `CarRacing-v2` 的离散动作控制：

开源仓库：<https://github.com/xzzzzc217/carracing-rl-course-project>

最新编译版报告：`report/carracing_rl_report.pdf`。仓库保留源码、配置、报告源码、自动图表和 CSV 日志；`.pt` 模型权重、提交压缩包和本地缓存不纳入版本控制。

```python
env = gym.make("CarRacing-v2", continuous=False)
```

项目实现了像素输入 DQN 系列方法，包括基础 DQN、Double DQN、Dueling 网络、优先经验回放、帧堆叠、奖励裁剪等开关，方便完成至少两项消融实验。

## 环境安装

建议使用 Python 3.10 或 3.11。Windows 下安装 Box2D 依赖时如果失败，先安装 `swig`，再执行安装命令。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

如果不需要可编辑安装，也可以：

```powershell
python -m pip install -r requirements.txt
```

## 训练

安装完成后先做依赖检查：

```powershell
python scripts/check_setup.py
```

如果所有依赖存在，再跑一次环境和网络烟测：

```powershell
python scripts/check_setup.py --runtime-smoke
python -m carracing_rl.train --config configs/smoke.yaml --overwrite
```

主实验配置为 `configs/full_dqn.yaml`，默认训练 200000 环境步，满足课程要求。

```powershell
python -m carracing_rl.train --config configs/full_dqn.yaml
```

推荐用批量脚本顺序运行主实验和至少两项消融实验：

```powershell
python scripts/run_experiments.py --configs configs/full_dqn.yaml configs/ablations/no_prioritized_replay.yaml configs/ablations/no_dueling.yaml --resume
```

如果机器较慢，可以分块运行，例如每次每个实验推进 5000 步：

```powershell
python scripts/run_experiments.py --configs configs/full_dqn.yaml configs/ablations/no_prioritized_replay.yaml configs/ablations/no_dueling.yaml --resume --limit-steps 5000 --no-progress
```

配置里默认将 PyTorch CPU 线程数设为 `num_threads: 4`、`num_interop_threads: 1`。在本机 benchmark 中，这比默认 24 线程更适合当前小型卷积网络。

如果训练中断，可直接用同一条命令加 `--resume` 继续；脚本会自动读取对应 run 的 `checkpoints/latest.pt`。单个实验也可以手动恢复：

```powershell
python -m carracing_rl.train --config configs/full_dqn.yaml --resume results/full_dqn_seed42/checkpoints/latest.pt
```

如果 latest 模型退化，可以从 best 权重重开一个低学习率微调分支：

```powershell
python -m carracing_rl.train --config configs/full_dqn_finetune.yaml --resume results/full_dqn_seed42/checkpoints/best.pt --resume-model-only --limit-steps 5000 --overwrite --no-progress
```

当前表现最好的分支使用更低探索率进行保守微调：

```powershell
python -m carracing_rl.train --config configs/full_dqn_low_eps_finetune.yaml --resume results/full_dqn_seed42/checkpoints/best.pt --resume-model-only --limit-steps 5000 --overwrite --no-progress
```

该分支 30000 步的 `best.pt` 已达到 10 回合平均奖励 200 以上；继续训练可能退化，因此配置里启用了 `early_stop_reward: 200` 和 `restore_best_on_early_stop: true`，复现实验会在达标后停止并把最优权重作为 `latest.pt` 导出。

也可以使用独立 run 名称做干净早停复现实验：

```powershell
python -m carracing_rl.train --config configs/full_dqn_low_eps_earlystop.yaml --resume results/full_dqn_seed42/checkpoints/best.pt --resume-model-only --limit-steps 5000 --overwrite --no-progress
```

更多消融实验示例：

```powershell
python -m carracing_rl.train --config configs/baseline_dqn.yaml
python -m carracing_rl.train --config configs/full_dqn_action_prior.yaml
python -m carracing_rl.train --config configs/full_dqn_low_eps_finetune.yaml --resume results/full_dqn_seed42/checkpoints/best.pt --resume-model-only
python -m carracing_rl.train --config configs/ablations/no_prioritized_replay.yaml
python -m carracing_rl.train --config configs/ablations/no_dueling.yaml
python -m carracing_rl.train --config configs/ablations/no_double_dqn.yaml
python -m carracing_rl.train --config configs/ablations/frame_stack_1.yaml
```

每个 run 会写入：

- `results/<run_name>/train.csv`
- `results/<run_name>/eval.csv`
- `results/<run_name>/config.yaml`
- `results/<run_name>/checkpoints/latest.pt`
- `results/<run_name>/checkpoints/best.pt`

## 评估

```powershell
python -m carracing_rl.evaluate --config configs/full_dqn_low_eps_finetune.yaml --checkpoint results/full_dqn_low_eps_finetune_seed42/checkpoints/best.pt --episodes 10
```

对最优检查点做多个评估种子偏移的独立复评：

```powershell
python scripts/evaluate_checkpoint_sweep.py --config configs/full_dqn_low_eps_finetune.yaml --checkpoint results/full_dqn_low_eps_finetune_seed42/checkpoints/best.pt --episodes 10 --seed-offsets 20000 30000 40000 --output results/full_dqn_low_eps_finetune_seed42/independent_eval.csv
```

## 生成报告图表

训练完成后运行：

```powershell
python scripts/aggregate_results.py --results-dir results --report-dir report
```

该脚本会根据 `eval.csv` 更新：

- `report/tables/results.tex`
- `report/figures/learning_curve.pdf`

然后编译报告：

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory report report/main.tex
```

## 打包提交

课程要求文件名为“学号+姓名”。训练和报告完成后执行：

```powershell
python scripts/aggregate_results.py --results-dir results --report-dir report
xelatex -interaction=nonstopmode -halt-on-error -output-directory report report/main.tex
python scripts/package_submission.py --student 57123117赵子辰
```

输出压缩包：

```text
57123117赵子辰.zip
```

压缩包包含源代码、配置、训练 CSV 日志、报告源码和已编译 PDF；默认不包含大型模型权重和视频。

## License

本项目以 MIT License 开源。课程资料 PDF、训练权重和本地提交压缩包不属于开源内容。
