# Reproducibility Notes

This repository contains source code, experiment configurations, report sources,
figures, and CSV logs for the CarRacing-v2 discrete-control course project.
Large model checkpoint files (`*.pt`) are intentionally excluded.

## Environment

Recommended Python version: 3.10 or 3.11.

```powershell
python -m pip install -U pip
python -m pip install -e .
```

Check the installation:

```powershell
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/check_setup.py --runtime-smoke
```

## Main Result

The best-performing branch is:

```text
results/full_dqn_low_eps_finetune_seed42
```

The key logged result is:

```text
30000 steps: 296.1 +/- 123.3 mean reward over 10 evaluation episodes
35000 steps: -93.1 mean reward, showing policy degradation
```

The clean early-stop reproduction is:

```text
results/full_dqn_low_eps_earlystop_seed42
```

Its final logged checkpoint evaluation remains:

```text
30000 steps: 296.1 +/- 123.3
```

## Independent Evaluation

The independent sweep is stored in:

```text
results/full_dqn_low_eps_finetune_seed42/independent_eval.csv
```

It covers 5 seed offsets and 50 total evaluation episodes. The current aggregate
is 272.2 mean reward across seed offsets, with 4/5 offsets above the 200-point
target.

To rerun the sweep:

```powershell
python scripts/evaluate_checkpoint_sweep.py --config configs/full_dqn_low_eps_finetune.yaml --checkpoint results/full_dqn_low_eps_finetune_seed42/checkpoints/best.pt --episodes 10 --seed-offsets 10000 20000 30000 40000 50000 --output results/full_dqn_low_eps_finetune_seed42/independent_eval.csv
```

This command requires the excluded checkpoint file to exist locally.

## Report Regeneration

Regenerate all tables and result figures:

```powershell
python scripts/aggregate_results.py --results-dir results --report-dir report
```

Compile the report:

```powershell
xelatex -interaction=nonstopmode -halt-on-error -jobname=main_updated -output-directory report report/main.tex
xelatex -interaction=nonstopmode -halt-on-error -jobname=main_updated -output-directory report report/main.tex
```

The repository stores the latest compiled public report as:

```text
report/carracing_rl_report.pdf
```
