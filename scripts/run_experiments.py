from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import yaml


DEFAULT_CONFIGS = [
    "configs/full_dqn.yaml",
    "configs/ablations/no_prioritized_replay.yaml",
    "configs/ablations/no_dueling.yaml",
]


def load_run_name(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "logging" in config and "run_name" in config["logging"]:
        return str(config["logging"]["run_name"])
    if "inherits" in config:
        parent = (config_path.parent / config["inherits"]).resolve()
        parent_name = load_run_name(parent)
        return f"{config_path.stem}_{parent_name}"
    return config_path.stem


def final_eval_step(eval_csv: Path) -> int | None:
    if not eval_csv.exists():
        return None
    with eval_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return int(float(rows[-1]["step"]))


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    return checkpoint if checkpoint.exists() else None


def run_config(
    config_path: Path,
    results_dir: Path,
    resume: bool,
    overwrite: bool,
    dry_run: bool,
    limit_steps: int | None,
    no_progress: bool,
) -> None:
    run_name = load_run_name(config_path)
    run_dir = results_dir / run_name
    command = [sys.executable, "-m", "carracing_rl.train", "--config", str(config_path), "--results-dir", str(results_dir)]
    if overwrite:
        command.append("--overwrite")
    elif resume:
        checkpoint = latest_checkpoint(run_dir)
        if checkpoint is not None:
            command.extend(["--resume", str(checkpoint)])
    if limit_steps is not None:
        command.extend(["--limit-steps", str(limit_steps)])
    if no_progress:
        command.append("--no-progress")

    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS, help="Config files to run sequentially.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints when present.")
    parser.add_argument("--overwrite", action="store_true", help="Restart each run from scratch.")
    parser.add_argument("--limit-steps", type=int, help="Run at most this many additional steps per config.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    for config in args.configs:
        run_config(
            Path(config),
            results_dir,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            limit_steps=args.limit_steps,
            no_progress=args.no_progress,
        )


if __name__ == "__main__":
    main()
