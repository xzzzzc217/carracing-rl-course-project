from __future__ import annotations

from pathlib import Path

from scripts.aggregate_results import (
    write_independent_eval_summary,
    write_independent_eval_table,
    write_results_table,
)


def test_results_table_marks_target_reached(tmp_path: Path) -> None:
    summaries = [
        {
            "run": "full_dqn_low_eps_finetune_seed42",
            "best": {"step": 30000, "mean_reward": 296.0, "std_reward": 12.0, "mean_length": 1000.0},
            "last": {"mean_reward": -93.1},
        }
    ]
    write_results_table(summaries, tmp_path)
    table = (tmp_path / "tables" / "results.tex").read_text(encoding="utf-8")
    assert "达标" in table
    assert "是" in table


def test_independent_eval_table_writes_rows(tmp_path: Path) -> None:
    rows = [
        {
            "run": "full_dqn_low_eps_finetune_seed42",
            "seed_offset": 20000,
            "checkpoint_step": 30000,
            "mean_reward": 210.0,
            "std_reward": 20.0,
            "min_reward": 180.0,
            "max_reward": 250.0,
            "mean_length": 1000.0,
            "episodes": 10,
        }
    ]
    write_independent_eval_table(rows, tmp_path)
    table = (tmp_path / "tables" / "independent_eval.tex").read_text(encoding="utf-8")
    assert "种子偏移" in table
    assert "20000" in table


def test_independent_eval_summary_writes_pass_ratio(tmp_path: Path) -> None:
    rows = [
        {
            "run": "full_dqn_low_eps_finetune_seed42",
            "seed_offset": 10000,
            "checkpoint_step": 30000,
            "mean_reward": 210.0,
            "std_reward": 10.0,
            "min_reward": 190.0,
            "max_reward": 240.0,
            "mean_length": 1000.0,
            "episodes": 10,
        },
        {
            "run": "full_dqn_low_eps_finetune_seed42",
            "seed_offset": 20000,
            "checkpoint_step": 30000,
            "mean_reward": 150.0,
            "std_reward": 20.0,
            "min_reward": 100.0,
            "max_reward": 180.0,
            "mean_length": 1000.0,
            "episodes": 10,
        },
    ]
    write_independent_eval_summary(rows, tmp_path)
    table = (tmp_path / "tables" / "independent_eval_summary.tex").read_text(encoding="utf-8")
    assert "达标比例" in table
    assert "1/2" in table
