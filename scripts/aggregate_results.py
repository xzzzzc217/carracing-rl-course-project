from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


RUN_LABELS = {
    "full_dqn_seed42": "Full DQN",
    "full_dqn_finetune_seed42": "Full DQN Finetune",
    "full_dqn_low_eps_finetune_seed42": "Low-Epsilon Finetune",
    "full_dqn_low_eps_earlystop_seed42": "Low-Epsilon EarlyStop",
    "full_dqn_action_prior_seed42": "Full DQN Action Prior",
    "baseline_dqn_seed42": "Baseline DQN",
    "ablation_no_prioritized_replay_seed42": "No PER",
    "ablation_no_dueling_seed42": "No Dueling",
    "ablation_no_double_dqn_seed42": "No Double DQN",
    "ablation_frame_stack_1_seed42": "Frame Stack 1",
    "ablation_reward_clip_1_seed42": "Reward Clip 1",
}

RUN_ORDER = {
    "full_dqn_seed42": 0,
    "full_dqn_action_prior_seed42": 1,
    "full_dqn_finetune_seed42": 2,
    "full_dqn_low_eps_finetune_seed42": 3,
    "full_dqn_low_eps_earlystop_seed42": 4,
    "baseline_dqn_seed42": 5,
    "ablation_no_prioritized_replay_seed42": 6,
    "ablation_no_dueling_seed42": 7,
    "ablation_no_double_dqn_seed42": 8,
    "ablation_frame_stack_1_seed42": 9,
    "ablation_reward_clip_1_seed42": 10,
}


PLOT_LABELS = {
    "Full DQN Action Prior": "Action Prior",
    "Full DQN Finetune": "Finetune",
    "Low-Epsilon Finetune": "Low-Eps Fine.",
    "Low-Epsilon EarlyStop": "Low-Eps Stop",
    "Baseline DQN": "Baseline",
    "No Double DQN": "No Double",
}


def display_name(run_name: str) -> str:
    return RUN_LABELS.get(run_name, run_name.replace("_", "\\_"))


def plot_label(run_name: str) -> str:
    label = RUN_LABELS.get(run_name, run_name.replace("_", " "))
    return PLOT_LABELS.get(label, label)


def read_eval_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_runs(results_dir: Path, include_smoke: bool = False) -> list[dict]:
    summaries = []
    for eval_path in sorted(results_dir.glob("*/eval.csv")):
        if not include_smoke and eval_path.parent.name.startswith("smoke"):
            continue
        rows = read_eval_csv(eval_path)
        if not rows:
            continue
        parsed = []
        for row in rows:
            parsed.append(
                {
                    "step": int(float(row["step"])),
                    "mean_reward": float(row["mean_reward"]),
                    "std_reward": float(row["std_reward"]),
                    "mean_length": float(row["mean_length"]),
                }
            )
        best = max(parsed, key=lambda item: item["mean_reward"])
        last = parsed[-1]
        summaries.append(
            {
                "run": eval_path.parent.name,
                "rows": parsed,
                "best": best,
                "last": last,
            }
        )
    return sorted(summaries, key=lambda item: (RUN_ORDER.get(item["run"], 999), item["run"]))


def write_results_table(summaries: list[dict], report_dir: Path) -> None:
    table_path = report_dir / "tables" / "results.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        handle.write("实验配置 & 最优步数 & 最优平均奖励 & 最终平均奖励 & 平均长度 & 达标 \\\\\n")
        handle.write("\\midrule\n")
        if not summaries:
            handle.write("待训练 & -- & -- & -- & -- & -- \\\\\n")
        else:
            for summary in summaries:
                reached_target = "是" if summary["best"]["mean_reward"] >= 200.0 else "否"
                handle.write(
                    f"{display_name(summary['run'])} & {summary['best']['step']} & "
                    f"{summary['best']['mean_reward']:.1f} $\\pm$ {summary['best']['std_reward']:.1f} & "
                    f"{summary['last']['mean_reward']:.1f} & {summary['best']['mean_length']:.1f} & "
                    f"{reached_target} \\\\\n"
                )
        handle.write("\\bottomrule\n\\end{tabular}\n")


def collect_independent_evals(results_dir: Path) -> list[dict]:
    rows = []
    for csv_path in sorted(results_dir.glob("*/independent_eval.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "run": csv_path.parent.name,
                        "seed_offset": int(float(row["seed_offset"])),
                        "checkpoint_step": int(float(row["checkpoint_step"])),
                        "mean_reward": float(row["mean_reward"]),
                        "std_reward": float(row["std_reward"]),
                        "min_reward": float(row["min_reward"]),
                        "max_reward": float(row["max_reward"]),
                        "mean_length": float(row["mean_length"]),
                        "episodes": int(float(row["episodes"])),
                    }
                )
    return sorted(rows, key=lambda item: (RUN_ORDER.get(item["run"], 999), item["run"], item["seed_offset"]))


def write_independent_eval_table(rows: list[dict], report_dir: Path) -> None:
    table_path = report_dir / "tables" / "independent_eval.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        handle.write("实验配置 & 种子偏移 & 检查点步数 & 平均奖励 & 奖励范围 & 回合数 \\\\\n")
        handle.write("\\midrule\n")
        if not rows:
            handle.write("待复评 & -- & -- & -- & -- & -- \\\\\n")
        else:
            for row in rows:
                reward_range = f"[{row['min_reward']:.1f}, {row['max_reward']:.1f}]"
                handle.write(
                    f"{display_name(row['run'])} & {row['seed_offset']} & {row['checkpoint_step']} & "
                    f"{row['mean_reward']:.1f} $\\pm$ {row['std_reward']:.1f} & "
                    f"{reward_range} & {row['episodes']} \\\\\n"
                )
        handle.write("\\bottomrule\n\\end{tabular}\n")


def write_independent_eval_summary(rows: list[dict], report_dir: Path) -> None:
    table_path = report_dir / "tables" / "independent_eval_summary.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["run"], []).append(row)

    with table_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        handle.write("实验配置 & 种子偏移数 & 总回合数 & 跨偏移均值 & 跨偏移标准差 & 达标比例 \\\\\n")
        handle.write("\\midrule\n")
        if not grouped:
            handle.write("待复评 & -- & -- & -- & -- & -- \\\\\n")
        else:
            for run_name in sorted(grouped, key=lambda name: (RUN_ORDER.get(name, 999), name)):
                run_rows = grouped[run_name]
                means = [row["mean_reward"] for row in run_rows]
                offset_std = statistics.stdev(means) if len(means) > 1 else 0.0
                pass_count = sum(mean >= 200.0 for mean in means)
                total_episodes = sum(row["episodes"] for row in run_rows)
                handle.write(
                    f"{display_name(run_name)} & {len(run_rows)} & {total_episodes} & "
                    f"{statistics.mean(means):.1f} & {offset_std:.1f} & "
                    f"{pass_count}/{len(run_rows)} \\\\\n"
                )
        handle.write("\\bottomrule\n\\end{tabular}\n")


def write_learning_curve(summaries: list[dict], report_dir: Path) -> None:
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "learning_curve.pdf"
    if not summaries:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.text(0.5, 0.5, "Run training to generate learning curves", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(figure_path)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for summary in summaries:
        steps = [row["step"] for row in summary["rows"]]
        rewards = [row["mean_reward"] for row in summary["rows"]]
        ax.plot(steps, rewards, marker="o", linewidth=1.5, markersize=3, label=display_name(summary["run"]))
    ax.axhline(200, color="tab:red", linestyle="--", linewidth=1, label="Target 200")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean evaluation reward")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def write_reward_summary_figure(summaries: list[dict], report_dir: Path) -> None:
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "reward_summary.pdf"
    if not summaries:
        return

    labels = [plot_label(summary["run"]) for summary in summaries]
    best_rewards = [summary["best"]["mean_reward"] for summary in summaries]
    best_stds = [summary["best"]["std_reward"] for summary in summaries]
    final_rewards = [summary["last"]["mean_reward"] for summary in summaries]
    final_stds = [summary["last"]["std_reward"] for summary in summaries]
    x_positions = list(range(len(summaries)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(
        [x - width / 2 for x in x_positions],
        best_rewards,
        width,
        yerr=best_stds,
        capsize=2,
        label="Best checkpoint",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        final_rewards,
        width,
        yerr=final_stds,
        capsize=2,
        label="Final checkpoint",
        color="#F58518",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.axhline(200, color="#D62728", linestyle="--", linewidth=1.1, label="Target 200")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.45)
    ax.set_ylabel("Mean evaluation reward")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Best and final checkpoint performance")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ymin = min(best_rewards + final_rewards + [-100]) - 35
    ymax = max([b + s for b, s in zip(best_rewards, best_stds)] + [240]) + 55
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def write_independent_eval_figure(rows: list[dict], report_dir: Path) -> None:
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "independent_eval.pdf"
    if not rows:
        return

    labels = [str(row["seed_offset"]) for row in rows]
    rewards = [row["mean_reward"] for row in rows]
    stds = [row["std_reward"] for row in rows]
    colors = ["#54A24B" if reward >= 200 else "#E45756" for reward in rewards]
    x_positions = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(
        x_positions,
        rewards,
        yerr=stds,
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.axhline(200, color="#D62728", linestyle="--", linewidth=1.1, label="Target 200")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.45)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Evaluation seed offset")
    ax.set_ylabel("Mean reward")
    ax.set_title("Independent evaluation of best checkpoint")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(min(rewards + [-50]) - max(stds + [0]) - 25, max([r + s for r, s in zip(rewards, stds)] + [240]) + 45)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def write_early_stop_figure(summaries: list[dict], report_dir: Path) -> None:
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "early_stop_effect.pdf"
    by_run = {summary["run"]: summary for summary in summaries}
    finetune = by_run.get("full_dqn_low_eps_finetune_seed42")
    early_stop = by_run.get("full_dqn_low_eps_earlystop_seed42")
    if finetune is None or early_stop is None:
        return

    labels = ["Finetune best\n(30k)", "Finetune final\n(35k)", "EarlyStop final\n(30k)"]
    rewards = [
        finetune["best"]["mean_reward"],
        finetune["last"]["mean_reward"],
        early_stop["last"]["mean_reward"],
    ]
    stds = [
        finetune["best"]["std_reward"],
        finetune["last"]["std_reward"],
        early_stop["last"]["std_reward"],
    ]
    colors = ["#54A24B", "#E45756", "#4C78A8"]

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.bar(
        range(len(labels)),
        rewards,
        yerr=stds,
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.axhline(200, color="#D62728", linestyle="--", linewidth=1.1, label="Target 200")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.45)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("Policy degradation and early stopping")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-130, max([r + s for r, s in zip(rewards, stds)] + [240]) + 55)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report-dir", default="report")
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke-test runs in the report table.")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    report_dir = Path(args.report_dir)
    summaries = collect_runs(results_dir, include_smoke=args.include_smoke)
    independent_rows = collect_independent_evals(results_dir)
    write_results_table(summaries, report_dir)
    write_independent_eval_table(independent_rows, report_dir)
    write_independent_eval_summary(independent_rows, report_dir)
    write_learning_curve(summaries, report_dir)
    write_reward_summary_figure(summaries, report_dir)
    write_independent_eval_figure(independent_rows, report_dir)
    write_early_stop_figure(summaries, report_dir)
    print(f"Collected {len(summaries)} run(s) and {len(independent_rows)} independent eval row(s). Updated {report_dir}.")


if __name__ == "__main__":
    main()
