"""
Benchmark Analysis: MTL Model vs Competitors
=============================================
Computes objective comparisons FROM the existing benchmark CSV, including:

  1. Average Rank per metric (fairer than win-count)
  2. Performance Consistency (std across tasks — lower = more reliable)
  3. Composite score: Borda count over ACC, AUC, MCC
  4. Coverage: tasks covered per "model deployment" cost

These comparisons work because our model produces ALL task predictions
simultaneously — comparing total capability, not just per-task peaks.

Usage:
    python benchmark_analysis.py \
        --csv "Benchmark Summary - Sheet1.csv" \
        --output_dir results/benchmark_analysis
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ── Model display config ────────────────────────────────────────────────
OUR_MODEL   = "Our model (option 3 - esm-2)"
MODEL_COLORS = {
    "Our model (option 3 - esm-2)": "#2ECC71",
    "PDeepPP":                       "#E74C3C",
    "UniDL4BioPep":                  "#3498DB",
    "UniDL4BioPep-FL":               "#85C1E9",
}
DEFAULT_COLOR = "#95A5A6"

# Metrics we compare (from the CSV columns)
METRICS = ["ACC", "AUC", "PR-AUC", "MCC"]

# Tasks covered by our model (the 19 + 1 anti-inflammatory)
# Maps bioactivity name (from CSV) → our internal task name
BIOACTIVITY_MAP = {
    "ACE inhibitory activity":                    "ACE_inhibitory",
    "DPP IV inhibitory activity":                 "DPPIV_inhibitory",
    "Bitter":                                     "Bitter",
    "Umami":                                      "Umami",
    "Antimicrobial activity":                     "Antimicrobial",
    "Antimalarial activity (alternative dataset)":"Antimalarial_alt",
    "Antimalarial activity (main dataset)":       "Antimalarial",
    "Quorum sensing activity":                    "Quorum_sensing",
    "Anticancer activity (alternative dataset)":  "Anticancer_alt",
    "Anticancer activity (main dataset)":         "Anticancer",
    "Anti-MRSA strains activity":                 "AntiMRSA",
    "Tumor T cell antigens":                      "TTCA",
    "Blood-Brain Barrier":                        "BBP",
    "Antiparasitic activity":                     "Anti_parasitic",
    "Neuropeptide":                               "NeuroPred",
    "Antibacterial activity":                     "Antibacterial",
    "Antifungal activity":                        "Antifungal",
    "Antiviral activity":                         "Antiviral",
    "Toxicity":                                   "Toxicity",
    "Anti-inflammatory activity (our dataset)":   "Anti_inflammatory",
}


# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================

def load_benchmark(csv_path: str) -> pd.DataFrame:
    """Load and clean the benchmark CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Forward-fill Dataset# and Bioactivity (merged cells in CSV)
    df["Dataset #"]   = df["Dataset #"].ffill()
    df["Bioactivity"] = df["Bioactivity"].ffill()

    # Drop rows without a Model entry
    df = df.dropna(subset=["Model"])
    df["Model"] = df["Model"].str.strip()

    # Numeric conversion (some cells have notes like "993" instead of "0.993")
    for col in ["ACC", "AUC", "PR-AUC", "BACC", "Sn", "Sp", "MCC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Fix entries > 1 that are percentages written as ints (e.g. 993 → 0.993)
            mask = df[col] > 1
            df.loc[mask, col] = df.loc[mask, col] / 1000

    df["Bioactivity_clean"] = df["Bioactivity"].str.strip()
    return df


def get_benchmark_tasks(df: pd.DataFrame) -> list:
    return sorted(df["Bioactivity_clean"].dropna().unique())


def get_models_with_full_coverage(df: pd.DataFrame, tasks: list, metric="ACC") -> list:
    """Return models that have entries for ALL benchmark tasks."""
    models_with_full = []
    all_models = df["Model"].unique()
    for m in all_models:
        sub = df[df["Model"] == m]
        covered = set(sub["Bioactivity_clean"].dropna().unique())
        if all(t in covered for t in tasks):
            models_with_full.append(m)
    return models_with_full


# ============================================================================
# ANALYSIS 1: Average Rank per metric
# ============================================================================

def compute_average_ranks(df: pd.DataFrame, tasks: list, metrics: list) -> pd.DataFrame:
    """
    For each task and metric, rank all models (1 = best).
    Average rank across all tasks = overall standing.
    Lower average rank = consistently better.
    """
    rank_records = []

    for task in tasks:
        sub = df[df["Bioactivity_clean"] == task].dropna(subset=["Model"])

        for metric in metrics:
            sub_m = sub.dropna(subset=[metric]).copy()
            if sub_m.empty:
                continue

            # Rank: higher metric value = better (rank 1)
            sub_m[f"rank_{metric}"] = sub_m[metric].rank(ascending=False, method="min")

            for _, row in sub_m.iterrows():
                rank_records.append({
                    "Task":   task,
                    "Model":  row["Model"],
                    "Metric": metric,
                    "Value":  row[metric],
                    "Rank":   row[f"rank_{metric}"],
                })

    rank_df = pd.DataFrame(rank_records)

    # Average rank per model per metric
    avg_ranks = (rank_df
                 .groupby(["Model", "Metric"])["Rank"]
                 .agg(["mean", "count", "std"])
                 .reset_index()
                 .rename(columns={"mean": "AvgRank", "count": "TasksCovered", "std": "StdRank"}))

    return rank_df, avg_ranks


def plot_average_ranks(avg_ranks: pd.DataFrame, output_dir: Path):
    pivot = avg_ranks.pivot(index="Model", columns="Metric", values="AvgRank")

    # Keep models present in multiple metrics
    pivot = pivot.dropna(thresh=2)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        col = pivot[metric].dropna().sort_values()
        colors = [MODEL_COLORS.get(m, DEFAULT_COLOR) for m in col.index]
        bars = ax.barh(range(len(col)), col.values, color=colors, edgecolor="white")

        # Highlight our model
        for i, (m, v) in enumerate(col.items()):
            if m == OUR_MODEL:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2)
                ax.text(v + 0.05, i, f"  {v:.2f}", va="center", fontsize=8, fontweight="bold")
            else:
                ax.text(v + 0.05, i, f"  {v:.2f}", va="center", fontsize=7)

        ax.set_yticks(range(len(col)))
        ax.set_yticklabels([m.replace("Our model (option 3 - esm-2)", "Ours (MTL)")
                            for m in col.index], fontsize=8)
        ax.set_xlabel("Average Rank (lower = better)", fontsize=9)
        ax.set_title(f"{metric}", fontsize=11, fontweight="bold")
        ax.invert_xaxis()
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # Legend
    patches = [mpatches.Patch(color=c, label=m.replace("Our model (option 3 - esm-2)", "Ours (MTL)"))
               for m, c in MODEL_COLORS.items() if m in avg_ranks["Model"].values]
    axes[0].legend(handles=patches, fontsize=7, loc="lower left")

    fig.suptitle("Average Rank Across All Benchmark Tasks\n(Lower = more consistently better across all tasks)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "1_average_rank_per_metric.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================================
# ANALYSIS 2: Performance Consistency (StdDev across tasks)
# ============================================================================

def compute_consistency(df: pd.DataFrame, tasks: list, metric="ACC") -> pd.DataFrame:
    """
    Models with low std across tasks are more RELIABLE/CONSISTENT.
    For a unified model (ours), consistency is especially valuable:
    users can trust predictions across diverse activities.
    """
    rows = []
    for model in df["Model"].unique():
        sub = df[(df["Model"] == model)].dropna(subset=[metric])
        if len(sub) < 3:
            continue
        rows.append({
            "Model":    model,
            "Mean_ACC": sub[metric].mean(),
            "Std_ACC":  sub[metric].std(),
            "N_Tasks":  len(sub),
            "Min_ACC":  sub[metric].min(),
            "Max_ACC":  sub[metric].max(),
        })
    return pd.DataFrame(rows).sort_values("Std_ACC")


def plot_consistency(consistency_df: pd.DataFrame, metric: str, output_dir: Path):
    df = consistency_df.copy()
    df = df[df["N_Tasks"] >= 3]
    df = df.sort_values("Std_ACC")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [MODEL_COLORS.get(m, DEFAULT_COLOR) for m in df["Model"]]
    x      = range(len(df))

    # Mean bars
    bars = ax.bar(x, df["Mean_ACC"], color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5, label="Mean ACC")

    # Error bars = std
    ax.errorbar(x, df["Mean_ACC"], yerr=df["Std_ACC"],
                fmt="none", color="black", capsize=4, linewidth=1.5, zorder=5)

    # Min line
    ax.scatter(x, df["Min_ACC"], marker="v", color="black", s=30, zorder=6,
               label="Min ACC (worst task)")

    ax.set_xticks(list(x))
    labels = [m.replace("Our model (option 3 - esm-2)", "Ours\n(MTL)")
              for m in df["Model"]]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric} (mean ± std across tasks)", fontsize=10)
    ax.set_title(
        f"Performance Consistency: Mean {metric} ± StdDev Across All Tasks\n"
        "(Error bars = variance across tasks; lower bar = worst single-task performance)",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    # Annotate our model
    our_idx = list(df["Model"]).index(OUR_MODEL) if OUR_MODEL in list(df["Model"]) else None
    if our_idx is not None:
        ax.annotate("★ Our model",
                    xy=(our_idx, df.iloc[our_idx]["Mean_ACC"] + df.iloc[our_idx]["Std_ACC"] + 0.01),
                    fontsize=8, ha="center", color="#27AE60", fontweight="bold")

    plt.tight_layout()
    out = output_dir / "2_performance_consistency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================================
# ANALYSIS 3: Borda Count Composite Score
# ============================================================================

def compute_borda_score(rank_df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Borda count: if there are N models competing on a task-metric,
    the model ranked k gets (N - k) points.
    Sum over all tasks and metrics.
    Higher total = better overall model.
    """
    records = []
    for (task, metric), group in rank_df.groupby(["Task", "Metric"]):
        n = len(group)
        for _, row in group.iterrows():
            borda = n - row["Rank"]
            records.append({"Model": row["Model"], "Task": task,
                            "Metric": metric, "Borda": borda, "N_competitors": n})

    borda_df = pd.DataFrame(records)
    summary  = (borda_df.groupby("Model")
                .agg(TotalBorda=("Borda", "sum"),
                     TasksCovered=("Task", "nunique"),
                     AvgBorda=("Borda", "mean"))
                .reset_index()
                .sort_values("TotalBorda", ascending=False))
    return borda_df, summary


def plot_borda(borda_summary: pd.DataFrame, output_dir: Path):
    df = borda_summary.sort_values("TotalBorda", ascending=True)
    colors = [MODEL_COLORS.get(m, DEFAULT_COLOR) for m in df["Model"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(df)), df["TotalBorda"], color=colors,
                   edgecolor="white", linewidth=0.5)

    # Highlight ours
    for i, (_, row) in enumerate(df.iterrows()):
        if row["Model"] == OUR_MODEL:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2)
        ax.text(row["TotalBorda"] + 0.3, i,
                f"  {row['TotalBorda']:.0f} pts ({row['TasksCovered']} tasks)",
                va="center", fontsize=8)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [m.replace("Our model (option 3 - esm-2)", "Ours (MTL)") for m in df["Model"]],
        fontsize=9
    )
    ax.set_xlabel("Total Borda Score (ACC + AUC + PR-AUC + MCC)", fontsize=10)
    ax.set_title(
        "Composite Borda Score: Overall Model Ranking\n"
        "(Points accumulated by outranking competitors across all tasks and metrics)",
        fontsize=11, fontweight="bold"
    )
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=c, label=m.replace("Our model (option 3 - esm-2)", "Ours (MTL)"))
               for m, c in MODEL_COLORS.items() if m in df["Model"].values]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = output_dir / "3_borda_composite_score.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================================
# ANALYSIS 4: Per-Task Performance Comparison (Radar/Spider)
# ============================================================================

def plot_per_task_bars(df: pd.DataFrame, tasks: list, metric: str, output_dir: Path):
    """
    For each task, show bars for each model. Highlight ours.
    Shows where we lead vs. fall behind — transparent and honest.
    """
    # Focus on main 3 models for clarity
    focus_models = [OUR_MODEL, "PDeepPP", "UniDL4BioPep"]
    df_f = df[df["Model"].isin(focus_models)]

    n_tasks = len(tasks)
    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    axes = axes.flatten()

    for idx, (ax, task) in enumerate(zip(axes, tasks)):
        sub = df_f[df_f["Bioactivity_clean"] == task].dropna(subset=[metric])
        if sub.empty:
            ax.set_visible(False)
            continue

        models = sub["Model"].tolist()
        values = sub[metric].tolist()
        colors = [MODEL_COLORS.get(m, DEFAULT_COLOR) for m in models]
        labels = [m.replace("Our model (option 3 - esm-2)", "Ours")
                    .replace("UniDL4BioPep", "UniDL4")
                  for m in models]

        bars = ax.bar(range(len(models)), values, color=colors,
                      edgecolor="white", linewidth=0.5)

        # Highlight winner
        best_idx = int(np.argmax(values))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
        ax.set_ylim(0.5, 1.05)
        ax.set_title(task.replace(" activity", "").replace(" Activity", ""),
                     fontsize=7, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Mark our model value
        for i, (m, v) in enumerate(zip(models, values)):
            if m == OUR_MODEL:
                ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=5.5,
                        fontweight="bold", color="#1A5276")

    # Hide unused axes
    for ax in axes[n_tasks:]:
        ax.set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=MODEL_COLORS[m],
                               label=m.replace("Our model (option 3 - esm-2)", "Ours (MTL)"))
               for m in focus_models if m in MODEL_COLORS]
    patches.append(mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2,
                                   label="Task winner"))
    fig.legend(handles=patches, fontsize=9, loc="lower right", ncol=4)
    fig.suptitle(f"Per-Task {metric} Comparison: Ours vs PDeepPP vs UniDL4BioPep\n"
                 "(Gold border = task winner; numbers above = our model's value)",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    out = output_dir / f"4_per_task_{metric.replace('-', '_')}_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================================
# ANALYSIS 5: Coverage vs Performance Trade-off
# ============================================================================

def plot_coverage_vs_performance(df: pd.DataFrame, tasks: list, output_dir: Path):
    """
    X-axis: Number of tasks the model covers (coverage/scope)
    Y-axis: Average ACC on covered tasks
    Bubble size: consistency (1/std) — larger = more consistent

    Our model is unique: 1 model instance → all 19 tasks.
    Others: each model covers 1 task → need 19 separate instances.
    """
    # For single-task competitors, count distinct tasks they appear in
    records = []
    for model in df["Model"].unique():
        sub = df[(df["Model"] == model)].dropna(subset=["ACC"])
        n_tasks_covered = sub["Bioactivity_clean"].nunique()
        mean_acc = sub["ACC"].mean()
        std_acc  = sub["ACC"].std() if len(sub) > 1 else 0

        # For task-specific specialist models that only appear on 1 task,
        # they implicitly cover only that 1 task in production
        # (but may have high ACC there)

        records.append({
            "Model":          model,
            "Tasks_Covered":  n_tasks_covered,
            "Mean_ACC":       mean_acc,
            "Std_ACC":        std_acc,
            "N_Models_Needed": n_tasks_covered,  # equals tasks for single-task models
        })

    cov_df = pd.DataFrame(records)

    # Our model: 1 model instance covers N tasks
    # Single-task models: N model instances to cover N tasks
    # Add "Models_Needed" column
    cov_df["Models_Needed"] = cov_df.apply(
        lambda r: 1 if r["Model"] == OUR_MODEL else r["Tasks_Covered"],
        axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in cov_df.iterrows():
        color = MODEL_COLORS.get(row["Model"], DEFAULT_COLOR)
        size  = max(50, 500 / max(row["Models_Needed"], 1))  # larger = fewer models needed
        label = (row["Model"]
                 .replace("Our model (option 3 - esm-2)", "Ours (MTL)")
                 .replace("UniDL4BioPep", "UniDL4"))

        ax.scatter(row["Models_Needed"], row["Mean_ACC"],
                   s=size, color=color, alpha=0.8, edgecolors="black",
                   linewidth=1.5 if row["Model"] == OUR_MODEL else 0.5,
                   zorder=5 if row["Model"] == OUR_MODEL else 3)
        ax.annotate(
            label, (row["Models_Needed"], row["Mean_ACC"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7.5,
            fontweight="bold" if row["Model"] == OUR_MODEL else "normal"
        )

    ax.set_xlabel("Number of Model Instances Needed\n(1 = single unified model; N = N separate deployments)",
                  fontsize=10)
    ax.set_ylabel("Average ACC Across Covered Tasks", fontsize=10)
    ax.set_title(
        "Task Coverage vs. Performance Trade-off\n"
        "Ideal = top-left (high accuracy, minimal deployment cost)\n"
        "Bubble size ∝ deployment efficiency (larger = fewer models needed)",
        fontsize=11, fontweight="bold"
    )
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate ideal quadrant
    ax.annotate("← Ideal:\nhigh ACC,\nfewer models",
                xy=(1.2, 0.96), fontsize=9, color="green", fontstyle="italic")

    plt.tight_layout()
    out = output_dir / "5_coverage_vs_performance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================================
# ANALYSIS 6: Ground-Truth Multi-Functional Recall
# (requires datasets directory — skipped gracefully if not available)
# ============================================================================

def compute_ground_truth_multifunctional(data_dir: str, output_dir: Path):
    """
    Find peptides that appear with POSITIVE labels in >= 2 task datasets.
    These are the GROUND TRUTH multi-functional peptides.

    For each such peptide:
      - Our model (single pass) predicts all N task activities at once
      - A single-task model for task A has NO knowledge of task B

    Metric: "Multi-activity recall" per peptide =
        (correctly predicted active tasks) / (true active tasks)

    This requires:
      - datasets/ directory with *_train.csv and *_test.csv
      - A pre-run raw_predictions.csv from multi_activity_profiling.py
    """
    data_path = Path(data_dir)
    predictions_csv = output_dir / "raw_predictions.csv"

    if not data_path.exists():
        print(f"  [SKIP] Dataset directory not found: {data_dir}")
        return None

    if not predictions_csv.exists():
        print(f"  [SKIP] Run multi_activity_profiling.py first to generate raw_predictions.csv")
        return None

    from mtl_peptide_classifier import get_all_peptide_tasks
    task_configs = get_all_peptide_tasks(str(data_path))
    task_names   = sorted(task_configs.keys())

    # Build ground-truth dict: sequence → {task: label}
    gt = defaultdict(dict)
    for task_name, cfg in task_configs.items():
        prefix = cfg["csv_prefix"]
        for split in ("train", "test"):
            csv = data_path / f"{prefix}_{split}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            seq_col   = "sequence" if "sequence" in df.columns else "Sequence"
            label_col = "label"    if "label"    in df.columns else "Label"
            df = df.dropna(subset=[seq_col, label_col])
            for _, row in df.iterrows():
                seq = str(row[seq_col]).strip().upper()
                gt[seq][task_name] = int(row[label_col])

    # Find multi-functional peptides (positive in ≥ 2 tasks)
    multi_functional = {
        seq: labels
        for seq, labels in gt.items()
        if sum(1 for v in labels.values() if v == 1) >= 2
    }

    print(f"  Ground-truth multi-functional peptides: {len(multi_functional):,}")

    # Load our predictions
    preds_df = pd.read_csv(predictions_csv)
    preds_df = preds_df.set_index("sequence")

    # Compute multi-activity recall for our model
    recalls = []
    for seq, labels in multi_functional.items():
        if seq not in preds_df.index:
            continue
        true_active  = [t for t, v in labels.items() if v == 1 and t in task_names]
        if not true_active:
            continue
        correct = sum(
            1 for t in true_active
            if f"pred_{t}" in preds_df.columns and preds_df.loc[seq, f"pred_{t}"] == 1
        )
        recalls.append(correct / len(true_active))

    if not recalls:
        print("  [SKIP] No matching sequences found in predictions.")
        return None

    mean_recall = np.mean(recalls)
    print(f"  Multi-activity recall (our model): {mean_recall:.3f}")
    print(f"  ({len(recalls):,} multi-functional peptides evaluated)")

    # Save summary
    result = {
        "n_multifunctional_peptides": len(multi_functional),
        "n_evaluated": len(recalls),
        "mean_multiactivity_recall": round(mean_recall, 4),
        "recall_distribution": {
            "perfect (1.0)":  sum(1 for r in recalls if r == 1.0),
            "high (≥0.8)":    sum(1 for r in recalls if r >= 0.8),
            "medium (≥0.5)":  sum(1 for r in recalls if r >= 0.5),
            "low (<0.5)":     sum(1 for r in recalls if r <  0.5),
        }
    }

    import json
    with open(output_dir / "6_multifunctional_recall.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, required=True,
                        help="Path to benchmark CSV file")
    parser.add_argument("--output_dir", type=str, default="results/benchmark_analysis")
    parser.add_argument("--data_dir",   type=str, default=None,
                        help="Optional: path to datasets/ for ground-truth multi-functional analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  BENCHMARK ANALYSIS")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading benchmark data...")
    df    = load_benchmark(args.csv)
    tasks = get_benchmark_tasks(df)
    print(f"  {len(tasks)} bioactivity tasks, {df['Model'].nunique()} models")
    print(f"  Models: {', '.join(df['Model'].unique())}")

    # ── Rank analysis ──────────────────────────────────────────────────
    print("\n[2/6] Computing average ranks...")
    rank_df, avg_ranks = compute_average_ranks(df, tasks, METRICS)
    avg_ranks.to_csv(output_dir / "1_average_ranks.csv", index=False)
    plot_average_ranks(avg_ranks, output_dir)

    # Print summary
    our_ranks = avg_ranks[avg_ranks["Model"] == OUR_MODEL]
    print(f"\n  Our model average ranks:")
    for _, row in our_ranks.iterrows():
        print(f"    {row['Metric']:8s}: avg rank {row['AvgRank']:.2f} over {int(row['TasksCovered'])} tasks")

    # ── Consistency ────────────────────────────────────────────────────
    print("\n[3/6] Computing performance consistency...")
    for metric in ["ACC", "AUC", "MCC"]:
        cons = compute_consistency(df, tasks, metric)
        cons.to_csv(output_dir / f"2_consistency_{metric}.csv", index=False)
        plot_consistency(cons, metric, output_dir)

    # ── Borda score ────────────────────────────────────────────────────
    print("\n[4/6] Computing composite Borda score...")
    borda_detail, borda_summary = compute_borda_score(rank_df, METRICS)
    borda_summary.to_csv(output_dir / "3_borda_score.csv", index=False)
    plot_borda(borda_summary, output_dir)

    print(f"\n  Borda score ranking:")
    for _, row in borda_summary.iterrows():
        marker = " ← OUR MODEL" if row["Model"] == OUR_MODEL else ""
        print(f"    {row['TotalBorda']:6.0f} pts ({int(row['TasksCovered'])} tasks)  "
              f"{row['Model']}{marker}")

    # ── Per-task bars ──────────────────────────────────────────────────
    print("\n[5/6] Plotting per-task comparisons...")
    for metric in ["ACC", "AUC", "MCC"]:
        plot_per_task_bars(df, tasks, metric, output_dir)

    # Coverage vs performance
    plot_coverage_vs_performance(df, tasks, output_dir)

    # ── Ground-truth multi-functional (if data available) ──────────────
    if args.data_dir:
        print("\n[6/6] Ground-truth multi-functional recall...")
        compute_ground_truth_multifunctional(args.data_dir, output_dir)
    else:
        print("\n[6/6] Ground-truth multi-functional: skipped (use --data_dir to enable)")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
