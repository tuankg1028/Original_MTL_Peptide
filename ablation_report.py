"""
Ablation Study Report Generator
Reads results from all ablation variant checkpoints and prints a comparison table.

Usage:
    python ablation_report.py
"""

import json
from pathlib import Path

# ============================================================================
# BASELINE: from comprehensive evaluation (full model, 50 epochs)
# ============================================================================

BASELINE_EVAL = Path("checkpoints/comprehensive_evaluation/MTL_20tasks_detailed_results.json")

# Ablation variants to collect (checkpoint subfolder name)
VARIANTS = [
    "full_model",
    "no_cnn",
    "no_transformer",
    "no_tim",
    "esm_ratio_1p0",
    "esm_ratio_0p5",
    "unfreeze_esm",
    "transformer_2L",
]


def load_baseline():
    """Load baseline metrics from comprehensive evaluation."""
    if not BASELINE_EVAL.exists():
        return None
    with open(BASELINE_EVAL) as f:
        r = json.load(f)
    m = r["metrics"]
    tasks = list(m.keys())
    return {
        "variant": "full_model (baseline)",
        "avg_acc":   sum(m[t]["ACC"]    for t in tasks) / len(tasks) * 100,
        "avg_auc":   sum(m[t]["AUC"]    for t in tasks) / len(tasks) * 100,
        "avg_prauc": sum(m[t]["PR_AUC"] for t in tasks) / len(tasks) * 100,
        "avg_mcc":   sum(m[t]["MCC"]    for t in tasks) / len(tasks) * 100,
        "avg_f1":    sum(m[t]["F1"]     for t in tasks) / len(tasks) * 100,
        "num_tasks": len(tasks),
        "source": "comprehensive_evaluation",
    }


def load_variant(variant_name):
    """Load metrics from a trained ablation variant's results.json."""
    results_path = Path("checkpoints") / variant_name / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        r = json.load(f)
    fm = r.get("final_metrics", {})
    if not fm:
        return None
    tasks = list(fm.keys())
    return {
        "variant": variant_name,
        "avg_acc":   sum(fm[t]["accuracy"] for t in tasks) / len(tasks) * 100,
        "avg_auc":   sum(fm[t]["auc"]      for t in tasks) / len(tasks) * 100,
        "avg_prauc": sum(fm[t].get("pr_auc", fm[t]["auc"]) for t in tasks) / len(tasks) * 100,
        "avg_mcc":   sum(fm[t]["mcc"]      for t in tasks) / len(tasks) * 100,
        "avg_f1":    sum(fm[t]["f1"]       for t in tasks) / len(tasks) * 100,
        "num_tasks": len(tasks),
        "source": str(results_path),
    }


def load_ablation_config(variant_name):
    """Load ablation config saved alongside checkpoint."""
    cfg_path = Path("checkpoints") / variant_name / "best_model" / "ablation_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def delta(val, baseline):
    """Format a delta vs baseline with sign and colour hint."""
    d = val - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


def print_report(rows, baseline_row):
    """Print formatted comparison table."""
    header = f"{'Variant':<28} {'ACC':>7} {'AUC':>7} {'PR-AUC':>8} {'MCC':>7} {'F1':>7}  {'dACC':>7} {'dAUC':>7} {'dMCC':>7}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  ABLATION STUDY - MTL Peptide Classifier (20 Tasks)")
    print("=" * len(header))
    print(header)
    print(sep)

    for row in rows:
        d_acc = delta(row["avg_acc"],   baseline_row["avg_acc"])
        d_auc = delta(row["avg_auc"],   baseline_row["avg_auc"])
        d_mcc = delta(row["avg_mcc"],   baseline_row["avg_mcc"])
        is_baseline = "baseline" in row["variant"]
        marker = " *" if is_baseline else "  "
        print(
            f"{marker}{row['variant']:<26} "
            f"{row['avg_acc']:>6.2f}% "
            f"{row['avg_auc']:>6.2f}% "
            f"{row['avg_prauc']:>7.2f}% "
            f"{row['avg_mcc']:>6.2f}% "
            f"{row['avg_f1']:>6.2f}%  "
            f"{d_acc:>7} "
            f"{d_auc:>7} "
            f"{d_mcc:>7}"
        )

    print(sep)
    print("  * = baseline (full model)")
    print()
    print("Interpretation:")
    print("  Negative dACC/dAUC/dMCC  --> removing that component hurts performance")
    print("  Positive delta            --> removing it had no cost (or helped)")
    print()


def main():
    rows = []

    # Load baseline
    baseline = load_baseline()
    if baseline:
        rows.append(baseline)
    else:
        print("[WARN] Baseline comprehensive evaluation not found.")
        baseline = {"avg_acc": 0, "avg_auc": 0, "avg_mcc": 0}

    # Load each ablation variant
    for v in VARIANTS:
        if v == "full_model":
            continue  # already loaded as baseline
        row = load_variant(v)
        if row:
            rows.append(row)
        else:
            print(f"[SKIP] {v}: results.json not found (not yet trained)")

    if not rows:
        print("No results found. Run ablation variants first.")
        return

    print_report(rows, baseline)

    # Per-task breakdown for completed variants
    print("=" * 60)
    print("  PER-TASK BREAKDOWN (ACC%)")
    print("=" * 60)

    # Collect per-task data
    task_data = {}

    # Baseline per-task
    if BASELINE_EVAL.exists():
        with open(BASELINE_EVAL) as f:
            b = json.load(f)
        for task, m in b["metrics"].items():
            task_data.setdefault(task, {})["full_model"] = m["ACC"] * 100

    # Variant per-task
    for v in VARIANTS:
        if v == "full_model":
            continue
        results_path = Path("checkpoints") / v / "results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            r = json.load(f)
        for task, m in r.get("final_metrics", {}).items():
            task_data.setdefault(task, {})[v] = m["accuracy"] * 100

    if task_data:
        completed_variants = ["full_model"] + [
            v for v in VARIANTS if v != "full_model"
            and (Path("checkpoints") / v / "results.json").exists()
        ]
        col_w = 10
        header2 = f"{'Task':<22}" + "".join(f"{v[:col_w]:>{col_w}}" for v in completed_variants)
        print(header2)
        print("-" * len(header2))
        for task in sorted(task_data.keys()):
            row_str = f"{task:<22}"
            for v in completed_variants:
                val = task_data[task].get(v)
                row_str += f"{val:>{col_w}.1f}" if val is not None else f"{'N/A':>{col_w}}"
            print(row_str)
        print()


if __name__ == "__main__":
    main()
