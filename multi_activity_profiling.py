"""
Multi-Activity Profiling for MTL Peptide Classifier
=====================================================
Demonstrates the unique capability of the MTL model: predicting all 19 (or 20)
peptide bioactivities in a SINGLE forward pass.

No single-task model can do this. This script:
  1. Pools all unique peptides from all task datasets
  2. Runs ONE backbone encoding per peptide (shared ESM-2 + Transformer + CNN)
  3. Passes the shared representation through all 19 task heads simultaneously
  4. Identifies "polypharmacological" peptides active in multiple tasks
  5. Produces a full analysis with heatmaps, distribution plots, and ranked tables

Analyses produced:
  A. Activity prevalence per task (bar chart)
  B. Activity co-occurrence matrix (19×19 heatmap)
  C. Multi-activity peptide distribution (histogram)
  D. Top polypharmacological peptides (ranked table)
  E. Biological cluster probability profiles (radar/heatmap)
  F. Efficiency benchmark: one pass vs N sequential models

Usage:
    python multi_activity_profiling.py \\
        --model_dir checkpoints \\
        --output_dir results/multi_activity \\
        --batch_size 8 \\
        --prob_threshold 0.5
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import EsmTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from mtl_peptide_classifier import MTLPeptideClassifier, get_all_peptide_tasks


# ============================================================================
# BIOLOGICAL CONTEXT
# ============================================================================

# Group tasks by biological mechanism for cluster analysis
TASK_GROUPS = {
    "Membrane-Active\n(Pathogen-Disrupting)": [
        "Antimicrobial", "Antibacterial", "Antifungal", "Antiviral", "AntiMRSA"
    ],
    "Cancer-Related": [
        "Anticancer", "Anticancer_alt", "TTCA"
    ],
    "Antiparasitic": [
        "Antimalarial", "Antimalarial_alt", "Anti_parasitic"
    ],
    "CNS / BBB": [
        "BBP", "NeuroPred"
    ],
    "Metabolic /\nEnzyme Inhibitor": [
        "ACE_inhibitory", "DPPIV_inhibitory"
    ],
    "Taste": [
        "Bitter", "Umami"
    ],
    "Signaling /\nOther": [
        "Quorum_sensing", "Toxicity"
    ],
}

# Short display names for plots
TASK_SHORT = {
    "ACE_inhibitory":   "ACE",
    "DPPIV_inhibitory": "DPPIV",
    "Bitter":           "Bitter",
    "Umami":            "Umami",
    "Antimicrobial":    "AMP",
    "Antimalarial":     "Malar",
    "Antimalarial_alt": "MalarAlt",
    "Quorum_sensing":   "QS",
    "Anticancer":       "ACP",
    "Anticancer_alt":   "ACPAlt",
    "AntiMRSA":         "MRSA",
    "TTCA":             "TTCA",
    "BBP":              "BBP",
    "Anti_parasitic":   "APar",
    "NeuroPred":        "Neuro",
    "Antibacterial":    "ABact",
    "Antifungal":       "AFung",
    "Antiviral":        "AV",
    "Toxicity":         "Tox",
    "Anti_inflammatory": "AInf",
}

# Group color palette
GROUP_COLORS = {
    "Membrane-Active\n(Pathogen-Disrupting)": "#E74C3C",
    "Cancer-Related":                         "#8E44AD",
    "Antiparasitic":                          "#E67E22",
    "CNS / BBB":                              "#2980B9",
    "Metabolic /\nEnzyme Inhibitor":          "#27AE60",
    "Taste":                                  "#F39C12",
    "Signaling /\nOther":                     "#7F8C8D",
}


# ============================================================================
# DATASET: Sequences only (no labels needed for profiling)
# ============================================================================

class SequenceOnlyDataset(Dataset):
    """Tokenizes peptide sequences for inference (no labels required)."""

    def __init__(self, sequences: list, tokenizer, max_length: int = 128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        tokens = " ".join(list(seq))
        enc = self.tokenizer(
            tokens,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ============================================================================
# MODEL LOADING (mirrors evaluate_mtl_comprehensive.py)
# ============================================================================

def load_model(checkpoint_dir: str, task_configs: dict, device: str):
    model = MTLPeptideClassifier(
        task_configs=task_configs,
        hidden_dim=1280,
        esm_ratio=0.9,
        num_transformer_layers=4,
        dropout=0.3,
    )

    best = Path(checkpoint_dir) / "best_model"

    backbone_path = best / "shared_backbone.pt"
    if backbone_path.exists():
        state = torch.load(backbone_path, map_location=device)
        model.base_embed.load_state_dict(state["base_embed"])
        model.transformer.load_state_dict(state["transformer"])
        model.cnn.load_state_dict(state["cnn"])
        model.layer_norm.load_state_dict(state["layer_norm"])
        print(f"  Loaded shared backbone from {backbone_path}")
    else:
        print(f"  [WARNING] Backbone not found at {backbone_path}")

    heads_path = best / "heads.pt"
    if heads_path.exists():
        heads_state = torch.load(heads_path, map_location=device)
        for name, head in model.heads.items():
            if name in heads_state:
                head.load_state_dict(heads_state[name])
        print(f"  Loaded {len(model.heads)} task heads from {heads_path}")
    else:
        print(f"  [WARNING] Heads not found at {heads_path}")

    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# CORE: ALL-TASK INFERENCE IN ONE PASS
# ============================================================================

@torch.no_grad()
def run_all_tasks_single_pass(model, dataloader, task_names, device):
    """
    The key MTL advantage: encode ONCE, apply all N task heads.

    For each batch:
      1. model.encode()  — ONE shared backbone forward pass
      2. head(repr) × N  — N cheap linear heads

    Returns:
        preds  : np.ndarray [n_seqs, n_tasks]  binary 0/1
        probs  : np.ndarray [n_seqs, n_tasks]  P(positive) per task
        timing : dict with backbone_time, heads_time
    """
    all_preds = defaultdict(list)
    all_probs = defaultdict(list)

    backbone_time = 0.0
    heads_time    = 0.0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # ── ONE backbone pass ──────────────────────────────────────────
        t0 = time.perf_counter()
        with autocast():
            shared_repr = model.encode(input_ids, attention_mask)  # [B, L, 2*D]
        backbone_time += time.perf_counter() - t0

        # ── All N task heads on the SAME shared_repr ───────────────────
        t1 = time.perf_counter()
        for task in task_names:
            with autocast():
                logits = model.heads[task](shared_repr, attention_mask)  # [B, 2]

            probs_t = torch.softmax(logits, dim=-1)[:, 1]  # P(positive)
            preds_t = (probs_t >= 0.5).long()

            all_probs[task].extend(probs_t.cpu().float().tolist())
            all_preds[task].extend(preds_t.cpu().tolist())
        heads_time += time.perf_counter() - t1

    # Convert to arrays [n_seqs, n_tasks]
    preds_matrix = np.column_stack([all_preds[t] for t in task_names])
    probs_matrix = np.column_stack([all_probs[t] for t in task_names])

    timing = {"backbone_s": backbone_time, "heads_s": heads_time}
    return preds_matrix, probs_matrix, timing


# ============================================================================
# POOL ALL UNIQUE PEPTIDES
# ============================================================================

def load_all_sequences(data_dir: str, task_configs: dict):
    """
    Collect every unique peptide from all train + test CSVs.
    Returns (sequences, source_info) where source_info tracks which tasks
    each peptide originally appeared in (and its true label there).
    """
    data_path = Path(data_dir)
    seq_source = defaultdict(lambda: {"tasks": set(), "labels": {}})

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
            df[seq_col] = df[seq_col].astype(str)

            for _, row in df.iterrows():
                seq = row[seq_col].strip().upper()
                if not seq or seq == "NAN":
                    continue
                seq_source[seq]["tasks"].add(task_name)
                seq_source[seq]["labels"][task_name] = int(row[label_col])

    sequences    = sorted(seq_source.keys())
    source_info  = seq_source
    print(f"  Pooled {len(sequences):,} unique peptides from all datasets")
    return sequences, source_info


# ============================================================================
# ANALYSIS A: Activity Prevalence
# ============================================================================

def plot_activity_prevalence(preds_matrix, task_names, short, output_dir):
    prevalence = preds_matrix.mean(axis=0) * 100  # % positive per task
    order      = np.argsort(prevalence)[::-1]

    # Assign colors by group
    task_to_group = {}
    for grp, tasks in TASK_GROUPS.items():
        for t in tasks:
            task_to_group[t] = grp
    colors = [GROUP_COLORS.get(task_to_group.get(task_names[i], "Signaling /\nOther"), "#95A5A6")
              for i in order]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(order)),
                  prevalence[order],
                  color=colors,
                  edgecolor="white",
                  linewidth=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short.get(task_names[i], task_names[i]) for i in order],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% Peptides Predicted Positive", fontsize=11)
    ax.set_title("Activity Prevalence Across All Pooled Peptides\n(MTL model, single-pass inference)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # Legend for groups
    patches = [mpatches.Patch(color=c, label=g.replace("\n", " "))
               for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              ncol=2, framealpha=0.9)

    plt.tight_layout()
    out = output_dir / "A_activity_prevalence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================================
# ANALYSIS B: Co-occurrence Matrix
# ============================================================================

def plot_cooccurrence_matrix(preds_matrix, task_names, short, output_dir):
    """
    Jaccard similarity between task predictions:
        J(A,B) = |A ∩ B| / |A ∪ B|
    Values near 1 → tasks are predicted positive together frequently.
    """
    n = len(task_names)
    jaccard = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = preds_matrix[:, i].astype(bool)
            b = preds_matrix[:, j].astype(bool)
            inter = (a & b).sum()
            union = (a | b).sum()
            jaccard[i, j] = inter / union if union > 0 else 0.0

    labels = [short.get(t, t) for t in task_names]
    df_jac = pd.DataFrame(jaccard, index=labels, columns=labels)

    # Reorder rows/cols by biological group
    ordered_tasks = []
    for grp, tasks in TASK_GROUPS.items():
        for t in tasks:
            if t in task_names:
                ordered_tasks.append(t)
    # append any not in groups
    for t in task_names:
        if t not in ordered_tasks:
            ordered_tasks.append(t)
    ordered_labels = [short.get(t, t) for t in ordered_tasks]
    df_jac = df_jac.loc[ordered_labels, ordered_labels]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.eye(len(ordered_labels), dtype=bool)  # hide diagonal
    sns.heatmap(
        df_jac,
        ax=ax,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0, vmax=0.6,
        linewidths=0.5,
        linecolor="#cccccc",
        cbar_kws={"label": "Jaccard Similarity", "shrink": 0.7},
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Activity Co-occurrence Matrix (Jaccard Similarity)\n"
        "Higher = tasks are frequently predicted positive on the same peptides",
        fontsize=12, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    # Draw group separators
    group_sizes = [sum(1 for t in tasks if t in task_names)
                   for tasks in TASK_GROUPS.values()]
    cumsum = 0
    for gs in group_sizes[:-1]:
        cumsum += gs
        ax.axhline(cumsum, color="navy", linewidth=1.5, alpha=0.7)
        ax.axvline(cumsum, color="navy", linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    out = output_dir / "B_cooccurrence_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # Also save raw data
    df_jac.to_csv(output_dir / "B_cooccurrence_data.csv")
    return df_jac


# ============================================================================
# ANALYSIS C: Multi-Activity Distribution
# ============================================================================

def plot_multiactivity_distribution(preds_matrix, task_names, output_dir):
    activity_counts = preds_matrix.sum(axis=1)  # how many tasks each peptide is positive in

    total = len(activity_counts)
    multi = (activity_counts >= 3).sum()
    hyper = (activity_counts >= 5).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    ax = axes[0]
    bins = range(0, len(task_names) + 2)
    counts, edges, patches = ax.hist(activity_counts, bins=bins,
                                     color="#3498DB", edgecolor="white", linewidth=0.5)
    # Color multi-active bars differently
    for patch, left_edge in zip(patches, edges[:-1]):
        if left_edge >= 3:
            patch.set_facecolor("#E74C3C")
        elif left_edge >= 1:
            patch.set_facecolor("#3498DB")
        else:
            patch.set_facecolor("#BDC3C7")

    ax.set_xlabel("Number of Active Tasks per Peptide", fontsize=11)
    ax.set_ylabel("Number of Peptides", fontsize=11)
    ax.set_title("Distribution of Multi-Activity Peptides", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(task_names) + 1))

    legend_patches = [
        mpatches.Patch(color="#BDC3C7", label="Inactive (0 tasks)"),
        mpatches.Patch(color="#3498DB", label="1-2 tasks"),
        mpatches.Patch(color="#E74C3C", label="≥3 tasks (polypharmacological)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)

    # Annotation
    ax.text(0.98, 0.95,
            f"Total peptides: {total:,}\n"
            f"Active in ≥3 tasks: {multi:,} ({100*multi/total:.1f}%)\n"
            f"Active in ≥5 tasks: {hyper:,} ({100*hyper/total:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Right: cumulative %
    ax2 = axes[1]
    thresholds = range(1, len(task_names) + 1)
    pct = [(activity_counts >= k).sum() / total * 100 for k in thresholds]
    ax2.plot(list(thresholds), pct, "o-", color="#E74C3C", linewidth=2, markersize=5)
    ax2.fill_between(list(thresholds), pct, alpha=0.15, color="#E74C3C")
    ax2.set_xlabel("Minimum Number of Active Tasks", fontsize=11)
    ax2.set_ylabel("% of Peptides Exceeding Threshold", fontsize=11)
    ax2.set_title("Polypharmacological Peptide Prevalence\nby Activity Threshold", fontsize=12, fontweight="bold")
    ax2.set_xticks(list(thresholds))
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 105)

    for k, p in zip(thresholds, pct):
        if k in (1, 3, 5):
            ax2.annotate(f"{p:.1f}%", (k, p), textcoords="offset points",
                         xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    out = output_dir / "C_multiactivity_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    return activity_counts


# ============================================================================
# ANALYSIS D: Top Polypharmacological Peptides
# ============================================================================

def build_top_peptides_table(sequences, preds_matrix, probs_matrix,
                              task_names, short, activity_counts,
                              source_info, output_dir, top_n=30):
    """Build ranked table of top multi-activity peptides."""

    # Score = number of active tasks + sum of all probabilities (tiebreaker)
    sum_probs = probs_matrix.sum(axis=1)
    score     = activity_counts * 100 + sum_probs  # lexicographic: count first

    order = np.argsort(score)[::-1][:top_n]

    rows = []
    for rank, idx in enumerate(order, 1):
        seq = sequences[idx]
        n_active = int(activity_counts[idx])
        active_tasks = [task_names[j] for j in range(len(task_names)) if preds_matrix[idx, j] == 1]
        active_short = [short.get(t, t) for t in active_tasks]

        # Get top 5 probabilities
        top5_idx   = np.argsort(probs_matrix[idx])[::-1][:5]
        top5_probs = [(short.get(task_names[i], task_names[i]), f"{probs_matrix[idx, i]:.3f}")
                      for i in top5_idx]
        top5_str   = ", ".join([f"{n}={p}" for n, p in top5_probs])

        original_tasks = source_info.get(seq, {}).get("tasks", set())

        rows.append({
            "Rank":             rank,
            "Sequence":         seq,
            "Length":           len(seq),
            "N_Active_Tasks":   n_active,
            "Active_Tasks":     "; ".join(active_short),
            "Top5_Probabilities": top5_str,
            "Sum_Probability":  round(float(sum_probs[idx]), 3),
            "Originally_From":  "; ".join(sorted(original_tasks)),
        })

    df = pd.DataFrame(rows)
    out_csv = output_dir / "D_top_polypharmacological_peptides.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")

    # Also print top 10 to console
    print(f"\n  {'─'*100}")
    print(f"  TOP 10 POLYPHARMACOLOGICAL PEPTIDES")
    print(f"  {'─'*100}")
    print(f"  {'Rank':>4}  {'Sequence':<30}  {'Len':>3}  {'N':>2}  Active Tasks")
    print(f"  {'─'*100}")
    for _, row in df.head(10).iterrows():
        seq_disp = row["Sequence"][:28] + ".." if len(row["Sequence"]) > 30 else row["Sequence"]
        print(f"  {row['Rank']:>4}  {seq_disp:<30}  {row['Length']:>3}  "
              f"{row['N_Active_Tasks']:>2}  {row['Active_Tasks']}")
    print(f"  {'─'*100}")

    return df


# ============================================================================
# ANALYSIS E: Probability Profile Heatmap for Top Peptides
# ============================================================================

def plot_probability_heatmap(sequences, probs_matrix, task_names,
                              short, activity_counts, output_dir, top_n=50):
    """Heatmap of prediction probabilities for top multi-active peptides."""

    order = np.argsort(activity_counts)[::-1][:top_n]
    sub_probs = probs_matrix[order]
    sub_seqs  = [sequences[i][:20] + ("…" if len(sequences[i]) > 20 else "")
                 for i in order]

    # Order tasks by biological group
    ordered_tasks = []
    for grp, tasks in TASK_GROUPS.items():
        for t in tasks:
            if t in task_names:
                ordered_tasks.append(t)
    for t in task_names:
        if t not in ordered_tasks:
            ordered_tasks.append(t)
    task_idx_order = [task_names.index(t) for t in ordered_tasks]
    ordered_labels = [short.get(t, t) for t in ordered_tasks]

    sub_probs_ordered = sub_probs[:, task_idx_order]

    fig, ax = plt.subplots(figsize=(16, max(8, top_n * 0.22)))
    im = ax.imshow(sub_probs_ordered, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sub_seqs, fontsize=6)
    ax.set_title(
        f"Prediction Probability Profile — Top {top_n} Multi-Active Peptides\n"
        "(Green=high probability; Red=low probability; columns ordered by biological group)",
        fontsize=11, fontweight="bold"
    )

    # Group separators on x-axis
    group_sizes = [sum(1 for t in tasks if t in task_names)
                   for tasks in TASK_GROUPS.values()]
    cumsum = -0.5
    for gs in group_sizes[:-1]:
        cumsum += gs
        ax.axvline(cumsum, color="navy", linewidth=1.5, alpha=0.8)

    plt.colorbar(im, ax=ax, label="P(active)", shrink=0.6)
    plt.tight_layout()
    out = output_dir / "E_probability_profile_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================================
# ANALYSIS F: Efficiency Benchmark
# ============================================================================

def report_efficiency(timing, n_sequences, n_tasks, output_dir):
    """
    Compare our single-pass inference vs sequential single-task inference.

    Assumptions for single-task comparison:
      - Each single-task model needs its own ESM-2 backbone pass
      - So sequential time ≈ N × backbone_time + N × head_time
      - Our time      = 1 × backbone_time + N × head_time
    """
    our_total     = timing["backbone_s"] + timing["heads_s"]
    sequential_est = n_tasks * timing["backbone_s"] + timing["heads_s"]  # N backbone passes

    speedup = sequential_est / max(our_total, 1e-9)

    our_per_seq       = our_total      / max(n_sequences, 1) * 1000  # ms
    sequential_per_seq = sequential_est / max(n_sequences, 1) * 1000

    report = (
        f"\n{'='*60}\n"
        f"  INFERENCE EFFICIENCY REPORT\n"
        f"{'='*60}\n"
        f"  Peptides processed : {n_sequences:,}\n"
        f"  Tasks predicted    : {n_tasks}\n"
        f"{'─'*60}\n"
        f"  Our MTL Model (single pass):\n"
        f"    Backbone time    : {timing['backbone_s']:.2f}s\n"
        f"    All heads time   : {timing['heads_s']:.2f}s\n"
        f"    Total            : {our_total:.2f}s\n"
        f"    Per peptide      : {our_per_seq:.2f}ms\n"
        f"{'─'*60}\n"
        f"  Sequential single-task models (estimated):\n"
        f"    {n_tasks}× backbone passes  : {n_tasks * timing['backbone_s']:.2f}s\n"
        f"    {n_tasks}× head passes      : {timing['heads_s']:.2f}s\n"
        f"    Total (est.)     : {sequential_est:.2f}s\n"
        f"    Per peptide      : {sequential_per_seq:.2f}ms\n"
        f"{'─'*60}\n"
        f"  Speedup factor     : {speedup:.1f}×\n"
        f"{'='*60}\n"
    )
    print(report)

    with open(output_dir / "F_efficiency_report.txt", "w") as f:
        f.write(report)

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Our MTL Model\n(1 backbone pass)", f"Sequential {n_tasks} Models\n({n_tasks} backbone passes)"],
        [our_total, sequential_est],
        color=["#2ECC71", "#E74C3C"],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, [our_total, sequential_est]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Total Inference Time (seconds)", fontsize=11)
    ax.set_title(
        f"Inference Efficiency: {n_sequences:,} Peptides × {n_tasks} Tasks\n"
        f"MTL model is {speedup:.1f}× faster",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, sequential_est * 1.2)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = output_dir / "F_efficiency_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    return speedup


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MTL Multi-Activity Profiling")
    parser.add_argument("--model_dir",      type=str, default="checkpoints",
                        help="Directory containing best_model/ subfolder")
    parser.add_argument("--data_dir",       type=str, default=None,
                        help="Path to datasets/ directory (default: auto-detect)")
    parser.add_argument("--output_dir",     type=str, default="results/multi_activity",
                        help="Output directory for all results")
    parser.add_argument("--batch_size",     type=int, default=8,
                        help="Inference batch size")
    parser.add_argument("--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for binary prediction")
    parser.add_argument("--top_n",          type=int, default=30,
                        help="Number of top peptides to report")
    parser.add_argument("--device",         type=str, default=None,
                        help="Device (cuda / mps / cpu)")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────
    script_dir = Path(__file__).parent
    data_dir   = args.data_dir or str(script_dir / "datasets")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("\n" + "=" * 70)
    print("  MTL MULTI-ACTIVITY PROFILING")
    print("=" * 70)
    print(f"  Model dir   : {args.model_dir}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Device      : {device}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Threshold   : {args.prob_threshold}")

    # ── Task configs ───────────────────────────────────────────────────
    print("\n[1/6] Loading task configurations...")
    task_configs = get_all_peptide_tasks(data_dir)
    task_names   = sorted(task_configs.keys())
    print(f"  {len(task_names)} tasks: {', '.join(task_names)}")

    # ── Load model ─────────────────────────────────────────────────────
    print("\n[2/6] Loading model...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model     = load_model(args.model_dir, task_configs, device)

    # ── Pool all unique peptides ───────────────────────────────────────
    print("\n[3/6] Pooling unique peptides from all datasets...")
    sequences, source_info = load_all_sequences(data_dir, task_configs)

    dataset    = SequenceOnlyDataset(sequences, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # ── Single-pass all-task inference ────────────────────────────────
    print(f"\n[4/6] Running single-pass inference on {len(sequences):,} peptides × {len(task_names)} tasks...")
    preds_matrix, probs_matrix, timing = run_all_tasks_single_pass(
        model, dataloader, task_names, device
    )

    # Apply custom threshold if != 0.5
    if args.prob_threshold != 0.5:
        preds_matrix = (probs_matrix >= args.prob_threshold).astype(int)

    activity_counts = preds_matrix.sum(axis=1)

    # Save raw results
    df_raw = pd.DataFrame(
        np.column_stack([probs_matrix, preds_matrix]),
        columns=(
            [f"prob_{t}" for t in task_names] +
            [f"pred_{t}" for t in task_names]
        )
    )
    df_raw.insert(0, "sequence", sequences)
    df_raw.insert(1, "n_active_tasks", activity_counts.astype(int))
    df_raw.to_csv(output_dir / "raw_predictions.csv", index=False)
    print(f"  Raw predictions saved → raw_predictions.csv")

    # ── Analyses ───────────────────────────────────────────────────────
    print("\n[5/6] Running analyses...")

    print("\n  ── Analysis A: Activity Prevalence ──")
    plot_activity_prevalence(preds_matrix, task_names, TASK_SHORT, output_dir)

    print("\n  ── Analysis B: Co-occurrence Matrix ──")
    plot_cooccurrence_matrix(preds_matrix, task_names, TASK_SHORT, output_dir)

    print("\n  ── Analysis C: Multi-activity Distribution ──")
    activity_counts = plot_multiactivity_distribution(preds_matrix, task_names, output_dir)

    print("\n  ── Analysis D: Top Polypharmacological Peptides ──")
    df_top = build_top_peptides_table(
        sequences, preds_matrix, probs_matrix,
        task_names, TASK_SHORT, activity_counts,
        source_info, output_dir, top_n=args.top_n
    )

    print("\n  ── Analysis E: Probability Profile Heatmap ──")
    plot_probability_heatmap(sequences, probs_matrix, task_names,
                             TASK_SHORT, activity_counts, output_dir, top_n=50)

    print("\n  ── Analysis F: Efficiency Benchmark ──")
    speedup = report_efficiency(timing, len(sequences), len(task_names), output_dir)

    # ── Summary statistics ─────────────────────────────────────────────
    print("\n[6/6] Summary")
    n_total   = len(sequences)
    n_multi3  = (activity_counts >= 3).sum()
    n_multi5  = (activity_counts >= 5).sum()
    n_inactive = (activity_counts == 0).sum()

    summary = {
        "total_unique_peptides":   n_total,
        "n_active_in_0_tasks":     int(n_inactive),
        "n_active_in_1plus_tasks": int((activity_counts >= 1).sum()),
        "n_active_in_3plus_tasks": int(n_multi3),
        "n_active_in_5plus_tasks": int(n_multi5),
        "pct_polypharmacological": round(100 * n_multi3 / n_total, 2),
        "inference_speedup_vs_sequential": round(speedup, 1),
        "tasks_analyzed": task_names,
    }

    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  MULTI-ACTIVITY PROFILING COMPLETE")
    print(f"{'='*70}")
    print(f"  Peptides analyzed          : {n_total:,}")
    print(f"  Inactive (0 tasks)         : {n_inactive:,} ({100*n_inactive/n_total:.1f}%)")
    print(f"  Active in ≥1 task          : {(activity_counts>=1).sum():,}")
    print(f"  Polypharmacological (≥3)   : {n_multi3:,} ({100*n_multi3/n_total:.1f}%)")
    print(f"  Hyper-active (≥5)          : {n_multi5:,} ({100*n_multi5/n_total:.1f}%)")
    print(f"  Inference speedup vs seq.  : {speedup:.1f}×")
    print(f"\n  All outputs saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
