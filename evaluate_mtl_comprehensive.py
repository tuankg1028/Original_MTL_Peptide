"""
Comprehensive MTL Evaluation Script
Computes all metrics including PR-AUC for all 19 peptide tasks.

This script evaluates a trained MTL model on all 19 UniDL4BioPep datasets
and generates detailed results including:
- Accuracy (ACC)
- ROC-AUC
- Precision-Recall AUC (PR_AUC)
- Balanced Accuracy (BACC)
- Sensitivity (Sn)
- Specificity (Sp)
- Matthews Correlation Coefficient (MCC)
- Precision, Recall, F1
- Confusion Matrix (TP, TN, FP, FN)

Usage:
    python evaluate_mtl_comprehensive.py \
        --model_dir "mtl_checkpoints" \
        --model_name "Original_MTL_19tasks_aggressive" \
        --batch_size 8
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)

# Import model components
from mtl_peptide_classifier import (
    MTLPeptideClassifier,
    PeptideDataset,
    get_all_peptide_tasks
)

# Import tokenizer
from transformers import EsmTokenizer


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_all_metrics(logits, labels, task_name):
    """
    Compute all classification metrics for a task.

    Returns dict with:
    - ACC: Accuracy
    - AUC: ROC-AUC
    - PR_AUC: Precision-Recall AUC
    - BACC: Balanced Accuracy
    - Sn: Sensitivity (Recall)
    - Sp: Specificity
    - MCC: Matthews Correlation Coefficient
    - Precision, F1, confusion matrix
    """
    # Get probabilities and predictions
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # For binary classification
    if probs.shape[1] == 2:
        pos_probs = probs[:, 1].cpu().numpy()
    else:
        pos_probs = probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs[:, 0].cpu().numpy()

    # Basic metrics
    accuracy = accuracy_score(labels_np, preds)
    precision = precision_score(labels_np, preds, average='binary', zero_division=0)
    recall = recall_score(labels_np, preds, average='binary', zero_division=0)
    f1 = f1_score(labels_np, preds, average='binary', zero_division=0)

    # Confusion matrix
    try:
        cm = confusion_matrix(labels_np, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except:
        # If one class missing
        cm = confusion_matrix(labels_np, preds)
        if cm.shape == (1, 1):
            if labels_np[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

    # Sensitivity (Recall) and Specificity
    sensitivity = recall  # TP / (TP + FN)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Balanced Accuracy
    balanced_accuracy = (sensitivity + specificity) / 2

    # ROC-AUC
    try:
        if len(np.unique(labels_np)) > 1:
            auc = roc_auc_score(labels_np, pos_probs)
        else:
            auc = 0.0
    except:
        auc = 0.0

    # PR-AUC (Average Precision)
    try:
        if len(np.unique(labels_np)) > 1:
            pr_auc = average_precision_score(labels_np, pos_probs)
        else:
            pr_auc = 0.0
    except:
        pr_auc = 0.0

    # MCC
    try:
        mcc = matthews_corrcoef(labels_np, preds)
    except:
        mcc = 0.0

    return {
        'ACC': accuracy,
        'AUC': auc,
        'PR_AUC': pr_auc,
        'BACC': balanced_accuracy,
        'Sn': sensitivity,
        'Sp': specificity,
        'MCC': mcc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_mtl_model(checkpoint_dir: str, task_configs: dict, device: str):
    """Load MTL model from checkpoint."""

    checkpoint_path = Path(checkpoint_dir) / "best_model"

    # Create model
    model = MTLPeptideClassifier(
        task_configs=task_configs,
        hidden_dim=1280,
        esm_ratio=0.9,
        num_transformer_layers=4,
        dropout=0.3
    )

    # Load shared backbone
    backbone_path = checkpoint_path / "shared_backbone.pt"
    if backbone_path.exists():
        backbone_state = torch.load(backbone_path, map_location=device)
        model.base_embed.load_state_dict(backbone_state['base_embed'])
        model.transformer.load_state_dict(backbone_state['transformer'])
        model.cnn.load_state_dict(backbone_state['cnn'])
        model.layer_norm.load_state_dict(backbone_state['layer_norm'])

    # Load task heads
    heads_path = checkpoint_path / "heads.pt"
    if heads_path.exists():
        heads_state = torch.load(heads_path, map_location=device)
        for name, head in model.heads.items():
            if name in heads_state:
                head.load_state_dict(heads_state[name])

    model = model.to(device)
    model.eval()

    print(f"* Loaded model from {checkpoint_dir}")
    return model


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_task(model, dataloader, task_name, device):
    """Evaluate model on a specific task."""
    model.eval()

    all_logits = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with autocast():
            logits = model(input_ids, attention_mask, task_name)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = compute_all_metrics(all_logits, all_labels, task_name)
    return metrics


@torch.no_grad()
def evaluate_all_tasks(model, val_datasets, task_names, device, batch_size=8):
    """Evaluate model on all validation tasks."""

    task_metrics = {}

    for task_name in task_names:
        if task_name not in val_datasets:
            continue

        dataset = val_datasets[task_name]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f"  Evaluating {task_name}...", end=" ")
        metrics = evaluate_task(model, dataloader, task_name, device)
        task_metrics[task_name] = metrics
        print(f"F1={metrics['F1']:.4f}, AUC={metrics['AUC']:.4f}, PR-AUC={metrics['PR_AUC']:.4f}")

    return task_metrics


# ============================================================================
# RESULTS FORMATTING
# ============================================================================

def format_results_table(task_metrics, model_name="Model"):
    """Format results as a markdown table."""

    print(f"\n{'='*120}")
    print(f"{model_name} - Comprehensive Results")
    print(f"{'='*120}\n")

    # Header
    header = f"{'Dataset':<20} {'ACC(%)':>8} {'AUC(%)':>8} {'PR-AUC(%)':>10} {'BACC(%)':>9} {'Sn(%)':>7} {'Sp(%)':>7} {'MCC(%)':>7}"
    print(header)
    print("-" * 120)

    # Rows
    for task_name, metrics in sorted(task_metrics.items()):
        row = f"{task_name:<20} "
        row += f"{metrics['ACC']*100:>7.2f} "
        row += f"{metrics['AUC']*100:>7.2f} "
        row += f"{metrics['PR_AUC']*100:>9.2f} "
        row += f"{metrics['BACC']*100:>8.2f} "
        row += f"{metrics['Sn']*100:>6.2f} "
        row += f"{metrics['Sp']*100:>6.2f} "
        row += f"{metrics['MCC']*100:>6.2f}"
        print(row)

    # Average
    print("-" * 120)
    avg_acc = np.mean([m['ACC'] for m in task_metrics.values()])
    avg_auc = np.mean([m['AUC'] for m in task_metrics.values()])
    avg_pr_auc = np.mean([m['PR_AUC'] for m in task_metrics.values()])
    avg_bacc = np.mean([m['BACC'] for m in task_metrics.values()])
    avg_sn = np.mean([m['Sn'] for m in task_metrics.values()])
    avg_sp = np.mean([m['Sp'] for m in task_metrics.values()])
    avg_mcc = np.mean([m['MCC'] for m in task_metrics.values()])

    avg_row = f"{'AVERAGE':<20} "
    avg_row += f"{avg_acc*100:>7.2f} "
    avg_row += f"{avg_auc*100:>7.2f} "
    avg_row += f"{avg_pr_auc*100:>9.2f} "
    avg_row += f"{avg_bacc*100:>8.2f} "
    avg_row += f"{avg_sn*100:>6.2f} "
    avg_row += f"{avg_sp*100:>6.2f} "
    avg_row += f"{avg_mcc*100:>6.2f}"
    print(avg_row)
    print("="*120)

    return {
        'avg_acc': avg_acc,
        'avg_auc': avg_auc,
        'avg_pr_auc': avg_pr_auc,
        'avg_bacc': avg_bacc,
        'avg_sn': avg_sn,
        'avg_sp': avg_sp,
        'avg_mcc': avg_mcc
    }


def save_detailed_results(task_metrics, model_name, output_dir):
    """Save detailed results to JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    results = {
        'model_name': model_name,
        'evaluation_date': datetime.now().isoformat(),
        'metrics': {}
    }

    for task_name, metrics in task_metrics.items():
        results['metrics'][task_name] = {
            'ACC': float(metrics['ACC']),
            'AUC': float(metrics['AUC']),
            'PR_AUC': float(metrics['PR_AUC']),
            'BACC': float(metrics['BACC']),
            'Sn': float(metrics['Sn']),
            'Sp': float(metrics['Sp']),
            'MCC': float(metrics['MCC']),
            'Precision': float(metrics['Precision']),
            'Recall': float(metrics['Recall']),
            'F1': float(metrics['F1']),
            'TP': int(metrics['TP']),
            'TN': int(metrics['TN']),
            'FP': int(metrics['FP']),
            'FN': int(metrics['FN'])
        }

    output_file = output_dir / f"{model_name}_detailed_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n* Detailed results saved to {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive MTL Evaluation")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--model_name", type=str, default="MTL_Model",
                        help="Name for this model in results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use")

    args = parser.parse_args()

    # Configuration
    # Data directory - relative path for portability
    script_dir = Path(__file__).parent
    data_dir = str(script_dir / "datasets")
    max_length = 128
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*120)
    print(f"COMPREHENSIVE MTL EVALUATION: {args.model_name}")
    print("="*120)
    print(f"\nModel directory: {args.model_dir}")
    print(f"Device: {device}")

    # Get task configurations
    print("\n" + "-"*120)
    print("Loading task configurations...")
    print("-"*120)

    task_configs = get_all_peptide_tasks(data_dir)
    print(f"\n* Detected {len(task_configs)} peptide tasks")

    # Import tokenizer
    print("\n" + "-"*120)
    print("Loading tokenizer...")
    print("-"*120)

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    print("* Tokenizer loaded")

    # Create test datasets
    print("\n" + "-"*120)
    print("Loading test datasets...")
    print("-"*120)

    test_datasets = {}
    task_names = []

    for task_name, cfg in task_configs.items():
        prefix = cfg['csv_prefix']
        test_path = Path(data_dir) / f"{prefix}_test.csv"

        if test_path.exists():
            test_datasets[task_name] = PeptideDataset(
                str(test_path),
                tokenizer,
                max_length
            )
            task_names.append(task_name)

    print(f"\n* Created {len(test_datasets)} test datasets")

    # Load model
    print("\n" + "-"*120)
    print("Loading model...")
    print("-"*120)

    model = load_mtl_model(args.model_dir, task_configs, device)

    # Evaluate all tasks
    print("\n" + "-"*120)
    print("Evaluating on all tasks...")
    print("-"*120)

    task_metrics = evaluate_all_tasks(
        model,
        test_datasets,
        task_names,
        device,
        args.batch_size
    )

    # Format and print results
    print("\n" + "-"*120)
    print("Results Summary")
    print("-"*120)

    averages = format_results_table(task_metrics, args.model_name)

    # Save detailed results
    output_dir = Path(args.model_dir) / "comprehensive_evaluation"
    save_detailed_results(task_metrics, args.model_name, output_dir)

    print("\n" + "="*120)
    print(f"EVALUATION COMPLETED: {args.model_name}")
    print("="*120)
    print(f"\nOverall Performance:")
    print(f"  - Average Accuracy: {averages['avg_acc']*100:.2f}%")
    print(f"  - Average AUC: {averages['avg_auc']*100:.2f}%")
    print(f"  - Average PR-AUC: {averages['avg_pr_auc']*100:.2f}%")
    print(f"  - Average MCC: {averages['avg_mcc']*100:.2f}%")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
