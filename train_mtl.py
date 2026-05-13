"""
MTL Training Script - PDeepPP Architecture
Trains all 19 peptide tasks jointly with frozen ESM-2 backbone.

This is the training script used for Original_MTL_19tasks_aggressive model.
The "aggressive" variant uses more aggressive hyperparameters for better performance.

Usage:
    python train_mtl.py --batch_size 16 --lr 1e-4 --epochs 50 --dropout 0.3

Aggressive Training Configuration:
    - Learning rate: 1e-4 (standard)
    - Batch size: 16
    - Epochs: 50 (aggressive - more epochs)
    - Dropout: 0.3
    - TIM Loss: Enabled
    - Label smoothing: 0.1
    - Gradient clipping: 1.0
    - Mixed precision: Enabled
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from tqdm import tqdm

# Import model components
from mtl_peptide_classifier import (
    MTLPeptideClassifier,
    MultiTaskDataLoader,
    PeptideDataset,
    TIMLoss,
    get_all_peptide_tasks
)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class MTLConfig:
    """Configuration for MTL training."""

    def __init__(self):
        # Data
        # Data directory - relative path for portability
        script_dir = Path(__file__).parent
        self.data_dir = str(script_dir / "datasets")
        self.max_length = 128
        self.batch_size = 16

        # Model
        self.hidden_dim = 1280
        self.esm_ratio = 0.9
        self.num_transformer_layers = 4
        self.dropout = 0.3

        # Ablation flags
        self.use_transformer = True
        self.use_cnn = True
        self.unfreeze_esm = False
        self.ablation_name = ""   # auto-derived when empty

        # Training
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 50
        self.warmup_epochs = 5
        self.grad_clip = 1.0

        # Loss
        self.use_tim_loss = True
        self.label_smoothing = 0.1

        # System
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision = True
        self.num_workers = 2

        # Validation split (fraction of training data used for checkpoint selection)
        self.val_split = 0.2

        # Paths - use relative path for portability
        self.output_dir = str(script_dir / "checkpoints")
        self.log_interval = 50

    def get_variant_name(self) -> str:
        """Auto-derive a readable variant name from ablation flags."""
        if self.ablation_name:
            return self.ablation_name
        parts = []
        if not self.use_transformer:
            parts.append("no_transformer")
        if not self.use_cnn:
            parts.append("no_cnn")
        if self.unfreeze_esm:
            parts.append("unfreeze_esm")
        if abs(self.esm_ratio - 0.9) > 1e-6:
            parts.append(f"esm_ratio_{self.esm_ratio:.1f}".replace(".", "p"))
        if not self.use_tim_loss:
            parts.append("no_tim")
        if self.num_transformer_layers != 4:
            parts.append(f"transformer_{self.num_transformer_layers}L")
        return "_".join(parts) if parts else "full_model"


# ============================================================================
# METRICS & EVALUATION
# ============================================================================

def compute_metrics(logits, labels, task_name):
    """Compute classification metrics for a task."""
    # Get predictions
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Basic metrics
    accuracy = accuracy_score(labels_np, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds, average='binary', zero_division=0
    )

    # AUC (if binary)
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(labels_np, probs[:, 1].cpu().numpy())
        else:
            auc = 0.0
    except:
        auc = 0.0

    # MCC
    try:
        mcc = matthews_corrcoef(labels_np, preds)
    except:
        mcc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc
    }


@torch.no_grad()
def evaluate_task(model, dataloader, task_name, device, config):
    """Evaluate model on a specific task."""
    model.eval()

    all_logits = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with autocast('cuda', enabled=config.mixed_precision and config.device == 'cuda'):
            logits = model(input_ids, attention_mask, task_name)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_logits, all_labels, task_name)
    return metrics


@torch.no_grad()
def evaluate_all_tasks(model, val_datasets, device, config):
    """Evaluate model on all validation tasks."""
    model.eval()

    task_metrics = {}

    for task_name, dataset in val_datasets.items():
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )

        metrics = evaluate_task(model, dataloader, task_name, device, config)
        task_metrics[task_name] = metrics

    return task_metrics


# ============================================================================
# TRAINING LOOP
# ============================================================================

class MTLTrainer:
    """Multi-Task Learning Trainer."""

    def __init__(self, model, train_loader, val_datasets, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_datasets = val_datasets
        self.config = config

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.total_steps = len(train_loader) * config.num_epochs
        self.warmup_steps = len(train_loader) * config.warmup_epochs

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - self.warmup_steps,
            eta_min=config.learning_rate * 0.01
        )

        # Loss
        # reduction='none' needed for per-sample TIM loss weighting
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            reduction='none'
        )
        self.criterion_mean = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )

        if config.use_tim_loss:
            self.tim_loss = TIMLoss(len(train_loader.task_names))
        else:
            self.tim_loss = None

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.mixed_precision and config.device == 'cuda' else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_avg_f1 = 0.0
        self.history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rate': []
        }

        # Task name to index mapping for TIM loss
        self.task_to_idx = {name: i for i, name in enumerate(train_loader.task_names)}

        print(f"\n* Trainer initialized:")
        print(f"  - Device: {config.device}")
        print(f"  - Total steps: {self.total_steps}")
        print(f"  - Warmup steps: {self.warmup_steps}")
        print(f"  - TIM Loss: {config.use_tim_loss}")

    def train(self):
        """Main training loop."""

        print("\n" + "="*80)
        print("MTL TRAINING STARTED")
        print("="*80)

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"{'='*80}")

            # Training
            epoch_loss = self.train_epoch()

            # Validation
            val_metrics = evaluate_all_tasks(
                self.model,
                self.val_datasets,
                self.config.device,
                self.config
            )

            # Log metrics
            self.log_metrics(epoch_loss, val_metrics)

            # Save checkpoint
            self.save_checkpoint(val_metrics)

        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)

        # Save final results
        self.save_final_results()

        return self.history

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for step, (batch, task_name) in pbar:
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['label'].to(self.config.device)

            # Forward
            with autocast('cuda', enabled=self.scaler is not None):
                logits = self.model(input_ids, attention_mask, task_name)
                if self.tim_loss is not None:
                    # Per-sample losses for TIM weighting
                    per_sample_losses = self.criterion(logits, labels)
                    task_idx = self.task_to_idx[task_name]
                    task_idx_tensor = torch.full(
                        (len(labels),), task_idx,
                        dtype=torch.long, device=self.config.device
                    )
                    loss = self.tim_loss(per_sample_losses, task_idx_tensor)
                else:
                    loss = self.criterion_mean(logits, labels)

            # Backward
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.optimizer.step()

            # Scheduler after warmup
            if self.global_step >= self.warmup_steps:
                self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            # Progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'task': task_name
            })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def log_metrics(self, train_loss, val_metrics):
        """Log training and validation metrics."""

        # Compute average metrics
        avg_acc = np.mean([m['accuracy'] for m in val_metrics.values()])
        avg_f1 = np.mean([m['f1'] for m in val_metrics.values()])
        avg_auc = np.mean([m['auc'] for m in val_metrics.values()])

        # Save history
        self.history['train_loss'].append(train_loss)
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

        # Print summary
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Avg - ACC: {avg_acc:.4f} | F1: {avg_f1:.4f} | AUC: {avg_auc:.4f}")
        print("\nPer-Task Metrics:")
        for task_name, metrics in val_metrics.items():
            print(f"  {task_name:25s}: ACC={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

    def save_checkpoint(self, val_metrics):
        """Save model checkpoint."""

        # Compute average F1
        avg_f1 = np.mean([m['f1'] for m in val_metrics.values()])

        # Save best model
        if avg_f1 > self.best_avg_f1:
            self.best_avg_f1 = avg_f1

            variant = self.config.get_variant_name()
            checkpoint_dir = Path(self.config.output_dir) / variant / "best_model"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save full checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
                'best_avg_f1': self.best_avg_f1,
                'val_metrics': val_metrics
            }, checkpoint_dir / "checkpoint.pt")

            # Save heads only (for inference)
            heads_state = {name: head.state_dict() for name, head in self.model.heads.items()}
            torch.save(heads_state, checkpoint_dir / "heads.pt")

            # Save shared components (conditional on ablation flags)
            backbone_state = {'base_embed': self.model.base_embed.state_dict()}
            if self.model.use_transformer:
                backbone_state['transformer'] = self.model.transformer.state_dict()
            if self.model.use_cnn:
                backbone_state['cnn'] = self.model.cnn.state_dict()
                backbone_state['layer_norm'] = self.model.layer_norm.state_dict()
            torch.save(backbone_state, checkpoint_dir / "shared_backbone.pt")

            # Save ablation config alongside checkpoint for reproducibility
            ablation_cfg = {
                'variant': variant,
                'use_transformer': self.model.use_transformer,
                'use_cnn': self.model.use_cnn,
                'unfreeze_esm': self.model.unfreeze_esm,
                'esm_ratio': self.model.esm_ratio,
                'feature_dim': self.model.feature_dim,
                'use_tim_loss': self.config.use_tim_loss,
                'label_smoothing': self.config.label_smoothing,
                'num_transformer_layers': self.config.num_transformer_layers,
            }
            with open(checkpoint_dir / "ablation_config.json", 'w') as f:
                json.dump(ablation_cfg, f, indent=2)

            print(f"\n* Saved best model [{variant}] (Avg F1: {avg_f1:.4f})")

    def save_final_results(self):
        """Save final training results."""

        variant = self.config.get_variant_name()
        output_dir = Path(self.config.output_dir) / variant
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save training history
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

        # Save final metrics
        final_metrics = self.history['val_metrics'][-1] if self.history['val_metrics'] else {}

        # Create summary
        summary = {
            'variant': variant,
            'best_avg_f1': float(self.best_avg_f1),
            'final_epoch': self.current_epoch + 1,
            'final_metrics': {
                task: {k: float(v) for k, v in metrics.items()}
                for task, metrics in final_metrics.items()
            },
            'config': {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.num_epochs,
                'dropout': self.config.dropout,
                'use_tim_loss': self.config.use_tim_loss,
                'label_smoothing': self.config.label_smoothing,
                'use_transformer': self.config.use_transformer,
                'use_cnn': self.config.use_cnn,
                'unfreeze_esm': self.config.unfreeze_esm,
                'esm_ratio': self.config.esm_ratio,
                'num_transformer_layers': self.config.num_transformer_layers,
            }
        }

        with open(output_dir / "results.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n* Results saved to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training function."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="MTL Training for Peptide Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation Study Examples:
  Full model (baseline):
      python train_mtl.py

  Without CNN branch:
      python train_mtl.py --no_cnn

  Without Transformer branch:
      python train_mtl.py --no_transformer

  Transformer only (2 layers):
      python train_mtl.py --no_cnn --transformer_layers 2

  Without TIM loss:
      python train_mtl.py --no_tim

  Unfrozen ESM-2 backbone:
      python train_mtl.py --unfreeze_esm --lr 1e-5

  ESM ratio 0.5 (more base embedding):
      python train_mtl.py --esm_ratio 0.5

  Pure ESM, no learnable base embedding:
      python train_mtl.py --esm_ratio 1.0

  Custom ablation name:
      python train_mtl.py --no_cnn --ablation_name my_no_cnn_run
        """
    )
    # Standard hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of train data held out for validation/checkpoint selection (default 0.1)")

    # Loss ablations
    parser.add_argument("--no_tim", action="store_true", help="Disable TIM loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default 0.1; set 0.0 to disable)")

    # Architecture ablations
    parser.add_argument("--no_transformer", action="store_true",
                        help="[Ablation] Remove shared Transformer encoder")
    parser.add_argument("--no_cnn", action="store_true",
                        help="[Ablation] Remove shared CNN branch")
    parser.add_argument("--unfreeze_esm", action="store_true",
                        help="[Ablation] Unfreeze ESM-2 backbone (use lower lr, e.g. 1e-5)")
    parser.add_argument("--esm_ratio", type=float, default=0.9,
                        help="[Ablation] ESM-2 weight in embedding mix (default 0.9; 1.0=pure ESM, 0.0=pure base)")
    parser.add_argument("--transformer_layers", type=int, default=4,
                        help="[Ablation] Number of shared Transformer layers (default 4)")

    # Naming
    parser.add_argument("--ablation_name", type=str, default="",
                        help="Custom name for this run's checkpoint dir (auto-derived if omitted)")

    args = parser.parse_args()

    # Configuration
    config = MTLConfig()

    # Standard hyperparameters
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_epochs = args.epochs
    config.dropout = args.dropout
    if args.device:
        config.device = args.device

    config.val_split = args.val_split

    # Loss
    config.use_tim_loss = not args.no_tim
    config.label_smoothing = args.label_smoothing

    # Architecture ablations
    config.use_transformer = not args.no_transformer
    config.use_cnn = not args.no_cnn
    config.unfreeze_esm = args.unfreeze_esm
    config.esm_ratio = args.esm_ratio
    config.num_transformer_layers = args.transformer_layers
    config.ablation_name = args.ablation_name

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    variant = config.get_variant_name()

    print("\n" + "="*80)
    print("MTL PEPTIDE CLASSIFIER - TRAINING")
    print("="*80)
    print(f"\nVariant: {variant}")
    print(f"\nConfiguration:")
    print(f"  - Batch size:          {config.batch_size}")
    print(f"  - Learning rate:       {config.learning_rate}")
    print(f"  - Epochs:              {config.num_epochs}")
    print(f"  - Dropout:             {config.dropout}")
    print(f"  - Device:              {config.device}")
    print(f"\nArchitecture (Ablation):")
    print(f"  - Transformer:         {'ON' if config.use_transformer else 'OFF (ablated)'} ({config.num_transformer_layers} layers)")
    print(f"  - CNN:                 {'ON' if config.use_cnn else 'OFF (ablated)'}")
    print(f"  - ESM-2:               {'Unfrozen (fine-tune)' if config.unfreeze_esm else 'Frozen'}")
    print(f"  - ESM ratio:           {config.esm_ratio}")
    print(f"\nLoss:")
    print(f"  - TIM Loss:            {'ON' if config.use_tim_loss else 'OFF (ablated)'}")
    print(f"  - Label smoothing:     {config.label_smoothing}")

    # Get task configurations
    print("\n" + "-"*80)
    print("Loading datasets...")
    print("-"*80)

    task_configs = get_all_peptide_tasks(config.data_dir)
    print(f"\n* Detected {len(task_configs)} peptide tasks:")
    for name, cfg in task_configs.items():
        print(f"  - {name}: {cfg['num_classes']} classes")

    # Import tokenizer
    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Create datasets — 3-way split following UniDL4BioPep methodology:
    #   train_datasets : 90% of *_train.csv  — model learns from this
    #   val_datasets   : 10% of *_train.csv  — used only to select best checkpoint
    #   test_datasets  : *_test.csv           — touched once after training for final reporting
    train_datasets = {}
    val_datasets = {}
    test_datasets = {}

    rng = torch.Generator().manual_seed(42)

    for task_name, cfg in task_configs.items():
        prefix = cfg['csv_prefix']
        train_path = Path(config.data_dir) / f"{prefix}_train.csv"
        test_path = Path(config.data_dir) / f"{prefix}_test.csv"

        if train_path.exists():
            full_train = PeptideDataset(str(train_path), tokenizer, config.max_length)
            n_val = max(1, int(len(full_train) * config.val_split))
            n_train = len(full_train) - n_val
            train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=rng)
            train_datasets[task_name] = train_subset
            val_datasets[task_name] = val_subset

        if test_path.exists():
            test_datasets[task_name] = PeptideDataset(str(test_path), tokenizer, config.max_length)

    print(f"\n* Created datasets:")
    print(f"  - Train tasks:  {len(train_datasets)} ({1 - config.val_split:.0%} of train CSV)")
    print(f"  - Val tasks:    {len(val_datasets)} ({config.val_split:.0%} of train CSV, for checkpoint selection)")
    print(f"  - Test tasks:   {len(test_datasets)} (held-out, evaluated once after training)")

    # Create model
    print("\n" + "-"*80)
    print("Creating model...")
    print("-"*80)

    model = MTLPeptideClassifier(
        task_configs=task_configs,
        hidden_dim=config.hidden_dim,
        esm_ratio=config.esm_ratio,
        num_transformer_layers=config.num_transformer_layers,
        dropout=config.dropout,
        use_transformer=config.use_transformer,
        use_cnn=config.use_cnn,
        unfreeze_esm=config.unfreeze_esm,
    )

    trainable = model.get_trainable_params()
    total = sum(p.numel() for p in model.parameters())
    print(f"\n* Model parameters:")
    print(f"  - Total: {total:,}")
    print(f"  - Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # Create dataloader
    train_loader = MultiTaskDataLoader(train_datasets, config.batch_size)
    print(f"\n* Created dataloader:")
    print(f"  - Approx batches/epoch: {len(train_loader)}")

    # Create trainer
    trainer = MTLTrainer(model, train_loader, val_datasets, config)

    # Train
    history = trainer.train()

    print("\n" + "="*80)
    print("MTL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nVariant:          {variant}")
    print(f"Best Val Avg F1:  {trainer.best_avg_f1:.4f}  (on validation split — used for checkpoint selection)")
    print(f"Results saved to: {Path(config.output_dir) / variant}")

    # ---- One-shot test evaluation on held-out test set ----
    best_checkpoint_path = Path(config.output_dir) / variant / "best_model" / "checkpoint.pt"
    if test_datasets and best_checkpoint_path.exists():
        print("\n" + "-"*80)
        print("Final evaluation on held-out test set (best checkpoint)")
        print("-"*80)

        checkpoint = torch.load(best_checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.device)

        test_metrics = evaluate_all_tasks(model, test_datasets, config.device, config)

        test_avg_acc = np.mean([m['accuracy'] for m in test_metrics.values()])
        test_avg_f1  = np.mean([m['f1']       for m in test_metrics.values()])
        test_avg_auc = np.mean([m['auc']       for m in test_metrics.values()])
        test_avg_mcc = np.mean([m['mcc']       for m in test_metrics.values()])

        print(f"\nTest Avg — ACC: {test_avg_acc:.4f} | F1: {test_avg_f1:.4f} | AUC: {test_avg_auc:.4f} | MCC: {test_avg_mcc:.4f}")
        print("\nPer-Task Test Metrics:")
        for task_name, m in test_metrics.items():
            print(f"  {task_name:25s}: ACC={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} MCC={m['mcc']:.4f}")

        # Save test results alongside the checkpoint
        test_results = {
            'best_val_avg_f1': float(trainer.best_avg_f1),
            'test_avg_acc':    float(test_avg_acc),
            'test_avg_f1':     float(test_avg_f1),
            'test_avg_auc':    float(test_avg_auc),
            'test_avg_mcc':    float(test_avg_mcc),
            'test_metrics': {
                task: {k: float(v) for k, v in m.items()}
                for task, m in test_metrics.items()
            }
        }
        results_path = Path(config.output_dir) / variant / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\n* Test results saved to {results_path}")


if __name__ == "__main__":
    main()
