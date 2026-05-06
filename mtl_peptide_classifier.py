"""
MTL Peptide Classifier - PDeepPP Architecture
All 19 peptide activity datasets trained jointly with frozen ESM-2 backbone.

This is the model architecture used for Original_MTL_19tasks_aggressive training.
"""

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import random


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MTLPeptideClassifier(nn.Module):
    """
    Multi-Task Learning classifier for peptide activities.
    Based on PDeepPP architecture with frozen ESM-2 backbone.

    Architecture:
    - Frozen ESM-2 (650M params) as base encoder
    - Learnable base embedding for amino acids
    - Weighted combination of ESM-2 and base embeddings
    - Shared transformer encoder for global context  [ablatable]
    - Shared CNN for local features                  [ablatable]
    - Task-specific classification heads

    Ablation flags:
        use_transformer: include shared transformer encoder (default True)
        use_cnn:         include shared CNN branch (default True)
        unfreeze_esm:    allow ESM-2 gradients to flow (default False)
    """

    def __init__(
        self,
        task_configs: Dict[str, Dict],
        hidden_dim: int = 1280,
        esm_ratio: float = 0.9,
        num_transformer_layers: int = 4,
        dropout: float = 0.3,
        use_transformer: bool = True,
        use_cnn: bool = True,
        unfreeze_esm: bool = False,
    ):
        """
        Args:
            task_configs: {task_name: {'num_classes': int, 'csv_prefix': str}}
            hidden_dim: Hidden dimension for base embedding
            esm_ratio: Weight for ESM-2 vs base embedding (0-1)
            num_transformer_layers: Layers in shared transformer
            dropout: Dropout rate
            use_transformer: Enable shared transformer encoder
            use_cnn: Enable shared CNN branch
            unfreeze_esm: Unfreeze ESM-2 backbone for fine-tuning
        """
        super().__init__()

        self.use_transformer = use_transformer
        self.use_cnn = use_cnn
        self.unfreeze_esm = unfreeze_esm

        # 1. Shared Encoder - ESM-2
        self.esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        if unfreeze_esm:
            self.esm.requires_grad_(True)   # Fine-tune backbone
        else:
            self.esm.requires_grad_(False)  # Freeze ESM-2 completely

        # 2. Learnable Base Embedding (amino acid embeddings)
        self.base_embed = nn.Embedding(33, hidden_dim)  # 33 amino acids
        self.esm_ratio = esm_ratio

        # 3. Shared Feature Extractor (conditional on ablation flags)
        if use_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_transformer_layers
            )

        if use_cnn:
            self.cnn = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=7,
                padding=3
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)

        # Head input dimension depends on active branches
        if use_transformer and use_cnn:
            self.feature_dim = hidden_dim * 2   # concat global + local
        else:
            self.feature_dim = hidden_dim        # single branch or pass-through

        # 4. Task-Specific Heads (all sequence-level for peptides)
        self.heads = nn.ModuleDict()
        for name, cfg in task_configs.items():
            self.heads[name] = SequenceHead(
                input_dim=self.feature_dim,
                num_classes=cfg['num_classes'],
                dropout=dropout
            )

        esm_status = "Unfrozen" if unfreeze_esm else "Frozen"
        branches = []
        if use_transformer:
            branches.append(f"{num_transformer_layers}-layer Transformer")
        if use_cnn:
            branches.append("CNN")
        if not branches:
            branches.append("Pass-through (embedding only)")
        print(f"* MTL Model initialized with {len(task_configs)} tasks")
        print(f"  - ESM-2: {esm_status} (650M params), ratio={esm_ratio}")
        print(f"  - Base Embedding: {hidden_dim} dim")
        print(f"  - Shared Backbone: {' + '.join(branches)}")
        print(f"  - Feature dim: {self.feature_dim}")
        print(f"  - Task Heads: {len(task_configs)} sequence-level")

    def encode(self, input_ids, attention_mask):
        """
        Encode sequences through shared backbone.
        Returns: [B, L, feature_dim]
          - feature_dim = 2*hidden_dim  when both Transformer + CNN active
          - feature_dim =   hidden_dim  when only one branch active
        """
        # ESM-2 embeddings (frozen or unfrozen depending on ablation flag)
        if self.unfreeze_esm:
            esm_out = self.esm(input_ids, attention_mask).last_hidden_state
        else:
            with torch.no_grad():
                esm_out = self.esm(input_ids, attention_mask).last_hidden_state

        # Base embeddings (learnable)
        base_out = self.base_embed(input_ids)

        # Weighted combination
        x = self.esm_ratio * esm_out + (1 - self.esm_ratio) * base_out

        # Feature extraction — conditional on ablation flags
        if self.use_transformer and self.use_cnn:
            # Full model: parallel Transformer + CNN, then concat
            global_feat = self.transformer(x)
            local_feat = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            local_feat = self.layer_norm(local_feat)
            shared_repr = torch.cat([global_feat, local_feat], dim=-1)  # [B, L, 2*hidden]
        elif self.use_transformer:
            # Ablation: Transformer only (no CNN)
            shared_repr = self.transformer(x)
        elif self.use_cnn:
            # Ablation: CNN only (no Transformer)
            shared_repr = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            shared_repr = self.layer_norm(shared_repr)
        else:
            # Ablation: pass-through (ESM embedding only, no feature extractor)
            shared_repr = x

        return shared_repr

    def forward(self, input_ids, attention_mask, task_name):
        """
        Forward pass for specific task.
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            task_name: which task head to use
        Returns:
            logits: [B, num_classes]
        """
        shared_repr = self.encode(input_ids, attention_mask)
        return self.heads[task_name](shared_repr, attention_mask)

    def get_trainable_params(self):
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SequenceHead(nn.Module):
    """
    Sequence-level classification head with masked pooling.
    For binary peptide activity classification.
    """

    def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, attention_mask):
        """
        Args:
            x: [B, L, D] shared representation
            attention_mask: [B, L] mask for pooling
        Returns:
            logits: [B, num_classes]
        """
        # Masked average pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        x_masked = x * mask_expanded
        x_pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        return self.fc(x_pooled)


# ============================================================================
# DATASET & DATALOADER
# ============================================================================

class PeptideDataset(Dataset):
    """Single peptide dataset for MTL training."""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        df = pd.read_csv(csv_path)
        # Handle both lowercase and capitalized column names
        seq_col = 'sequence' if 'sequence' in df.columns else 'Sequence'
        label_col = 'label' if 'label' in df.columns else 'Label'

        # Drop rows with NaN in sequence or label
        df = df.dropna(subset=[seq_col, label_col])

        # Convert sequences to strings and filter out non-string values
        df[seq_col] = df[seq_col].astype(str)
        df = df[df[seq_col] != 'nan']

        self.sequences = df[seq_col].tolist()
        self.labels = df[label_col].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = int(self.labels[idx])

        # Tokenize: space-separated amino acids
        tokens = " ".join(list(sequence))
        encoded = self.tokenizer(
            tokens,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class MultiTaskDataLoader:
    """
    Multi-task dataloader with task sampling.
    Samples batches from random tasks each iteration.
    """

    def __init__(self, task_datasets: Dict[str, Dataset], batch_size: int = 16):
        """
        Args:
            task_datasets: {task_name: Dataset}
            batch_size: batch size per task
        """
        self.task_loaders = {}
        import platform
        nw = 0 if platform.system() == "Windows" else 2
        for task_name, dataset in task_datasets.items():
            self.task_loaders[task_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                pin_memory=True
            )
        self.task_names = list(task_datasets.keys())
        self.batch_size = batch_size

    def __iter__(self):
        """Yield batches from randomly sampled tasks."""
        # Create iterators for each task
        iters = {name: iter(loader) for name, loader in self.task_loaders.items()}

        while iters:
            # Sample random task
            task = random.choice(list(iters.keys()))

            try:
                batch = next(iters[task])
                yield batch, task
            except StopIteration:
                # Task exhausted, restart iterator
                del iters[task]
                if len(iters) == 0:
                    break

    def __len__(self):
        # Approximate total batches per epoch
        return sum(len(loader) for loader in self.task_loaders.values())

    def get_task_batches(self, task_name: str):
        """Get all batches for a specific task (for validation)."""
        return list(self.task_loaders[task_name])


# ============================================================================
# TASK CONFIGURATION (20 Peptide Activities)
# ============================================================================

def get_all_peptide_tasks(data_dir: str) -> Dict[str, Dict]:
    """
    Auto-detect all 20 peptide tasks from UniDL4BioPep data directory.
    Returns task_configs for MTL model.

    20 Tasks:
    1. ACE_inhibitory - ACE inhibitory activity
    2. DPPIV_inhibitory - DPPIV inhibitory activity
    3. Bitter - Bitter taste peptides
    4. Umami - Umami taste peptides
    5. Antimicrobial - Antimicrobial activity
    6. Antimalarial - Antimalarial activity (main)
    7. Antimalarial_alt - Antimalarial activity (alternative)
    8. Quorum_sensing - Quorum sensing activity
    9. Anticancer - Anticancer activity (main)
    10. Anticancer_alt - Anticancer activity (alternative)
    11. AntiMRSA - Anti-MRSA strains activity
    12. TTCA - Therapeutic peptides for cancer
    13. BBP - Blood-Brain Barrier peptides
    14. Anti_parasitic - Anti-parasitic peptides
    15. NeuroPred - Neuroprotective peptides
    16. Antibacterial - Antibacterial peptides
    17. Antifungal - Antifungal peptides
    18. Antiviral - Antiviral peptides
    19. Toxicity - Toxicity prediction
    20. Anti_inflammatory - Anti-inflammatory peptides
    """
    data_path = Path(data_dir)

    # Define task mappings (includes both main and alternative datasets)
    task_mappings = {
        "1__ACE_inhibitory_activity": "ACE_inhibitory",
        "2__DPPIV_inhibitory_activity": "DPPIV_inhibitory",
        "3__Bitter": "Bitter",
        "4__Umami": "Umami",
        "5__Antimicrobial_activity": "Antimicrobial",
        "6__Antimalarial_activity-main": "Antimalarial",
        "6__Antimalarial_activity-alternative": "Antimalarial_alt",
        "7__Quorum_sensing_activity": "Quorum_sensing",
        "8__ACP_Anticancer_activity-main": "Anticancer",
        "8__ACP_Anticancer_activity-alternative": "Anticancer_alt",
        "9__Anti-MRSA_strains_activity": "AntiMRSA",
        "10__TTCA": "TTCA",
        "11__BBP_Blood-Brain_Barrier_Peptides": "BBP",
        "12__APP__Anti-parasitic": "Anti_parasitic",
        "13_NeuroPred": "NeuroPred",
        "14__antibacterial_AB": "Antibacterial",
        "15__Antifungal_AF": "Antifungal",
        "16__AV_Antiviral": "Antiviral",
        "17__Toxicity_2021_Dataset": "Toxicity",
        "18__Anti_inflammatory_peptides": "Anti_inflammatory",
        "19__Signal_peptides": "Signal_peptide"
    }

    task_configs = {}
    for csv_file in data_path.glob("*_train.csv"):
        prefix = csv_file.stem.replace("_train", "")

        if prefix in task_mappings:
            task_name = task_mappings[prefix]

            # Read full file to correctly detect classes (sampling can miss minority class)
            df = pd.read_csv(csv_file)
            label_col = 'label' if 'label' in df.columns else 'Label'
            n_classes = df[label_col].nunique() if label_col in df.columns else 2
            n_classes = max(n_classes, 2)  # enforce minimum 2 for binary classification

            task_configs[task_name] = {
                'num_classes': n_classes,
                'csv_prefix': prefix
            }

    return task_configs


# ============================================================================
# TIM LOSS (Threshold-independent Multi-task loss)
# ============================================================================

class TIMLoss(nn.Module):
    """
    Threshold-Independent Multi-task Loss for imbalanced datasets.
    Reference: https://arxiv.org/abs/2008.10599

    Uses learnable task-specific weights (log variances) to balance
    losses across tasks with different scales and difficulties.
    """

    def __init__(self, num_tasks: int):
        super().__init__()
        # Learnable task weights (log variances)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: torch.Tensor, task_indices: torch.Tensor):
        """
        Args:
            losses: [B] per-sample losses
            task_indices: [B] which task each sample belongs to
        Returns:
            weighted_loss: scalar
        """
        # Get precision for each task
        precision = torch.exp(-self.log_vars)

        # Weight each sample's loss by its task's precision
        weighted_losses = []
        for i, loss in enumerate(losses):
            task_idx = task_indices[i].item()
            weighted_loss = precision[task_idx] * loss + self.log_vars[task_idx]
            weighted_losses.append(weighted_loss)

        return torch.stack(weighted_losses).mean()


# ============================================================================
# MAIN TRAINING UTILS
# ============================================================================

def create_model_and_loaders(
    data_dir: str,
    batch_size: int = 16,
    max_length: int = 128
):
    """Create MTL model, datasets, and dataloaders."""

    # Get task configurations
    task_configs = get_all_peptide_tasks(data_dir)
    print(f"\n* Detected {len(task_configs)} peptide tasks:")
    for name, cfg in task_configs.items():
        print(f"  - {name}: {cfg['num_classes']} classes")

    # Initialize tokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Create model
    model = MTLPeptideClassifier(
        task_configs=task_configs,
        hidden_dim=1280,
        esm_ratio=0.9,
        num_transformer_layers=4,
        dropout=0.3
    )

    # Create datasets
    train_datasets = {}
    val_datasets = {}

    for task_name, cfg in task_configs.items():
        prefix = cfg['csv_prefix']
        train_path = Path(data_dir) / f"{prefix}_train.csv"
        val_path = Path(data_dir) / f"{prefix}_test.csv"  # Using test as val

        if train_path.exists():
            train_datasets[task_name] = PeptideDataset(
                str(train_path),
                tokenizer,
                max_length
            )
        if val_path.exists():
            val_datasets[task_name] = PeptideDataset(
                str(val_path),
                tokenizer,
                max_length
            )

    # Create multi-task dataloaders
    train_loader = MultiTaskDataLoader(train_datasets, batch_size)

    print(f"\n* Created dataloaders:")
    print(f"  - Train tasks: {len(train_datasets)}")
    print(f"  - Val tasks: {len(val_datasets)}")
    print(f"  - Approx batches/epoch: {len(train_loader)}")

    # Count trainable parameters
    trainable = model.get_trainable_params()
    total = sum(p.numel() for p in model.parameters())
    print(f"\n* Model parameters:")
    print(f"  - Total: {total:,}")
    print(f"  - Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    return model, train_loader, val_datasets, task_configs


if __name__ == "__main__":
    # Test model creation
    # Data directory - relative path for portability
    script_dir = Path(__file__).parent
    data_dir = str(script_dir / "datasets")

    model, train_loader, val_datasets, task_configs = create_model_and_loaders(
        data_dir,
        batch_size=16
    )

    # Test forward pass
    print("\n" + "="*60)
    print("Testing forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get a batch
    for batch, task_name in train_loader:
        print(f"  Task: {task_name}")
        print(f"  Batch size: {batch['input_ids'].shape[0]}")

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, task_name)

        print(f"  Logits shape: {logits.shape}")
        print("  * Forward pass successful!")
        break
