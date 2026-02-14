# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Task Learning (MTL) peptide classifier for 19 UniDL4BioPep peptide activity datasets. Uses PDeepPP-inspired architecture with frozen ESM-2 backbone for efficient joint training across diverse peptide classification tasks.

### Model Performance (Original_MTL_19tasks_aggressive)
- Average Accuracy: 89.49%
- Average AUC: 94.15%
- Average PR-AUC: 92.85%
- Average MCC: 78.88%

This is the **best performing variant** among all MTL models tested.

### The 19 Peptide Activity Tasks
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

## Architecture

```
Input: Peptide Sequence
         ↓
    Shared Encoder (Frozen)
    - ESM-2 (650M params): facebook/esm2_t33_650M_UR50D
    - Base Embedding (33 aa → 1280 dim, learnable)
    - Weighted Combination (ESM ratio: 0.9)
         ↓
    Parallel Feature Extraction
    - Transformer (4 layers, 8 heads) for global context
    - CNN (kernel=7) for local features
         ↓
    Concatenated Features [2560 dim]
         ↓
    Task-Specific Heads (19 binary classifiers)
    - SequenceHead: 2560 → 256 → 128 → 2
    - Masked average pooling
    - 2 FC layers with ReLU + Dropout(0.3)
         ↓
    Output: Binary logits
```

## File Structure

```
Original_MTL_19tasks_aggressive/
├── train_mtl.py                    # Main training script
├── mtl_peptide_classifier.py       # Model architecture + dataset classes
├── evaluate_mtl_comprehensive.py   # Evaluation script with full metrics
├── datasets/                       # UniDL4BioPep data (train/test CSVs)
│   ├── *_train.csv                # Training data per task
│   └── *_test.csv                 # Test data per task
└── checkpoints/                    # Model checkpoints (created during training)
    └── best_model/
        ├── checkpoint.pt           # Full checkpoint
        ├── heads.pt                # Task-specific heads
        └── shared_backbone.pt      # Shared components
```

## Common Commands

### Training

```bash
# Default aggressive training (recommended)
python train_mtl.py --batch_size 16 --lr 1e-4 --epochs 50 --dropout 0.3

# Without TIM loss
python train_mtl.py --no_tim

# Custom configuration
python train_mtl.py --batch_size 32 --lr 5e-5 --epochs 30 --dropout 0.2
```

### Evaluation

```bash
# Comprehensive evaluation (all metrics: ACC, AUC, PR-AUC, MCC, etc.)
python evaluate_mtl_comprehensive.py \
    --model_dir "checkpoints" \
    --model_name "Original_MTL_19tasks_aggressive" \
    --batch_size 8
```

### Testing Model Components

```bash
# Test model architecture and forward pass
python mtl_peptide_classifier.py
```

## Key Architecture Components

### MTLPeptideClassifier (`mtl_peptide_classifier.py`)

**Core Design Principles:**
- **Frozen ESM-2 Backbone**: Leverages pre-trained protein representations without catastrophic forgetting
- **Learnable Base Embedding**: Allows task-specific adaptation via weighted combination with ESM-2
- **Parallel Feature Extraction**: Transformer for global context + CNN for local patterns
- **Task-Specific Heads**: Separate binary classifier for each of 19 peptide activities

**Key Parameters:**
- `esm_ratio=0.9`: Weight of ESM-2 vs base embedding (higher = more reliance on pre-trained features)
- `num_transformer_layers=4`: Shared transformer layers for global context
- `hidden_dim=1280`: Hidden dimension (matches ESM-2 output)
- `dropout=0.3`: Dropout rate in heads

### MultiTaskDataLoader (`mtl_peptide_classifier.py`)

Samples batches randomly from all 19 tasks each iteration. This ensures all tasks are trained jointly with balanced sampling across the entire epoch.

### TIMLoss (`mtl_peptide_classifier.py`)

Threshold-Independent Multi-task loss for balancing losses across tasks with different scales. Uses learnable task-specific log variances to weight losses automatically.

## Dataset Format

CSV files in `datasets/` directory with columns:
- `sequence` or `Sequence`: Peptide amino acid sequence
- `label` or `Label`: Binary label (0/1)

Dataset naming convention maps UniDL4BioPep folder names to task names (defined in `get_all_peptide_tasks()`).

### Task Name Mapping

```
1__ACE_inhibitory_activity        → ACE_inhibitory
2__DPPIV_inhibitory_activity      → DPPIV_inhibitory
3__Bitter                         → Bitter
4__Umami                          → Umami
5__Antimicrobial_activity         → Antimicrobial
6__Antimalarial_activity-main     → Antimalarial
6__Antimalarial_activity-alternative → Antimalarial_alt
7__Quorum_sensing_activity        → Quorum_sensing
8__ACP_Anticancer_activity-main   → Anticancer
8__ACP_Anticancer_activity-alternative → Anticancer_alt
9__Anti-MRSA_strains_activity     → AntiMRSA
10__TTCA                          → TTCA
11__BBP_Blood-Brain_Barrier_Peptides → BBP
12__APP__Anti-parasitic           → Anti_parasitic
13_NeuroPred                      → NeuroPred
14__antibacterial_AB              → Antibacterial
15__Antifungal_AF                 → Antifungal
16__AV_Antiviral                  → Antiviral
17__Toxicity_2021_Dataset         → Toxicity
```

## Training Configuration

### Aggressive Configuration (Recommended)

```python
Learning Rate: 1e-4
Batch Size: 16
Epochs: 50
Dropout: 0.3
Weight Decay: 1e-5
Warmup Epochs: 5
Gradient Clipping: 1.0
Label Smoothing: 0.1
Mixed Precision: Enabled
TIM Loss: Enabled
ESM Ratio: 0.9
Transformer Layers: 4
Max Sequence Length: 128
```

### Training Process

1. **Task Sampling**: Each batch samples from a random task (MultiTaskDataLoader)
2. **Forward Pass**: Shared encoder → task-specific head
3. **Loss Computation**: CrossEntropy with label smoothing + optional TIM loss weighting
4. **Optimization**: AdamW with cosine annealing after warmup
5. **Validation**: Evaluate all 19 tasks after each epoch
6. **Checkpointing**: Save best model based on average F1-score

## Model Loading for Inference

```python
from mtl_peptide_classifier import MTLPeptideClassifier, get_all_peptide_tasks
from transformers import EsmTokenizer
import torch

# Load tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Get task configs
task_configs = get_all_peptide_tasks("datasets")

# Create model
model = MTLPeptideClassifier(
    task_configs=task_configs,
    hidden_dim=1280,
    esm_ratio=0.9,
    num_transformer_layers=4,
    dropout=0.3
)

# Load checkpoint components
checkpoint_dir = "checkpoints/best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load shared backbone
backbone = torch.load(f"{checkpoint_dir}/shared_backbone.pt", map_location=device)
model.base_embed.load_state_dict(backbone['base_embed'])
model.transformer.load_state_dict(backbone['transformer'])
model.cnn.load_state_dict(backbone['cnn'])
model.layer_norm.load_state_dict(backbone['layer_norm'])

# Load task heads
heads = torch.load(f"{checkpoint_dir}/heads.pt", map_location=device)
for name, head in model.heads.items():
    if name in heads:
        head.load_state_dict(heads[name])

model = model.to(device)
model.eval()

# Inference
sequence = "MKWVTFISLLFLFSSAYSRGVFRR"
tokens = " ".join(list(sequence))
inputs = tokenizer(tokens, return_tensors='pt', max_length=128, padding='max_length', truncation=True)

with torch.no_grad():
    logits = model(
        inputs['input_ids'].to(device),
        inputs['attention_mask'].to(device),
        task_name="Antimicrobial"
    )
    probs = torch.softmax(logits, dim=-1)
```

## Dependencies

```
torch>=2.0.0
transformers>=4.30.0
esm
numpy
pandas
scikit-learn
tqdm
```

## Key Implementation Details

### Sequence Tokenization
Sequences are tokenized as space-separated amino acids for ESM-2 tokenizer:
```python
tokens = " ".join(list(sequence))  # "ACDEF" → "A C D E F"
```

### Masked Pooling
Sequence heads use masked average pooling to handle variable-length sequences:
```python
mask_expanded = attention_mask.unsqueeze(-1).float()
x_masked = x * mask_expanded
x_pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
```

### Auto-Detection of Tasks
The `get_all_peptide_tasks()` function auto-detects available datasets from the `datasets/` directory by scanning for `*_train.csv` files and mapping them to task names.

## Performance Considerations

- **GPU Memory**: Model requires ~8GB GPU memory with batch size 16
- **Training Time**: ~6-8 hours on RTX 4070 Ti for 50 epochs
- **ESM-2**: Frozen backbone reduces trainable parameters significantly
- **Mixed Precision**: Enabled by default for faster training

## Comparison with Other MTL Variants

| Model | Avg ACC | Avg AUC | Avg MCC |
|-------|---------|---------|---------|
| Original_MTL_19tasks | 87.79% | 93.80% | 74.43% |
| Original_MTL_19tasks_finetuned | 88.21% | 93.64% | 75.19% |
| Safe_MTL_19tasks | 88.20% | 93.36% | 74.89% |
| **Original_MTL_19tasks_aggressive** | **89.49%** | **94.15%** | **78.88%** |
| GradNorm_MTL | 87.45% | 92.94% | 73.66% |
