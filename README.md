# Original_MTL_19tasks_aggressive

Multi-Task Learning (MTL) peptide classifier trained on 19 UniDL4BioPep peptide activity datasets using the PDeepPP architecture with frozen ESM-2 backbone.

## Model Performance

| Metric | Score |
|--------|-------|
| Average Accuracy | **89.49%** |
| Average AUC | **94.15%** |
| Average PR-AUC | **92.85%** |
| Average MCC | **78.88%** |

This is the **best performing model** among all MTL variants tested.

### Per-Dataset Performance

| Dataset | ACC | AUC | PR_AUC | MCC | F1 |
|---------|-----|-----|--------|-----|-----|
| ACE_inhibitory | 87.31% | 91.82% | 89.53% | 74.65% | 87.50% |
| AntiMRSA | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Anti_parasitic | 77.17% | 92.72% | 87.94% | 59.68% | 81.08% |
| Antibacterial | 95.49% | 96.70% | 93.69% | 91.10% | 95.60% |
| Anticancer | 72.97% | 82.20% | 80.17% | 46.80% | 70.10% |
| Anticancer_alt | 97.16% | 98.93% | 97.81% | 94.39% | 97.22% |
| Antifungal | 94.88% | 99.28% | 99.24% | 90.02% | 95.07% |
| Antimalarial | 98.02% | 98.00% | 99.85% | 82.12% | 98.95% |
| Antimalarial_alt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Antimicrobial | 98.07% | 98.56% | 99.04% | 95.28% | 98.65% |
| Antiviral | 85.47% | 90.04% | 86.70% | 71.86% | 86.54% |
| BBP | 84.21% | 93.35% | 92.68% | 72.11% | 86.36% |
| Bitter | 90.62% | 97.36% | 97.40% | 81.89% | 90.00% |
| DPPIV_inhibitory | 84.21% | 92.88% | 93.33% | 68.43% | 84.09% |
| NeuroPred | 90.52% | 95.15% | 94.08% | 81.03% | 90.50% |
| Quorum_sensing | 95.00% | 96.38% | 94.13% | 90.45% | 95.24% |
| TTCA | 64.97% | 73.31% | 64.48% | 33.94% | 62.70% |
| Toxicity | 94.31% | 97.47% | 96.91% | 88.66% | 94.60% |
| Umami | 89.89% | 94.70% | 97.11% | 76.33% | 93.02% |

## Architecture

```
Input: Peptide Sequence
         ↓
    ┌─────────────────────────────────────────────────┐
    │              Shared Encoder (Frozen)             │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  ESM-2 (650M params) - Frozen               │ │
    │  │  facebook/esm2_t33_650M_UR50D               │ │
    │  └─────────────────────────────────────────────┘ │
    │                      ↓                           │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  Base Embedding (33 aa → 1280 dim)          │ │
    │  └─────────────────────────────────────────────┘ │
    │                      ↓                           │
    │  Weighted Combination (ESM ratio: 0.9)          │
    │                      ↓                           │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  Parallel Feature Extraction                 │ │
    │  │  ├─ Transformer (4 layers, 8 heads)         │ │
    │  │  └─ CNN (kernel=7, padding=3)               │ │
    │  └─────────────────────────────────────────────┘ │
    │                      ↓                           │
    │         Concatenated Features [2560 dim]        │
    └─────────────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────────────┐
    │          Task-Specific Heads (19 tasks)          │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  SequenceHead: 2560 → 256 → 128 → 2        │ │
    │  │  - Masked average pooling                   │ │
    │  │  - 2 FC layers with ReLU + Dropout(0.3)     │ │
    │  └─────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────┘
                      ↓
                 Output: Binary logits
```

## Training Configuration

### Aggressive Training Parameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Epochs | 50 (aggressive) |
| Dropout | 0.3 |
| Weight Decay | 1e-5 |
| Warmup Epochs | 5 |
| Gradient Clipping | 1.0 |
| Label Smoothing | 0.1 |
| Mixed Precision | Enabled |
| TIM Loss | Enabled |

### Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 1280 |
| ESM Ratio | 0.9 |
| Transformer Layers | 4 |
| Transformer Heads | 8 |
| CNN Kernel Size | 7 |

## Files

- **`mtl_peptide_classifier.py`** - Model architecture and data utilities
- **`train_mtl.py`** - Training script with aggressive configuration
- **`evaluate_mtl_comprehensive.py`** - Comprehensive evaluation script

## 19 Peptide Activity Tasks

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

## Usage

### Training

```bash
python train_mtl.py --batch_size 16 --lr 1e-4 --epochs 50 --dropout 0.3
```

### Evaluation

```bash
python evaluate_mtl_comprehensive.py \
    --model_dir "path/to/checkpoint" \
    --model_name "Original_MTL_19tasks_aggressive" \
    --batch_size 8
```

### Inference

```python
from mtl_peptide_classifier import MTLPeptideClassifier, get_all_peptide_tasks
from transformers import EsmTokenizer

# Load model and tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
task_configs = get_all_peptide_tasks("data/unidl4biopep_processed")

model = MTLPeptideClassifier(
    task_configs=task_configs,
    hidden_dim=1280,
    esm_ratio=0.9,
    num_transformer_layers=4,
    dropout=0.3
)

# Load checkpoint weights
# ... (loading code)

# Predict
sequence = "MKWVTFISLLFLFSSAYSRGVFRR"
tokens = " ".join(list(sequence))
inputs = tokenizer(tokens, return_tensors='pt')

with torch.no_grad():
    logits = model(
        inputs['input_ids'],
        inputs['attention_mask'],
        task_name="Antimicrobial"
    )
    probs = torch.softmax(logits, dim=-1)
```

## Key Features

1. **Frozen ESM-2 Backbone**: Leverages pre-trained protein representations without catastrophic forgetting
2. **Learnable Base Embedding**: Allows task-specific adaptation of amino acid representations
3. **Parallel Feature Extraction**: Combines global context (transformer) and local features (CNN)
4. **TIM Loss**: Threshold-Independent Multi-task loss for balanced training across tasks
5. **Masked Pooling**: Handles variable-length sequences properly

## Comparison with Other Models

| Model | Avg ACC | Avg AUC | Avg MCC |
|-------|---------|---------|---------|
| Original_MTL_19tasks | 87.79% | 93.80% | 74.43% |
| Original_MTL_19tasks_finetuned | 88.21% | 93.64% | 75.19% |
| Safe_MTL_19tasks | 88.20% | 93.36% | 74.89% |
| **Original_MTL_19tasks_aggressive** | **89.49%** | **94.15%** | **78.88%** |
| GradNorm_MTL | 87.45% | 92.94% | 73.66% |

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
esm
numpy
pandas
scikit-learn
tqdm
```

## References

- ESM-2: Lin et al. (2023) - "Evolutionary Scale Prediction of Protein Function with Language Models"
- PDeepPP: Original architecture inspiration
- TIM Loss: Kendall et al. (2018) - "Multi-Task Learning Using Uncertainty to Weigh Losses"
- UniDL4BioPep: Benchmark dataset for peptide activity prediction
