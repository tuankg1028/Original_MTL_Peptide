# MTL Peptide Classifier

Multi-Task Learning (MTL) peptide classifier trained on UniDL4BioPep peptide activity datasets using a PDeepPP-inspired architecture with ESM-2 backbone.

## Architecture

```
Input: Peptide Sequence
         ↓
    ┌─────────────────────────────────────────────────┐
    │              Shared Encoder                      │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  ESM-2 (650M params)                        │ │
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
    │  │  Parallel Feature Extraction [ablatable]     │ │
    │  │  ├─ Transformer (4 layers, 8 heads)         │ │
    │  │  └─ CNN (kernel=7, padding=3)               │ │
    │  └─────────────────────────────────────────────┘ │
    │                      ↓                           │
    │         Concatenated Features [2560 dim]        │
    └─────────────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────────────┐
    │          Task-Specific Heads (20 tasks)          │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  SequenceHead: 2560 → 256 → 128 → 2        │ │
    │  │  - Masked average pooling                   │ │
    │  │  - 2 FC layers with ReLU + Dropout(0.3)     │ │
    │  └─────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────┘
                      ↓
                 Output: Binary logits
```

## Peptide Activity Tasks

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
20. Signal_peptide - Signal peptides

## Files

- **`mtl_peptide_classifier.py`** - Model architecture and data utilities (supports ablation flags)
- **`train_mtl.py`** - Training script with ablation study CLI
- **`evaluate_mtl_comprehensive.py`** - Comprehensive evaluation script
- **`ablation_report.py`** - Ablation study reporting and comparison
- **`download_datasets.py`** - Dataset download utilities
- **`process_signal_peptides.py`** - Signal peptide dataset preprocessing
- **`SignalPeptides_dattaset_balanced.xlsx`** - Balanced signal peptide dataset

## Training Configuration

### Default Parameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Epochs | 50 |
| Dropout | 0.3 |
| Weight Decay | 1e-5 |
| Warmup Epochs | 5 |
| Gradient Clipping | 1.0 |
| Label Smoothing | 0.1 |
| Mixed Precision | Enabled |
| TIM Loss | Enabled |
| ESM Ratio | 0.9 |
| Transformer Layers | 4 |

## Usage

### Training

```bash
# Default training
python train_mtl.py --batch_size 16 --lr 1e-4 --epochs 50 --dropout 0.3

# Without TIM loss
python train_mtl.py --no_tim

# Custom label smoothing
python train_mtl.py --label_smoothing 0.05
```

### Ablation Studies

The model supports fine-grained ablation via CLI flags. Each variant is saved to its own checkpoint directory named automatically from the active flags.

```bash
# Full model (baseline)
python train_mtl.py

# Without CNN branch
python train_mtl.py --no_cnn

# Without Transformer branch
python train_mtl.py --no_transformer

# Transformer only, 2 layers
python train_mtl.py --no_cnn --transformer_layers 2

# Unfreeze ESM-2 backbone (use lower lr)
python train_mtl.py --unfreeze_esm --lr 1e-5

# ESM ratio 0.5 (equal mix of ESM + base embedding)
python train_mtl.py --esm_ratio 0.5

# Without TIM loss
python train_mtl.py --no_tim

# Custom run name
python train_mtl.py --no_cnn --ablation_name my_experiment
```

### Evaluation

```bash
python evaluate_mtl_comprehensive.py \
    --model_dir "checkpoints/full_model/best_model" \
    --model_name "full_model" \
    --batch_size 8
```

### Ablation Report

```bash
python ablation_report.py --results_dir checkpoints/
```

### Inference

```python
from mtl_peptide_classifier import MTLPeptideClassifier, get_all_peptide_tasks
from transformers import EsmTokenizer
import torch

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
task_configs = get_all_peptide_tasks("datasets")

model = MTLPeptideClassifier(
    task_configs=task_configs,
    hidden_dim=1280,
    esm_ratio=0.9,
    num_transformer_layers=4,
    dropout=0.3,
    use_transformer=True,
    use_cnn=True,
    unfreeze_esm=False,
)

checkpoint_dir = "checkpoints/full_model/best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

backbone = torch.load(f"{checkpoint_dir}/shared_backbone.pt", map_location=device)
model.base_embed.load_state_dict(backbone['base_embed'])
if 'transformer' in backbone:
    model.transformer.load_state_dict(backbone['transformer'])
if 'cnn' in backbone:
    model.cnn.load_state_dict(backbone['cnn'])
    model.layer_norm.load_state_dict(backbone['layer_norm'])

heads = torch.load(f"{checkpoint_dir}/heads.pt", map_location=device)
for name, head in model.heads.items():
    if name in heads:
        head.load_state_dict(heads[name])

model = model.to(device).eval()

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

## Ablation Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--no_transformer` | Remove shared Transformer encoder | Off |
| `--no_cnn` | Remove shared CNN branch | Off |
| `--unfreeze_esm` | Allow ESM-2 gradients (fine-tuning) | Off |
| `--esm_ratio` | ESM-2 weight in embedding mix (0–1) | 0.9 |
| `--transformer_layers` | Number of shared Transformer layers | 4 |
| `--no_tim` | Disable TIM multi-task loss | Off |
| `--label_smoothing` | Label smoothing factor | 0.1 |
| `--ablation_name` | Custom checkpoint directory name | auto |

Each run saves an `ablation_config.json` alongside the checkpoint for full reproducibility.

## Key Features

- **Ablatable Architecture**: Transformer and CNN branches can be independently disabled via CLI
- **ESM-2 Backbone**: Frozen by default; can be unfrozen for fine-tuning
- **TIM Loss**: Threshold-Independent Multi-task loss with learnable per-task log variances
- **Masked Pooling**: Handles variable-length peptide sequences
- **Auto Variant Naming**: Checkpoint directories named automatically from active ablation flags
- **Windows Compatible**: DataLoader `num_workers` auto-set to 0 on Windows

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
esm
numpy
pandas
scikit-learn
tqdm
openpyxl
```

## References

- ESM-2: Lin et al. (2023) - Evolutionary Scale Prediction of Protein Function with Language Models
- PDeepPP: Original architecture inspiration
- TIM Loss: Kendall et al. (2018) - Multi-Task Learning Using Uncertainty to Weigh Losses
- UniDL4BioPep: Benchmark dataset for peptide activity prediction
