"""
Script to upload the MTL Peptide Classifier to HuggingFace Hub.
"""

import os
import shutil
from huggingface_hub import HfApi, login
from pathlib import Path

# Configuration
MODEL_NAME = "tuankg1028/MTL-Peptide-Classifier"
CHECKPOINT_DIR = "checkpoints/best_model"
REPO_ID = MODEL_NAME

def create_model_card():
    """Create a README.md for the model."""
    return """---
license: mit
base_model: facebook/esm2_t33_650M_UR50D
tags:
- biology
- peptide
- multi-task-learning
- protein
- classification
---

# MTL Peptide Classifier (19 Tasks)

## Model Description

Multi-Task Learning (MTL) peptide classifier for 19 UniDL4BioPep peptide activity datasets.
Uses PDeepPP-inspired architecture with frozen ESM-2 backbone.

### Performance
- Average Accuracy: 89.49%
- Average AUC: 94.15%
- Average PR-AUC: 92.85%
- Average MCC: 78.88%

### Architecture
- **Shared Encoder**: Frozen ESM-2 (650M params) + learnable base embedding
- **Feature Extraction**: 4-layer Transformer + CNN (parallel)
- **Task Heads**: 19 binary classifiers (one per peptide activity)

## The 19 Peptide Activity Tasks

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

```python
from huggingface_hub import hf_hub_download
import torch

# Download model files
checkpoint_dir = "MTL-Peptide-Classifier-19tasks"
os.makedirs(checkpoint_dir, exist_ok=True)

for file in ["checkpoint.pt", "heads.pt", "shared_backbone.pt"]:
    hf_hub_download(
        repo_id="tuankg1028/MTL-Peptide-Classifier",
        filename=file,
        local_dir=checkpoint_dir
    )

# Load model (requires mtl_peptide_classifier.py)
from mtl_peptide_classifier import MTLPeptideClassifier, get_all_peptide_tasks
from transformers import EsmTokenizer

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
task_configs = get_all_peptide_tasks("datasets")

model = MTLPeptideClassifier(
    task_configs=task_configs,
    hidden_dim=1280,
    esm_ratio=0.9,
    num_transformer_layers=4,
    dropout=0.3
)

# Load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = torch.load(f"{checkpoint_dir}/shared_backbone.pt", map_location=device)
heads = torch.load(f"{checkpoint_dir}/heads.pt", map_location=device)

# Load weights
model.base_embed.load_state_dict(backbone['base_embed'])
model.transformer.load_state_dict(backbone['transformer'])
model.cnn.load_state_dict(backbone['cnn'])
model.layer_norm.load_state_dict(backbone['layer_norm'])

for name, head in model.heads.items():
    if name in heads:
        head.load_state_dict(heads[name])

model = model.to(device)
model.eval()
```

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
esm
numpy
pandas
scikit-learn
```

## Training Details

- Base Model: facebook/esm2_t33_650M_UR50D (frozen)
- Training Datasets: UniDL4BioPep benchmark
- Batch Size: 16
- Learning Rate: 1e-4
- Epochs: 50
- Dropout: 0.3
- Mixed Precision: Enabled
"""

def upload_to_huggingface():
    """Upload model to HuggingFace Hub."""

    api = HfApi()

    # Check if logged in
    try:
        user_info = api.whoami()
        print(f"Logged in as: {user_info}")
    except Exception as e:
        print("Not logged in. Please run: huggingface-cli login")
        return

    # Create repository
    try:
        repo_url = api.create_repo(
            repo_id=REPO_ID.split('/')[-1],
            repo_type="model",
            private=False,
            exist_ok=True
        )
        print(f"Repository created/exists: {repo_url}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Create temporary directory for upload
    upload_dir = Path("hf_upload_temp")
    upload_dir.mkdir(exist_ok=True)

    # Copy checkpoint files
    print("Copying model files...")
    checkpoint_files = ["checkpoint.pt", "heads.pt", "shared_backbone.pt"]

    for file in checkpoint_files:
        src = Path(CHECKPOINT_DIR) / file
        if src.exists():
            shutil.copy(src, upload_dir / file)
            print(f"  Copied {file}")
        else:
            print(f"  Warning: {file} not found")

    # Create README
    readme_path = upload_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(create_model_card())
    print("Created README.md")

    # Copy model architecture file
    if Path("mtl_peptide_classifier.py").exists():
        shutil.copy("mtl_peptide_classifier.py", upload_dir / "mtl_peptide_classifier.py")
        print("Copied mtl_peptide_classifier.py")

    # Upload files
    print("\nUploading files to HuggingFace Hub...")
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(upload_dir),
        repo_type="model"
    )

    print(f"\n✅ Model uploaded successfully!")
    print(f"🔗 View at: https://huggingface.co/{REPO_ID}")

    # Cleanup
    shutil.rmtree(upload_dir)
    print("Cleaned up temporary files")

if __name__ == "__main__":
    upload_to_huggingface()
