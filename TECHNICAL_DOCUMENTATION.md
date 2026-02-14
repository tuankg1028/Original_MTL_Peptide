# Technical Documentation: Multi-Task Learning Peptide Classifier

## 1. Project Overview

### 1.1 Research Objective
This project implements a deep learning-based multi-task learning (MTL) system for simultaneous classification of 20 distinct peptide bioactivity types. The system leverages pre-trained protein language models with task-specific heads to achieve state-of-the-art performance across diverse peptide classification tasks from the UniDL4BioPep benchmark.

### 1.2 Core Innovation
The architecture combines a frozen ESM-2 backbone (650M parameters) with learnable task-specific heads, enabling efficient knowledge transfer across 20 different peptide bioactivity prediction tasks while preventing catastrophic forgetting of pre-trained protein representations.

---

## 2. System Architecture

### 2.1 High-Level Design
```
Input Layer: Peptide Amino Acid Sequence
    ↓
Shared Encoding Layer (Frozen)
    ├── ESM-2 Pre-trained Encoder (facebook/esm2_t33_650M_UR50D)
    ├── Learnable Base Embedding (33 amino acids → 1280 dimensions)
    └── Weighted Feature Combination (90% ESM-2, 10% Learnable)
    ↓
Parallel Feature Extraction
    ├── Global Context Branch: 4-layer Transformer (8 attention heads)
    └── Local Pattern Branch: 1D Convolutional Network (kernel size 7)
    ↓
Feature Fusion: Concatenation (2560-dimensional vector)
    ↓
Task-Specific Classification Heads (20 binary classifiers)
    ├── Masked Average Pooling (variable-length handling)
    ├── Fully Connected Layer 1: 2560 → 256 (ReLU + Dropout 0.3)
    ├── Fully Connected Layer 2: 256 → 128 (ReLU + Dropout 0.3)
    └── Output Layer: 128 → 2 (Binary logits)
```

### 2.2 Key Architectural Components

#### 2.2.1 Frozen Backbone Strategy
- **ESM-2 Encoder**: Utilizes meta-MSA trained transformer from Meta AI
- **Parameter Freeze**: All ESM-2 parameters remain frozen during training
- **Rationale**: Preserves pre-trained protein language understanding while allowing task-specific adaptation through heads
- **Efficiency**: Reduces trainable parameters from 650M to approximately 12M

#### 2.2.2 Dual-Branch Feature Extraction
- **Transformer Branch**: Captures long-range dependencies and global sequence context
- **CNN Branch**: Extracts local motifs and short-range patterns
- **Fusion**: Concatenation enables complementary feature representation

#### 2.2.3 Task-Specific Heads
- **Independent Classifiers**: Each of 20 tasks has dedicated head architecture
- **Masked Pooling**: Handles variable-length sequences via attention-weighted averaging
- **Regularization**: Dropout (0.3) mitigates overfitting

---

## 3. Dataset Configuration

### 3.1 Task Composition
The system classifies 20 peptide bioactivity types:

| Task ID | Bioactivity Type | Dataset Source |
|---------|-----------------|----------------|
| 1 | ACE Inhibitory | UniDL4BioPpeptide ACE inhibitory activity |
| 2 | DPPIV Inhibitory | UniDL4BioPpeptide DPPIV inhibitory activity |
| 3 | Bitter Taste | UniDL4BioPpeptide Bitter peptides |
| 4 | Umami Taste | UniDL4BioPpeptide Umami peptides |
| 5 | Antimicrobial | UniDL4BioPpeptide Antimicrobial activity |
| 6 | Antimalarial (Main) | UniDL4BioPpeptide Antimalarial-main |
| 7 | Antimalarial (Alt) | UniDL4BioPpeptide Antimalarial-alternative |
| 8 | Quorum Sensing | UniDL4BioPpeptide Quorum sensing activity |
| 9 | Anticancer (Main) | UniDL4BioPpeptide ACP Anticancer-main |
| 10 | Anticancer (Alt) | UniDL4BioPpeptide ACP Anticancer-alternative |
| 11 | Anti-MRSA | UniDL4BioPpeptide Anti-MRSA strains |
| 12 | TTCA | UniDL4BioPpeptide Therapeutic peptides for cancer |
| 13 | BBP | UniDL4BioPpeptide Blood-Brain Barrier peptides |
| 14 | Anti-Parasitic | UniDL4BioPpeptide Anti-parasitic peptides |
| 15 | Neuroprotective | UniDL4BioPpeptide NeuroPred |
| 16 | Antibacterial | UniDL4BioPpeptide Antibacterial AB |
| 17 | Antifungal | UniDL4BioPpeptide Antifungal AF |
| 18 | Antiviral | UniDL4BioPpeptide Antiviral AV |
| 19 | Toxicity | UniDL4BioPpeptide Toxicity 2021 |
| 20 | Anti-inflammatory | Custom processed dataset (14,400 train / 3,600 test) |

### 3.2 Data Format Specification
- **File Format**: CSV files with train/test splits
- **Required Columns**:
  - `sequence`: Amino acid sequence (single-letter code)
  - `label`: Binary classification label (0 or 1)
- **Naming Convention**: `{task_id}__{task_name}_peptides_{train,test}.csv`
- **Split Ratio**: 80% training, 20% testing (stratified by label)

### 3.3 Data Preprocessing Pipeline
1. **Sequence Tokenization**: Convert amino acid sequences to space-separated format for ESM-2 tokenizer
2. **Padding/Truncation**: Standardize to maximum length of 128 amino acids
3. **Attention Mask Generation**: Create binary masks for valid vs padded positions
4. **Multi-Task Sampling**: Random task selection per batch with equal probability

---

## 4. Training Methodology

### 4.1 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 1e-4 | Optimal for fine-tuning large pre-trained models |
| Batch Size | 16 | Balances memory constraints and gradient stability |
| Epochs | 50 | Sufficient for convergence without overfitting |
| Dropout Rate | 0.3 | Strong regularization for diverse tasks |
| Weight Decay | 1e-5 | L2 regularization to prevent overfitting |
| Gradient Clipping | 1.0 | Stabilizes training, prevents gradient explosion |
| Label Smoothing | 0.1 | Reduces overconfidence, improves calibration |
| Warmup Epochs | 5 | Gradual learning rate ramp-up for stability |
| Mixed Precision | Enabled | Reduces memory usage, accelerates training |
| ESM Ratio | 0.9 | Balances pre-trained features vs task adaptation |

### 4.2 Loss Function Architecture

#### 4.2.1 Base Loss
- **Cross-Entropy Loss**: Standard binary classification loss with label smoothing
- **Formula**: L_CE = -Σ y*log(p) + (1-y)*log(1-p)

#### 4.2.2 Threshold-Independent Multi-task (TIM) Loss
- **Purpose**: Dynamically balances loss scales across 20 tasks with different difficulties
- **Mechanism**: Learnable task-specific log variance parameters
- **Formula**: L_TIM = Σ (1/(2σ²)) * L_task + log(σ)
- **Benefit**: Prevents dominant tasks from overwhelming learning

### 4.3 Optimization Strategy
- **Optimizer**: AdamW (Adam with decoupled weight decay)
- **Learning Rate Schedule**: Cosine annealing after warmup period
- **Gradient Accumulation**: Effectively doubles batch size
- **Early Stopping**: Monitors average F1-score across all tasks

### 4.4 Training Process
1. **Task Sampling**: Randomly select one of 20 tasks for each batch
2. **Batch Assembly**: Sample from selected task's training data
3. **Forward Pass**: Encode sequence → extract features → task head prediction
4. **Loss Computation**: Compute task-specific loss with TIM weighting
5. **Backward Pass**: Compute gradients for task head and shared components
6. **Parameter Update**: Apply AdamW optimizer with gradient clipping
7. **Validation**: Evaluate all 20 tasks after each epoch
8. **Checkpointing**: Save best model based on average validation F1-score

---

## 5. Performance Results

### 5.1 Overall Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Average Accuracy | 88.89% | Correct classification rate across all tasks |
| Average AUC-ROC | 94.10% | Area under ROC curve (excellent discrimination) |
| Average PR-AUC | 93.32% | Area under Precision-Recall curve (robust to imbalance) |
| Average MCC | 78.88% | Matthews Correlation Coefficient (balanced measure) |
| Best Task F1-Score | 99.41% (AntiMRSA) | Near-perfect classification on Anti-MRSA task |

### 5.2 Top Performing Tasks

| Rank | Task | Accuracy | AUC | PR-AUC | MCC |
|------|------|----------|-----|--------|-----|
| 1 | AntiMRSA | 98.99% | 99.86% | 99.86% | 96.07% |
| 2 | Anti-inflammatory | 98.81% | 99.57% | 99.29% | 97.61% |
| 3 | Antimalarial_alt | 98.77% | 99.95% | 99.93% | 95.66% |
| 4 | Antimicrobial | 96.74% | 98.47% | 99.31% | 92.21% |
| 5 | Antimalarial | 97.36% | 93.79% | 97.94% | 75.23% |

### 5.3 Performance Distribution Analysis

**High Performance Tasks (>95% accuracy)**: 7/20 tasks
- AntiMRSA, Anti-inflammatory, Antimalarial_alt, Antimicrobial, Antifungal, Antibacterial, Quorum_sensing

**Medium Performance Tasks (85-95% accuracy)**: 9/20 tasks
- NeuroPred, Toxicity, Anticancer_alt, Bitter, Umami, DPPIV_inhibitory, Antiviral, ACE_inhibitory, Anti_parasitic

**Challenging Tasks (<85% accuracy)**: 4/20 tasks
- BBP (76.32%), TTCA (72.08%), Anti_parasitic (65.22%), Anticancer (73.84%)

### 5.4 Comparative Analysis

| Model Variant | Avg Accuracy | Avg AUC | Avg MCC |
|--------------|--------------|---------|---------|
| Original MTL (19 tasks) | 87.79% | 93.80% | 74.43% |
| Safe MTL | 88.20% | 93.36% | 74.89% |
| GradNorm MTL | 87.45% | 92.94% | 73.66% |
| **Aggressive MTL (20 tasks)** | **88.89%** | **94.10%** | **78.88%** |

**Key Finding**: The aggressive training configuration with 20 tasks achieves superior performance across all metrics compared to previous variants.

---

## 6. Technical Implementation Details

### 6.1 Sequence Encoding
- **Tokenization Scheme**: Space-separated amino acids (e.g., "ACDEFGH" → "A C D E F G H")
- **Maximum Length**: 128 amino acids (covers 99th percentile of dataset)
- **Vocabulary**: Standard 20 amino acids + special tokens (padding, unknown, mask)

### 6.2 Variable-Length Handling
- **Attention Mechanism**: Binary masks distinguish valid tokens from padding
- **Masked Pooling**: Average pooling over only valid positions
- **Formula**: Pooled = Σ (feature × mask) / Σ mask

### 6.3 Memory Optimization
- **Mixed Precision Training**: FP16 for forward pass, FP32 for gradients
- **Gradient Accumulation**: Effective batch size 32 with memory for 16
- **Frozen Backbone**: Zero gradients for ESM-2 parameters
- **Checkpointing**: Save only task-specific heads and shared components

### 6.4 Computational Requirements
- **GPU Memory**: Minimum 8GB (tested on RTX 4070 Ti 16GB)
- **Training Time**: ~9 hours for 50 epochs (20 tasks, 140K+ total sequences)
- **Inference Speed**: ~25 sequences/second (batch size 16)

---

## 7. Model Deployment

### 7.1 Checkpoint Structure
```
checkpoints/best_model/
├── checkpoint.pt              # Full model state
├── heads.pt                   # Task-specific head weights
├── shared_backbone.pt         # Shared encoder weights
├── training_history.json      # Training metrics per epoch
└── results.json              # Final test set performance
```

### 7.2 Inference Pipeline
1. **Load Pre-trained Components**: ESM-2 tokenizer and model checkpoint
2. **Initialize Model**: Reconstruct architecture with trained weights
3. **Tokenize Input**: Convert peptide sequence to model input format
4. **Task Selection**: Specify which bioactivity task to predict
5. **Forward Pass**: Generate logits through encoder → task head
6. **Probability Extraction**: Apply softmax to get class probabilities
7. **Decision**: Binary classification based on probability threshold (default 0.5)

---

## 8. Key Contributions

### 8.1 Scientific Contributions
1. **First MTL System for 20 Peptide Tasks**: Comprehensive bioactivity prediction across diverse functional categories
2. **Frozen Backbone Strategy**: Demonstrates effectiveness of preserving pre-trained protein representations
3. **TIM Loss Adaptation**: Successfully applied threshold-independent multi-task loss to peptide classification
4. **Anti-inflammatory Dataset Integration**: Extended benchmark with novel therapeutic peptide dataset

### 8.2 Engineering Contributions
1. **Modular Architecture**: Easy addition of new tasks without retraining existing heads
2. **Efficient Training**: Reduced trainable parameters by 98% while maintaining performance
3. **Robust Evaluation**: Comprehensive metrics (ACC, AUC, PR-AUC, MCC) for thorough assessment
4. **Reproducibility**: Complete training history and checkpoint management

### 8.3 Performance Advantages
1. **State-of-the-Art Results**: Achieves 88.89% average accuracy across 20 diverse tasks
2. **Superior to Single-Task**: Multi-task learning improves generalization through shared representations
3. **Best Variant**: Aggressive configuration outperforms all previous MTL variants
4. **Scalability**: Architecture supports addition of new peptide bioactivity tasks

---

## 9. Future Directions

### 9.1 Potential Enhancements
1. **Task Expansion**: Add remaining UniDL4BioPep tasks (currently 22 available, using 20)
2. **Hierarchical Classification**: Leverage task relationships (e.g., antimicrobial → antibacterial/antifungal/antiviral)
3. **Attention Visualization**: Interpret model decisions via attention weight analysis
4. **Active Learning**: Iteratively improve by querying uncertain predictions

### 9.2 Research Opportunities
1. **Transfer Learning**: Study cross-species peptide bioactivity prediction
2. **Few-Shot Learning**: Enable prediction for novel peptide activities with minimal data
3. **Multi-Label Classification**: Handle peptides with multiple bioactivities
4. **Sequence Generation**: Generative modeling for novel peptide design

---

## 10. Conclusion

This project demonstrates the effectiveness of multi-task learning with frozen pre-trained backbones for peptide bioactivity prediction. The aggressive training configuration achieves state-of-the-art performance (88.89% accuracy) across 20 diverse peptide classification tasks, establishing a robust foundation for computational peptide research and therapeutic discovery.
