"""
Process Anti-inflammatory peptide dataset for MTL training.
Splits the dataset into train and test sets following UniDL4BioPep format.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Read the dataset
script_dir = Path(__file__).parent
input_file = script_dir / "Anti inflamatory peptide_dataset.csv"
datasets_dir = script_dir / "datasets"

# Create datasets directory if not exists
datasets_dir.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} sequences from {input_file.name}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['Label'].value_counts()}")

# Rename columns to match expected format (lowercase)
df = df.rename(columns={'Sequence': 'sequence', 'Label': 'label'})

# Split into train and test (80/20 split like other datasets)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Use prefix consistent with UniDL4BioPep naming
# Adding as 18__Anti_inflammatory_peptides (continuing from 17__Toxicity)
prefix = "18__Anti_inflammatory_peptides"

train_file = datasets_dir / f"{prefix}_train.csv"
test_file = datasets_dir / f"{prefix}_test.csv"

# Save with expected column names (lowercase)
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"\nCreated training set: {train_file.name} ({len(train_df)} sequences)")
print(f"Created test set: {test_file.name} ({len(test_df)} sequences)")
print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
print(f"\nTest label distribution:\n{test_df['label'].value_counts()}")
