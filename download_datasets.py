"""
Download UniDL4BioPep datasets from GitHub and convert .xlsx → .csv
Output files are named to match get_all_peptide_tasks() expectations.

Usage:
    pip install requests openpyxl pandas
    python download_datasets.py
"""

import requests
import pandas as pd
from pathlib import Path
import sys

BASE_URL = "https://github.com/dzjxzyd/UniDL4BioPep/raw/main"
OUTPUT_DIR = Path(__file__).parent / "datasets"

# (github_folder, train_file, test_file, output_prefix)
TASKS = [
    ("1. ACE inhibitory activity",           "AHT_train.xlsx",                "AHT_test.xlsx",                "1__ACE_inhibitory_activity"),
    ("2. DPPIV inhibitory activity",          "DPPIV_train.xlsx",              "DPPIV_test.xlsx",              "2__DPPIV_inhibitory_activity"),
    ("3. Bitter",                             "bitter_train.xlsx",             "bitter_test.xlsx",             "3__Bitter"),
    ("4. Umami",                              "umami_train.xlsx",              "umami_test.xlsx",              "4__Umami"),
    ("5. Antimicrobial activity",             "AMP_train.xlsx",                "AMP_test.xlsx",                "5__Antimicrobial_activity"),
    ("6. Antimalarial activity-main",         "AMAP_train_main.xlsx",          "AMAP_test_main.xlsx",          "6__Antimalarial_activity-main"),
    ("6. Antimalarial activity-alternative",  "AMAP_train_alternative.xlsx",   "AMAP_test_alternative.xlsx",   "6__Antimalarial_activity-alternative"),
    ("7. Quorum sensing activity",            "QS_train.xlsx",                 "QS_test.xlsx",                 "7__Quorum_sensing_activity"),
    ("8. ACP Anticancer activity-main",       "main_ACP_train.xlsx",           "main_ACP_test.xlsx",           "8__ACP_Anticancer_activity-main"),
    ("8. ACP Anticancer activity-alternative","alternative_ACP_train.xlsx",    "alternative_ACP_test.xlsx",    "8__ACP_Anticancer_activity-alternative"),
    ("9. Anti-MRSA strains activity",         "SCMRSA_train.xlsx",             "SCMRSA_test.xlsx",             "9__Anti-MRSA_strains_activity"),
    ("10. TTCA",                              "TTCA_train.xlsx",               "TTCA_test.xlsx",               "10__TTCA"),
    ("11. BBP Blood-Brain Barrier Peptides",  "BBP_train.xlsx",                "BBP_test.xlsx",                "11__BBP_Blood-Brain_Barrier_Peptides"),
    ("12. APP  Anti-parasitic",               "APP_train.xlsx",                "APP_test.xlsx",                "12__APP__Anti-parasitic"),
    ("13.NeuroPred",                          "neuro_train.xlsx",              "neuro_test.xlsx",              "13_NeuroPred"),
    ("14. antibacterial AB",                  "AB_train.xlsx",                 "AB_test.xlsx",                 "14__antibacterial_AB"),
    ("15. Antifungal AF",                     "AF_train.xlsx",                 "AF_test.xlsx",                 "15__Antifungal_AF"),
    ("16. AV Antiviral",                      "AV_train.xlsx",                 "AV_test.xlsx",                 "16__AV_Antiviral"),
    ("17. Toxicity 2021 Dataset",             "train_3284.xlsx",               "test_580.xlsx",                "17__Toxicity_2021_Dataset"),
]

# Possible column names in source files → standardized name
SEQ_ALIASES  = {"sequence", "Sequence", "seq", "Seq", "peptide", "Peptide"}
LABEL_ALIASES = {"label", "Label", "activity", "Activity", "class", "Class", "y"}


def standardize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rename columns to 'sequence' and 'label', drop everything else."""
    cols = set(df.columns)

    seq_col = next((c for c in df.columns if c in SEQ_ALIASES), None)
    label_col = next((c for c in df.columns if c in LABEL_ALIASES), None)

    if seq_col is None or label_col is None:
        print(f"    [!] Could not auto-detect columns in {source}")
        print(f"        Available columns: {list(df.columns)}")
        # Fall back: assume first col = sequence, second col = label
        seq_col, label_col = df.columns[0], df.columns[1]
        print(f"        Falling back to: seq='{seq_col}', label='{label_col}'")

    out = df[[seq_col, label_col]].copy()
    out.columns = ["sequence", "label"]
    out = out.dropna(subset=["sequence", "label"])
    out["sequence"] = out["sequence"].astype(str)
    out = out[out["sequence"] != "nan"]
    out["label"] = out["label"].astype(int)
    return out


def download_xlsx(folder: str, filename: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{folder}/{filename}".replace(" ", "%20")
    print(f"  Downloading {folder}/{filename} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        # Write to temp bytes buffer and read with pandas
        import io
        df = pd.read_excel(io.BytesIO(r.content))
        print(f"OK ({len(df)} rows, cols: {list(df.columns)})")
        return df
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    success, failed = 0, []

    for folder, train_file, test_file, prefix in TASKS:
        print(f"[{prefix}]")

        for split, filename in [("train", train_file), ("test", test_file)]:
            df = download_xlsx(folder, filename)
            if df is None:
                failed.append(f"{prefix}_{split}")
                continue

            df = standardize(df, filename)
            out_path = OUTPUT_DIR / f"{prefix}_{split}.csv"
            df.to_csv(out_path, index=False)
            print(f"    Saved -> {out_path.name} ({len(df)} rows)")
            success += 1

        print()

    print("=" * 60)
    print(f"Done: {success} files saved to {OUTPUT_DIR}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    else:
        print("All downloads successful!")


if __name__ == "__main__":
    main()
