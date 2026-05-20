import os
import zipfile
import urllib.request
import shutil
import random
import csv
from pathlib import Path

ZIP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
DEST = Path("datasets/SMAP_MSL")
TMP_DIR = Path("tmp_smap")

def download_file(url, out_path):
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)

def main():
    if DEST.exists() and any(DEST.iterdir()):
        print(f"{DEST} already exists and is not empty, skipping download.")
        return
    # ensure clean env
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    zip_path = TMP_DIR / "data.zip"
    download_file(ZIP_URL, zip_path)
    # unzip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(TMP_DIR)
    # after extraction the zip contains a folder named 'data'
    extracted_data = TMP_DIR / "data"
    if not extracted_data.exists():
        raise RuntimeError("Expected 'data' folder not found after unzip")
    # move data folder to destination
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_data), str(DEST / "data"))
    # download labels csv
    download_file(LABELS_URL, DEST / "labeled_anomalies.csv")
    # generate simple train/valid/test splits (70/15/15) based on .npy files inside each split folder
    splits = {"train": 0.7, "valid": 0.15, "test": 0.15}
    for split_name, ratio in splits.items():
        split_dir = DEST / "data" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists (may already be empty)
        npy_files = list((split_dir).rglob("*.npy"))
        random.seed(42)
        random.shuffle(npy_files)
        csv_path = DEST / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for p in npy_files:
                writer.writerow([str(p), 0])  # placeholder label 0 (real labels are in labeled_anomalies.csv)
    # clean tmp
    shutil.rmtree(TMP_DIR)
    print(f"SMAP_MSL dataset prepared at {DEST}")

if __name__ == "__main__":
    main()
