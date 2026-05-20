import subprocess
import shutil
from pathlib import Path

REPO_URL = "https://github.com/khundman/telemanom.git"
DEST = Path("datasets/SMAP_MSL")
TMP_DIR = Path("tmp_tele")
ZIP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"

def run(cmd: str):
    subprocess.check_call(cmd, shell=True)

def main():
    if DEST.exists():
        print(f"{DEST} already exists, skipping download.")
        return
    # create temp directory
    TMP_DIR.mkdir(exist_ok=True)
    # download zip
    zip_path = TMP_DIR / "data.zip"
    run(f"curl -L -o {zip_path} {ZIP_URL}")
    # unzip
    run(f"tar -xf {zip_path} -C {TMP_DIR}")
    # move extracted folder (it contains a 'data' folder)
    extracted = TMP_DIR / "data"
    if not extracted.exists():
        raise RuntimeError("Extracted data folder not found")
    shutil.move(str(extracted), str(DEST))
    # download labels csv
    run(f"curl -L -o {DEST}/labeled_anomalies.csv {LABELS_URL}")
    # generate simple train/valid/test CSV splits (70/15/15) using file list
    import random, csv
    splits = {"train": 0.7, "valid": 0.15, "test": 0.15}
    for split_name, ratio in splits.items():
        split_dir = DEST / "data" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        npy_files = list((split_dir).rglob("*.npy"))
        random.seed(42)
        random.shuffle(npy_files)
        csv_path = DEST / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for p in npy_files:
                # placeholder label 0
                writer.writerow([str(p), 0])
    # cleanup temp
    shutil.rmtree(TMP_DIR)
    print(f"SMAP_MSL dataset prepared at {DEST}")

if __name__ == "__main__":
    main()
