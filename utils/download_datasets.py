import os, sys, shutil, zipfile, argparse, subprocess, urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.kaggle_auth import ensure_kaggle_credentials

DATASETS_DIR = REPO_ROOT / "datasets"

def download_file(url, out_path):
    print("  Downloading:", url)
    urllib.request.urlretrieve(url, out_path)

def run_cmd(cmd):
    subprocess.check_call(cmd, shell=True)

def kaggle_copy(kaggle_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for item in os.listdir(kaggle_path):
        s = os.path.join(kaggle_path, item)
        d = dest / item
        if os.path.isdir(s):
            shutil.copytree(s, str(d), dirs_exist_ok=True)
        else:
            shutil.copy2(s, str(d))
    print("  Copied to", dest)

def download_smd():
    dest = DATASETS_DIR / "SMD"
    if dest.exists() and any(dest.iterdir()):
        print("[SMD] Already exists, skipping.")
        return
    print("[SMD] Cloning OmniAnomaly repo (ServerMachineDataset)...")
    tmp = Path("tmp_smd_clone")
    try:
        run_cmd("git clone --depth 1 https://github.com/NetManAIOps/OmniAnomaly.git " + str(tmp))
        src = tmp / "ServerMachineDataset"
        if not src.exists():
            raise RuntimeError("ServerMachineDataset not found in clone.")
        shutil.move(str(src), str(dest))
        print("[SMD] Done:", dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

def download_smap_msl():
    dest = DATASETS_DIR / "SMAP_MSL"
    if dest.exists() and any(dest.iterdir()):
        print("[SMAP/MSL] Already exists, skipping.")
        return
    # Method 1: kagglehub
    try:
        import kagglehub
        if not ensure_kaggle_credentials(REPO_ROOT):
            raise RuntimeError("Kaggle credentials missing")
        print("[SMAP/MSL] Trying Kaggle (kagglehub)...")
        try:
            path = kagglehub.dataset_download("patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
        except Exception:
            print("[SMAP/MSL] Primary slug failed, trying alternate...")
            path = kagglehub.dataset_download("drscarlat/smap-and-msl-anomaly-detection")
        kaggle_copy(path, dest)
        labels_url = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
        download_file(labels_url, dest / "labeled_anomalies.csv")
        print("[SMAP/MSL] Done:", dest)
        return
    except ImportError:
        print("[SMAP/MSL] kagglehub not installed, falling back to S3...")
    except Exception as e:
        print("[SMAP/MSL] Kaggle failed:", e, "- falling back to S3...")
    # Method 2: Direct S3
    tmp = Path("tmp_smap_msl")
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        zip_path = tmp / "data.zip"
        download_file("https://s3-us-west-2.amazonaws.com/telemanom/data.zip", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        extracted = tmp / "data"
        if not extracted.exists():
            raise RuntimeError("data folder not found after unzip.")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(dest / "data"))
        labels_url = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
        download_file(labels_url, dest / "labeled_anomalies.csv")
        print("[SMAP/MSL] Done:", dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

def download_urpc():
    dest = DATASETS_DIR / "URPC2020"
    if dest.exists() and any(dest.iterdir()):
        print("[URPC2020] Already exists, skipping.")
        return
    if not ensure_kaggle_credentials(REPO_ROOT):
        print("[URPC2020] ERROR: Cần KAGGLE_API_TOKEN (export hoặc .env).")
        print("           Manual: https://www.kaggle.com/datasets/lywang777/urpc2020")
        return
    try:
        import kagglehub
        print("[URPC2020] Downloading via kagglehub...")
        try:
            path = kagglehub.dataset_download("lywang777/urpc2020")
        except Exception:
            print("[URPC2020] Primary slug failed, trying alternate...")
            path = kagglehub.dataset_download("slmhvn/urpc-2020")
        kaggle_copy(path, dest)
        print("[URPC2020] Done:", dest)
    except ImportError:
        print("[URPC2020] ERROR: kagglehub required.")
        print("           Install: pip install kagglehub")
        print("           Manual:  https://www.kaggle.com/datasets/lywang777/urpc2020")
    except Exception as e:
        print("[URPC2020] Failed:", e)

def main():
    parser = argparse.ArgumentParser(description="Download FedKDL datasets.")
    parser.add_argument("--smd",      action="store_true", help="SMD only")
    parser.add_argument("--smap-msl", action="store_true", help="SMAP and MSL only")
    parser.add_argument("--urpc",     action="store_true", help="URPC2020 only")
    parser.add_argument("--all",      action="store_true", help="All datasets")
    args = parser.parse_args()
    run_all = args.all or not (args.smd or args.smap_msl or args.urpc)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if run_all or args.smd:
        print("\n" + "=" * 50)
        download_smd()
    if run_all or args.smap_msl:
        print("\n" + "=" * 50)
        download_smap_msl()
    if run_all or args.urpc:
        print("\n" + "=" * 50)
        download_urpc()
    print("\nAll done!")

if __name__ == "__main__":
    main()
