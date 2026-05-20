import subprocess
import shutil
from pathlib import Path

REPO_URL = "https://github.com/NetManAIOps/OmniAnomaly.git"
DEST = Path("datasets/SMD")
TMP_DIR = Path("tmp_omni")

def run(cmd: str):
    subprocess.check_call(cmd, shell=True)

def main():
    if DEST.exists():
        print(f"{DEST} already exists, skipping download.")
        return
    # shallow clone
    run(f"git clone --depth 1 {REPO_URL} {TMP_DIR}")
    src = TMP_DIR / "ServerMachineDataset"
    if not src.exists():
        raise RuntimeError("ServerMachineDataset not found in cloned repo")
    shutil.move(str(src), str(DEST))
    shutil.rmtree(TMP_DIR)
    print(f"SMD dataset copied to {DEST}")

if __name__ == "__main__":
    main()
