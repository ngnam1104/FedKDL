#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

REPO_URL="${GIT_REPO_URL:-https://github.com/ngnam1104/FedKDL.git}"
WORKDIR="${WORKDIR:-/workspace/FedKDL}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv}"

echo "[entrypoint] Preparing SSH for Bitvise/SSH clients..."
apt-get update
apt-get install -y --no-install-recommends \
  git \
  openssh-server \
  python3-venv \
  python3-pip \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  tmux \
  htop \
  rsync

mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh

if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
  echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

if [[ -n "${ROOT_PASSWORD:-}" ]]; then
  echo "root:${ROOT_PASSWORD}" | chpasswd
  sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config
else
  sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
fi
sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config
/usr/sbin/sshd || true

echo "[entrypoint] Syncing repository..."
mkdir -p "$(dirname "$WORKDIR")"
if [[ -d "$WORKDIR/.git" ]]; then
  git -C "$WORKDIR" pull --ff-only || true
else
  git clone "$REPO_URL" "$WORKDIR"
fi

cd "$WORKDIR"

echo "[entrypoint] Creating Python environment..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo "[entrypoint] Installing Python requirements..."
pip install -r requirements.txt

chmod +x run_fedkdl.sh run_kd_logit.sh run_kd_logit_proj.sh 2>/dev/null || true
mkdir -p results results/train_logs results/logs

cat <<'MSG'
[entrypoint] Done.

Next steps inside SSH:
  cd /workspace/FedKDL
  source .venv/bin/activate
  python utils/download_datasets.py --urpc
  python utils/generate_all_envs.py --dataset URPC --n 30 --m-relays 8 --seeds 1109
  python scripts/fedkdl/train_student_warmup.py --mode warmup
  GPU=0 ./run_fedkdl.sh

Bitvise:
  Host: your Vast.ai public IP
  Port: the Vast.ai SSH mapped port, not necessarily 22
  User: root
  Auth: SSH_PUBLIC_KEY or ROOT_PASSWORD configured in Vast.ai env
MSG

tail -f /dev/null
