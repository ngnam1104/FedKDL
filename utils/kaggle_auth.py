"""
Xác thực Kaggle cho kagglehub (URPC, SMAP/MSL).

Token KHÔNG được commit. Dùng một trong các cách:
  export KAGGLE_API_TOKEN=...   # https://www.kaggle.com/settings → API
  hoặc file .env (đã gitignore) — xem .env.example
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> bool:
    """Nạp KEY=VALUE từ .env vào os.environ (không ghi đè biến đã set)."""
    p = Path(path)
    if not p.is_file():
        return False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
    return True


def kaggle_token_configured() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    access = Path.home() / ".kaggle" / "access_token"
    return legacy.is_file() or access.is_file()


def ensure_kaggle_credentials(repo_root: str | Path | None = None) -> bool:
    """
    Chuẩn bị credential trước khi gọi kagglehub.dataset_download.
    Returns True nếu có token, False nếu thiếu.
    """
    root = Path(repo_root or Path.cwd())
    load_dotenv(root / ".env")

    if kaggle_token_configured():
        masked = "(set)" if os.environ.get("KAGGLE_API_TOKEN") else "(legacy/json)"
        print(f"[Kaggle] Credentials OK {masked}")
        return True

    print(
        "[Kaggle] Thiếu token — URPC/SMAP qua Kaggle sẽ thất bại.\n"
        "  Cách 1: export KAGGLE_API_TOKEN=<token từ kaggle.com/settings>\n"
        "  Cách 2: cp .env.example .env  rồi điền token (file .env không được commit)\n"
        "  Lấy token: https://www.kaggle.com/settings → API → Generate New Token"
    )
    return False
