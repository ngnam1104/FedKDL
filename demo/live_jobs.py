"""Bounded background jobs for optional live FedKDL demo rounds."""

from __future__ import annotations

import re
import subprocess
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_AUV_PATTERN = re.compile(r"\[AUV\s+(\d+)\]", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveRoundJobManager:
    """Run at most one real training process per configured worker."""

    def __init__(self, repo_root: Path, max_workers: int = 1):
        self.repo_root = Path(repo_root).resolve()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="fedkdl-live-round",
        )
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(self, command: list[str], baseline: str) -> dict[str, Any]:
        with self._lock:
            active = [
                job
                for job in self._jobs.values()
                if job["status"] in {"queued", "running", "cancelling"}
            ]
            if active:
                raise RuntimeError(
                    f"Live training job {active[0]['job_id']} is already active."
                )

            job_id = uuid.uuid4().hex[:12]
            job = {
                "job_id": job_id,
                "baseline": baseline,
                "status": "queued",
                "created_at": _utc_now(),
                "started_at": None,
                "finished_at": None,
                "return_code": None,
                "current_auv": None,
                "completed_auvs": [],
                "message": "Waiting for the bounded training worker.",
                "command": list(command),
                "logs": deque(maxlen=200),
                "process": None,
                "cancel_requested": False,
            }
            self._jobs[job_id] = job
            self._executor.submit(self._run, job_id)
            return self._public(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public(job) if job is not None else None

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] in {"completed", "failed", "cancelled"}:
                return self._public(job)
            job["cancel_requested"] = True
            job["status"] = "cancelling"
            job["message"] = "Cancellation requested."
            process = job.get("process")

        if process is not None and process.poll() is None:
            process.terminate()
        return self.get(job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job["cancel_requested"]:
                job["status"] = "cancelled"
                job["finished_at"] = _utc_now()
                return
            job["status"] = "running"
            job["started_at"] = _utc_now()
            job["message"] = "Real one-round training is running."
            command = list(job["command"])

        try:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._lock:
                job = self._jobs[job_id]
                job["process"] = process

            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                match = _AUV_PATTERN.search(line)
                with self._lock:
                    job = self._jobs[job_id]
                    job["logs"].append(line)
                    if match:
                        auv_id = int(match.group(1))
                        previous = job["current_auv"]
                        if (
                            previous is not None
                            and previous != auv_id
                            and previous not in job["completed_auvs"]
                        ):
                            job["completed_auvs"].append(previous)
                        job["current_auv"] = auv_id
                        job["message"] = f"AUV {auv_id} is training."
                    cancel_requested = job["cancel_requested"]
                if cancel_requested and process.poll() is None:
                    process.terminate()

            return_code = process.wait()
            with self._lock:
                job = self._jobs[job_id]
                current_auv = job["current_auv"]
                if (
                    current_auv is not None
                    and return_code == 0
                    and current_auv not in job["completed_auvs"]
                ):
                    job["completed_auvs"].append(current_auv)
                job["return_code"] = return_code
                job["finished_at"] = _utc_now()
                if job["cancel_requested"]:
                    job["status"] = "cancelled"
                    job["message"] = "Live training was cancelled."
                elif return_code == 0:
                    job["status"] = "completed"
                    job["message"] = "Real one-round training completed."
                else:
                    job["status"] = "failed"
                    job["message"] = f"Training process exited with code {return_code}."
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["message"] = str(exc)
                job["finished_at"] = _utc_now()
        finally:
            with self._lock:
                self._jobs[job_id]["process"] = None

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job["job_id"],
            "baseline": job["baseline"],
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "return_code": job["return_code"],
            "current_auv": job["current_auv"],
            "completed_auvs": list(job["completed_auvs"]),
            "message": job["message"],
            "logs": list(job["logs"]),
        }
