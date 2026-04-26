from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import psutil
from tqdm import tqdm



# ============================================================
# PROMPTS (now preserve your original architecture requirements)
# ============================================================

PHASE0_PROMPT = r"""
You are a senior distributed-systems architect, Kotlin backend engineer, and production readiness reviewer.

OBJECTIVE:
Design a production-grade, fault-tolerant, horizontally scalable event-driven system for a global financial trading platform.

THIS PHASE IS ARCHITECTURE ONLY.
DO NOT GENERATE CODE.
DO NOT GENERATE JSON.
DO NOT GENERATE FILES.

============================================================
ARCHITECTURE REQUIREMENTS (MUST BE FULLY ADDRESSED)
============================================================
1) Capacity targets:
   - Peak throughput: 10 million events/second
   - Latency: sub-50ms at p99.99
   - Multi-region active-active deployment (≥3 regions)
   - Exactly-once processing semantics end-to-end
2) Functional requirements:
   - Real-time risk checks
   - Order ingestion and basic matching logic
   - Regulatory-grade, immutable audit logging
   - Replayable historical analytics
3) Hard constraints:
   - Partial region outages must not cause data loss
   - Explicit handling of network partitions (state tradeoffs)
   - Backpressure and load shedding must be both explained AND implemented later in code
   - Cold-start behavior must be addressed (startup + replay)
4) Technology choices:
   - Name specific messaging, storage, coordination technologies
   - Provide tradeoffs and alternatives
   - Justify the final design
5) Failures:
   - At least 10 concrete failure scenarios
   - Detection, mitigation, recovery paths
6) Performance model:
   - Throughput, latency, memory cost modeling
7) Security:
   - Authentication, authorization
   - Encryption in transit + at rest
   - Key management and rotation
8) Observability:
   - Metrics, tracing, structured logs
   - Debugging tail latency

============================================================
OUTPUT REQUIREMENT
============================================================
Produce a clear, structured architecture document with headings.
Include: assumptions, dataflow, storage/messaging choices, tradeoffs, failure analysis,
performance model, security, observability.

STOP AFTER ARCHITECTURE IS COMPLETE.
"""

# Phase1: runnable scaffold + manifest, but must reflect Phase0 decisions.
PHASE1_PROMPT_TEMPLATE = r"""
You are a senior Kotlin backend engineer.

ARCHITECTURE CONTEXT (summary from Phase 0):
{ARCH_SUMMARY}

TASK:
Generate the RUNNABLE SCAFFOLD for a multi-module Kotlin + Gradle project that demonstrates this architecture.

HARD RULES:
- Output MUST be a SINGLE JSON object and NOTHING else.
- No Markdown. No backticks. No commentary.
- JSON schema:

{{
  "project": {{ "name": "<name>", "description": "<one paragraph>" }},
  "files": [ {{ "path": "...", "content": "..." }}, ... ]
}}

SCAFFOLD REQUIREMENTS (ONLY):
1) Multi-module Gradle Kotlin DSL project:
   - settings.gradle.kts
   - root build.gradle.kts
   - gradle.properties
   - modules: app, core, infra (each with build.gradle.kts)
2) Include Gradle Wrapper text files:
   - gradlew, gradlew.bat, gradle/wrapper/gradle-wrapper.properties
   - DO NOT include binary gradle-wrapper.jar
3) Java 17 toolchain must be configured.
4) Minimal runnable Ktor server in :app:
   - GET /health -> 200 OK JSON
   - GET /metrics -> Prometheus plaintext (placeholder is OK in Phase1)
   - POST /ingest, POST /orders, POST /replay routes must exist but may be stubbed (return 501) in Phase1
   - Must start with: ./gradlew :app:run
5) Minimal domain types in :core
6) Minimal infra stubs in :infra
7) README.md with exact commands:
   - ./gradlew test
   - ./gradlew :app:run
   - Include environment variables (PORT etc)
8) MANIFEST.md with:
   - Path, approx line count, purpose, key dependencies
   - A section named exactly: PENDING_FILES
   - Under PENDING_FILES list at least 20 additional file paths (one per line with "- ")
   - Those paths must cover: api, domain, service, infra, persistence, observability, util
   - Include integration test and load test script paths in PENDING_FILES

IMPORTANT:
- Do NOT generate full implementation yet.
- Keep output small enough to not truncate.

Now output the JSON object.
"""

# Phase2: generate specific files in batches, still aligned to architecture summary.
PHASE2_PROMPT_TEMPLATE = r"""
You are implementing a production-grade Kotlin prototype for a distributed event-driven trading system.

ARCHITECTURE CONTEXT (summary from Phase 0):
{ARCH_SUMMARY}

TASK:
Generate the following files EXACTLY as specified.

OUTPUT RULES:
- Output MUST be a SINGLE JSON object and NOTHING else.
- No Markdown. No backticks. No commentary.
- Output schema:

{{ "files": [ {{ "path": "...", "content": "..." }}, ... ] }}

- Output ONLY these exact file paths (no extras):
{FILE_LIST}

QUALITY RULES:
- Production-grade Kotlin; no TODO, no placeholders.
- Must compile under Java 17 toolchain configured by the scaffold.
- Must implement:
  - bounded backpressure queue + load shedding (429/503 behavior)
  - /metrics Prometheus-like output with at least one real counter and one histogram-like metric
  - POST /ingest accepts JSON and returns 2xx on success
  - audit log append-only with hash chaining
  - replay reads audit log and replays through pipeline
  - idempotency/dedup so retries do not duplicate processing
  - structured logging
  - at least 1 integration test (boots server and hits /health, /metrics, /ingest)
  - at least 3 unit tests in :core
  - a runnable load test script (k6 JS or Kotlin client)

Stop ONLY after ALL files listed are produced.
Now output the JSON object.
"""


# ============================================================
# Phase0: architecture generation + summary extraction
# ============================================================

def _summarize_for_injection(full_arch_text: str, max_chars: int = 1800) -> str:
    """
    Keep the injected summary short to avoid blowing context in Phase 1/2.
    We keep the first ~max_chars plus a tail marker.
    """
    t = (full_arch_text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "\n\n[...truncated summary for prompt injection...]"


def phase0_architecture(
    *,
    model: str,
    run_dir: Path,
    temperature: float,
    timeout: int,
    num_ctx: int,
    num_predict: int,
    max_hops: int,
) -> tuple[str, str]:
    """
    Runs Phase 0 and stores architecture to disk. Returns (full_text, summary_text).
    Uses explicit num_ctx to prevent prompt truncation. [1](https://github.com/D4RK-777/Qwen3-LLM-Coder)[2](https://docs.continue.dev/guides/ollama-guide)
    """
    resp = generate_with_autocontinue(
        model=model,
        prompt=PHASE0_PROMPT,
        temperature=temperature,
        timeout=timeout,
        num_ctx=num_ctx,
        num_predict=num_predict,
        force_json=False,
        max_hops=max_hops,
    )

    text = (resp.get("response") or "").strip()

    model_dir = run_dir / "models" / safe_name(model)
    model_dir.mkdir(parents=True, exist_ok=True)

    arch_path = model_dir / f"ARCHITECTURE_{run_dir.name}.md"
    arch_path.write_text(text, encoding="utf-8")

    summary = _summarize_for_injection(text)
    summary_path = model_dir / f"ARCH_SUMMARY_{run_dir.name}.txt"
    summary_path.write_text(summary, encoding="utf-8")

    return text, summary




# ============================================================
# Defaults / directories
# ============================================================

OUTPUT_DIR = Path("ollama_benchmarks")
RUNS_DIR = OUTPUT_DIR / "runs"
PROJECTS_DIR = OUTPUT_DIR / "projects"
OUTPUT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

DEFAULT_TIMEOUT_SEC = 1800
DEFAULT_COOLDOWN_SEC = 15
DEFAULT_TEMPERATURE = 0.2

# IMPORTANT: large prompts require explicit num_ctx
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT_PHASE1 = 2048
DEFAULT_NUM_PREDICT_PHASE2 = 4096
DEFAULT_PHASE2_BATCH = 8

# Auto-continue settings
DEFAULT_MAX_HOPS = 6
DEFAULT_MIN_BATCH = 2


# ============================================================
# Env helpers
# ============================================================

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v in (None, ""):
        return default
    try:
        return int(v)
    except ValueError:
        return default

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError:
        return default


# ============================================================
# Utilities
# ============================================================

import os
import subprocess
import time
from pathlib import Path
import requests
import json
import re
import time


_PROM_METRIC_LINE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*({.*})?\s+[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$")


def is_prometheus_like(text: str) -> tuple[bool, str]:
    """
    Minimal Prometheus exposition format sanity check:
    - Must be non-empty text
    - Must contain at least one metric sample line: name{labels} value
    - Optional comment lines starting with '#'
    """
    if not text or not text.strip():
        return False, "empty metrics body"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False, "no non-empty lines"

    metric_samples = 0
    for ln in lines:
        if ln.startswith("#"):
            continue
        if _PROM_METRIC_LINE.match(ln):
            metric_samples += 1
            if metric_samples >= 1:
                return True, "found metric sample line"

    return False, "no metric sample lines found"


def _http_get(url: str, timeout_sec: int = 3) -> tuple[bool, int | None, str]:
    try:
        r = requests.get(url, timeout=timeout_sec)
        return (200 <= r.status_code < 300), r.status_code, r.text or ""
    except Exception as e:
        logging.error("Error in _http_get for URL %s: %s", url, str(e))
        return False, None, str(e)


def _http_post_json(url: str, payload: dict, timeout_sec: int = 5) -> tuple[bool, int | None, str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout_sec)
        body = r.text or ""
        return (200 <= r.status_code < 300), r.status_code, body
    except Exception as e:
        logging.error("Error in _http_post_json for URL %s: %s", url, str(e))
        return False, None, str(e)


def _wait_for_healthy(health_url: str, total_wait_sec: int = 60, interval_sec: float = 1.0) -> tuple[bool, str]:
    deadline = time.time() + total_wait_sec
    last = ""
    while time.time() < deadline:
        ok, code, body = _http_get(health_url, timeout_sec=3)
        if ok:
            return True, f"OK {code}"
        last = f"HTTP {code}: {body[:200]}" if code is not None else body
        time.sleep(interval_sec)
    return False, last


def _gradlew_cmd(project_dir: Path) -> list[str]:
    """
    Returns an absolute path to the Gradle Wrapper.
    This avoids WinError 2 on Windows subprocess calls.
    """
    if os.name == "nt":
        return [str((project_dir / "gradlew.bat").resolve())]
    return ["./gradlew"]

def run_command(
    cmd: list[str],
    cwd: Path,
    timeout_sec: int,
    extra_env: dict | None = None,
) -> dict:
    """
    Run a command and capture stdout/stderr/exit code.
    Uses subprocess.run (recommended) to capture output. [1](https://markaicode.com/continue-dev-ollama-ai-code-completion-tutorial/)[2](https://inferencerig.com/fix/ollama-not-working-fix-errors-crashes-slow-performance-2026-guide/)
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            shell=(os.name == "nt")
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": p.returncode,
            "ok": p.returncode == 0,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
    except subprocess.TimeoutExpired as e:
        logging.error("Timeout in run_command for cmd %s: %s", " ".join(cmd), str(e))
        return {
            "cmd": " ".join(cmd),
            "returncode": None,
            "ok": False,
            "stdout": (e.stdout or ""),
            "stderr": f"TIMEOUT after {timeout_sec}s",
        }


def _terminate_process_tree(proc: subprocess.Popen, grace_sec: int = 10) -> dict:
    """
    Best-effort terminate for Gradle run tasks that don't exit on their own.
    - On Windows: send CTRL_BREAK_EVENT to the process group if possible, else terminate().
    - On POSIX: terminate(), then kill() if needed.
    """
    try:
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=1)
            return {"stopped": True, "method": "already_exited", "stdout": out or "", "stderr": err or ""}

        if os.name == "nt":
            # If started with CREATE_NEW_PROCESS_GROUP, we can CTRL_BREAK_EVENT
            try:
                proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                out, err = proc.communicate(timeout=grace_sec)
                return {"stopped": True, "method": "CTRL_BREAK_EVENT", "stdout": out or "", "stderr": err or ""}
            except Exception:
                logging.error("Failed to send CTRL_BREAK_EVENT, terminating process")
                proc.terminate()
                out, err = proc.communicate(timeout=grace_sec)
                return {"stopped": True, "method": "terminate", "stdout": out or "", "stderr": err or ""}

        # POSIX
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=grace_sec)
            return {"stopped": True, "method": "terminate", "stdout": out or "", "stderr": err or ""}
        except subprocess.TimeoutExpired:
            logging.error("Terminate timed out, killing process")
            proc.kill()
            out, err = proc.communicate(timeout=5)
            return {"stopped": True, "method": "kill", "stdout": out or "", "stderr": err or ""}

    except Exception as e:
        logging.error("Error in _terminate_process_tree: %s", str(e))
        return {"stopped": False, "method": "error", "error": str(e), "stdout": "", "stderr": ""}


def run_gradle_tests(project_dir: Path) -> dict:
    """
    Runs: ./gradlew test
    Returns pass/fail + stdout/stderr + returncode.

    Uses wrapper + plain console (Gradle CLI guidance) [3](https://learn.microsoft.com/en-us/answers/questions/2260870/we-have-gpu-servers-and-we-are-using-ollama-models)
    """
    gradlew = _gradlew_cmd(project_dir)
    cmd = gradlew + ["test", "--no-daemon", "--console=plain"]
    gradlew_path = Path(cmd[0])
    if os.name == "nt" and not gradlew_path.exists():
        return {
            "cmd": " ".join(cmd),
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": f"gradlew.bat not found at {gradlew_path}",
        }
    # 15 minutes is generous for first-time dependency download + tests
    return run_command(cmd=cmd, cwd=project_dir, timeout_sec=900)


def _wait_for_health(url: str, total_wait_sec: int = 45, interval_sec: float = 1.0) -> tuple[bool, str]:
    """
    Polls a health URL until success or timeout.
    """
    deadline = time.time() + total_wait_sec
    last_err = ""
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if 200 <= r.status_code < 300:
                return True, f"OK {r.status_code}"
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(interval_sec)
    return False, last_err


import os
import subprocess
from pathlib import Path


def run_app_smoke(project_dir: Path) -> dict:
    """
    Smoke run:
      - Starts ./gradlew :app:run
      - Waits for /health to succeed
      - Verifies /metrics looks Prometheus-like
      - POSTs a sample /ingest and expects 2xx
      - Stops the process
      - Returns pass/fail + details + captured logs
    """
    gradlew = ["gradlew.bat"] if os.name == "nt" else ["./gradlew"]

    # Port used by the generated server (allow override)
    port = int(os.getenv("BENCH_APP_PORT", "8080"))
    host = os.getenv("BENCH_APP_HOST", "127.0.0.1")

    health_url = f"http://{host}:{port}/health"
    metrics_url = f"http://{host}:{port}/metrics"
    ingest_url = f"http://{host}:{port}/ingest"

    gradlew_path = project_dir / "gradlew.bat" if os.name == "nt" else project_dir / "gradlew"
    if not gradlew_path.exists():
        return {
            "cmd": str(gradlew),
            "returncode": None,
            "ok": False,
            "started": False,
            "health_ok": False,
            "health_detail": "",
            "metrics_ok": False,
            "ingest_ok": False,
            "stdout": f"gradlew executable not found at {gradlew_path}",
            "stderr": "",
            "stop_method": None,
        }

    cmd = gradlew + [":app:run", "--no-daemon", "--console=plain"]

    env = os.environ.copy()
    # Many Ktor setups read PORT; this helps standardize smoke checks.
    env.setdefault("PORT", str(port))

    started = False
    health_ok = False
    metrics_ok = False
    ingest_ok = False
    health_detail = ""
    metrics_detail = ""
    ingest_detail = ""
    metrics_http = None
    ingest_http = None

    try:
        popen_kwargs = dict(
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Windows: create new process group so CTRL_BREAK_EVENT can work if you implemented it
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        proc = subprocess.Popen(cmd, **popen_kwargs)
        started = True

        # Wait for /health
        ok, detail = _wait_for_healthy(health_url, total_wait_sec=75, interval_sec=1.0)
        health_ok = ok
        health_detail = detail

        # If health failed, stop immediately and return failure
        if not health_ok:
            # Drain logs and stop
            stop_info = _terminate_process_tree(proc, grace_sec=10)
            return {
                "cmd": " ".join(cmd),
                "ok": False,
                "returncode": proc.returncode,
                "started": started,
                "health_ok": False,
                "health_detail": health_detail,
                "metrics_ok": False,
                "metrics_detail": "skipped (health failed)",
                "ingest_ok": False,
                "ingest_detail": "skipped (health failed)",
                "stdout": stop_info.get("stdout", ""),
                "stderr": stop_info.get("stderr", ""),
                "stop_method": stop_info.get("method"),
            }

        # Check /metrics
        ok_m, code_m, body_m = _http_get(metrics_url, timeout_sec=5)
        metrics_http = code_m
        if ok_m:
            prom_ok, prom_detail = is_prometheus_like(body_m)
            metrics_ok = prom_ok
            metrics_detail = prom_detail if prom_ok else f"not prometheus-like: {prom_detail}"
        else:
            metrics_ok = False
            metrics_detail = f"HTTP {code_m}: {body_m[:200]}" if code_m is not None else body_m

        # POST /ingest with a sample payload
        sample_event = {
            "eventId": f"smoke-{int(time.time()*1000)}",
            "type": "SMOKE_TEST",
            "timestamp": int(time.time()),
            "payload": {"hello": "world", "n": 1},
        }
        ok_i, code_i, body_i = _http_post_json(ingest_url, sample_event, timeout_sec=8)
        ingest_http = code_i
        ingest_ok = ok_i
        ingest_detail = f"OK {code_i}" if ok_i else (f"HTTP {code_i}: {body_m[:200]}" if code_i is not None else body_m)

        # Stop server
        stop_info = _terminate_process_tree(proc, grace_sec=10)

        # Overall pass requires all checks
        overall_ok = bool(health_ok and metrics_ok and ingest_ok)

        return {
            "cmd": " ".join(cmd),
            "ok": overall_ok,
            "returncode": 0 if overall_ok else (proc.returncode if proc.returncode is not None else None),
            "started": started,

            "health_ok": health_ok,
            "health_detail": health_detail,

            "metrics_ok": metrics_ok,
            "metrics_http": metrics_http,
            "metrics_detail": metrics_detail,

            "ingest_ok": ingest_ok,
            "ingest_http": ingest_http,
            "ingest_detail": ingest_detail,

            "stdout": stop_info.get("stdout", ""),
            "stderr": stop_info.get("stderr", ""),
            "stop_method": stop_info.get("method"),
        }

    except Exception as e:
        logging.exception("Exception in run_app_smoke")
        return {
            "cmd": " ".join(cmd),
            "ok": False,
            "returncode": None,
            "started": started,
            "health_ok": False,
            "health_detail": "exception",
            "metrics_ok": False,
            "metrics_detail": "exception",
            "ingest_ok": False,
            "ingest_detail": "exception",
            "stdout": "",
            "stderr": str(e),
            "stop_method": None,
        }

def run_cmd(cmd, check=False):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)

def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "unknown"

def get_pulled_models():
    r = run_cmd(["ollama", "list"], check=True)
    lines = r.stdout.splitlines()
    models = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models

def get_ollama_version():
    try:
        r = run_cmd(["ollama", "--version"])
        if r.returncode == 0:
            return (r.stdout.strip() or r.stderr.strip() or "").strip()
    except Exception as e:
        logging.error("Error getting Ollama version: %s", str(e))
    return None

def get_python_env_info():
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def _fmt_mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
    

def _write_phase_error(run_dir: Path, model: str, phase: str, error: Exception):
    phase_dir = run_dir / "models" / safe_name(model)
    phase_dir.mkdir(parents=True, exist_ok=True)
    path = phase_dir / f"{phase}_ERROR.log"
    path.write_text(str(error), encoding="utf-8")


# ============================================================
# Memory measurement (RSS + optional NVIDIA)
# ============================================================

def get_ollama_rss_mb():
    total = 0
    for p in psutil.process_iter(attrs=["name", "exe", "cmdline", "memory_info"]):
        try:
            name = (p.info.get("name") or "").lower()
            exe = (p.info.get("exe") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in exe or "ollama" in cmd:
                mi = p.info.get("memory_info")
                if mi:
                    total += mi.rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return round(total / (1024 * 1024), 2)

def get_nvidia_vram_used_mb():
    try:
        r = run_cmd(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if r.returncode != 0:
            return None
        vals = []
        for line in r.stdout.strip().splitlines():
            if line.strip():
                vals.append(int(line.strip()))
        return sum(vals) if vals else 0
    except FileNotFoundError:
        logging.info("nvidia-smi not found, assuming no NVIDIA GPU")
        return None

def snapshot_memory():
    return {
        "ollama_rss_mb": get_ollama_rss_mb(),
        "nvidia_vram_used_mb": get_nvidia_vram_used_mb(),
        "timestamp": time.time(),
    }


@dataclass
class Peak:
    peak_rss: float | None = None
    peak_vram: int | None = None
    samples: int = 0


class PeakSampler:
    def __init__(self, interval: float):
        self.interval = interval
        self._stop = threading.Event()
        self.peak = Peak()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            m = snapshot_memory()
            rss = m["ollama_rss_mb"]
            vram = m["nvidia_vram_used_mb"]
            if rss is not None:
                self.peak.peak_rss = rss if self.peak.peak_rss is None else max(self.peak.peak_rss, rss)
            if vram is not None:
                self.peak.peak_vram = vram if self.peak.peak_vram is None else max(self.peak.peak_vram, vram)
            self.peak.samples += 1
            time.sleep(self.interval)


# ============================================================
# Ollama API: correct api_generate() + auto-continue
# ============================================================

def ollama_generate_url() -> str:
    host = env_str("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    return host.rstrip("/") + "/api/generate"


def api_generate(
    model: str,
    prompt: str,
    *,
    temperature: float,
    timeout: int,
    num_ctx: int,
    num_predict: int | None = None,
    keep_alive: int | str = -1,
    context: list[int] | None = None,
    force_json: bool = False,
    stop: list[str] | None = None,
    stream: bool = True,
) -> dict:
    """
    Robust /api/generate caller.

    - Uses streaming NDJSON by default and reconstructs full response from chunks.
    - Sets num_ctx explicitly to prevent prompt truncation.
    - Supports 'context' continuation between calls.
    - Optional JSON mode: format="json".
    """
    url = ollama_generate_url()

    options: dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": num_ctx,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict
    if stop:
        options["stop"] = stop

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": keep_alive,
        "options": options,
    }
    if context is not None:
        payload["context"] = context
    if force_json:
        payload["format"] = "json"

    if stream:
        r = requests.post(url, json=payload, timeout=timeout, stream=True)
        r.raise_for_status()

        full_parts = []
        last = {}
        for raw in r.iter_lines():
            if not raw:
                continue
            obj = json.loads(raw.decode("utf-8"))
            last = obj
            chunk = obj.get("response", "")
            if chunk:
                full_parts.append(chunk)
            if obj.get("done"):
                break

        last["response"] = "".join(full_parts)
        return last

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def generate_with_autocontinue(
    *,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    num_ctx: int,
    num_predict: int,
    force_json: bool,
    max_hops: int,
) -> dict:
    """
    Auto-continue if done_reason == 'length' using returned 'context'.
    """
    ctx = None
    full = []
    last = {}

    for hop in range(max_hops):
        # First hop uses full prompt; later hops request continuation.
        hop_prompt = prompt if hop == 0 else (
            "Continue exactly where you left off. Output must remain valid JSON only. "
            "Do not repeat any already emitted text."
        )

        resp = api_generate(
            model=model,
            prompt=hop_prompt,
            temperature=temperature,
            timeout=timeout,
            num_ctx=num_ctx,
            num_predict=num_predict,
            keep_alive=-1,
            context=ctx,
            force_json=force_json,
            stream=True,
        )

        # Detect prompt truncation: prompt_eval_count too small means the model didn't see the prompt.
        if resp.get("prompt_eval_count") is not None and resp.get("prompt_eval_count", 0) < 200 and hop == 0:
            raise RuntimeError(
                f"Prompt appears truncated for model {model} (prompt_eval_count={resp.get('prompt_eval_count')}). "
                f"Increase num_ctx (try 32768) or shorten Phase 1 prompt."
            )

        text = resp.get("response", "") or ""
        if text:
            full.append(text)

        last = resp
        ctx = resp.get("context", ctx)

        if resp.get("done_reason") != "length":
            break

    last["response"] = "".join(full)
    return last


# ============================================================
# JSON extraction + safe project writing
# ============================================================

def extract_json_object(text: str) -> dict | None:
    t = text.strip()

    # Strip code fences if present
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    # Try direct parse
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        logging.error("Error parsing JSON directly: %s", str(e))

    # Try substring parse
    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        sub = t[i:j + 1]
        try:
            obj = json.loads(sub)
            return obj if isinstance(obj, dict) else None
        except Exception as e:
            logging.error("Error parsing JSON substring: %s", str(e))

    return None


def write_files_to_dir(project_dir: Path, files: list[dict]) -> int:
    """
    Writes file entries safely (no traversal). Returns count written.
    """
    written = 0
    root = project_dir.resolve()
    for entry in files:
        if not isinstance(entry, dict):
            continue
        rel_path = str(entry.get("path") or "").strip()
        content = entry.get("content")
        if not rel_path or content is None:
            continue

        rel_path = rel_path.replace("\\", "/").lstrip("/")
        if ".." in rel_path.split("/"):
            continue

        out_path = (project_dir / rel_path).resolve()
        if root not in out_path.parents and out_path != root:
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(str(content), encoding="utf-8")
        written += 1
    return written


def parse_pending_files(manifest_text: str) -> list[str]:
    """
    Extracts PENDING_FILES section with lines prefixed "- ".
    """
    lines = manifest_text.splitlines()
    pending = []
    in_section = False
    for line in lines:
        if line.strip() == "PENDING_FILES":
            in_section = True
            continue
        if in_section:
            # Stop at next heading-like line
            if line.strip().startswith("#"):
                break
            m = re.match(r"^\s*-\s+(.+?)\s*$", line)
            if m:
                path = m.group(1).strip()
                if path and not path.startswith("/"):
                    pending.append(path)
    # De-dupe preserve order
    seen = set()
    out = []
    for p in pending:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ============================================================
# Phase 1 + Phase 2 orchestration
# ============================================================


# Function to sanitize the JSON string by removing invalid control characters
def sanitize_json_string(json_str: str) -> str:
    """
    Removes invalid control characters from the JSON string.
    """
    # Remove non-printable ASCII characters (0-31 and 127)
    sanitized = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
    return sanitized

def phase1_scaffold(
    *,
    model: str,
    run_dir: Path,
    temperature: float,
    timeout: int,
    num_ctx: int,
    num_predict: int,
    max_hops: int,
) -> tuple[dict, Path]:
    """
    Generates scaffold JSON and writes to projects/<model>/<project>_<runid>.
    Returns (phase1_json, project_dir).
    """
    resp = generate_with_autocontinue(
        model=model,
        prompt=PHASE1_PROMPT,
        temperature=temperature,
        timeout=timeout,
        num_ctx=num_ctx,
        num_predict=num_predict,
        force_json=False,   # prompt already enforces JSON-only; json mode can still be used if desired
        max_hops=max_hops,
    )

    raw_text = resp.get("response", "") or ""
    phase1_raw_path = run_dir / "models" / safe_name(model) / f"phase1_raw_{run_dir.name}.txt"
    phase1_raw_path.parent.mkdir(parents=True, exist_ok=True)
    phase1_raw_path.write_text(raw_text, encoding="utf-8")

    # Sanitize the raw JSON response
    sanitized_text = sanitize_json_string(raw_text)

    obj = extract_json_object(sanitized_text)
    if not obj or "files" not in obj or "project" not in obj:
        raise RuntimeError(f"Phase 1 did not produce valid JSON with project/files for model {model}")

    project = obj.get("project") or {}
    proj_name = safe_name(str(project.get("name") or "kotlin-project"))
    project_dir = PROJECTS_DIR / safe_name(model) / f"{proj_name}_{run_dir.name}"
    project_dir.mkdir(parents=True, exist_ok=True)

    files = obj.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Phase 1 files[] missing/empty for model {model}")

    write_files_to_dir(project_dir, files)

    # Save sanitized phase1 JSON for debugging
    phase1_json_path = run_dir / "models" / safe_name(model) / f"phase1_{run_dir.name}.json"
    phase1_json_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    # Check if MANIFEST.md has PENDING_FILES section
    manifest_path = project_dir / "MANIFEST.md"
    if not manifest_path.exists():
        raise RuntimeError("Phase 1 missing MANIFEST.md.")

    pending_files = parse_pending_files(manifest_path.read_text(encoding="utf-8"))
    if len(pending_files) < 20:
        raise RuntimeError(f"MANIFEST.md has no PENDING_FILES list or fewer than 20 files. Found {len(pending_files)} files.")

    return obj, project_dir


# phase2_generate_files.py

import re

def phase2_generate_files(
    *,
    model: str,
    run_dir: Path,
    project_dir: Path,
    pending_files: list[str],
    temperature: float,
    timeout: int,
    num_ctx: int,
    num_predict: int,
    max_hops: int,
    batch_size: int,
    arch_summary: str,
) -> dict:
    """
    Generates pending files in batches and writes them to the project directory.
    Auto-retries by shrinking batch if length issues persist.
    Returns stats.
    """
    stats = {
        "pending_total": len(pending_files),
        "batches": 0,
        "files_written": 0,
        "errors": [],
    }

    model_key = safe_name(model)
    model_run_dir = run_dir / "models" / model_key
    model_run_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    batch_size = max(batch_size, DEFAULT_MIN_BATCH)

    while i < len(pending_files):
        current = pending_files[i:i + batch_size]
        file_list_str = "\n".join(current)
        prompt = PHASE2_PROMPT_TEMPLATE.replace("{ARCH_SUMMARY}", arch_summary).replace("{FILE_LIST}", file_list_str)

        # Generate chunk with autocontinue
        try:
            resp = generate_with_autocontinue(
                model=model,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout,
                num_ctx=num_ctx,
                num_predict=num_predict,
                force_json=False,
                max_hops=max_hops,
            )
            raw_text = resp.get("response", "") or ""

            chunk_raw_path = model_run_dir / f"phase2_chunk_{stats['batches']+1}_{run_dir.name}.txt"
            chunk_raw_path.write_text(raw_text, encoding="utf-8")

            # Sanitize the JSON response
            sanitized_text = sanitize_json_string(raw_text)

            obj = extract_json_object(sanitized_text)
            if not obj or "files" not in obj:
                raise RuntimeError("Phase 2 chunk not valid JSON with files[]")

            files = obj.get("files")
            if not isinstance(files, list) or not files:
                raise RuntimeError("Phase 2 files[] empty")

            # Write to disk
            wrote = write_files_to_dir(project_dir, files)
            stats["files_written"] += wrote
            stats["batches"] += 1

            # Save JSON chunk
            chunk_json_path = model_run_dir / f"phase2_chunk_{stats['batches']}_{run_dir.name}.json"
            chunk_json_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

            i += batch_size

        except Exception as e:
            logging.error("Error generating phase2 files for batch %d: %s", stats['batches']+1, str(e))
            stats["errors"].append(str(e))

            # If we fail, shrink batch size and retry same index.
            if batch_size > DEFAULT_MIN_BATCH:
                batch_size = max(DEFAULT_MIN_BATCH, batch_size // 2)
                continue

            # If already at minimum batch size, skip this file to avoid infinite loop.
            i += 1

    return stats


# ============================================================
# Benchmark per model (load/unload + metrics)
# ============================================================

def preload_model(model: str, temperature: float, timeout: int, num_ctx: int):
    # small warmup + ensures model loads with desired context size
    return api_generate(
        model=model,
        prompt=" ",
        temperature=temperature,
        timeout=timeout,
        num_ctx=num_ctx,
        num_predict=1,
        keep_alive=-1,
        stream=True,
    )

def unload_model(model: str):
    # Prefer CLI stop (unloads immediately)
    try:
        r = run_cmd(["ollama", "stop", model])
        if r.returncode == 0:
            return {"method": "ollama stop", "ok": True}
    except Exception as e:
        logging.error("Error stopping model %s with ollama stop: %s", model, str(e))
    # Fallback: API unload (keep_alive=0 is supported by /api/generate)
    try:
        r = requests.post(
            ollama_generate_url(),
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=60,
        )
        r.raise_for_status()
        return {"method": "api keep_alive=0", "ok": True}
    except Exception as e:
        logging.error("Error unloading model %s via API: %s", model, str(e))
        return {"method": "api keep_alive=0", "ok": False, "detail": str(e)}




def run_model_end_to_end(
    *,
    model: str,
    run_dir: Path,
    temperature: float,
    timeout: int,
    num_ctx: int,
    phase0_num_predict: int,
    phase1_num_predict: int,
    phase2_num_predict: int,
    max_hops: int,
    batch_size: int,
) -> dict:
    sampler = PeakSampler(interval=0.5)

    result: dict[str, Any] = {
        "model": model,
        "run_id": run_dir.name,
        "temperature": temperature,
        "timeout_sec": timeout,
        "num_ctx": num_ctx,
        "phase0_num_predict": phase0_num_predict,
        "phase1_num_predict": phase1_num_predict,
        "phase2_num_predict": phase2_num_predict,
        "phase2_batch": batch_size,
        "max_hops": max_hops,

        "project_dir": None,
        "phase2_stats": None,

        "gradle_test": None,
        "app_smoke": None,

        "memory_baseline": None,
        "memory_after": None,
        "peak_rss_mb": None,
        "peak_vram_mb": None,

        "wall_time_sec": None,
        "error": None,
        "unload": None,
    }

    t0 = time.time()
    model_log_dir = run_dir / "models" / safe_name(model) / "validation_logs"
    model_log_dir.mkdir(parents=True, exist_ok=True)

    try:
        result["memory_baseline"] = snapshot_memory()
        sampler.start()

        # Load model once (ensure context is set)
        preload_model(model, temperature=temperature, timeout=timeout, num_ctx=num_ctx)

        # --- Phase 0: Architecture (TEXT) ---
    
        try:
            full_arch, arch_summary = phase0_architecture(
                model=model,
                run_dir=run_dir,
                temperature=temperature,
                timeout=timeout,
                num_ctx=num_ctx,
                num_predict=phase0_num_predict,
                max_hops=max_hops,
            )
            result["phase0_response"] = full_arch  # Save raw response
        except Exception as e:
            _write_phase_error(run_dir, model, "PHASE0", e)
            result["error"] = f"Phase 0 (architecture) failed: {e}"
            raise


        # --- Phase 1: Scaffold (JSON) ---
        phase1_prompt = PHASE1_PROMPT_TEMPLATE.replace("{ARCH_SUMMARY}", arch_summary)

        phase1_resp = generate_with_autocontinue(
            model=model,
            prompt=phase1_prompt,
            temperature=temperature,
            timeout=timeout,
            num_ctx=num_ctx,
            num_predict=phase1_num_predict,
            force_json=False,
            max_hops=max_hops,
        )

        phase1_text = (phase1_resp.get("response") or "").strip()
        result["phase1_response"] = phase1_text  # Save raw response

        phase1_obj = extract_json_object(phase1_text)
        if not phase1_obj or "files" not in phase1_obj or "project" not in phase1_obj:
            raise RuntimeError("Phase 1 did not produce valid JSON with project/files.")

        proj_name = safe_name(str((phase1_obj.get("project") or {}).get("name") or "kotlin-project"))
        project_dir = PROJECTS_DIR / safe_name(model) / f"{proj_name}_{run_dir.name}"
        project_dir.mkdir(parents=True, exist_ok=True)
        result["project_dir"] = str(project_dir)

        wrote1 = write_files_to_dir(project_dir, phase1_obj["files"])
        # Save phase1 raw/json for debugging
        model_dir = run_dir / "models" / safe_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"phase1_raw_{run_dir.name}.txt").write_text(phase1_text, encoding="utf-8")
        (model_dir / f"phase1_{run_dir.name}.json").write_text(json.dumps(phase1_obj, indent=2), encoding="utf-8")

        # --- Phase 2: Implementation (JSON batches) ---
        manifest_path = project_dir / "MANIFEST.md"
        if not manifest_path.exists():
            raise RuntimeError("Phase 1 missing MANIFEST.md.")
        pending_files = parse_pending_files(manifest_path.read_text(encoding="utf-8"))
        if not pending_files:
            raise RuntimeError("MANIFEST.md has no PENDING_FILES list.")

        try:
            phase2_stats = phase2_generate_files(
                model=model,
                run_dir=run_dir,
                project_dir=project_dir,
                pending_files=pending_files,
                temperature=temperature,
                timeout=timeout,
                num_ctx=num_ctx,
                num_predict=phase2_num_predict,
                max_hops=max_hops,
                batch_size=batch_size,
                arch_summary=arch_summary,
            )
            result["phase2_stats"] = phase2_stats
        except Exception as e:
            _write_phase_error(run_dir, model, "PHASE2", e)
            result["error"] = f"Phase 2 (implementation) failed: {e}"
            raise

        # --- Validation: Gradle tests + app smoke ---
        try:
            tests_result = run_gradle_tests(project_dir)
            app_result = run_app_smoke(project_dir)

            (model_log_dir / "gradle_test_stdout.txt").write_text(tests_result.get("stdout", ""), encoding="utf-8")
            (model_log_dir / "gradle_test_stderr.txt").write_text(tests_result.get("stderr", ""), encoding="utf-8")
            (model_log_dir / "app_run_stdout.txt").write_text(app_result.get("stdout", ""), encoding="utf-8")
            (model_log_dir / "app_run_stderr.txt").write_text(app_result.get("stderr", ""), encoding="utf-8")

            result["gradle_test"] = {"ok": bool(tests_result.get("ok")), "returncode": tests_result.get("returncode")}
            result["app_smoke"] = {
                "ok": bool(app_result.get("ok")),
                "health_ok": app_result.get("health_ok"),
                "metrics_ok": app_result.get("metrics_ok"),
                "ingest_ok": app_result.get("ingest_ok"),
                "returncode": app_result.get("returncode"),
            }

            if not tests_result.get("ok"):
                raise RuntimeError("Gradle tests failed (./gradlew test).")
            if not app_result.get("ok"):
                raise RuntimeError("App smoke validation failed (health/metrics/ingest).")
        except Exception as e:
            _write_phase_error(run_dir, model, "VALIDATION", e)
            result["error"] = f"Validation failed: {e}"
            raise

    except Exception as e:
        logging.exception("Exception in run_model_end_to_end for model %s", model)
        result["error"] = str(e)

    finally:
        try:
            sampler.stop()
        except Exception:
            pass
        result["peak_rss_mb"] = sampler.peak.peak_rss
        result["peak_vram_mb"] = sampler.peak.peak_vram
        result["memory_after"] = snapshot_memory()
        result["wall_time_sec"] = round(time.time() - t0, 3)
        result["unload"] = unload_model(model)

    return result



# ============================================================
# Run artifacts (timestamp in filenames)
# ============================================================

def write_run_artifacts(run_dir: Path, results: list[dict], env_info: dict):
    run_id = run_dir.name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Construct file names with timestamp
    results_json = run_dir / f"results_{run_id}_{timestamp}.json"
    results_csv = run_dir / f"results_{run_id}_{timestamp}.csv"
    report_md = run_dir / f"report_{run_id}_{timestamp}.md"
    env_json = run_dir / f"env_{run_id}_{timestamp}.json"

    # Save the raw JSON responses
    for i, result in enumerate(results):
        if "phase0_response" in result:
            phase0_raw_path = run_dir / f"phase0_raw_{run_id}_{timestamp}_model{i}.txt"
            phase0_raw_path.write_text(result["phase0_response"], encoding="utf-8")
        
        if "phase1_response" in result:
            phase1_raw_path = run_dir / f"phase1_raw_{run_id}_{timestamp}_model{i}.txt"
            phase1_raw_path.write_text(result["phase1_response"], encoding="utf-8")

    # Write the results and environment info to respective files
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    env_json.write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    # CSV
    fieldnames = sorted({k for r in results for k in r.keys()})
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fieldnames})

    # Markdown summary
    ok = [r for r in results if not r.get("error")]
    bad = [r for r in results if r.get("error")]

    md = []
    md.append("# Ollama Benchmark Report\n\n")
    md.append(f"- Run ID: `{run_id}`\n")
    md.append(f"- Python: `{env_info.get('python')}`\n")
    md.append(f"- Platform: `{env_info.get('platform')}`\n")
    if env_info.get("ollama_version"):
        md.append(f"- Ollama: `{env_info.get('ollama_version')}`\n")
    md.append(f"- Projects dir: `{PROJECTS_DIR}`\n\n")

    md.append("## Successful models\n\n")
    for r in ok:
        md.append(f"- `{r['model']}` → project: `{r.get('project_dir')}`; phase2 files written: `{(r.get('phase2_stats') or {}).get('files_written')}`\n")

    if bad:
        md.append("\n## Failures\n\n")
        for r in bad:
            md.append(f"- `{r['model']}`: {r.get('error')}\n")

    report_md.write_text("".join(md), encoding="utf-8")


# ============================================================
# CLI entrypoint
# ============================================================

def parse_args(argv):
    p = argparse.ArgumentParser(description="2-phase Ollama benchmark that materializes runnable Kotlin projects.")

    p.add_argument("--timeout", type=int, default=env_int("BENCH_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))
    p.add_argument("--cooldown", type=int, default=env_int("BENCH_COOLDOWN_SEC", DEFAULT_COOLDOWN_SEC))
    p.add_argument("--temperature", type=float, default=env_float("BENCH_TEMPERATURE", DEFAULT_TEMPERATURE))

    p.add_argument("--num-ctx", type=int, default=env_int("BENCH_NUM_CTX", DEFAULT_NUM_CTX))
    
    
    p.add_argument(
        "--phase0-num-predict",
        type=int,
        default=env_int("BENCH_PHASE0_NUM_PREDICT", 4096),
    )

    p.add_argument("--phase1-num-predict", type=int, default=env_int("BENCH_PHASE1_NUM_PREDICT", DEFAULT_NUM_PREDICT_PHASE1))
    p.add_argument("--phase2-num-predict", type=int, default=env_int("BENCH_PHASE2_NUM_PREDICT", DEFAULT_NUM_PREDICT_PHASE2))

    p.add_argument("--phase2-batch", type=int, default=env_int("BENCH_PHASE2_BATCH", DEFAULT_PHASE2_BATCH))
    p.add_argument("--max-hops", type=int, default=env_int("BENCH_MAX_HOPS", DEFAULT_MAX_HOPS))

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    models = get_pulled_models()
    if not models:
        print("No local models found. Run `ollama list` to verify.")
        return 1

    env_info = get_python_env_info()
    env_info["ollama_version"] = get_ollama_version()
    env_info["run_id"] = run_id
    env_info["ollama_host"] = env_str("OLLAMA_HOST", "http://localhost:11434")

    results = []
    pbar = tqdm(
        models,
        desc="Benchmarking models",
        unit="model",
        mininterval=0.5,   # avoid excessive redraws
    )

    for model in pbar:
        # ✅ reset per-model timer
        model_start = time.monotonic()

        # left side description (stable)
        pbar.set_description("Benchmarking models", refresh=False)

        # right side dynamic info
        pbar.set_postfix_str(f"{model}, model_time=00:00", refresh=True)

        r = run_model_end_to_end(
            model=model,
            run_dir=run_dir,
            temperature=args.temperature,
            timeout=args.timeout,
            num_ctx=args.num_ctx,
            phase0_num_predict=args.phase0_num_predict,
            phase1_num_predict=args.phase1_num_predict,
            phase2_num_predict=args.phase2_num_predict,
            max_hops=args.max_hops,
            batch_size=args.phase2_batch,
        )
        results.append(r)

        # ✅ update final timer once model finishes
        elapsed = time.monotonic() - model_start
        pbar.set_postfix_str(
            f"{model} | {'ERROR' if r.get('error') else 'OK'}, "
            f"model_time={_fmt_mmss(elapsed)}",
            refresh=True,
        )

        time.sleep(args.cooldown)

    write_run_artifacts(run_dir, results, env_info)
    print(f"\n✅ Run saved to: {run_dir}")
    print(f"✅ Projects saved under: {PROJECTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())