from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import psutil
from tqdm import tqdm


DEFAULT_PROMPT = """
You are a senior distributed-systems architect, Kotlin backend engineer, and production readiness reviewer.

OBJECTIVE:
Design a production-grade, fault-tolerant, horizontally scalable event-driven system for a global financial trading platform AND produce a fully runnable local Kotlin prototype that demonstrates this design with real, compiled code.

THIS IS NOT A MOCKUP.
THIS IS NOT A SKETCH.
THE RESULT MUST COMPILE AND RUN LOCALLY WITHOUT MODIFICATION.

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
   - Backpressure and load shedding must be both explained AND implemented
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
KOTLIN PROTOTYPE — NON‑NEGOTIABLE REQUIREMENTS
============================================================
You MUST generate a complete, runnable Kotlin project that builds and runs with Gradle.

A) Build:
   - Multi-module Gradle build using Kotlin DSL
   - Modules:
     - :app    (Ktor HTTP server)
     - :core   (domain + business logic)
     - :infra  (queues, audit log, metrics, persistence abstractions)
   - Include Gradle Wrapper files (gradlew, gradlew.bat, gradle/wrapper/*)
   - Explicit Kotlin and Java toolchain versions (Java 17)

B) Server:
   - Ktor server using a stable engine (CIO or Netty)
   - Endpoints:
     - POST /ingest
     - POST /orders
     - POST /replay
     - GET /metrics   (Prometheus format)
     - GET /health
   - Graceful shutdown and queue draining
   - Configurable via environment variables

C) Backpressure:
   - Bounded queue/channel
   - Explicit behavior when queue is full (429 or 503)
   - Structured logging of backpressure events

D) Exactly-once demo semantics:
   - Idempotency key handling
   - In-memory or file-backed dedup store
   - Retry-safe logic

E) Audit log:
   - Append-only file log
   - Hash chaining for tamper evidence
   - Replay reads from audit log

F) Observability:
   - Structured logs
   - Metrics counters and histograms
   - Explicit metric names

G) Tests:
   - At least 1 integration test that:
     - boots the server
     - sends HTTP requests
     - validates responses
   - At least 3 unit tests in :core

H) Load testing:
   - Include a runnable load test (k6 JS or Kotlin/JVM client)

I) Documentation:
   - README.md with:
     - Prerequisites
     - How to build
     - How to run
     - How to test
     - Failure modes

J) Code quality:
   - NO placeholders (“TODO”, “omitted”, “…”)
   - NO pseudocode
   - All code must compile
   - Exceptions handled explicitly
   - Cancellation supported where applicable

============================================================
OUTPUT SIZE + COMPLETENESS CONSTRAINTS (ENFORCED)
============================================================
- Minimum 1,500 lines of Kotlin code
- Minimum 25 total files
- At least 15 Kotlin source files
- Multi-package structure:
  api, domain, service, infra, persistence, observability, util
- If these constraints are not met, CONTINUE GENERATING CODE

============================================================
DIRECTORY ASSUMPTION
============================================================
The project will be written by a tool to:
  ollama_benchmarks/projects/<model-name>/<project>/

All file paths must be relative to the project root.

============================================================
OUTPUT FORMAT (ABSOLUTE REQUIREMENT)
============================================================
Return ONE AND ONLY ONE JSON object.
NO Markdown. NO prose outside JSON. NO backticks.

The JSON MUST be exactly:

{
  "project": {
    "name": "<short-project-name>",
    "description": "<one paragraph>"
  },
  "files": [
    { "path": "README.md", "content": "..." },
    { "path": "settings.gradle.kts", "content": "..." },
    { "path": "build.gradle.kts", "content": "..." },
    { "path": "gradlew", "content": "..." },
    { "path": "gradlew.bat", "content": "..." },
    { "path": "gradle/wrapper/gradle-wrapper.properties", "content": "..." },
    { "path": "app/build.gradle.kts", "content": "..." },
    { "path": "core/build.gradle.kts", "content": "..." },
    { "path": "infra/build.gradle.kts", "content": "..." },
    { "path": "app/src/main/kotlin/.../Main.kt", "content": "..." },
    { "path": "app/src/test/kotlin/...IntegrationTest.kt", "content": "..." },
    { "path": "core/src/test/kotlin/...Test.kt", "content": "..." },
    { "path": "scripts/loadtest.js", "content": "..." },
    { "path": "MANIFEST.md", "content": "..." }
  ]
}

============================================================
MANIFEST.md REQUIREMENTS
============================================================
The MANIFEST.md file MUST list EVERY file and include:
- Path
- Approx line count
- Purpose
- Key dependencies

============================================================
VALIDATION BEFORE FINISHING (REQUIRED)
============================================================
Before you finish:
- Verify all Kotlin syntax is valid
- Verify Gradle builds without modification
- Verify ./gradlew :app:run starts the server
- Verify ./gradlew test runs tests
- Verify the JSON parses correctly

DO NOT STOP UNTIL ALL REQUIREMENTS ARE SATISFIED.
NOW PRODUCE THE JSON OBJECT.
"""


DEFAULT_TIMEOUT_SEC = 1200
DEFAULT_COOLDOWN_SEC = 15
DEFAULT_SAMPLE_INTERVAL_SEC = 0.5
DEFAULT_TEMPERATURE = 0.2

OUTPUT_DIR = Path("ollama_benchmarks")
RUNS_DIR = OUTPUT_DIR / "runs"
PROJECTS_DIR = OUTPUT_DIR / "projects"
OUTPUT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)


# -------------------------
# Env helpers
# -------------------------
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


# -------------------------
# Utilities
# -------------------------
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
    except Exception:
        pass
    return None


def get_python_env_info():
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
    }


# -------------------------
# GPU memory measurement
# -------------------------
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
        r = run_cmd(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        if r.returncode != 0:
            return None
        vals = []
        for line in r.stdout.strip().splitlines():
            if line.strip():
                vals.append(int(line.strip()))
        return sum(vals) if vals else 0
    except FileNotFoundError:
        return None


def snapshot_memory():
    return {
        "ollama_rss_mb": get_ollama_rss_mb(),
        "nvidia_vram_used_mb": get_nvidia_vram_used_mb(),
        "timestamp": time.time(),
    }


# -------------------------
# Ollama API helpers
# -------------------------
def ollama_generate_url() -> str:
    host = env_str("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    return host.rstrip("/") + "/api/generate"


def api_generate(model, prompt, keep_alive=-1, temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT_SEC, num_predict=None):
    url = ollama_generate_url()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,       # ← critical
            "num_predict": 8192,
        },
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    
    for line in resp.iter_lines():
        print(line)

    return resp.json()


def preload_model(model, temperature, timeout):
    return api_generate(model=model, prompt=" ", keep_alive=-1, num_predict=1, temperature=temperature, timeout=timeout)


def unload_model(model):
    try:
        r = run_cmd(["ollama", "stop", model])
        if r.returncode == 0:
            return {"method": "ollama stop", "ok": True, "detail": ""}
    except Exception:
        pass

    try:
        resp = api_generate(model=model, prompt="", keep_alive=0, num_predict=8192, timeout=60)
        return {"method": "api keep_alive=0", "ok": True, "detail": resp.get("done_reason", "")}
    except Exception as e:
        return {"method": "api keep_alive=0", "ok": False, "detail": str(e)}


# -------------------------
# Peak sampler
# -------------------------
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


# -------------------------
# Project extraction + writing
# -------------------------
def extract_json_object(text: str) -> dict | None:
    t = text.strip()

    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        sub = t[i:j+1]
        try:
            obj = json.loads(sub)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def write_project_from_response(model_name: str, response_text: str, run_id: str) -> dict:
    """
    Writes a Kotlin project file bundle to:
      ollama_benchmarks/projects/<model-name>/<project>_<run_id>/
    """
    meta = {"written": False, "project_dir": None, "reason": None}

    obj = extract_json_object(response_text)
    if not obj:
        meta["reason"] = "Response was not valid JSON"
        return meta

    project = obj.get("project") or {}
    project_name = safe_name(str(project.get("name") or "kotlin-prototype"))
    files = obj.get("files")

    if not isinstance(files, list) or not files:
        meta["reason"] = "JSON missing files[]"
        return meta

    model_dir = PROJECTS_DIR / safe_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    project_dir = model_dir / f"{project_name}_{run_id}"
    project_dir.mkdir(parents=True, exist_ok=True)

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
        if project_dir.resolve() not in out_path.parents and out_path != project_dir.resolve():
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(str(content), encoding="utf-8")

    meta["written"] = True
    meta["project_dir"] = str(project_dir)
    return meta


# -------------------------
# Benchmark per model
# -------------------------
def benchmark_model(model, prompt, temperature, timeout, sample_interval, num_predict, run_dir: Path):
    run_id = run_dir.name

    result = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "timeout_sec": timeout,
        "num_predict": num_predict,

        "memory_baseline": None,
        "memory_after_load": None,
        "memory_after_run": None,
        "memory_after_unload": None,
        "peak_ollama_rss_mb": None,
        "peak_nvidia_vram_used_mb": None,

        "load_duration_sec": None,
        "wall_time_sec": None,
        "prompt_tokens": None,
        "output_tokens": None,
        "tokens_per_sec": None,
        "response_chars": None,

        "unload": None,
        "error": None,

        "response_file": None,
        "project_write": None,
    }

    sampler = PeakSampler(interval=sample_interval)

    try:
        result["memory_baseline"] = snapshot_memory()

        load_resp = preload_model(model, temperature=temperature, timeout=timeout)
        if "load_duration" in load_resp and load_resp["load_duration"] is not None:
            result["load_duration_sec"] = round(load_resp["load_duration"] / 1e9, 3)

        result["memory_after_load"] = snapshot_memory()

        sampler.start()
        t0 = time.time()

        resp = api_generate(
            model=model,
            prompt=prompt,
            keep_alive=-1,
            temperature=temperature,
            timeout=timeout,
            num_predict=8192,
        )

        wall = time.time() - t0
        sampler.stop()

        result["wall_time_sec"] = round(wall, 3)
        result["memory_after_run"] = snapshot_memory()
        result["peak_ollama_rss_mb"] = sampler.peak.peak_rss
        result["peak_nvidia_vram_used_mb"] = sampler.peak.peak_vram

        result["prompt_tokens"] = resp.get("prompt_eval_count")
        result["output_tokens"] = resp.get("eval_count")
        response_text = resp.get("response", "") or ""
        result["response_chars"] = len(response_text)

        eval_dur_ns = resp.get("eval_duration")
        out_tokens = result["output_tokens"]
        if out_tokens and eval_dur_ns and eval_dur_ns > 0:
            result["tokens_per_sec"] = round(out_tokens / (eval_dur_ns / 1e9), 3)

        # Save raw response per-model (timestamped filename)
        responses_dir = run_dir / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)
        resp_file = responses_dir / f"{safe_name(model)}_{run_id}.txt"
        resp_file.write_text(response_text, encoding="utf-8")
        result["response_file"] = str(resp_file)

        # Write project bundle (timestamped folder)
        result["project_write"] = write_project_from_response(model, response_text, run_id=run_id)

    except Exception as e:
        try:
            sampler.stop()
        except Exception:
            pass
        result["error"] = str(e)

    finally:
        result["unload"] = unload_model(model)
        result["memory_after_unload"] = snapshot_memory()

    return result


# -------------------------
# Run-level saving (timestamped filenames)
# -------------------------
def write_run_artifacts(run_dir: Path, results: list[dict], env_info: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name

    results_json = run_dir / f"results_{run_id}.json"
    results_csv = run_dir / f"results_{run_id}.csv"
    report_md = run_dir / f"report_{run_id}.md"
    env_json = run_dir / f"env_{run_id}.json"

    # JSON
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # CSV
    fieldnames = [
        "model",
        "tokens_per_sec",
        "wall_time_sec",
        "load_duration_sec",
        "prompt_tokens",
        "output_tokens",
        "response_chars",
        "peak_ollama_rss_mb",
        "peak_nvidia_vram_used_mb",
        "error",
        "response_file",
        "project_dir",
    ]
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            proj_dir = None
            if isinstance(r.get("project_write"), dict):
                proj_dir = r["project_write"].get("project_dir")
            w.writerow({
                "model": r.get("model"),
                "tokens_per_sec": r.get("tokens_per_sec"),
                "wall_time_sec": r.get("wall_time_sec"),
                "load_duration_sec": r.get("load_duration_sec"),
                "prompt_tokens": r.get("prompt_tokens"),
                "output_tokens": r.get("output_tokens"),
                "response_chars": r.get("response_chars"),
                "peak_ollama_rss_mb": r.get("peak_ollama_rss_mb"),
                "peak_nvidia_vram_used_mb": r.get("peak_nvidia_vram_used_mb"),
                "error": r.get("error"),
                "response_file": r.get("response_file"),
                "project_dir": proj_dir,
            })

    # Markdown report
    ok = [r for r in results if not r.get("error") and r.get("tokens_per_sec") is not None]
    ok_sorted = sorted(ok, key=lambda r: r["tokens_per_sec"], reverse=True)

    md = []
    md.append("# Ollama Benchmark Report\n\n")
    md.append(f"- Run dir: `{run_dir}`\n")
    md.append(f"- Run ID: `{run_id}`\n")
    md.append(f"- Python: `{env_info.get('python')}`\n")
    md.append(f"- Platform: `{env_info.get('platform')}`\n")
    if env_info.get("ollama_version"):
        md.append(f"- Ollama: `{env_info.get('ollama_version')}`\n")

    md.append("\n## Results (ranked by tokens/sec)\n")
    md.append("| Rank | Model | tok/s | Wall (s) | Out tokens | Peak RSS (MB) | Peak VRAM (MB) | Project |\n")
    md.append("|---:|---|---:|---:|---:|---:|---:|---|\n")

    for i, r in enumerate(ok_sorted, 1):
        proj_dir = ""
        if isinstance(r.get("project_write"), dict):
            proj_dir = r["project_write"].get("project_dir") or ""
        md.append(
            f"| {i} | `{r['model']}` | {r.get('tokens_per_sec','')} | {r.get('wall_time_sec','')} | "
            f"{r.get('output_tokens','')} | {r.get('peak_ollama_rss_mb','')} | {r.get('peak_nvidia_vram_used_mb','')} | "
            f"{proj_dir} |\n"
        )

    failures = [r for r in results if r.get("error")]
    if failures:
        md.append("\n## Failures\n")
        for r in failures:
            md.append(f"- `{r['model']}`: {r['error']}\n")

    report_md.write_text("".join(md), encoding="utf-8")

    # Environment snapshot
    env_json.write_text(json.dumps(env_info, indent=2), encoding="utf-8")


# -------------------------
# CLI
# -------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="Benchmark all pulled Ollama models and materialize Kotlin projects.")
    p.add_argument("--timeout", type=int, default=env_int("BENCH_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))
    p.add_argument("--cooldown", type=int, default=env_int("BENCH_COOLDOWN_SEC", DEFAULT_COOLDOWN_SEC))
    p.add_argument("--sample-interval", type=float, default=env_float("BENCH_SAMPLE_INTERVAL_SEC", DEFAULT_SAMPLE_INTERVAL_SEC))
    p.add_argument("--temperature", type=float, default=env_float("BENCH_TEMPERATURE", DEFAULT_TEMPERATURE))
    p.add_argument("--prompt", type=str, default=env_str("BENCH_PROMPT", DEFAULT_PROMPT))

    max_tokens = env_int("BENCH_MAX_TOKENS", 0)
    p.add_argument("--max-tokens", type=int, default=max_tokens, help="0 means unlimited (omit num_predict).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    models = get_pulled_models()
    if not models:
        print("No local models found. Run `ollama list` to verify.")
        return 1

    num_predict = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None

    env_info = get_python_env_info()
    env_info["ollama_version"] = get_ollama_version()
    env_info["run_id"] = run_id
    env_info["ollama_host"] = env_str("OLLAMA_HOST", "http://localhost:11434")
    env_info["num_predict"] = num_predict

    results = []
    pbar = tqdm(models, desc="Benchmarking models", unit="model")

    for model in pbar:
        pbar.set_postfix_str(model)
        r = benchmark_model(
            model=model,
            prompt=args.prompt,
            temperature=args.temperature,
            timeout=args.timeout,
            sample_interval=args.sample_interval,
            num_predict=num_predict,
            run_dir=run_dir,
        )
        results.append(r)

        if r.get("error"):
            pbar.set_postfix_str(f"{model} | ERROR")
        else:
            pbar.set_postfix_str(f"{model} | {r.get('tokens_per_sec')} tok/s")

        time.sleep(args.cooldown)

    write_run_artifacts(run_dir, results, env_info)
    print(f"\n✅ Run saved to: {run_dir}")
    print(f"✅ Projects saved under: {PROJECTS_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())