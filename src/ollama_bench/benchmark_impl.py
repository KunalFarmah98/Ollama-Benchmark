from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import threading
import statistics
from pathlib import Path
from typing import Any

import requests
import psutil
from tqdm import tqdm

# ============================================================
# PROMPT
# ============================================================

TICTACTOE_PROMPT = r"""
Task:
Generate a complete, runnable native Android Tic-Tac-Toe game using Kotlin.
The application MUST be a functional project that can be opened in Android Studio and built with Gradle.

Output Rules:
- Output MUST be a SINGLE JSON object and NOTHING else.
- No Markdown code blocks, no backticks, no commentary before or after the JSON.
- JSON schema:
{
  "project": { "name": "TicTacToe", "description": "A fully functional native Android Tic-Tac-Toe game." },
  "files": [ { "path": "<relative path>", "content": "<file content>" }, ... ]
}

Requirements:
1. Native Android project (XML layouts, ViewBinding, Kotlin). No Jetpack Compose.
2. Package: com.example.tictactoe
3. Logic: 3x3 board, 2 players (X/O), alternating turns, win/draw detection, and a Reset button.
4. Essential Files MUST be included:
   - settings.gradle.kts (include(":app"))
   - build.gradle.kts (Project level)
   - app/build.gradle.kts (Module level: viewBinding = true, minSdk 26, targetSdk 34, Java 17)
   - app/src/main/AndroidManifest.xml (MainActivity as Launcher)
   - app/src/main/res/layout/activity_main.xml (The UI with 9 buttons in a GridLayout and a Reset button)
   - app/src/main/kotlin/com/example/tictactoe/MainActivity.kt (Game logic and ViewBinding handling)
   - app/src/main/res/values/strings.xml, colors.xml, themes.xml
5. Everything must be complete and syntactically correct. No placeholders.

Now output the JSON object.
"""

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("../ollama_benchmarks")
RUNS_DIR = OUTPUT_DIR / "runs"
PROJECTS_DIR = OUTPUT_DIR / "projects"
OUTPUT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

DEFAULT_COOLDOWN_SEC = 10
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_CTX = 16384
DEFAULT_MAX_HOPS = 10

ALLOWED_MODELS = [
    "devstral-16k:latest",
    "gemma4:26b",
    "gpt-oss:20b",
    "qwen2.5-coder:14b",
    "qwen3:14b",
]

# ============================================================
# Metrics Monitoring
# ============================================================

def get_gpu_metrics() -> tuple[float, float]:
    """Returns (gpu_util_percent, vram_used_mb) using nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0:
            parts = res.stdout.strip().split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
    except:
        pass
    return 0.0, 0.0

def get_ollama_process():
    """Finds the ollama process."""
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'] or ""
            if 'ollama' in name.lower():
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

class BackgroundMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self._thread = None
        self._ollama_proc = None

    def _collect(self):
        self._ollama_proc = get_ollama_process()
        # Initialize CPU percent calculation
        psutil.cpu_percent(interval=None)
        if self._ollama_proc:
            try: self._ollama_proc.cpu_percent(interval=None)
            except: pass

        while not self.stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)

                ollama_cpu = 0.0
                if self._ollama_proc:
                    try:
                        # process.cpu_percent() returns % of a single core (e.g. 100.0 = 1 core)
                        ollama_cpu = self._ollama_proc.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                gpu, vram = get_gpu_metrics()

                # DRAM usage: track system-wide used physical memory in MB
                dram = psutil.virtual_memory().used / (1024 * 1024)

                self.samples.append({
                    "cpu": cpu,
                    "ollama_cpu": ollama_cpu,
                    "gpu": gpu,
                    "vram": vram,
                    "dram": dram
                })
            except:
                pass
            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

        if not self.samples:
            return {}

        metrics = {}
        for key in ["cpu", "ollama_cpu", "gpu", "vram", "dram"]:
            vals = [s[key] for s in self.samples if key in s]
            if not vals:
                metrics.update({f"{key}_min": 0.0, f"{key}_max": 0.0, f"{key}_avg": 0.0})
                continue
            metrics[f"{key}_min"] = min(vals)
            metrics[f"{key}_max"] = max(vals)
            metrics[f"{key}_avg"] = statistics.mean(vals)
        return metrics

# ============================================================
# Ollama API
# ============================================================

def ollama_generate_url() -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    return host.rstrip("/") + "/api/generate"

def api_generate(
    model: str,
    prompt: str,
    num_ctx: int,
    temperature: float,
    num_gpu: int,
    context: list[int] | None = None,
) -> dict:
    url = ollama_generate_url()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "context": context,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "num_predict": -1,
            "num_gpu": num_gpu
        }
    }

    start_time = time.time()
    # Use a longer timeout for large models or slow prompt processing.
    # Connection timeout: 60s
    # Read timeout: 900s (15 mins). Large models (26B+) can take several minutes
    # for the initial prompt prefill/KV cache allocation on some hardware.
    r = requests.post(url, json=payload, timeout=(60, 900), stream=True)
    r.raise_for_status()

    full_text, last = [], {}
    first_token_time = None

    for line in r.iter_lines():
        if line:
            obj = json.loads(line.decode("utf-8"))
            last = obj
            chunk = obj.get("response", "")
            if chunk:
                if first_token_time is None:
                    first_token_time = time.time()
                full_text.append(chunk)
            if obj.get("done"):
                break

    last["response"] = "".join(full_text)
    if first_token_time:
        last["time_to_first_token"] = round(first_token_time - start_time, 3)
    else:
        last["time_to_first_token"] = 0.0

    return last

def generate_with_autocontinue(model: str, prompt: str, num_ctx: int, temperature: float, num_gpu: int, max_hops: int) -> tuple[str, dict]:
    ctx, full = None, []
    stats = {
        "hops": 0,
        "total_eval_count": 0,
        "total_prompt_eval_count": 0,
        "total_eval_duration_ns": 0,
        "ttft": 0.0
    }

    for hop in range(max_hops):
        stats["hops"] += 1
        print(f"  [HOP {hop+1}/{max_hops}] Generating...")
        h_prompt = prompt if hop == 0 else "Continue exactly where you left off. Output ONLY the remaining part of the JSON object. Do not repeat any previously output text."
        resp = api_generate(model, h_prompt, num_ctx, temperature, num_gpu, ctx)

        full.append(resp.get("response", ""))
        ctx = resp.get("context")

        stats["total_eval_count"] += resp.get("eval_count", 0)
        stats["total_prompt_eval_count"] += resp.get("prompt_eval_count", 0)
        stats["total_eval_duration_ns"] += resp.get("eval_duration", 0)

        if hop == 0:
            stats["ttft"] = resp.get("time_to_first_token", 0.0)

        if resp.get("done_reason") != "length": break

    if stats["total_eval_duration_ns"] > 0:
        stats["avg_tokens_per_sec"] = round(stats["total_eval_count"] / (stats["total_eval_duration_ns"] / 1e9), 2)
    else:
        stats["avg_tokens_per_sec"] = 0.0

    stats["total_tokens_used"] = stats["total_eval_count"] + stats["total_prompt_eval_count"]

    return "".join(full), stats

# ============================================================
# Utilities
# ============================================================

def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "unknown"

def get_pulled_models():
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        return [line.split()[0] for line in r.stdout.splitlines()[1:] if line.strip()]
    except Exception:
        return []

def validate_project(project_dir: Path) -> dict:
    req = [
        "app/src/main/AndroidManifest.xml",
        "app/src/main/res/layout/activity_main.xml",
        "app/src/main/kotlin/com/example/tictactoe/MainActivity.kt",
        "app/build.gradle.kts",
        "settings.gradle.kts"
    ]
    status = {f: (project_dir / f).exists() for f in req}
    status["ok"] = all(status.values())
    status["total_files"] = sum(1 for _ in project_dir.rglob("*") if _.is_file())
    return status

def write_project(project_dir: Path, files: list[dict]) -> int:
    written = 0
    root = project_dir.resolve()
    for f in files:
        rel = str(f.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel: continue
        out = (project_dir / rel).resolve()
        if root in out.parents or out == root:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(str(f.get("content", "")), encoding="utf-8")
            written += 1
    return written

# ============================================================
# Pipeline
# ============================================================

def run_benchmark(model: str, run_dir: Path, args: argparse.Namespace) -> dict:
    print(f"\n[BENCHMARK] Model: {model}")

    # Pre-warm model (loading from disk can take time)
    try:
        requests.post(ollama_generate_url(), json={"model": model, "prompt": "", "stream": False, "options": {"num_predict": 1}}, timeout=300)
    except: pass

    monitor = BackgroundMonitor()
    monitor.start()

    t0 = time.time()
    res = {"model": model, "success": False, "files_written": 0}
    model_run_dir = run_dir / "models" / safe_name(model)
    model_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        response, gen_stats = generate_with_autocontinue(model, TICTACTOE_PROMPT, args.num_ctx, args.temperature, args.num_gpu, args.max_hops)
        res.update(gen_stats)
        res["total_time_sec"] = round(time.time() - t0, 3)

        (model_run_dir / "raw_response.txt").write_text(response, encoding="utf-8")

        # Extract and parse JSON
        text = re.sub(r'[\x00-\x1f\x7f]', '', response.strip())
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        obj = None
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try: obj = json.loads(text[start:end+1])
            except: pass

        if not obj or "files" not in obj:
            raise RuntimeError("Invalid JSON output or missing files array")

        project_dir = PROJECTS_DIR / safe_name(model) / f"TicTacToe_{run_dir.name}"
        project_dir.mkdir(parents=True, exist_ok=True)

        res["files_written"] = write_project(project_dir, obj.get("files", []))
        res["validation"] = validate_project(project_dir)
        res["success"] = res["validation"]["ok"]
        res["project_dir"] = str(project_dir)

        (model_run_dir / "project.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"  [DONE] Written {res['files_written']} files. Success: {res['success']}")
        print(f"  [METRICS] {res.get('avg_tokens_per_sec', 0):.2f} t/s | TTFT: {res.get('ttft', 0.0):.3f}s | Time: {res.get('total_time_sec', 0):.2f}s | Tokens: {res.get('total_tokens_used', 0)}")
    except Exception as e:
        res["error"] = str(e)
        print(f"  [ERROR] {e}")

    resource_stats = monitor.stop()
    res.update(resource_stats)

    # Check for CPU offloading
    # If the ollama process uses significant CPU, it indicates offloading.
    # Threshold: 50.0 (equivalent to half a core being pegged).
    if resource_stats.get("ollama_cpu_avg", 0) > 50.0:
        res["success"] = False
        res["error"] = f"CPU Offloading detected ({resource_stats['ollama_cpu_avg']:.1f}% Ollama CPU). Aborting to ensure GPU-only results."
        print(f"  [WARNING] {res['error']}")

    # Unload model from VRAM
    try:
        requests.post(ollama_generate_url(), json={"model": model, "keep_alive": 0}, timeout=30)
    except: pass

    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Specific model name")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument("--num-gpu", type=int, default=99, help="Number of layers to offload to GPU")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    args = parser.parse_args()

    print(f"System: {platform.platform()}")
    gpu, vram = get_gpu_metrics()
    if vram > 0:
        print(f"GPU Detected: Utilization {gpu}%, VRAM Used {vram}MB")
    else:
        print("Warning: No NVIDIA GPU detected via nvidia-smi. Benchmark may run on CPU.")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pulled = get_pulled_models()
    if args.model:
        models = [m for m in pulled if args.model in m]
    else:
        models = [m for m in pulled if any(a in m for a in ALLOWED_MODELS)]

    if not models:
        print("No matching models found.")
        return

    print(f"Benchmarking models: {models}")
    results = []
    for model in tqdm(models, desc="Overall Progress"):
        results.append(run_benchmark(model, run_dir, args))
        time.sleep(DEFAULT_COOLDOWN_SEC)

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [f"# Tic-Tac-Toe Benchmark Report {run_id}\n\n"]
    md.append(f"- Date: {time.ctime()}\n")
    md.append(f"- System: {platform.platform()}\n\n")

    md.append("## Resource Usage (Min / Max / Avg)\n\n")
    md.append("| Model | CPU (%) | GPU (%) | VRAM (MB) | DRAM (MB) |\n")
    md.append("|---|---|---|---|---|\n")
    for r in results:
        def fmt(k): return f"{r.get(k+'_min',0.0):.1f} / {r.get(k+'_max',0.0):.1f} / {r.get(k+'_avg',0.0):.1f}"
        md.append(f"| `{r['model']}` | {fmt('cpu')} | {fmt('gpu')} | {fmt('vram')} | {fmt('dram')} |\n")

    md.append("\n## Performance Metrics\n\n")
    md.append("| Model | Avg Tokens/s | TTFT (s) | Total Time (s) | Total Tokens Used |\n")
    md.append("|---|---|---|---|---|\n")
    for r in results:
        md.append(f"| `{r['model']}` | {r.get('avg_tokens_per_sec', 0):.2f} | {r.get('ttft', 0.0):.3f} | {r.get('total_time_sec', 0):.2f} | {r.get('total_tokens_used', 0)} |\n")

    (run_dir / "REPORT.md").write_text("".join(md), encoding="utf-8")
    print(f"\n✅ Benchmark complete. Results saved to: {run_dir}")

    # Console Summary Table
    print("\n" + "="*100)
    print(f"{'Model':<30} | {'T/s':<6} | {'TTFT':<6} | {'Time':<8} | {'Tokens':<8} | {'GPU Avg':<8} | {'VRAM Max':<8}")
    print("-" * 100)
    for r in results:
        print(f"{r['model']:<30} | {r.get('avg_tokens_per_sec', 0):<6.2f} | {r.get('ttft', 0.0):<6.2f} | {r.get('total_time_sec', 0):<8.1f} | {r.get('total_tokens_used', 0):<8} | {r.get('gpu_avg', 0.0):<8.1f} | {r.get('vram_max', 0.0):<8.0f}")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
