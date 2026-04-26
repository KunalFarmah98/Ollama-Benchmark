"""
benchmark.py (wrapper)

Loads .env / env vars and then delegates to the "real" benchmark script
stored in the same folder (e.g., benchmark_impl.py).

How it delegates:
1) If the impl exposes a `main()` callable, we call it with passthrough argv.
2) Otherwise, we execute the impl as a script via runpy.run_path(..., run_name="__main__")

This keeps your project entrypoint clean:
    python -m ollama_bench
or
    python -m ollama_bench.benchmark --pin
"""

from __future__ import annotations

import sys
import runpy
import importlib.util
import subprocess
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
except Exception as e:
    raise SystemExit(
        "Missing dependency: python-dotenv.\n"
        "Install it in your venv: pip install python-dotenv\n"
        f"Original error: {e}"
    )

# List of allowed models
ALLOWED_MODELS = [
    "deepseek-coder-v2:16b",
    "gemma4:26b",
    "qwen3-coder:latest",
    "qwen2.5-coder:14b"
]

IMPL_CANDIDATES = (
    "benchmark_impl.py",   # recommended
    "benchmark_script.py",
    "benchmark_core.py",
    "bench_impl.py",
    "bench.py",
)

def _load_env() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _find_impl_path() -> Path:
    here = Path(__file__).resolve().parent
    for name in IMPL_CANDIDATES:
        p = here / name
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find the benchmark implementation script in:\n"
        f"  {here}\n\n"
        "Expected one of:\n"
        + "\n".join(f"  - {n}" for n in IMPL_CANDIDATES)
        + "\n\n"
        "Fix: Put your actual benchmark script in the same folder and name it "
        "`benchmark_impl.py` (recommended)."
    )

def _import_impl_module(impl_path: Path):
    module_name = "ollama_bench_impl"
    spec = importlib.util.spec_from_file_location(module_name, str(impl_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import implementation module from {impl_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        models = []
        for line in lines:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception as e:
        print(argparse.ArgumentTypeError(f"Failed to list Ollama models: {e}"))
        return []

def main(argv: list[str] | None = None) -> int:
    _load_env()
    impl_path = _find_impl_path()

    if argv is None:
        argv = sys.argv[1:]

    # Get all models currently in Ollama
    installed_models = _get_ollama_models()
    
    # Intersection of installed models and allowed models
    # We only want to run models that are BOTH installed and in our allowed list
    models_to_run = [m for m in installed_models if m in ALLOWED_MODELS]

    if not models_to_run:
        print("No allowed models found installed in Ollama.")
        return 0

    # Prepare base arguments (excluding any user-passed --model overrides)
    base_argv = [arg for arg in argv if not arg.startswith('--model=')]

    # Try to load implementation once to check for main()
    impl_mod = None
    impl_main = None
    try:
        impl_mod = _import_impl_module(impl_path)
        impl_main = getattr(impl_mod, "main", None)
    except Exception:
        pass

    for model in models_to_run:
        print(f"\n{'='*60}\nStarting benchmark for: {model}\n{'='*60}")
        
        # Construct argv for this specific model
        current_argv = base_argv + [f"--model={model}"]
        
        try:
            if callable(impl_main):
                try:
                    impl_main(current_argv)
                except TypeError:
                    impl_main()
            else:
                # Fallback: run as script
                old_sys_argv = sys.argv[:]
                sys.argv = [str(impl_path)] + current_argv
                runpy.run_path(str(impl_path), run_name="__main__")
                sys.argv = old_sys_argv
        except Exception as e:
            print(f"Error running benchmark for {model}: {e}")
            continue

    return 0

if __name__ == "__main__":
    raise SystemExit(main())