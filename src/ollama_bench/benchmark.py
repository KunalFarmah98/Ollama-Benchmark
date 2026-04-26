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
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
except Exception as e:
    raise SystemExit(
        "Missing dependency: python-dotenv.\n"
        "Install it in your venv: pip install python-dotenv\n"
        f"Original error: {e}"
    )


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


def main(argv: list[str] | None = None) -> int:
    _load_env()
    impl_path = _find_impl_path()

    if argv is None:
        argv = sys.argv[1:]

    # Try import-and-call main() if present
    try:
        impl_mod = _import_impl_module(impl_path)
        impl_main = getattr(impl_mod, "main", None)
        if callable(impl_main):
            try:
                rc = impl_main(argv)
                return int(rc) if rc is not None else 0
            except TypeError:
                rc = impl_main()
                return int(rc) if rc is not None else 0
    except Exception:
        pass

    # Fallback: run file as __main__
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(impl_path)] + argv
        runpy.run_path(str(impl_path), run_name="__main__")
        return 0
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
