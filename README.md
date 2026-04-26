# Ollama Model Benchmark Project

## Setup (Windows PowerShell)
1) Copy env template:
   - `copy .env.example .env`
2) Bootstrap:
   - `.\scripts\bootstrap.ps1`
3) Run:
   - `.\scripts\run.ps1`

## Notes
- `.env` is loaded by python-dotenv (`load_dotenv()`), so secrets stay out of git.
- Results are written to `ollama_benchmarks/` by the benchmark script.
``