$ErrorActionPreference = "Stop"

# Activate venv
.\.venv\Scripts\Activate.ps1

# Optional: also export .env variables into PowerShell session
# (python-dotenv already handles .env inside Python, so this is extra)
if (Test-Path ".\.env") {
  Get-Content ".\.env" | ForEach-Object {
    if ($_ -match "^\s*#" -or $_ -match "^\s*$") { return }
    $parts = $_ -split "=", 2
    if ($parts.Length -eq 2) {
      $name = $parts[0].Trim()
      $value = $parts[1].Trim()
      [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
  }
}

python -m ollama_bench