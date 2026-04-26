
# Bootstrap a clean venv + install dependencies + pin them
# If PowerShell blocks scripts, you can temporarily allow in this session:
# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

$ErrorActionPreference = "Stop"

if (!(Test-Path ".\.venv")) {
  python -m venv .venv
}

# Activate venv (PowerShell activation script) [2](https://stackoverflow.com/questions/1365081/virtualenv-in-powershell)
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.in

# --- Optional: CUDA-aware PyTorch install (uses nvidia-smi if available) ---
function Get-CudaVersionFromNvidiaSmi {
  try {
    $out = & nvidia-smi 2>$null
    if (!$out) { return $null }
    $m = [regex]::Match($out, "CUDA Version:\s*([0-9]+)\.([0-9]+)")
    if ($m.Success) { return "$($m.Groups[1].Value).$($m.Groups[2].Value)" }
    return $null
  } catch { return $null }
}

$cuda = Get-CudaVersionFromNvidiaSmi
$indexUrl = "https://download.pytorch.org/whl/cpu"
if ($cuda) {
  # Choose a supported wheel index similar to PyTorch's official index-url approach [9](https://docs.pytorch.org/get-started/locally/)[10](https://pytorch.org/get-started/previous-versions/)
  # Prefer newest <= driver
  $majorMinor = $cuda.Split(".")
  $maj = [int]$majorMinor[0]
  $min = [int]$majorMinor[1]
  if ($maj -gt 13 -or ($maj -eq 13 -and $min -ge 0)) { $indexUrl = "https://download.pytorch.org/whl/cu130" }
  elseif ($maj -gt 12 -or ($maj -eq 12 -and $min -ge 8)) { $indexUrl = "https://download.pytorch.org/whl/cu128" }
  elseif ($maj -gt 12 -or ($maj -eq 12 -and $min -ge 6)) { $indexUrl = "https://download.pytorch.org/whl/cu126" }
  else { $indexUrl = "https://download.pytorch.org/whl/cpu" }
}

Write-Host "Installing PyTorch from index: $indexUrl"
python -m pip install torch torchvision torchaudio --index-url $indexUrl

# Pin exact versions (pip freeze outputs requirements format) [3](https://pip.pypa.io/en/stable/cli/pip_freeze/)
python -m pip freeze > requirements.txt

Write-Host "✅ Done. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "✅ Run with: scripts\run.ps1"
