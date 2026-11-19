# Detached launcher for the sweep (headline, leakage-free feature set).
#
#   powershell -ExecutionPolicy Bypass -File launch_sweep.ps1 -Person Ahmed        # your share
#   powershell -ExecutionPolicy Bypass -File launch_sweep.ps1                       # all 14 models
#   powershell -ExecutionPolicy Bypass -File launch_sweep.ps1 -Models segformer_laca # a subset
#
# The process is started with Start-Process so it is not a child of the shell
# that launched it. Re-running is safe: finished models (results.json) are
# skipped and an interrupted model resumes from its last completed epoch.
# Progress: outputs\sweep.log, outputs\log_<model>__<feature_set>.txt,
# or `python status.py`.

param(
    [string[]]$Models = @(),
    [int]$Laptop = 0,
    [string]$Person = "",
    [string]$DataRoot = ""
)

# Four shares balanced on measured RTX 3050 (4 GB) training time per epoch
# (minutes, micro-batches from run_all_local.py). Each share has one heavy
# backbone plus lighter ones and totals ~10 GPU-min per epoch, i.e. roughly
# 3 - 3.5 h per laptop for 15 epochs + XAI. SegFormer-B5 alone costs ~11
# min/epoch (27% of the whole sweep) and cannot be split, so it is a share
# on its own.
#   1 Jana         segformer_laca 11.1                                          = 11.1
#   2 Malak        swinv2_tab 4.0, maxvit 3.3, davit 1.4, convnext_bilinear 1.3 = 10.0
#   3 Ahmed        swinv2_gat 3.8, gnn_maxvit 2.9, vit_early 2.7, unet 0.8      = 10.2
#   4 Abdelrahman  biovil 2.8, hybrid_vit 2.6, dinov2 1.5, effnetv2 1.4,
#                  convnext_bicross 1.3                                         =  9.6
$shares = @{
    1 = @("segformer_laca");
    2 = @("swinv2_tabtransformer", "maxvit_crossattn", "davit_gated", "convnext_bilinear");
    3 = @("swinv2_gat", "gnn_maxvit", "vit_early_fusion", "unet_crossattn");
    4 = @("biovil_lcsa", "hybrid_vit_mlp", "dinov2_crossattn", "effnetv2_crossattn", "convnext_bicross")
}
$people = @{ "jana" = 1; "malak" = 2; "ahmed" = 3; "abdelrahman" = 4 }
if ($Person) {
    $key = $Person.ToLower()
    if (-not $people.ContainsKey($key)) { throw "-Person must be Ahmed, Jana, Malak or Abdelrahman" }
    $Laptop = $people[$key]
}
if ($Laptop -gt 0) {
    if (-not $shares.ContainsKey($Laptop)) { throw "-Laptop must be 1, 2, 3 or 4" }
    $Models = $shares[$Laptop]
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out  = Join-Path $here "outputs"
New-Item -ItemType Directory -Force $out | Out-Null

# model-weight caches: keep the original location on this machine, otherwise
# fall back to the user profile defaults
if (Test-Path "D:\ml-cache") {
    $env:HF_HOME = "D:\ml-cache\hf"
    $env:TORCH_HOME = "D:\ml-cache\torch"
}
$env:PYTHONIOENCODING = "utf-8"
$env:TOKENIZERS_PARALLELISM = "false"
$env:TF_CPP_MIN_LOG_LEVEL = "3"

$py = (Get-Command python).Source
$pyArgs = @("-u", "`"$(Join-Path $here 'run_all_local.py')`"", "--epochs", "15")
if ($DataRoot) { $pyArgs += @("--data-root", "`"$DataRoot`"") }
if ($Models.Count -gt 0) { $pyArgs += @("--models") + $Models }

Start-Process -FilePath $py -ArgumentList $pyArgs -WorkingDirectory $here `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $out "sweep.log") `
    -RedirectStandardError  (Join-Path $out "sweep.err.log")
$which = if ($Models.Count -gt 0) { $Models -join ", " } else { "all models" }
Write-Output "sweep started ($which); log: $out\sweep.log"
