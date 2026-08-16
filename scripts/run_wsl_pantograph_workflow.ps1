param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$LinuxUser = "lean",
    [string]$LinuxProject = "/home/lean/projects/formal-theorem-prover",
    [string]$WindowsRoot = "E:\python_project",
    [string]$HfEndpoint = "https://hf-mirror.com",
    [switch]$InstallSystemPackages
)

$ErrorActionPreference = "Stop"

Write-Host "Windows user:"
whoami

Write-Host "`nWSL distributions:"
wsl --list --verbose

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "/mnt/e/python_project/outputs/linux_pantograph_$runId"
$script = "/mnt/e/python_project/scripts/linux_pantograph_workflow.sh"

Write-Host "`nRunning Linux Pantograph workflow..."
Write-Host "Distro: $Distro"
Write-Host "Linux user: $LinuxUser"
Write-Host "Linux project: $LinuxProject"
Write-Host "Log dir: $logDir"
Write-Host "HF endpoint: $HfEndpoint"

if ($InstallSystemPackages) {
    Write-Host "`nInstalling required WSL system packages as root..."
    wsl -d $Distro -u root -e bash -lc "apt-get update && apt-get install -y python3-venv python3-pip"
}

wsl -d $Distro -u $LinuxUser -e bash -lc "WINDOWS_ROOT='/mnt/e/python_project' LINUX_PROJECT='$LinuxProject' RUN_ID='$runId' LOG_DIR='$logDir' HF_ENDPOINT='$HfEndpoint' bash '$script'"

Write-Host "`nWorkflow finished."
Write-Host "Windows result dir: E:\python_project\outputs\linux_pantograph_$runId"
