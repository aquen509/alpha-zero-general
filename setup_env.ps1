#!/usr/bin/env pwsh
# Exit on error
$ErrorActionPreference = "Stop"

param (
    [string] $PythonVersion = "3.14",
    [string] $VenvPath = ".venv"
)

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "The uv package manager is required. Please install it from https://github.com/astral-sh/uv."
    exit 1
}

Write-Host "Creating virtual environment with Python $PythonVersion at $VenvPath"

# Create virtual environment
uv venv --python $PythonVersion $VenvPath

# Activate the virtual environment
$activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Activation script not found at $activateScript"
    exit 1
}
. $activateScript

# Install dependencies
uv pip install -r requirements.txt

Write-Host "Environment ready. Activate it with:"
Write-Host ". $activateScript"