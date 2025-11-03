#!/usr/bin/env python3
"""Create and populate a uv-managed virtual environment.

This script mirrors the behaviour of the historical ``setup_env.sh`` helper while
being portable across platforms, including Windows.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a uv virtual environment and install requirements.")
    parser.add_argument(
        "python_version",
        nargs="?",
        default="3.14",
        help="Python version to use when creating the virtual environment (default: 3.14)",
    )
    parser.add_argument(
        "venv_path",
        nargs="?",
        default=".venv",
        help="Target directory for the virtual environment (default: .venv)",
    )
    return parser.parse_args()


def ensure_uv_available() -> None:
    if shutil.which("uv") is None:
        print("The uv package manager is required. Please install it from https://github.com/astral-sh/uv.", file=sys.stderr)
        sys.exit(1)


def create_virtualenv(python_version: str, venv_path: Path) -> None:
    print(f"Creating virtual environment with Python {python_version} at {venv_path}")
    subprocess.run(["uv", "venv", "--python", python_version, str(venv_path)], check=True)


def venv_python_path(venv_path: Path) -> Path:
    candidates = [
        venv_path / "bin" / "python",
        venv_path / "Scripts" / "python",
        venv_path / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def install_requirements(python_executable: Path) -> None:
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python_executable), "-r", "requirements.txt"],
        check=True,
    )


def activation_message(venv_path: Path) -> str:
    posix_activate = f"source {venv_path.as_posix()}/bin/activate"
    windows_cmd = f"{venv_path}\\Scripts\\activate"
    windows_powershell = f"{venv_path}\\Scripts\\Activate.ps1"
    return (
        "Environment ready. Activate it with:\n"
        f"  POSIX shells: {posix_activate}\n"
        f"  cmd.exe: {windows_cmd}\n"
        f"  PowerShell: {windows_powershell}"
    )


def main() -> None:
    args = parse_args()
    venv_path = Path(args.venv_path)

    ensure_uv_available()
    create_virtualenv(args.python_version, venv_path)

    python_executable = venv_python_path(venv_path)
    install_requirements(python_executable)

    print(activation_message(venv_path))


if __name__ == "__main__":
    main()
