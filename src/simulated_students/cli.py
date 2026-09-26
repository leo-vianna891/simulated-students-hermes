"""Minimal project diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import platform


def doctor() -> None:
    """Print the local Python and optional ML dependency status."""
    print(f"Python: {platform.python_version()}")
    for package in ("torch", "transformers", "trl", "peft"):
        status = "installed" if importlib.util.find_spec(package) else "not installed"
        print(f"{package}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sim-student")
    parser.add_argument("command", choices=["doctor"], nargs="?")
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    else:
        parser.print_help()
