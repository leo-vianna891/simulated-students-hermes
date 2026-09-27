"""Minimal project commands."""

from __future__ import annotations

import argparse
import importlib.util
import platform
from pathlib import Path


def doctor() -> None:
    """Print the local Python and optional ML dependency status."""
    print(f"Python: {platform.python_version()}")
    for package in ("torch", "transformers", "peft", "yaml"):
        status = "installed" if importlib.util.find_spec(package) else "not installed"
        print(f"{package}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sim-student")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("doctor", help="inspect the local environment")

    train = commands.add_parser("train", help="train the Llama 3.2 student adapter")
    train.add_argument("dataset_root", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    train.add_argument("--model-id")
    train.add_argument("--model-revision")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--max-train-samples", type=int)
    train.add_argument("--max-validation-samples", type=int)

    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
        return
    if args.command == "train":
        from simulated_students.training import train_llama

        train_llama(
            args.dataset_root,
            args.output_dir,
            config_path=args.config,
            model_id_override=args.model_id,
            model_revision=args.model_revision,
            max_steps=args.max_steps,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
        )
        return
    parser.print_help()
