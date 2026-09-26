"""Project maintenance command line interface."""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Annotated

import typer

from simulated_students.artifacts import RunLayout, artifact_root, git_commit, project_root
from simulated_students.config import load_yaml
from simulated_students.dataset import write_audit_report

app = typer.Typer(no_args_is_help=True, help="Simulated-students research utilities.")


@app.command()
def doctor() -> None:
    """Print non-secret local readiness information."""
    root = artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(project_root()),
        "artifact_root": str(root),
        "artifact_free_gib": round(usage.free / 2**30, 2),
        "git_commit": git_commit(),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("validate-config")
def validate_config(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Verify that a YAML file is a non-empty mapping."""
    config = load_yaml(path)
    typer.echo(json.dumps({"path": str(path), "top_level_keys": sorted(config)}, indent=2))


@app.command("init-run")
def init_run(
    run_id: Annotated[str, typer.Argument(help="Stable lowercase experiment run identifier")],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Resolved config")],
) -> None:
    """Create an ignored run layout and write its initial receipt."""
    layout = RunLayout.create(run_id)
    receipt = layout.write_receipt(config)
    typer.echo(json.dumps(receipt, indent=2, sort_keys=True))


@app.command("audit-dataset")
def audit_dataset_command(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    revision: Annotated[str, typer.Option(help="Immutable Hugging Face dataset revision")],
    output: Annotated[
        Path,
        typer.Option(help="Aggregate JSON report path"),
    ] = Path("results/summary/dataset-audit.json"),
) -> None:
    """Audit the pinned Eedi CSV snapshot without exposing dialogue contents."""
    report = write_audit_report(root, revision, output)
    typer.echo(
        json.dumps(
            {
                "output": str(output.resolve()),
                "train_dialogues": report["splits"]["train"]["unique_dialogues"],
                "test_dialogues": report["splits"]["test"]["unique_dialogues"],
            },
            indent=2,
            sort_keys=True,
        )
    )
