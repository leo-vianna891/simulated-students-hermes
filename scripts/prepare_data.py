"""Download the reference's annotated Eedi CSVs at the configured commit."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import yaml

from simulated_students.dataset import load_eedi_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))["data"]
    root = Path("artifacts/datasets/annotated") / data["revision"]
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{data['source_repo']}/{data['revision']}/data/annotated/eedi"
    for split in ("train", "val", "test"):
        path = root / f"{split}_{data['annotation_model']}.csv"
        temporary = path.with_suffix(".csv.part")
        try:
            with urllib.request.urlopen(f"{base}/{path.name}", timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    splits = load_eedi_splits(root)
    print(f"Annotated dataset: {root}")
    print(f"Train/validation/test: {len(splits.train)}/{len(splits.validation)}/{len(splits.test)}")


if __name__ == "__main__":
    main()
