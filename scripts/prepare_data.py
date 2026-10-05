"""Download/extract official UCI 447 and save row-local features and frozen splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from nextcheck.data.download import download_archive, extract_required, sha256_file  # noqa: E402
from nextcheck.data.extract import prepare  # noqa: E402
from nextcheck.data.split import save_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Existing directory containing original sensor text files")
    parser.add_argument("--artifacts", type=Path, default=Path(os.getenv("NEXTCHECK_ARTIFACTS", ROOT / "artifacts")))
    args = parser.parse_args()
    output = args.artifacts / "data"
    output.mkdir(parents=True, exist_ok=True)
    if args.raw_dir is None:
        archive = download_archive(output / "uci447.zip")
        extracted = extract_required(output / "uci447.zip", output / "raw")
        raw = output / "raw"
    else:
        archive = None
        extracted = None
        raw = args.raw_dir
    metadata = prepare(raw, output)
    if archive is not None:
        metadata["archive"] = archive
        metadata["extraction"] = extracted
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    with np.load(output / "features.npz") as data:
        manifest = json.loads((ROOT / "demo_manifest.json").read_text(encoding="utf-8"))
        expected = [name for panel in manifest["panels"] for name in panel["features"]]
        if data["feature_names"].tolist() != expected:
            raise ValueError("Extracted feature order disagrees with demo manifest")
        splits = save_splits(data["y"], output / "splits.json")
    splits["metadata"]["features_sha256"] = metadata["features_sha256"]
    splits["metadata"]["manifest_sha256"] = sha256_file(ROOT / "demo_manifest.json")
    (output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(json.dumps({"rows": metadata["rows"], "features": metadata["features"], "features_sha256": metadata["features_sha256"], "selected_counts": splits["metadata"]["selected_counts"], "unused_count": splits["metadata"]["unused_count"]}, indent=2))


if __name__ == "__main__":
    main()
