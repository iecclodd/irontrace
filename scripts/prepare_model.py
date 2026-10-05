"""Fetch the exact official checkpoint into the cache used by IRONTRACE.

Review the upstream model terms and obtain access yourself before running this
command. Hugging Face's existing local authentication is used; no terms are
accepted here. A cached checkpoint is verified without a network request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from nextcheck.data.download import sha256_file  # noqa: E402
from nextcheck.model.tabpfn35 import EXPECTED_CHECKPOINT  # noqa: E402

REPOSITORY = "Prior-Labs/tabpfn_3_5"
EXPECTED_SHA256 = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"


def prepare_checkpoint(checkpoint: Path, *, check_only: bool = False) -> dict:
    """Download when absent, then verify bytes; do not imply inference readiness."""
    if checkpoint.name != EXPECTED_CHECKPOINT:
        raise ValueError(f"Unexpected checkpoint selected: {checkpoint.name}")
    existed = checkpoint.is_file()
    if not existed:
        if check_only:
            raise FileNotFoundError("Checkpoint is not cached; run without --check-only after obtaining model access.")
        from huggingface_hub import hf_hub_download

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(hf_hub_download(
            repo_id=REPOSITORY,
            filename=EXPECTED_CHECKPOINT,
            local_dir=checkpoint.parent,
        ))
        if downloaded.resolve() != checkpoint.resolve():
            raise ValueError("Download path did not match the model cache location")
    digest = sha256_file(checkpoint)
    if digest != EXPECTED_SHA256:
        raise ValueError("Checkpoint SHA-256 differs from the verified release. It was not overwritten; check the file and upstream revision before using it.")
    return {
        "checkpoint_verified": True,
        "source": f"https://huggingface.co/{REPOSITORY}",
        "checkpoint": checkpoint.name,
        "sha256": digest,
        "bytes": checkpoint.stat().st_size,
        "already_cached": existed,
        "next_step": "Run scripts/prepare_data.py, then scripts/doctor.py to verify real inference.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify a cached checkpoint without downloading")
    args = parser.parse_args()
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        model = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)
        checkpoint = Path(model.model_path).expanduser().resolve()
        print(json.dumps(prepare_checkpoint(checkpoint, check_only=args.check_only), indent=2))
        return 0
    except Exception as exc:
        # Do not print HTTP headers, auth values, or raw remote exception text.
        print(f"Model preparation failed ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, (ValueError, FileNotFoundError)):
            print(str(exc), file=sys.stderr)
        print("Review access at https://huggingface.co/Prior-Labs/tabpfn_3_5. If authentication is required, run .venv/Scripts/hf.exe auth login in your own terminal, then retry. Never paste tokens into chat or commit them. Check network access and available disk space if access is already granted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
