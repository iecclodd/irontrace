"""Model setup must fail closed and keep offline cache checks offline."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_model.py"
SPEC = importlib.util.spec_from_file_location("prepare_model", SCRIPT)
prepare_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_model)


def test_missing_cache_check_does_not_download(tmp_path, monkeypatch):
    import huggingface_hub

    def forbidden(**kwargs):
        pytest.fail("Offline cache check attempted a download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    with pytest.raises(FileNotFoundError, match="not cached"):
        prepare_model.prepare_checkpoint(tmp_path / prepare_model.EXPECTED_CHECKPOINT, check_only=True)


def test_mismatched_cached_weights_are_preserved_and_rejected(tmp_path):
    checkpoint = tmp_path / prepare_model.EXPECTED_CHECKPOINT
    checkpoint.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        prepare_model.prepare_checkpoint(checkpoint, check_only=True)
    assert checkpoint.read_bytes() == b"corrupt checkpoint"


def test_exact_download_is_verified_and_reused_offline(tmp_path, monkeypatch):
    import huggingface_hub

    payload = b"explicit test fixture, not model weights"
    monkeypatch.setattr(prepare_model, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(payload)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    checkpoint = tmp_path / prepare_model.EXPECTED_CHECKPOINT
    result = prepare_model.prepare_checkpoint(checkpoint)
    assert result["checkpoint_verified"] and not result["already_cached"]
    assert result["checkpoint"] == prepare_model.EXPECTED_CHECKPOINT
    assert str(tmp_path) not in str(result)
    assert calls == [{"repo_id": "Prior-Labs/tabpfn_3_5", "filename": prepare_model.EXPECTED_CHECKPOINT, "local_dir": tmp_path}]
    assert prepare_model.prepare_checkpoint(checkpoint, check_only=True)["already_cached"]
    assert len(calls) == 1


def test_unexpected_checkpoint_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unexpected checkpoint"):
        prepare_model.prepare_checkpoint(tmp_path / "another-model.safetensors")
