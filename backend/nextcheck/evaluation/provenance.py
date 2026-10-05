"""Small, inspectable run metadata helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(target)


def environment_info() -> dict[str, str]:
    import numpy
    import sklearn

    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": numpy.__version__, "scikit_learn": sklearn.__version__,
    }
