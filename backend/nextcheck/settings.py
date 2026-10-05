from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Settings:
    root: Path = ROOT
    artifacts: Path = Path(os.environ.get('NEXTCHECK_ARTIFACTS', str(ROOT / 'artifacts')))
    seed: int = 20261004
