"""Fit one-step value and no-residual ablation from real development predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "backend"))

from nextcheck.evaluation.runner import prepare_utility  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--utility-limit", type=int)
    parser.add_argument("--selection-limit", type=int)
    parser.add_argument("--smoke", action="store_true", help="Explicit small run: 64 context, 4 utility and 4 selection rows")
    args = parser.parse_args()
    if args.smoke:
        args.context_limit = args.context_limit or 64
        args.utility_limit = args.utility_limit or 4
        args.selection_limit = args.selection_limit or 4
    try:
        result = prepare_utility(REPOSITORY, data_dir=args.data_dir, output_dir=args.output_dir,
            context_limit=args.context_limit, utility_limit=args.utility_limit,
            selection_limit=args.selection_limit)
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__, "blocker": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "scope": result["scope"],
        "counts": result["counts"], "tuning": result["tuning"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
