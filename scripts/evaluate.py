"""Select policies on development rows, freeze choices, then score final test rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "backend"))

from nextcheck.evaluation.benchmark import BUDGETS, LAMBDAS  # noqa: E402
from nextcheck.evaluation.runner import evaluate_frozen  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--utility-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--utility-limit", type=int)
    parser.add_argument("--selection-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--budgets", type=float, nargs="+")
    parser.add_argument("--lambdas", type=float, nargs="+")
    parser.add_argument("--skip-null", action="store_true", help="Mark null task omitted from this run")
    parser.add_argument("--smoke", action="store_true", help="Explicit small run: 64/4/4/4 rows, budgets 0/6/11 and lambda .02")
    args = parser.parse_args()
    if args.smoke:
        args.context_limit = args.context_limit or 64
        args.utility_limit = args.utility_limit or 4
        args.selection_limit = args.selection_limit or 4
        args.test_limit = args.test_limit or 4
    budgets = tuple(args.budgets) if args.budgets is not None else ((0.0, 6.0, 11.0) if args.smoke else BUDGETS)
    lambdas = tuple(args.lambdas) if args.lambdas is not None else ((0.02,) if args.smoke else LAMBDAS)
    if not budgets or any(b < 0 for b in budgets) or not lambdas or any(l < 0 for l in lambdas):
        parser.error("budgets and lambdas must be nonnegative and nonempty")
    try:
        result = evaluate_frozen(REPOSITORY, data_dir=args.data_dir, utility_dir=args.utility_dir,
            output_dir=args.output_dir, context_limit=args.context_limit,
            utility_limit=args.utility_limit, selection_limit=args.selection_limit,
            test_limit=args.test_limit, budgets=budgets, lambdas=lambdas,
            include_null=not args.skip_null)
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__, "blocker": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
        "scope": result["scope"], "selected_defaults": result["selected_defaults"],
        "null_task": result["null_task"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
