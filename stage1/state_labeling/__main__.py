"""Read-only review of the state CSV against the R1/B1 proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rules import review_dictionary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dictionary", type=Path)
    parser.add_argument("--details", action="store_true", help="include all unique CSV rows")
    args = parser.parse_args()
    report = review_dictionary(args.dictionary)
    if not args.details:
        report.pop("review_rows")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready_for_joint_review"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
