#!/usr/bin/env python3
"""CLI wrapper for Odysseus Doctor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics.doctor import format_json_report, format_text_report, run_doctor_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only Odysseus install diagnostics.")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    parser.add_argument("--no-network", action="store_true", help="skip HTTP reachability checks")
    parser.add_argument("--no-docker", action="store_true", help="skip Docker checks")
    parser.add_argument("--timeout", type=float, default=3.0, help="subprocess/network timeout in seconds")
    args = parser.parse_args(argv)

    checks = run_doctor_checks(
        root=ROOT,
        include_network=not args.no_network,
        include_docker=not args.no_docker,
        timeout=max(args.timeout, 0.1),
    )
    print(format_json_report(checks) if args.json else format_text_report(checks))
    return 1 if any(check.status == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
