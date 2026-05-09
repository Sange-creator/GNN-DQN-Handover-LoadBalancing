#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder for publication figure generation.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(f"Figure generation is not implemented yet for {args.run_dir}; run training/evaluation first.")


if __name__ == "__main__":
    main()
