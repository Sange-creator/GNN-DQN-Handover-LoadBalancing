#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a drive-test CSV and mark PRB as unavailable.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    lower_cols = {c.lower(): c for c in df.columns}
    for required in ["rsrp", "rsrq"]:
        if not any(required in c for c in lower_cols):
            raise ValueError(f"Missing required drive-test field containing {required!r}")
    df["prb_utilization"] = 0.0
    df["prb_available"] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote UE-only processed drive-test CSV: {args.output}")


if __name__ == "__main__":
    main()
