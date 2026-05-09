"""Download and filter OpenCellID data for Nepal LTE cells.

Usage:
    # With API token (register free at https://opencellid.org):
    python3 tools/download_opencellid.py --token YOUR_API_TOKEN

    # Or if you already downloaded the CSV manually:
    python3 tools/download_opencellid.py --input path/to/cell_towers.csv

The script filters for Nepal (MCC=429) LTE cells and saves region-specific files.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from handover_gnn_dqn.topology import REGIONS, latlon_to_xy


def download_nepal_cells(token: str, output_path: Path) -> None:
    """Download Nepal cell data from OpenCellID API."""
    import urllib.request

    url = f"https://opencellid.org/cell/getInArea?key={token}&BBOX=26.3,80.0,30.5,88.2&format=csv&radio=LTE&mcc=429&limit=10000"
    print(f"Downloading from OpenCellID...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: Download manually from https://opencellid.org/downloads.php")
        print("Select: Country=Nepal, Radio=LTE, Format=CSV")
        print(f"Save the file as: {output_path}")
        sys.exit(1)


def filter_region(input_path: Path, region: str, output_path: Path) -> int:
    """Filter OpenCellID CSV for a specific region."""
    reg = REGIONS[region]
    rows = []

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                radio = row.get("radio", "")
                if radio != "LTE":
                    continue
                lat = float(row.get("lat", 0))
                lon = float(row.get("lon", 0))
                if reg.lat_min <= lat <= reg.lat_max and reg.lon_min <= lon <= reg.lon_max:
                    rows.append(row)
            except (ValueError, KeyError):
                continue

    if not rows:
        print(f"  No LTE cells found for {region} in {input_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lats = np.array([float(r["lat"]) for r in rows])
    lons = np.array([float(r["lon"]) for r in rows])
    x, y = latlon_to_xy(lats, lons)
    positions = np.column_stack((x, y))

    pos_path = output_path.with_suffix(".positions.csv")
    np.savetxt(pos_path, positions, delimiter=",", header="x_m,y_m", comments="")
    print(f"  {region}: {len(rows)} LTE cells → {output_path}")
    print(f"  Positions (meters): {pos_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Download/filter OpenCellID Nepal data")
    parser.add_argument("--token", type=str, help="OpenCellID API token")
    parser.add_argument("--input", type=Path, help="Path to already-downloaded CSV")
    args = parser.parse_args()

    data_dir = Path("data/opencellid")
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / "nepal_lte_raw.csv"

    if args.input:
        raw_path = args.input
    elif args.token:
        download_nepal_cells(args.token, raw_path)
    elif not raw_path.exists():
        print("No data found. Options:")
        print("  1. Register at https://opencellid.org and run:")
        print("     python3 tools/download_opencellid.py --token YOUR_TOKEN")
        print()
        print("  2. Download manually from https://opencellid.org/downloads.php")
        print("     (Select Nepal, LTE, CSV) and run:")
        print("     python3 tools/download_opencellid.py --input path/to/file.csv")
        print()
        print("  3. Use synthetic topology instead (already works without real data):")
        print("     python3 -c \"from handover_gnn_dqn.topology import load_topology; print(load_topology('pokhara', 50)[0].shape)\"")
        sys.exit(0)

    print(f"\nFiltering {raw_path} for Nepal regions...")
    total = 0
    for region in REGIONS:
        output = data_dir / f"{region}_cells.csv"
        total += filter_region(raw_path, region, output)

    print(f"\nTotal: {total} LTE cells across {len(REGIONS)} regions")


if __name__ == "__main__":
    main()
