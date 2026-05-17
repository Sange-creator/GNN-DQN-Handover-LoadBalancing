#!/usr/bin/env python3
"""Augment raw Pokhara drive-test CSV with derived features needed for
simulator calibration and shadow replay evaluation.

Outputs the following new columns:
  - gps_speed_smoothed   : Haversine-based speed (m/s), 5-sample rolling median
  - region               : 4-region segmentation (less_dense / dense_urban /
                           highway / shadow) per calibration report §6.1
  - rsrp_trend           : 5-sample linear regression slope (dB/sample)
  - rsrq_trend           : 5-sample linear regression slope (dB/sample)
  - a3_trigger           : 1 if best_neighbor_rsrp − rsrp ≥ 3 dB sustained 2 s
  - ho_ml_predicted      : 1 if GradientBoosting predicts a cell-ID change in
                           the next 5 s
  - ho_ml_score          : Probability score from the same classifier

The script is deterministic for a fixed random_state.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def gps_speed(df: pd.DataFrame) -> pd.Series:
    """Per-route Haversine speed, 5-sample rolling median to suppress jitter.

    Timestamps in the source CSV have 1-second resolution and contain many
    duplicates (multiple samples per second). dt is clamped to ≥1.0 s to avoid
    division by sub-second deltas which produce nonsense speeds. Final speeds
    are clipped to [0, 40] m/s (144 km/h ceiling).
    """
    out = np.zeros(len(df), dtype=float)
    for route, sub in df.groupby("route", sort=False):
        idx = sub.index.to_numpy()
        lat = sub["latitude"].to_numpy()
        lon = sub["longitude"].to_numpy()
        ts = pd.to_datetime(sub["timestamp"]).astype("int64").to_numpy() / 1e9
        speeds = np.zeros(len(sub))
        for i in range(1, len(sub)):
            dt = max(ts[i] - ts[i - 1], 1.0)
            d = haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
            speeds[i] = min(d / dt, 40.0)
        s = pd.Series(speeds).rolling(window=5, center=True, min_periods=1).median().to_numpy()
        out[idx] = s
    return pd.Series(out, index=df.index)


def classify_region(row) -> str:
    if row["rsrp"] < -105 and row["snr"] < 5:
        return "shadow"
    if row.get("gps_speed_smoothed", 0.0) > 8.0:
        return "highway"
    if row["num_neighbors"] >= 4 and row["rsrp"] > -100:
        return "dense_urban"
    return "less_dense"


def signal_trend(values: np.ndarray, window: int = 5) -> np.ndarray:
    """5-sample linear regression slope per index."""
    n = len(values)
    out = np.zeros(n)
    x = np.arange(window) - window / 2.0
    for i in range(window - 1, n):
        y = values[i - window + 1 : i + 1]
        # slope = cov(x,y)/var(x)
        out[i] = float(np.polyfit(x, y, 1)[0])
    return out


def a3_synthetic_triggers(df: pd.DataFrame, offset_db: float = 3.0, ttt_samples: int = 2) -> pd.Series:
    """Standard 3GPP A3 rule: best_neighbor_rsrp − serving_rsrp ≥ offset sustained for ttt_samples."""
    cond = (df["best_neighbor_rsrp"] - df["rsrp"]) >= offset_db
    sustained = cond.rolling(window=ttt_samples, min_periods=ttt_samples).sum() >= ttt_samples
    return sustained.fillna(False).astype(int)


def ml_future_ho_labels(df: pd.DataFrame, horizon: int = 5):
    """Train a GradientBoosting classifier on UE-side features to predict
    cell_id change within the next `horizon` samples. Used as dense supervised
    label since only 6 raw cell-ID changes exist in the full corpus."""
    from sklearn.ensemble import GradientBoostingClassifier

    # Per-route label generation to avoid cross-route leakage
    y = np.zeros(len(df), dtype=int)
    for route, sub in df.groupby("route", sort=False):
        idx = sub.index.to_numpy()
        cell = sub["cell_id"].to_numpy()
        for i, gi in enumerate(idx):
            end = min(i + horizon, len(cell) - 1)
            if cell[i] != cell[end]:
                y[gi] = 1

    features = [
        "rsrp", "rsrq", "snr", "num_neighbors", "best_neighbor_rsrp",
        "n1_rsrp", "n1_rsrq", "n2_rsrp", "n2_rsrq", "n3_rsrp", "n3_rsrq",
        "rsrp_trend", "rsrq_trend", "load_proxy_rsrq", "gps_speed_smoothed",
    ]
    X = df[features].fillna(0.0).to_numpy()

    # Time-respecting 75/25 split (no shuffle): match calibration report §4.1
    cut = int(0.75 * len(df))
    if y[:cut].sum() < 5:
        # Not enough positives for training — fall back to a3_trigger labels
        return df["a3_trigger"].astype(int), df["a3_trigger"].astype(float)

    pos_w = max(1.0, (y[:cut] == 0).sum() / max((y[:cut] == 1).sum(), 1))
    sample_weight = np.where(y[:cut] == 1, pos_w, 1.0)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    clf.fit(X[:cut], y[:cut], sample_weight=sample_weight)

    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pd.Series(pred, index=df.index), pd.Series(proba, index=df.index)


def augment(in_path: Path, out_path: Path) -> dict:
    df = pd.read_csv(in_path)
    df = df.copy()

    # 1. GPS-derived speed (the device-reported speed is 0 for 76% of rows)
    df["gps_speed_smoothed"] = gps_speed(df)

    # 2. Region segmentation (4-region tree)
    df["region"] = df.apply(classify_region, axis=1)

    # 3. Recompute RSRP/RSRQ trend (existing columns are all zero per report §2.4)
    df["rsrp_trend"] = signal_trend(df["rsrp"].astype(float).to_numpy())
    df["rsrq_trend"] = signal_trend(df["rsrq"].astype(float).to_numpy())

    # 4. Synthetic A3 trigger labels
    df["a3_trigger"] = a3_synthetic_triggers(df)

    # 5. ML-predicted future-5s handover labels (dense supervised signal)
    ho_pred, ho_score = ml_future_ho_labels(df)
    df["ho_ml_predicted"] = ho_pred
    df["ho_ml_score"] = ho_score

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return {
        "rows": int(len(df)),
        "regions": dict(df["region"].value_counts().to_dict()),
        "speed_nonzero_frac": float((df["gps_speed_smoothed"] > 0.1).mean()),
        "a3_trigger_count": int(df["a3_trigger"].sum()),
        "ho_ml_predicted_count": int(df["ho_ml_predicted"].sum()),
        "out_columns": list(df.columns),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    args = ap.parse_args()

    stats = augment(args.in_path, args.out_path)
    print("=== Augmentation summary ===")
    for k, v in stats.items():
        if k == "out_columns":
            print(f"  {k}: {len(v)} columns ({', '.join(v[-7:])} ...)")
        else:
            print(f"  {k}: {v}")
    print(f"\nWrote: {args.out_path}")


if __name__ == "__main__":
    main()
