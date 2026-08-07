#!/usr/bin/env python3
"""
Download and plot temperature readings from a PicoTemp-style sensor device.

The device serves a form at http://<host>/ with a GET endpoint at
/download?from_ts=...&to_ts=... that returns a CSV with columns:
Time,T0,T1,T2,T3,T4 (newest record first).

Readings accumulate in a local data file (default: temperatures.csv).
On each run:
  - If the data file doesn't exist yet, it's created by downloading
    DEFAULT_FROM_TS..DEFAULT_TO_TS (overridable with --from/--to).
  - If it exists, only the missing tail is downloaded: from one second
    after the newest stored record, through one hour from now (to
    safely cover any clock skew), and merged in, newest first.

The plot itself only shows the most recent 48 hours of stored data by
default (override with --window-hours), regardless of how much history
has accumulated in the data file.

Usage:
    # Download/update from the device and plot
    python3 update_temperatures.py

    # Specify host, and the backfill range used only on first run
    python3 update_temperatures.py --host 10.0.0.64 --from "2026-08-03T18:00" --to "2026-08-04T18:00"

    # Skip downloading and just plot a CSV you already have
    python3 update_temperatures.py --csv data.csv

Run this on a machine that's actually on the same local network as the device.
"""

import argparse
import os
import sys
import urllib.request
import urllib.parse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_HOST = "10.0.0.64"
DEFAULT_FROM_TS = "2026-01-01T00:00"
DEFAULT_TO_TS = "2100-01-01T00:00"
DATA_CSV_PATH = "temperatures.csv"
PLOT_WINDOW_HOURS = 48


def download_csv(host: str, from_ts: str, to_ts: str, dest_path: str, timeout: float = 15.0) -> str:
    """Fetch the CSV from the device's /download endpoint and save it locally."""
    params = urllib.parse.urlencode({"from_ts": from_ts, "to_ts": to_ts})
    url = f"http://{host}/download?{params}"
    print(f"Requesting {url}")

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()

    with open(dest_path, "wb") as f:
        f.write(data)

    print(f"Saved downloaded CSV to {dest_path}")
    return dest_path


def update_data_file(host: str, data_csv_path: str, bootstrap_from_ts: str, bootstrap_to_ts: str) -> str:
    """
    Bring the local data file up to date from the device.

    - If data_csv_path doesn't exist: download bootstrap_from_ts..bootstrap_to_ts
      and save it as the new data file.
    - If it exists: download only the records newer than what's already
      stored (top timestamp + 1 second) through one hour from now, and
      merge them in, keeping newest-first order like the device's own CSVs.

    Returns the path to the up-to-date data file.
    """
    if not os.path.exists(data_csv_path):
        download_csv(host, bootstrap_from_ts, bootstrap_to_ts, data_csv_path)
        return data_csv_path

    existing_df = pd.read_csv(data_csv_path, parse_dates=["Time"])
    if existing_df.empty:
        download_csv(host, bootstrap_from_ts, bootstrap_to_ts, data_csv_path)
        return data_csv_path

    latest_stored = existing_df["Time"].iloc[0]  # newest-first, so row 0 is newest
    from_ts = (latest_stored + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
    to_ts = (pd.Timestamp.now() + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

    tmp_path = data_csv_path + ".new"
    download_csv(host, from_ts, to_ts, tmp_path)
    new_df = pd.read_csv(tmp_path, parse_dates=["Time"])
    os.remove(tmp_path)

    if new_df.empty:
        print("No new records since last update.")
        return data_csv_path

    combined = pd.concat([new_df, existing_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="Time")
    combined = combined.sort_values("Time", ascending=False).reset_index(drop=True)
    combined.to_csv(data_csv_path, index=False)
    print(f"Added {len(new_df)} new record(s) to {data_csv_path}")
    return data_csv_path


def plot_temperatures(csv_path: str, output_path: str, window_hours: float = PLOT_WINDOW_HOURS) -> None:
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)  # file may be newest-first

    if not df.empty:
        cutoff = df["Time"].max() - pd.Timedelta(hours=window_hours)
        df = df[df["Time"] >= cutoff].reset_index(drop=True)

    latest_timestamp = df["Time"].max() if not df.empty else None

    sensor_cols = [c for c in df.columns if c != "Time"]

    # Insert a NaN row wherever the gap between consecutive readings exceeds
    # this threshold, so periods where the board was offline show up as a
    # break in the line instead of a straight interpolation across the
    # outage. Readings are expected roughly every minute, with a few
    # seconds of drift -- 3 minutes is comfortably above normal jitter but
    # well below a real outage.
    GAP_THRESHOLD_SECONDS = 3 * 60
    diffs = df["Time"].diff().dt.total_seconds()
    gap_threshold = GAP_THRESHOLD_SECONDS

    gap_rows = []
    for i in range(1, len(df)):
        gap = diffs.iloc[i]
        if gap > gap_threshold:
            midpoint = df["Time"].iloc[i - 1] + (df["Time"].iloc[i] - df["Time"].iloc[i - 1]) / 2
            blank = {c: (midpoint if c == "Time" else float("nan")) for c in df.columns}
            gap_rows.append(blank)

    if gap_rows:
        df = pd.concat([df, pd.DataFrame(gap_rows)], ignore_index=True)
        df = df.sort_values("Time").reset_index(drop=True)

    # A sensor read failure is reported as -273.2 C. Treat anything below
    # -50 C as a failed reading and drop it (as a gap) rather than plot it.
    FAILED_READING_THRESHOLD = -50
    df[sensor_cols] = df[sensor_cols].where(df[sensor_cols] >= FAILED_READING_THRESHOLD)

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = plt.cm.viridis([i / max(len(sensor_cols) - 1, 1) for i in range(len(sensor_cols))])

    for col, color in zip(sensor_cols, colors):
        ax.plot(df["Time"], df[col], label=col, color=color, linewidth=1.8)

    for threshold in (4, -18):
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_title("Temperature Sensor Readings", fontsize=14, fontweight="bold")
    if latest_timestamp is not None:
        ax.set_title(
            f"Latest reading: {latest_timestamp.strftime('%Y-%m-%d %H:%M')}",
            fontsize=9, color="gray", loc="right", pad=12,
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(title="Sensor", loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download (optional) and plot PicoTemp sensor data.")
    parser.add_argument("--csv", help="Path to an existing CSV file to plot (skips download)")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Device IP/hostname (default: {DEFAULT_HOST})")
    parser.add_argument("--from", dest="from_ts", default=DEFAULT_FROM_TS,
                         help=f"Range start, format YYYY-MM-DDTHH:MM (default: {DEFAULT_FROM_TS})")
    parser.add_argument("--to", dest="to_ts", default=DEFAULT_TO_TS,
                         help=f"Range end, format YYYY-MM-DDTHH:MM (default: {DEFAULT_TO_TS})")
    parser.add_argument("--out", default="temperatures_plot.png", help="Output image path")
    parser.add_argument("--data-csv", default=DATA_CSV_PATH,
                         help=f"Persistent local data file (default: {DATA_CSV_PATH})")
    parser.add_argument("--window-hours", type=float, default=PLOT_WINDOW_HOURS,
                         help=f"How many hours of the most recent data to plot (default: {PLOT_WINDOW_HOURS})")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        try:
            csv_path = update_data_file(args.host, args.data_csv, args.from_ts, args.to_ts)
        except Exception as e:
            print(f"Failed to update data from device: {e}", file=sys.stderr)
            sys.exit(1)

    plot_temperatures(csv_path, args.out, args.window_hours)


if __name__ == "__main__":
    main()
