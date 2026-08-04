#!/usr/bin/env python3
"""
Download and plot temperature readings from a PicoTemp-style sensor device.

The device serves a form at http://<host>/ with a GET endpoint at
/download?from_ts=...&to_ts=... that returns a CSV with columns:
Time,T0,T1,T2,T3,T4

Usage:
    # Download from the device and plot (uses default host/time range below)
    python3 plot_temps.py

    # Specify host and/or time range
    python3 plot_temps.py --host 10.0.0.64 --from "2026-08-03T16:04" --to "2026-08-03T18:16"

    # Skip downloading and just plot a CSV you already have
    python3 plot_temps.py --csv data.csv

Run this on a machine that's actually on the same local network as the
device (e.g. your own computer) -- it won't work from a sandboxed/cloud
environment that has no route to your LAN.
"""

import argparse
import sys
import urllib.request
import urllib.parse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_HOST = "10.0.0.64"
DEFAULT_FROM_TS = "2026-08-03T00:00"
DEFAULT_TO_TS = "2026-08-05T00:00"


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


def plot_temperatures(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    df = df.sort_values("Time")  # file may be newest-first; put it in chronological order

    sensor_cols = [c for c in df.columns if c != "Time"]

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = plt.cm.viridis([i / max(len(sensor_cols) - 1, 1) for i in range(len(sensor_cols))])

    for col, color in zip(sensor_cols, colors):
        ax.plot(df["Time"], df[col], label=col, color=color, linewidth=1.8)

    ax.set_title("Temperature Sensor Readings", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(title="Sensor", loc="best", frameon=True)
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
    parser.add_argument("--out", default="temperature_plot.png", help="Output image path")
    parser.add_argument("--downloaded-csv", default="downloaded.csv",
                         help="Where to save the downloaded CSV (default: downloaded.csv)")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        try:
            csv_path = download_csv(args.host, args.from_ts, args.to_ts, args.downloaded_csv)
        except Exception as e:
            print(f"Failed to download CSV from device: {e}", file=sys.stderr)
            sys.exit(1)

    plot_temperatures(csv_path, args.out)


if __name__ == "__main__":
    main()
