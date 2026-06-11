"""
Daily ingestion shape (demo).

In production the engine runs on an append-only daily feed: each day you land that
day's inflow (new signups) and outflow (that day's churn events), partitioned by
date. The diagnosis engine then trains on a trailing rolling window.

This module fakes that by partitioning the synthetic master table by signup_date and
exposing a rolling-window reader, so insights_feed / the dashboard can pretend to be
"as of" any date. Swap these two functions for real warehouse reads to productionise.
"""

import pandas as pd

from data_generator import generate_users


def write_daily_partitions(out_dir="feed", n=12000, seed=42):
    """Write one CSV per signup date: feed/YYYY-MM-DD.csv  (demo only)."""
    import os

    df = generate_users(n=n, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    for day, part in df.groupby(df["signup_date"].dt.date):
        part.to_csv(os.path.join(out_dir, f"{day}.csv"), index=False)
    print(f"Wrote {df['signup_date'].dt.date.nunique()} daily partitions to {out_dir}/")
    return df


def rolling_window(df, as_of=None, window_days=90):
    """Return users observed within the trailing window ending at `as_of`.

    Models "what the engine sees when it runs today": users whose signup falls in the
    window. (Real systems window on the observation/event date; signup is fine here.)
    """
    if as_of is None:
        as_of = df["signup_date"].max()
    as_of = pd.Timestamp(as_of)
    lo = as_of - pd.Timedelta(days=window_days)
    return df[(df["signup_date"] > lo) & (df["signup_date"] <= as_of)]


if __name__ == "__main__":
    df = write_daily_partitions()
    w = rolling_window(df, window_days=90)
    print(f"Most recent 90-day window: {len(w):,} users, {w['churned'].mean():.1%} churn")
