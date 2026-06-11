"""
Insight feed — the headless entry point for the churn diagnosis engine.

Pipeline:  daily inflow/outflow  ->  cohort engine  ->  per-cohort diagnosis  ->
ranked insight feed  +  drift alerts.

Usage:
    python insights_feed.py                 # generate fake data, print feed
    python insights_feed.py --json out.json # also dump structured records
    python insights_feed.py --top 10        # show top N insights

In production you'd replace generate_users() with a read from your daily feed
(see daily_ingest.py for the shape).
"""

import argparse
import json
import warnings

import numpy as np

from cohort_engine import build_cohorts
from data_generator import generate_users
from diagnosis import detect_drift, diagnose_all, render_statement

warnings.simplefilter("ignore")


def run(n=12000, seed=42):
    df = generate_users(n=n, seed=seed)
    cohorts = build_cohorts(df)
    recs = diagnose_all(cohorts)
    drift = detect_drift(df)
    return df, recs, drift


def _print_feed(recs, drift, top):
    actionable = [r for r in recs if r["top_driver"] is not None]
    actionable.sort(key=lambda r: r.get("est_saveable", 0), reverse=True)

    print("\n" + "=" * 78)
    print(" CHURN PAIN-POINT FEED  —  ranked by estimated saveable users")
    print("=" * 78)
    for i, r in enumerate(actionable[:top], 1):
        print(f"\n{i}. " + render_statement(r).replace("**", "").replace("*", ""))
        print(f"   ↳ recommended test: {r['top_action']}")

    if drift:
        print("\n" + "=" * 78)
        print(" ⚠️  DRIFT ALERTS  —  a cohort's #1 churn driver just changed")
        print("=" * 78)
        for a in drift:
            print(
                f"\n• {a['city']}: '{a['from_driver']}' → '{a['to_driver']}' "
                f"(HR {a['from_hr']:.2f}× → {a['to_hr']:.2f}×) over last {a['window_days']}d"
            )
    print()


def _to_jsonable(recs, drift):
    def clean(d):
        out = {}
        for k, v in d.items():
            if k in ("drivers_ranked",):
                out[k] = [
                    {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv)
                     for kk, vv in dr.items()}
                    for dr in v
                ]
            elif isinstance(v, float):
                out[k] = None if np.isnan(v) else round(v, 4)
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            else:
                out[k] = v
        return out

    return {"insights": [clean(r) for r in recs], "drift_alerts": drift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    df, recs, drift = run(n=args.n, seed=args.seed)
    print(
        f"\nIngested {len(df):,} users · {df['churned'].mean():.1%} churn · "
        f"{len([r for r in recs if r['top_driver']])} cohorts with a dominant driver "
        f"· {len(drift)} drift alert(s)."
    )
    _print_feed(recs, drift, args.top)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(_to_jsonable(recs, drift), f, indent=2)
        print(f"Structured feed written to {args.json}\n")


if __name__ == "__main__":
    main()
