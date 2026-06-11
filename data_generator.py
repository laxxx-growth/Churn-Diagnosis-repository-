"""
Synthetic data generator for the churn diagnosis engine.

Now built around a SURVIVAL simulation so "exit after N days" is real time-to-event
data the Cox model can use:

  - Each user gets driver covariates (engagement, feature adoption, MAU, etc.).
  - A Weibull event time is drawn whose hazard rises with churn_model.linear_predictor
    (which is cohort-varying: each market has a different dominant driver).
  - A censoring time (observation window since signup) decides who we've actually
    seen churn vs who is still active (censored).
  => duration + event_observed, plus signup_date for the daily feed / drift detection.

Run directly to dump CSVs:  python data_generator.py
"""

import numpy as np
import pandas as pd

import churn_model as cm

STUDY_END = pd.Timestamp("2026-06-11")   # "today" — right edge of the observation window


def tenure_band(days):
    for label, lo, hi in cm.TENURE_BANDS:
        if lo < days <= hi:
            return label
    return cm.TENURE_BANDS[0][0]


def generate_users(n=8000, seed=42, drift=True):
    rng = np.random.default_rng(seed)

    user_id = np.array([f"U{100000 + i}" for i in range(n)])
    city = rng.choice(cm.CITIES, size=n, p=_normalise([0.22, 0.18, 0.2, 0.12, 0.1, 0.1, 0.08]))
    age = np.clip(rng.normal(29, 9, n).round().astype(int), 13, 70)
    gender = rng.choice(cm.GENDERS, size=n, p=[0.47, 0.48, 0.05])
    price_tier = rng.choice(cm.PRICE_TIERS, size=n, p=[0.6, 0.4])

    # Signup date spread across the last ~18 months.
    days_ago = rng.integers(1, 540, size=n)
    signup_date = STUDY_END - pd.to_timedelta(days_ago, unit="D")

    # Feature adoption: discount users adopt fewer features.
    base_adopt = rng.normal(0, 1, (n, cm.MAX_FEATURES))
    bias = (-0.3 * (price_tier == "Discount conversion")).reshape(-1, 1)
    feature_matrix = (base_adopt + bias + 0.15 > 0).astype(int)
    feature_count = feature_matrix.sum(axis=1)

    # Engagement rises with feature breadth, but loosely (lots of independent noise)
    # so "low engagement" and "low feature adoption" stay statistically separable.
    avg_engagement = 38 + 5 * feature_count + rng.normal(0, 20, n)
    avg_engagement = np.clip(avg_engagement, 1, 300).round(1)

    # MAU: mostly an independent activation coin-flip with only a mild engagement tilt,
    # so "inactivity" doesn't just proxy for low engagement.
    mau_prob = 0.55 + 0.20 * cm._sigmoid((avg_engagement - 50) / 40) - 0.10
    mau = (rng.random(n) < np.clip(mau_prob, 0.05, 0.95)).astype(int)

    # Competitor pricing pressure. DRIFT: a competitor runs an aggressive promo over
    # the last ~120 days that hits BANGALORE hard (recent signups). Normally Bangalore's
    # pain is low engagement; the shock should let the drift detector catch competitor
    # pricing OVERTAKING engagement as Bangalore's #1 driver in the recent window.
    recent = days_ago < 120
    comp_prob = np.full(n, 0.30)
    if drift:
        comp_prob = np.where((city == "Bangalore") & recent, 0.62, comp_prob)
        comp_prob = np.where((city == "Delhi") & recent, 0.70, comp_prob)
    competitor = (rng.random(n) < comp_prob).astype(int)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "city": city,
            "age": age,
            "gender": gender,
            "price_tier": price_tier,
            "signup_date": signup_date,
            "avg_engagement_mins": avg_engagement,
            "feature_count": feature_count,
            "mau": mau,
            "competitor_pricing_on_exit": competitor,
        }
    )
    for i, f in enumerate(cm.FEATURES):
        df[f"feat_{f}"] = feature_matrix[:, i]

    # ---- Survival simulation ----
    eta = cm.linear_predictor(df, market_aware=True)
    if drift:
        # The Bangalore promo shock makes competitor pricing genuinely more lethal
        # (not just more common) for recent signups there.
        shock = (city == "Bangalore") & recent
        eta = eta + 1.7 * competitor * shock
    u = rng.random(n)
    k = cm.WEIBULL_SHAPE
    # Proportional-hazards Weibull: larger eta => shorter event time.
    event_time = (-np.log(u) / (cm.WEIBULL_LAMBDA0 * np.exp(eta))) ** (1.0 / k)

    # Censoring = how long we've observed them (days since signup, capped).
    censor_time = np.minimum(days_ago, 540).astype(float)

    churned = (event_time <= censor_time).astype(int)
    duration = np.minimum(event_time, censor_time)

    df["age_on_platform_days"] = duration.round().astype(int)  # observed tenure to date
    df["time_spent_hours"] = (avg_engagement * np.maximum(duration, 1) / 60.0).round(1)
    df["churned"] = churned
    df["duration_days"] = duration.round().astype(int)         # survival duration
    df["event_observed"] = churned                              # 1 = churn seen, 0 = censored
    df["exit_after_days"] = np.where(churned == 1, duration.round().astype(int), np.nan)
    df["exit_after_renewals"] = np.floor(df["age_on_platform_days"] / 30.0).astype(int)

    safe_count = np.maximum(feature_count, 1)
    df["engagement_per_feature_mins"] = (df["avg_engagement_mins"] / safe_count).round(1)
    df.loc[df["feature_count"] == 0, "engagement_per_feature_mins"] = 0.0

    # Cohort keys
    df["tenure_band"] = df["age_on_platform_days"].map(tenure_band)
    df["cohort"] = df["city"] + " · " + df["price_tier"] + " · " + df["tenure_band"]
    df["churn_prob"] = cm.churn_probability(df)
    return df


def to_inflow(df):
    return pd.DataFrame(
        {
            "user_id": df["user_id"],
            "city": df["city"],
            "age": df["age"],
            "gender": df["gender"],
            "mau": df["mau"],
            "non_mau": 1 - df["mau"],
            "price_tier": df["price_tier"],
            "age_on_platform_days": df["age_on_platform_days"],
            "time_spent_hours": df["time_spent_hours"],
            "signup_date": df["signup_date"].dt.date,
        }
    )


def to_outflow(df):
    c = df[df["churned"] == 1].copy()
    out = pd.DataFrame(
        {
            "user_id": c["user_id"],
            "city": c["city"],
            "age": c["age"],
            "gender": c["gender"],
            "price_tier": c["price_tier"],
            "exit_after_days": c["exit_after_days"].astype(int),
            "avg_engagement_mins": c["avg_engagement_mins"],
        }
    )
    for f in cm.FEATURES:
        out[f"feat_{f}"] = c[f"feat_{f}"]
    out["engagement_per_feature_mins"] = c["engagement_per_feature_mins"]
    out["exit_after_renewals"] = c["exit_after_renewals"]
    out["competitor_pricing_on_exit"] = c["competitor_pricing_on_exit"]
    return out


def _normalise(p):
    p = np.array(p, dtype=float)
    return p / p.sum()


if __name__ == "__main__":
    df = generate_users()
    to_inflow(df).to_csv("inflow.csv", index=False)
    to_outflow(df).to_csv("outflow.csv", index=False)
    df.to_csv("users_master.csv", index=False)
    print(f"Generated {len(df)} users across {df['cohort'].nunique()} cohorts.")
    print(f"Overall churn (events observed): {df['churned'].mean():.1%}")
    print(f"Median exit-after-days (churned): {df.loc[df.churned==1,'exit_after_days'].median():.0f}")
