"""
Shared churn model + driver definitions.

This module is the single source of truth used by THREE things:
  1. data_generator.py  - plants the (cohort-varying) churn drivers into fake data
  2. diagnosis.py        - tries to *recover* those drivers per cohort (Cox + logistic)
  3. app.py (simulator)  - projects intervention impact

Key design choice: each MARKET has a DIFFERENT dominant churn driver (see
MARKET_PROFILES). That's what makes the diagnosis engine interesting — it should
report "feature adoption is the pain in Mumbai, competitor pricing is the pain in
Delhi", not one global answer.

Candidate drivers are defined ONCE in DRIVERS, each as a risk signal oriented so
that HIGHER = MORE churn risk. The generator builds hazard from these signals; the
diagnosis engine regresses on these same signals. So recovery is honest.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
FEATURES = ["daylist", "blend", "jam", "ai_dj", "radio", "mix"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
GENDERS = ["Female", "Male", "Non-binary"]
PRICE_TIERS = ["Full price", "Discount conversion"]

TENURE_BANDS = [
    ("New (0-90d)", -1, 90),
    ("Growing (90-365d)", 90, 365),
    ("Established (1-2y)", 365, 730),
    ("Loyal (2y+)", 730, 10_000),
]

ENGAGEMENT_REF_MINS = 90.0
AGE_REF_YEARS = 60.0
MAX_FEATURES = len(FEATURES)


# ---------------------------------------------------------------------------
# Candidate drivers. Each maps to a risk signal in [0, ~1], higher = riskier.
# `weight` is the base hazard weight planted by the generator.
# These are the ONLY things the diagnosis engine is allowed to name as a "pain point".
# ---------------------------------------------------------------------------
DRIVERS = {
    "low_feature_adoption": {
        "label": "Lack of feature adoption",
        "weight": 0.7,
        "action": "drive feature adoption (onboarding nudges to AI DJ / blend / daylist)",
    },
    "low_engagement": {
        "label": "Low engagement",
        "weight": 0.8,
        "action": "re-engagement push (personalised mixes, listening streaks)",
    },
    "inactivity": {
        "label": "Inactivity (non-MAU)",
        "weight": 0.7,
        "action": "win-back / re-activation campaign",
    },
    "competitor_pricing": {
        "label": "Competitor pricing pressure",
        "weight": 0.5,
        "action": "targeted retention offer ahead of renewal",
    },
    "young_audience": {
        "label": "Younger-skewing audience",
        "weight": 0.3,
        "action": "youth-oriented content & social (jam) features",
    },
}

# Each market's dominant pain point gets an extra hazard boost on that one driver.
MARKET_PROFILES = {
    "Mumbai": "low_feature_adoption",
    "Delhi": "competitor_pricing",
    "Bangalore": "low_engagement",
    "Chennai": "inactivity",
    "Kolkata": "low_feature_adoption",
    "Hyderabad": "competitor_pricing",
    "Pune": "low_engagement",
}
MARKET_BOOST = 1.9          # extra hazard weight on the market's dominant driver
DISCOUNT_WEIGHT = 0.5       # discount tier adds churn risk (a cohort key, not a driver)

# Weibull baseline for the survival simulation (tuned for ~25-35% events).
WEIBULL_SHAPE = 1.4
WEIBULL_LAMBDA0 = 1.35e-5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def risk_signals(df):
    """Return a dict of the candidate-driver risk signals for each row.

    All oriented so higher = more churn risk, all roughly in [0, 1].
    Expects columns: feature_count, avg_engagement_mins, mau,
    competitor_pricing_on_exit, age.
    """
    return {
        "low_feature_adoption": (MAX_FEATURES - df["feature_count"]) / MAX_FEATURES,
        "low_engagement": np.clip(
            (ENGAGEMENT_REF_MINS - df["avg_engagement_mins"]) / ENGAGEMENT_REF_MINS, 0, 1
        ),
        "inactivity": (df["mau"] == 0).astype(float),
        "competitor_pricing": df["competitor_pricing_on_exit"].astype(float),
        "young_audience": np.clip((35 - df["age"]) / 35.0, 0, 1),
    }


def linear_predictor(df, market_aware=True):
    """Hazard linear predictor eta (higher = faster churn).

    Built from the candidate-driver signals with base weights, plus a per-market
    boost on that market's dominant driver, plus a discount-tier term.
    """
    sig = risk_signals(df)
    eta = np.zeros(len(df), dtype=float)
    for name, spec in DRIVERS.items():
        eta = eta + spec["weight"] * np.asarray(sig[name], dtype=float)

    if market_aware and "city" in df.columns:
        for market, dom in MARKET_PROFILES.items():
            m = (df["city"] == market).to_numpy()
            eta = eta + MARKET_BOOST * np.asarray(sig[dom], dtype=float) * m

    is_discount = (df["price_tier"] == "Discount conversion").astype(float)
    eta = eta + DISCOUNT_WEIGHT * np.asarray(is_discount, dtype=float)
    return eta


def churn_probability(df, retention_discount=0.0, market_aware=True):
    """Logistic-style churn probability for the dashboard simulator.

    Derived from the same hazard linear predictor so interventions (more
    engagement, more features, etc.) move it in the right direction.

    `retention_discount` (0..1, scalar or per-row) models a retention price cut:
    a deeper discount lowers churn hazard (people stay), scaled by
    DISCOUNT_RETENTION_BETA. This is separate from the *acquisition* discount tier.
    """
    eta = linear_predictor(df, market_aware=market_aware)
    eta = eta - DISCOUNT_RETENTION_BETA * np.asarray(retention_discount, dtype=float)
    # Centre/scale so the average propensity tracks the realised ~30% annual churn
    # (keeps the simulator consistent with the Overview tab and the LTV maths sane).
    return _sigmoid(1.0 * (eta - 2.4))


# ---------------------------------------------------------------------------
# Revenue / Lifetime Value
# ---------------------------------------------------------------------------
# Monthly ARPU by acquisition tier (₹). Discount-conversion users already pay less.
ARPU_MONTHLY = {"Full price": 119.0, "Discount conversion": 66.0}

# How strongly a retention discount lowers churn hazard (per unit discount fraction).
DISCOUNT_RETENTION_BETA = 3.5

# LTV is computed over a FIXED horizon (months), not the naive ARPU/churn formula.
# A finite horizon is what makes discounting a genuine trade-off: it's accretive for
# flight-risk users but dilutive for loyal ones (where you'd just be giving away margin).
LTV_HORIZON_MONTHS = 36


def monthly_arpu(df, extra_discount=0.0):
    """Per-user monthly revenue after an optional extra retention discount (0..1)."""
    base = df["price_tier"].map(ARPU_MONTHLY).to_numpy(dtype=float)
    return base * (1.0 - np.asarray(extra_discount, dtype=float))


def _monthly_churn(prob):
    """Convert the model's churn propensity (treated as annual) to a monthly rate."""
    c = np.clip(prob, 0.005, 0.95)
    return 1.0 - (1.0 - c) ** (1.0 / 12.0)


def ltv(df, retention_discount=0.0, extra_discount=0.0, horizon=LTV_HORIZON_MONTHS,
        market_aware=True):
    """Expected lifetime value per user over a fixed horizon.

    LTV = monthly_arpu × E[months retained within horizon]. A retention discount
    BOTH lowers churn (via retention_discount) AND lowers ARPU (via extra_discount) —
    pass the same fraction to both to model a price cut.
    """
    prob = churn_probability(df, retention_discount=retention_discount, market_aware=market_aware)
    c_m = _monthly_churn(prob)
    exp_months = (1.0 - (1.0 - c_m) ** horizon) / c_m
    return monthly_arpu(df, extra_discount=extra_discount) * exp_months
