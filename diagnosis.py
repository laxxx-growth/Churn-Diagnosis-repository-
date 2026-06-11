"""
Per-cohort churn diagnosis.

For each cohort we answer: "which candidate driver is the biggest churn pain here,
and what's the typical exit timing?" Two complementary models on the SAME driver
signals (defined once in churn_model.DRIVERS):

  - Cox proportional hazards (lifelines): hazard ratios + timing. This is the
    primary ranking — it uses the full time-to-event data ("exit after N days").
  - Logistic regression (sklearn): churn yes/no driver importance, as a cross-check.

The top driver = the one with the largest *significant* positive hazard effect,
scored by  effect x prevalence  so a big effect on a tiny slice doesn't win.

Each diagnosis is a structured record (see diagnose_cohort) plus a templated
plain-English statement (render_statement).
"""

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression

import churn_model as cm

# "Exposed" thresholds — who counts as high-risk on each driver (for prevalence,
# the headline churn-multiple, and the saveable-user estimate).
EXPOSED = {
    "low_feature_adoption": lambda d: d["feature_count"] < 3,
    "low_engagement": lambda d: d["avg_engagement_mins"] < 40,
    "inactivity": lambda d: d["mau"] == 0,
    "competitor_pricing": lambda d: d["competitor_pricing_on_exit"] == 1,
    "young_audience": lambda d: d["age"] < 25,
}

SIG_STRONG = 0.01
SIG_OK = 0.05


def _design(frame):
    """Standardised driver-signal design matrix; drops near-constant columns."""
    sig = cm.risk_signals(frame)
    X = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in sig.items()}, index=frame.index)
    keep = [c for c in X.columns if X[c].std() > 1e-6]
    X = X[keep]
    Xz = (X - X.mean()) / X.std()
    return Xz, keep


def _cox(frame, cols):
    """Fit Cox PH; return {driver: (coef, hr, p)} or {} on failure."""
    d = pd.DataFrame(index=frame.index)
    Xz, keep = _design(frame)
    if not keep:
        return {}
    d[keep] = Xz
    d["_dur"] = frame["duration_days"].clip(lower=1).values
    d["_evt"] = frame["event_observed"].values
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(d, duration_col="_dur", event_col="_evt")
        s = cph.summary
        return {
            drv: (float(s.loc[drv, "coef"]), float(s.loc[drv, "exp(coef)"]), float(s.loc[drv, "p"]))
            for drv in keep
        }
    except Exception:
        return {}


def _logit(frame):
    """Standardised logistic coefficients per driver (cross-check)."""
    Xz, keep = _design(frame)
    y = frame["churned"].values
    if not keep or y.sum() < 5 or y.sum() == len(y):
        return {}
    try:
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(Xz.values, y)
        return {k: float(c) for k, c in zip(keep, clf.coef_[0])}
    except Exception:
        return {}


def diagnose_cohort(cohort):
    """Diagnose one cohort dict (from cohort_engine). Returns a structured record."""
    frame = cohort["frame"]
    n = len(frame)
    events = int(frame["event_observed"].sum())
    base_churn = frame["churned"].mean()

    cox = _cox(frame, None)
    logit = _logit(frame)

    ranked = []
    for drv, spec in cm.DRIVERS.items():
        if drv not in cox:
            continue
        coef, hr, p = cox[drv]
        if coef <= 0:  # only risk-increasing drivers are "pain points"
            continue
        exposed_mask = EXPOSED[drv](frame)
        prevalence = float(exposed_mask.mean())
        if prevalence == 0:
            continue
        # Rank mainly by hazard effect, with a mild preference for broader drivers
        # (so a strong effect on a tiny slice doesn't outrank a near-as-strong,
        # cohort-wide one).
        score = coef * (0.5 + 0.5 * prevalence)
        ranked.append(
            {
                "driver": drv,
                "label": spec["label"],
                "action": spec["action"],
                "hazard_ratio": hr,
                "coef": coef,
                "p_value": p,
                "prevalence": prevalence,
                "logit_coef": logit.get(drv, np.nan),
                "score": score,
            }
        )

    ranked.sort(key=lambda r: r["score"], reverse=True)

    rec = {
        "cohort": cohort["key"],
        "level": cohort["level"],
        "city": cohort["city"],
        "tier": cohort["tier"],
        "band": cohort["band"],
        "n_users": n,
        "n_events": events,
        "baseline_churn": base_churn,
        "drivers_ranked": ranked,
    }

    if not ranked or ranked[0]["p_value"] > SIG_OK:
        rec.update(
            {
                "top_driver": None,
                "top_label": "No dominant driver (signals weak / cohort stable)",
                "confidence": "low",
            }
        )
        return rec

    top = ranked[0]
    drv = top["driver"]
    exposed = EXPOSED[drv](frame)
    churn_exposed = frame.loc[exposed, "churned"].mean()
    churn_rest = frame.loc[~exposed, "churned"].mean() if (~exposed).any() else 0.0
    churn_mult = (churn_exposed / base_churn) if base_churn > 0 else np.nan
    exposed_churned = frame.loc[exposed & (frame["churned"] == 1)]
    median_exit = (
        float(exposed_churned["exit_after_days"].median()) if len(exposed_churned) else np.nan
    )
    est_saveable = float(exposed.sum() * max(0.0, churn_exposed - churn_rest))

    if top["p_value"] < SIG_STRONG and n >= 400 and events >= 40:
        confidence = "high"
    elif top["p_value"] < SIG_OK:
        confidence = "medium"
    else:
        confidence = "low"

    rec.update(
        {
            "top_driver": drv,
            "top_label": top["label"],
            "top_action": top["action"],
            "hazard_ratio": top["hazard_ratio"],
            "p_value": top["p_value"],
            "prevalence": top["prevalence"],
            "churn_exposed": float(churn_exposed),
            "churn_multiple": float(churn_mult),
            "median_exit_days": median_exit,
            "users_exposed": int(exposed.sum()),
            "est_saveable": est_saveable,
            "confidence": confidence,
        }
    )
    return rec


def render_statement(rec):
    """One plain-English line for the insight feed."""
    if rec["top_driver"] is None:
        return f"**{rec['cohort']}** — no dominant churn driver detected (n={rec['n_users']:,})."
    exit_txt = (
        f"typically exiting around **day {rec['median_exit_days']:.0f}**"
        if not np.isnan(rec.get("median_exit_days", np.nan))
        else "with elevated early exit"
    )
    return (
        f"**{rec['cohort']}** — *{rec['top_label']}* is the #1 churn driver "
        f"(hazard ratio **{rec['hazard_ratio']:.2f}×**; exposed users churn at "
        f"**{rec['churn_multiple']:.1f}× the cohort baseline**, {exit_txt}). "
        f"Affects ~{rec['users_exposed']:,} users · est. **{rec['est_saveable']:.0f} saveable**. "
        f"Confidence: **{rec['confidence']}**."
    )


def diagnose_all(cohorts):
    return [diagnose_cohort(c) for c in cohorts]


def _frame_as_cohort(frame, key, city, tier="(all tiers)", band="(window)", level="window"):
    return {"key": key, "level": level, "city": city, "tier": tier, "band": band, "frame": frame}


def detect_drift(df, window_days=120, min_users=200, min_events=20):
    """Compare each market's #1 churn driver in the RECENT signup window vs the PRIOR
    window. Returns alerts where the dominant driver changed (or its hazard jumped).

    This is what makes the engine "dynamic": run it daily and it tells you when a
    cohort's pain point shifts (e.g. a competitor promo overtaking engagement).
    """
    if "signup_date" not in df.columns:
        return []
    cutoff = df["signup_date"].max() - pd.Timedelta(days=window_days)
    recent = df[df["signup_date"] >= cutoff]
    prior = df[df["signup_date"] < cutoff]

    alerts = []
    for city in sorted(df["city"].unique()):
        r = recent[recent["city"] == city]
        p = prior[prior["city"] == city]
        if (
            len(r) < min_users or len(p) < min_users
            or r["event_observed"].sum() < min_events
            or p["event_observed"].sum() < min_events
        ):
            continue
        rec_r = diagnose_cohort(_frame_as_cohort(r, f"{city} · recent", city))
        rec_p = diagnose_cohort(_frame_as_cohort(p, f"{city} · prior", city))
        if rec_r["top_driver"] is None or rec_p["top_driver"] is None:
            continue
        if rec_r["top_driver"] != rec_p["top_driver"]:
            alerts.append(
                {
                    "city": city,
                    "from_driver": rec_p["top_label"],
                    "to_driver": rec_r["top_label"],
                    "from_hr": rec_p.get("hazard_ratio"),
                    "to_hr": rec_r.get("hazard_ratio"),
                    "window_days": window_days,
                    "severity": "high",
                }
            )
    return alerts
