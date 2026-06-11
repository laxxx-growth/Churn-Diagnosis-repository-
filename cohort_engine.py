"""
Cohort engine.

Auto-generates analysis cohorts and enforces a minimum sample-size guard so we never
emit an "insight" from a handful of users. Thin cohorts roll up to their parent
(market x tier, dropping the tenure band) so they still get a — coarser — diagnosis
instead of being dropped silently.
"""

import churn_model as cm

MIN_USERS = 250
MIN_EVENTS = 25


def _full_key(city, tier, band):
    return f"{city} · {tier} · {band}"


def _parent_key(city, tier):
    return f"{city} · {tier} · (all tenures)"


def build_cohorts(df, min_users=MIN_USERS, min_events=MIN_EVENTS):
    """Return a list of cohort dicts: {key, level, city, tier, band, frame}.

    level is "full" (city x tier x tenure) or "rolled_up" (city x tier).
    A user is assigned to exactly one emitted cohort.
    """
    cohorts = []
    assigned = set()

    # 1) Try full-granularity cohorts first.
    grp = df.groupby(["city", "price_tier", "tenure_band"], observed=True)
    thin = []
    for (city, tier, band), idx in grp.groups.items():
        sub = df.loc[idx]
        if len(sub) >= min_users and int(sub["event_observed"].sum()) >= min_events:
            cohorts.append(
                {
                    "key": _full_key(city, tier, band),
                    "level": "full",
                    "city": city,
                    "tier": tier,
                    "band": band,
                    "frame": sub,
                }
            )
            assigned.update(idx)
        else:
            thin.append((city, tier))

    # 2) Roll up the leftover (thin) users to market x tier.
    leftover = df.loc[~df.index.isin(assigned)]
    if len(leftover):
        for (city, tier), idx in leftover.groupby(["city", "price_tier"], observed=True).groups.items():
            sub = df.loc[idx]
            if len(sub) >= min_users and int(sub["event_observed"].sum()) >= min_events:
                cohorts.append(
                    {
                        "key": _parent_key(city, tier),
                        "level": "rolled_up",
                        "city": city,
                        "tier": tier,
                        "band": "(all tenures)",
                        "frame": sub,
                    }
                )

    cohorts.sort(key=lambda c: len(c["frame"]), reverse=True)
    return cohorts
