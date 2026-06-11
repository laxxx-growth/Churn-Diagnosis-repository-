"""
Churn Test-and-Learn dashboard (Streamlit).

Tabs:
  1. Cohort Diagnosis - the engine: per-cohort #1 churn driver, exit timing, saveable
                        users, drift alerts (Cox PH + logistic under the hood)
  2. Overview         - KPIs and the inflow/outflow line-item tables (+ CSV download)
  3. Churn drivers    - where the pain points are, sliced by each driver
  4. Feature usage    - adoption & engagement per feature, churned vs retained
  5. Test & Learn     - pick a segment, apply an intervention, see projected churn lift

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import churn_model as cm
from cohort_engine import build_cohorts
from data_generator import generate_users, to_inflow, to_outflow
from diagnosis import detect_drift, diagnose_all, render_statement

st.set_page_config(page_title="Churn Test & Learn", layout="wide", page_icon="📉")


@st.cache_data
def load(n, seed):
    df = generate_users(n=n, seed=seed)
    return df


@st.cache_data
def run_diagnosis(n, seed):
    """Full cohort-diagnosis pipeline (cached). Returns (records, drift_alerts)."""
    import warnings

    warnings.simplefilter("ignore")
    df = generate_users(n=n, seed=seed)
    recs = diagnose_all(build_cohorts(df))
    drift = detect_drift(df)
    return recs, drift


# ----------------------------- Sidebar ------------------------------------
st.sidebar.title("⚙️ Data controls")
n_users = st.sidebar.slider("Number of users", 1000, 20000, 10000, step=1000)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
df = load(n_users, int(seed))

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
sel_cities = st.sidebar.multiselect("City", cm.CITIES, default=cm.CITIES)
sel_tiers = st.sidebar.multiselect("Price tier", cm.PRICE_TIERS, default=cm.PRICE_TIERS)
age_range = st.sidebar.slider("Age", 13, 70, (13, 70))

mask = (
    df["city"].isin(sel_cities)
    & df["price_tier"].isin(sel_tiers)
    & df["age"].between(*age_range)
)
fdf = df[mask].copy()

st.title("📉 Churn Test-and-Learn Model")
st.caption(
    "Synthetic data with realistic latent churn drivers. Use the **Test & Learn** "
    "tab to simulate interventions and project churn reduction."
)

if fdf.empty:
    st.warning("No users match the current filters.")
    st.stop()


def _bar(frame, col, title):
    """Churn rate by a categorical column, as a labelled bar chart."""
    g = frame.groupby(col, observed=True)["churned"].agg(["mean", "count"]).reset_index()
    g.columns = [col, "churn_rate", "users"]
    fig = px.bar(
        g, x=col, y="churn_rate", text=g["churn_rate"].map("{:.0%}".format),
        hover_data=["users"], title=title,
    )
    fig.update_yaxes(tickformat=".0%", title="churn rate")
    fig.update_traces(textposition="outside")
    fig.update_xaxes(type="category")
    return fig

tab_diag, tab_overview, tab_drivers, tab_features, tab_tl = st.tabs(
    [
        "🧭 Cohort Diagnosis",
        "📊 Overview",
        "🔍 Churn drivers",
        "🎚️ Feature usage",
        "🧪 Test & Learn",
    ]
)

# ----------------------------- Overview -----------------------------------
with tab_overview:
    churn_rate = fdf["churned"].mean()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users", f"{len(fdf):,}")
    c2.metric("Churned", f"{int(fdf['churned'].sum()):,}")
    c3.metric("Churn rate", f"{churn_rate:.1%}")
    c4.metric("MAU share", f"{fdf['mau'].mean():.1%}")
    c5.metric(
        "Discount mix",
        f"{(fdf['price_tier'] == 'Discount conversion').mean():.1%}",
    )

    st.markdown("### Inflow line items")
    inflow = to_inflow(fdf)
    st.dataframe(inflow.head(200), width="stretch", height=260)
    st.download_button(
        "⬇️ Download inflow.csv",
        inflow.to_csv(index=False).encode(),
        "inflow.csv",
        "text/csv",
    )

    st.markdown("### Outflow line items (churned users)")
    outflow = to_outflow(fdf)
    st.dataframe(outflow.head(200), width="stretch", height=260)
    st.download_button(
        "⬇️ Download outflow.csv",
        outflow.to_csv(index=False).encode(),
        "outflow.csv",
        "text/csv",
    )

# ----------------------------- Churn drivers ------------------------------
with tab_drivers:
    st.markdown("### Where is the churn pain?")
    st.caption("Churn rate broken down by each candidate driver. Bigger gaps = bigger lever.")

    fdf["mau_label"] = np.where(fdf["mau"] == 1, "MAU", "Non-MAU")
    fdf["competitor_label"] = np.where(
        fdf["competitor_pricing_on_exit"] == 1, "Competitor promo", "No promo"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_bar(fdf, "price_tier", "By price tier"), width="stretch")
        st.plotly_chart(_bar(fdf, "mau_label", "By MAU status"), width="stretch")
        st.plotly_chart(
            _bar(fdf, "competitor_label", "By competitor pricing pressure"),
            width="stretch",
        )
    with col2:
        eng_bins = pd.cut(
            fdf["avg_engagement_mins"], [0, 20, 40, 60, 90, 1000],
            labels=["0-20", "20-40", "40-60", "60-90", "90+"],
        )
        tmp = fdf.assign(eng_bucket=eng_bins)
        st.plotly_chart(
            _bar(tmp, "eng_bucket", "By avg engagement (mins/day)"),
            width="stretch",
        )
        st.plotly_chart(
            _bar(fdf, "feature_count", "By # features adopted"),
            width="stretch",
        )
        ten_bins = pd.cut(
            fdf["age_on_platform_days"], [-1, 0, 90, 365, 730, 10000],
            labels=["New (0)", "1-90d", "90-365d", "1-2y", "2y+"],
        )
        tmp2 = fdf.assign(tenure_bucket=ten_bins)
        st.plotly_chart(
            _bar(tmp2, "tenure_bucket", "By tenure on platform"),
            width="stretch",
        )

# ----------------------------- Feature usage ------------------------------
with tab_features:
    st.markdown("### Feature adoption & engagement")
    rows = []
    for f in cm.FEATURES:
        col = f"feat_{f}"
        adopt = fdf[col].mean()
        churn_if_used = fdf.loc[fdf[col] == 1, "churned"].mean()
        churn_if_not = fdf.loc[fdf[col] == 0, "churned"].mean()
        rows.append(
            {
                "feature": f,
                "adoption": adopt,
                "churn_if_used": churn_if_used,
                "churn_if_not_used": churn_if_not,
                "stickiness_gap": (churn_if_not or 0) - (churn_if_used or 0),
            }
        )
    fstats = pd.DataFrame(rows).sort_values("stickiness_gap", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            fstats, x="feature", y="adoption",
            text=fstats["adoption"].map("{:.0%}".format), title="Adoption rate by feature",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")
    with col2:
        m = fstats.melt(
            id_vars="feature", value_vars=["churn_if_used", "churn_if_not_used"],
            var_name="cohort", value_name="churn_rate",
        )
        fig = px.bar(
            m, x="feature", y="churn_rate", color="cohort", barmode="group",
            title="Churn: used vs not used",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Per-feature stickiness")
    st.caption("Higher gap = users who adopt it churn much less (a good adoption target).")
    st.dataframe(
        fstats.style.format(
            {
                "adoption": "{:.0%}",
                "churn_if_used": "{:.0%}",
                "churn_if_not_used": "{:.0%}",
                "stickiness_gap": "{:+.0%}",
            }
        ),
        width="stretch",
    )

# ----------------------------- Test & Learn -------------------------------
with tab_tl:
    st.markdown("### 🧪 Test-and-Learn simulator")
    st.caption(
        "Pick a target segment, apply a treatment, and project the churn impact. "
        "Projection recomputes churn probability with the **same model that generated "
        "the data**, so the lift is internally consistent."
    )

    seg_col1, seg_col2 = st.columns(2)
    with seg_col1:
        target = st.selectbox(
            "Target segment",
            [
                "All filtered users",
                "Discount-conversion users",
                "Non-MAU users",
                "Low engagement (<40 min/day)",
                "Low feature adoption (<3 features)",
                "Exposed to competitor pricing",
            ],
        )
    with seg_col2:
        treatment = st.selectbox(
            "Treatment / intervention",
            [
                "Lift avg engagement by N minutes",
                "Drive feature adoption (+N features)",
                "Convert discount users to full price",
                "Neutralise competitor pricing",
                "Win-back to MAU (re-activate)",
            ],
        )

    magnitude = st.slider("Treatment magnitude (N)", 1, 60, 15)
    adoption_rate = st.slider(
        "Treatment take-up rate (%)", 5, 100, 60,
        help="Not everyone in the segment responds — what share actually takes the treatment?",
    ) / 100.0

    # Build the segment mask
    seg = {
        "All filtered users": pd.Series(True, index=fdf.index),
        "Discount-conversion users": fdf["price_tier"] == "Discount conversion",
        "Non-MAU users": fdf["mau"] == 0,
        "Low engagement (<40 min/day)": fdf["avg_engagement_mins"] < 40,
        "Low feature adoption (<3 features)": fdf["feature_count"] < 3,
        "Exposed to competitor pricing": fdf["competitor_pricing_on_exit"] == 1,
    }[target]

    treated = fdf.copy()
    # Who actually takes the treatment (random take-up within the segment)
    rng = np.random.default_rng(int(seed) + 7)
    takeup = seg & (pd.Series(rng.random(len(fdf)), index=fdf.index) < adoption_rate)

    if treatment == "Lift avg engagement by N minutes":
        treated.loc[takeup, "avg_engagement_mins"] = (
            treated.loc[takeup, "avg_engagement_mins"] + magnitude
        ).clip(upper=300)
    elif treatment == "Drive feature adoption (+N features)":
        treated.loc[takeup, "feature_count"] = (
            treated.loc[takeup, "feature_count"] + min(magnitude, 6)
        ).clip(upper=len(cm.FEATURES))
    elif treatment == "Convert discount users to full price":
        treated.loc[takeup, "price_tier"] = "Full price"
    elif treatment == "Neutralise competitor pricing":
        treated.loc[takeup, "competitor_pricing_on_exit"] = 0
    elif treatment == "Win-back to MAU (re-activate)":
        treated.loc[takeup, "mau"] = 1

    base_prob = cm.churn_probability(fdf)
    new_prob = cm.churn_probability(treated)

    seg_idx = takeup
    base_seg = base_prob[seg_idx.values].mean() if seg_idx.any() else 0
    new_seg = new_prob[seg_idx.values].mean() if seg_idx.any() else 0
    overall_base = base_prob.mean()
    overall_new = new_prob.mean()
    users_treated = int(seg_idx.sum())
    expected_saved = float((base_prob[seg_idx.values] - new_prob[seg_idx.values]).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users treated", f"{users_treated:,}")
    m2.metric(
        "Segment churn prob",
        f"{new_seg:.1%}",
        delta=f"{(new_seg - base_seg):.1%}",
        delta_color="inverse",
    )
    m3.metric(
        "Overall churn prob",
        f"{overall_new:.1%}",
        delta=f"{(overall_new - overall_base):.1%}",
        delta_color="inverse",
    )
    m4.metric("Expected users retained", f"{expected_saved:,.0f}")

    comp = pd.DataFrame(
        {
            "scenario": ["Baseline", "After treatment"],
            "segment_churn": [base_seg, new_seg],
            "overall_churn": [overall_base, overall_new],
        }
    )
    fig = px.bar(
        comp.melt(id_vars="scenario", var_name="metric", value_name="churn"),
        x="metric", y="churn", color="scenario", barmode="group",
        title="Projected churn: baseline vs treatment",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"**Read-out:** treating **{users_treated:,}** users in *{target}* with "
        f"*{treatment.lower()}* (N={magnitude}, {adoption_rate:.0%} take-up) is projected "
        f"to cut that segment's churn from **{base_seg:.1%} → {new_seg:.1%}**, retaining "
        f"~**{expected_saved:,.0f}** users. Next step in a real test-and-learn: run this "
        f"as a randomised holdout (treatment vs control) and measure the actual lift."
    )


# ----------------------------- Cohort Diagnosis ---------------------------
with tab_diag:
    st.markdown("### 🧭 Automated cohort diagnosis")
    st.caption(
        "For every cohort (market × price tier × tenure), the engine fits a **Cox "
        "proportional-hazards** model + logistic cross-check on the candidate drivers "
        "and reports the **#1 churn pain point**, the typical exit timing, and an "
        "estimated saveable-user count. Runs on the full population (not the sidebar "
        "filters). In production this runs daily on the inflow/outflow feed."
    )

    recs, drift = run_diagnosis(n_users, int(seed))
    actionable = sorted(
        [r for r in recs if r["top_driver"] is not None],
        key=lambda r: r.get("est_saveable", 0),
        reverse=True,
    )
    total_saveable = sum(r.get("est_saveable", 0) for r in actionable)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cohorts analysed", f"{len(recs)}")
    k2.metric("With a dominant driver", f"{len(actionable)}")
    k3.metric("Est. total saveable users", f"{total_saveable:,.0f}")
    k4.metric("Drift alerts", f"{len(drift)}")

    if drift:
        st.markdown("#### ⚠️ Drift alerts — a cohort's #1 driver just changed")
        for a in drift:
            st.warning(
                f"**{a['city']}** — dominant churn driver shifted from "
                f"**{a['from_driver']}** to **{a['to_driver']}** "
                f"(hazard {a['from_hr']:.2f}× → {a['to_hr']:.2f}×) over the last "
                f"{a['window_days']} days. Likely a competitor promo — investigate retention offers."
            )

    st.markdown("#### 🔎 Churn pain-point feed")
    top_n = st.slider("Show top N cohorts (by saveable users)", 3, max(3, len(actionable)),
                      min(10, len(actionable)))
    conf_colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    for r in actionable[:top_n]:
        with st.container(border=True):
            st.markdown(
                f"{conf_colour.get(r['confidence'],'')} " + render_statement(r)
            )
            st.caption(f"💡 Recommended test-and-learn: {r['top_action']}")

    st.markdown("#### 🗺️ Where each driver dominates (cohorts by market)")
    matrix = pd.DataFrame(
        [{"market": r["city"], "driver": r["top_label"], "saveable": r.get("est_saveable", 0)}
         for r in actionable]
    )
    if not matrix.empty:
        piv = matrix.pivot_table(
            index="market", columns="driver", values="saveable", aggfunc="sum", fill_value=0
        )
        fig = px.imshow(
            piv, text_auto=".0f", aspect="auto", color_continuous_scale="Reds",
            labels=dict(color="est. saveable"),
        )
        fig.update_layout(title="Estimated saveable users by market × dominant driver")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### 🔬 Cohort drill-down")
    keys = [r["cohort"] for r in actionable]
    pick = st.selectbox("Inspect a cohort's full driver ranking", keys)
    chosen = next(r for r in recs if r["cohort"] == pick)
    dd = pd.DataFrame(chosen["drivers_ranked"])
    if not dd.empty:
        show = dd[["label", "hazard_ratio", "p_value", "prevalence", "logit_coef"]].copy()
        show.columns = ["Driver", "Hazard ratio", "p-value", "Prevalence", "Logistic coef"]
        st.dataframe(
            show.style.format(
                {"Hazard ratio": "{:.2f}×", "p-value": "{:.3f}",
                 "Prevalence": "{:.0%}", "Logistic coef": "{:+.2f}"}
            ),
            width="stretch",
        )
        st.caption(
            f"Cohort n={chosen['n_users']:,} · events={chosen['n_events']} · "
            f"baseline churn={chosen['baseline_churn']:.1%}. Hazard ratio > 1 means the "
            f"driver raises churn risk; p-value is its statistical significance within this cohort."
        )
