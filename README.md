# Churn Diagnosis Engine + Test-and-Learn (synthetic)

A churn **diagnosis engine** for a music-streaming-style platform: it ingests daily
inflow/outflow data and, **for each cohort, tells you the #1 churn pain point**, the
typical exit timing, and how many users are saveable — then you act on it with the
test-and-learn simulator. All on **fake data with realistic, cohort-varying churn
drivers** planted in.

> Example output:
> *"Delhi · Full price · Growing (90–365d) — **Competitor pricing pressure** is the #1
> churn driver (hazard ratio 1.93×; exposed users churn at 1.9× the cohort baseline,
> typically exiting around **day 179**). ~220 users · est. **100 saveable**. Confidence:
> high."*

## How it works (the honest version)

1. **Plant** — `data_generator.py` runs a **survival simulation**: each user's
   "exit after N days" is a Weibull event time whose hazard rises with real drivers.
   Crucially, **each market has a different dominant driver** (`churn_model.MARKET_PROFILES`),
   so the engine has something market-specific to find.
2. **Recover** — `diagnosis.py` fits, **per cohort**, a **Cox proportional-hazards**
   model (timing + hazard ratios) and a **logistic** cross-check on the same candidate
   drivers, then ranks them by `effect × prevalence`.
3. **Validate** — on a 12k-user run the engine recovers the planted dominant driver in
   **~96% of cohorts**.

## Files

| File | Purpose |
|------|---------|
| `churn_model.py` | Single source of truth: candidate **DRIVERS**, per-market dominant driver, hazard weights. |
| `data_generator.py` | Survival-based synthetic data → inflow / outflow line items + cohort keys. |
| `cohort_engine.py` | Auto-builds `market × tier × tenure` cohorts with a sample-size guard + roll-up for thin cohorts. |
| `diagnosis.py` | **The core.** Per-cohort Cox PH + logistic, driver ranking, plain-English statements, **drift detection**. |
| `daily_ingest.py` | Shows the daily-feed shape (partitions + rolling window). Swap for real warehouse reads. |
| `insights_feed.py` | Headless CLI: runs the whole pipeline, prints the ranked feed + drift, exports JSON. |
| `app.py` | Streamlit dashboard — the **🧭 Cohort Diagnosis** tab surfaces all of the above. |

## Line items modelled

**Inflow** — User ID, City, Age, Gender, MAU, Non-MAU, Price tier, Age on platform
(0 = new), Time spent, Signup date.
**Outflow** — User ID, City, Age, Gender, Price tier, Exit-after-days, Avg engagement,
the 6 features 1/0 (daylist, blend, jam, AI DJ, Radio, Mix), Engagement-per-feature,
Exit-after-renewals, Competitor pricing on exit (0/1).

## Candidate drivers (what the engine is allowed to name)

`Lack of feature adoption` · `Low engagement` · `Inactivity (non-MAU)` ·
`Competitor pricing pressure` · `Younger-skewing audience`. (Price tier and tenure are
cohort *keys*, so they're held constant within a cohort, not diagnosed as drivers.)

## Drift detection

The engine compares each market's #1 driver in the recent signup window vs the prior
window and flags changes — e.g. it catches a planted competitor-promo shock as
*"Bangalore: Low engagement → Competitor pricing pressure"*. That's what makes it
**dynamic**: run it daily and it tells you when a pain point shifts.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Headless engine — prints the ranked feed + drift, optional JSON export
python insights_feed.py --top 10 --json feed.json

# Full dashboard (open the 🧭 Cohort Diagnosis tab)
streamlit run app.py

# Raw CSVs
python data_generator.py    # inflow.csv, outflow.csv, users_master.csv
```

## Caveats (say these to your boss before they ask)

- The engine ranks **associations, not proven causes** — that's exactly why each insight
  ships with a *recommended test*. Confirm with a **randomised holdout** (treatment vs
  control), measure the actual lift, feed it back.
- Real deployment needs a feature store + scheduled retraining on a rolling window; the
  synthetic version fakes the daily append.
- Thin cohorts roll up or are suppressed so you never get an "insight" from a handful of
  users.
