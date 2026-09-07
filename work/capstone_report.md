# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Sameer
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Sameer99-star/flyrank-ml-internship-starter
- **Date:** September 2026

## 1. Problem framing

This capstone asks whether a simple, interpretable scoring approach can identify content pages that should be prioritized for human review and possible refresh using content age and recent performance trend signals.

The unit of analysis is an individual content page. The output is a ranked prioritization score and action queue. The human action supported by the output is to review higher-ranked pages first and decide whether a refresh, further investigation, or monitoring is appropriate.

The cost of a wrong call is primarily wasted editorial effort: a page may be reviewed or refreshed even though it does not need intervention, or a page needing attention may receive a lower priority. Therefore, the score is used as decision support rather than automatic execution.

Data and ML help by making the prioritization process repeatable and transparent across a large content dataset instead of relying only on manual selection.

## 2. Data safety

The analysis uses the anonymized FlyRank content refresh dataset containing 30,000 content-level rows and 44 columns.

The primary signals used in the scoring analysis are:

- `days_since_last_update`
- `trend_pct`

The analysis also references `trend_direction`, `freshness_tier`, `ctr`, and `avg_position` for interpretation and reporting.

Pseudonymous `content_id` values are used only to identify and rank rows. They are not used as predictive features.

No client names, domains, private search queries, credentials, or raw private exports are included in the public research paper.

A leakage check was considered because `trend_direction` and `trend_pct` are derived performance signals. They are treated as observed/current signals for prioritization rather than as a future outcome label. No future-window outcome is used to train or evaluate a predictive model.

This work does not claim to predict Google's ranking algorithm or prove that refreshing a page causes improved search performance.

## 3. Baseline

The baseline is an age-only prioritization score:

`baseline_score = days_since_last_update / 30`

Older pages therefore receive higher baseline priority.

The capstone analysis extends this transparent baseline with a recent-decline component:

`model_score = days_since_last_update / 30 + abs(negative trend_pct) / 10`

Only negative `trend_pct` contributes to the decline component.

This is a fair and interpretable comparison because the extended score preserves the baseline age signal and adds one clearly defined performance signal.

On the 30,000-row dataset, the baseline score ranged from 0.033 to 12.433, while the extended score ranged from 0.033 to 22.433.

The comparison focuses on how adding recent decline changes prioritization ranks. It is not presented as evidence of causal business impact.

## 4. Model / analysis

The method is an interpretable rule-based prioritization score rather than a trained predictive model.

The exact features used in the score are:

- `days_since_last_update`
- `trend_pct`

The score is:

`score = max(days_since_last_update, 0) / 30 + abs(min(trend_pct, 0)) / 10`

The first component increases with content age. The second component increases when recent performance trend is negative.

Pages are sorted by descending score, with `days_since_last_update` used as a secondary ordering signal.

The resulting ranking is intended to answer:

> Which content pages should receive earlier human review based on age and recent decline?

No separate predictive target is defined. The output is therefore a prioritization score, not a probability of future performance or a predicted refresh outcome.

The approach was deliberately kept simple so that an editor can understand why a page received a high score.

## 5. Evaluation

This capstone is a ranking and decision-support analysis rather than a supervised prediction task. Therefore, classification metrics such as accuracy, precision, recall, or AUC are not appropriate for the current analysis.

The evaluation compares the age-only baseline ranking with the extended age-plus-decline ranking on the same 30,000 rows.

The extended score changes rankings by incorporating recent negative trend in addition to content age. The highest-ranked candidates are generally pages that are both relatively old and experiencing strong negative recent trends.

Examples from the top-ranked output include pages with approximately 300–373 days since their last update and trend values close to -100.

The score distribution also shows that the extended approach can substantially increase prioritization for pages with negative trends. The extended score reaches 22.433 compared with a maximum baseline score of 12.433.

This evaluation measures ranking differences, not prediction accuracy or business impact. No randomized experiment was performed, so the results should be interpreted as directional decision-support evidence.

## 6. Interpretation

The analysis finds that combining content age with recent negative performance creates a stronger prioritization signal than age alone.

The highest-ranked candidates tend to have two characteristics:

1. They have not been updated for a relatively long period.
2. Their recent trend is strongly negative.

For example, the highest-ranked candidate in the analysis had:

- `days_since_last_update`: 373
- `trend_pct`: -100
- baseline score: 12.433
- extended score: 22.433

Other high-ranked candidates show similar combinations of older content and negative trend.

The freshness summary also shows a clear difference across freshness tiers. The 181+ day group contains 174 pages with a median score of approximately 10.03 and median days since update of 211.

An important observation is that the baseline produces a prioritization based only on age, while the extended score gives additional weight to recent decline.

The analysis does not establish that these pages will improve after a refresh. It only identifies them as higher-priority candidates for human investigation.

## 7. Recommendation

The ranked output should be used as an editorial review queue.

A practical workflow is:

1. Start with the highest-ranked pages.
2. Review the page's current content, search intent, and business relevance.
3. Check whether the observed decline is meaningful or caused by another factor.
4. Decide whether the page should be refreshed, investigated further, or monitored.
5. Record the editorial decision and monitor future performance.

The action labels and reason codes are:

- **REFRESH** — candidate for human review because of age and negative trend.
- **AGE_AND_DECLINE** — the reason code explaining the prioritization.

The score should not automatically trigger a refresh. Human review remains necessary because the dataset does not capture every factor that may influence search performance.

The current baseline assigns a positive prioritization score to all rows because the scoring formula incorporates age and/or negative trend. Therefore, the result should be interpreted as a continuous ranking of review priority rather than a claim that all 30,000 pages actually require refreshing.

Confidence is moderate for the narrow decision-support task of ranking pages using the selected signals. Confidence is low for any claim about causal refresh impact or universal search behavior.

## 8. Reproducibility

The analysis is implemented in the capstone notebook:

`work/notebooks/capstone.ipynb`

Repository:

https://github.com/Sameer99-star/flyrank-ml-internship-starter

The notebook loads:

`data/raw/content_refresh_anonymized.csv`

and writes analysis artifacts under:

- `work/figures/`
- `work/outputs/`

The analysis is deterministic because the scoring method does not involve randomized model training. No random seed is required for the scoring calculations.

The main scoring logic is:

```python
model_df["score"] = (
    model_df["days_since_last_update"].fillna(0).clip(lower=0) / 30
    + model_df["trend_pct"].fillna(0).clip(upper=0).abs() / 10
)
