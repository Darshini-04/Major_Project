# PulseIQ research evaluation

## Research questions and hypotheses

**RQ1:** Do multi-window temporal RFM features improve next-month relative RFM segment prediction over a static 12-month RFM representation when the learner, chronological test population, and evaluation protocol are held constant?

**RQ2:** Do basket and other available behavioural features add consistent value beyond temporal RFM for next-segment and 90-day inactivity-proxy prediction?

**RQ3:** Are observed differences stable across chronological test periods and feature lookback windows?

**RQ4:** Do evidence-grounded LLM recommendations score better than deterministic rule-based recommendations when both receive identical customer evidence?

Hypotheses to test (not assumed conclusions): multi-window RFM and behavioural trend features may help distinguish future segment transitions and inactivity risk; gains may vary by period, class and target. Persistence may be a strong next-segment baseline because adjacent relative RFM states can remain unchanged. RQ4 remains untested because no LLM key is configured.

## Dataset and target definitions

- Dataset: UCI Online Retail II, `data/raw/online_retail_II.xlsx`; one UK online retailer, with observed transactions ending **2011-12-09**. No cross-retailer dataset or retention-treatment outcomes are available.
- The experiment constructed **88,515** customer-month rows covering **5,878** customers. There were **0 duplicate customer-month states**.
- Input windows use invoices observed before a month-end snapshot cutoff. Invoice lines are aggregated before customer-level features are calculated. Positive-purchase features exclude cancellation invoices and non-positive quantities or prices.
- Next-segment target: the same customer's heuristic relative RFM segment in the immediately following monthly state. It is not a business-ground-truth class, and monthly relative quintile cutoffs can change with the customer population.
- 90-day inactivity target: no positive purchase invoice in the half-open period `[snapshot cutoff, cutoff + 90 days)`. Rows are labeled only if the whole outcome horizon is covered by observations. This is an inactivity proxy, not contractual churn.
- No customer is removed solely for recurring across periods; results estimate temporal continuation among a partially known population, not cold-start performance for unseen customers.

## Protocol

The controlled experiment uses the same XGBoost configuration across feature ablations: 250 estimators, depth 4, learning rate 0.05, subsample and column sample 0.85, L2 regularization 1.0, histogram trees, four jobs, and seed 42. The earlier balanced logistic RFM result is retained as a separate legacy-estimator comparator; because its learner differs, it is not used to attribute gains to features.

Feature conditions are nested and distinct:

| Condition | Inputs |
|---|---|
| Static RFM | `R_12m`, `F_12m`, `M_12m` |
| Temporal RFM | R, F and M over 3, 6 and 12 months (9 features) |
| Temporal RFM + basket | Temporal RFM plus mean invoice value per customer-window over 3, 6 and 12 months (12 features) |
| Full current features | The preceding 12 plus 3-month frequency/spend changes, 12-month interpurchase interval, and 12-month tenure (16 features) |

All four feature conditions for a target and split use the same train and test rows. Test rows whose target class is absent from training are excluded consistently and counted. The next-segment persistence baseline predicts that a customer's current segment remains their next segment.

Test periods use expanding chronological training. Training labels are purged at the test boundary: next-segment training targets must be dated before the first test snapshot month; inactivity training horizons must end on or before the first test snapshot cutoff. Inactivity training rows without a complete horizon are excluded. Scores use the default 0.5 threshold for binary classification. Raw probability calibration metrics are diagnostics only; no probability calibration transform was applied.

The report includes three chronological test windows per target. The latest next-segment test is the existing July–November 2011 period. The inactivity test is necessarily shorter because of its 90-day horizon. Alternative lookback experiments compare 3-, 6- and 12-month feature windows on the latest eligible test window while keeping target definitions and customer-month population fixed.

## Controlled temporal ablation results

### Next-segment macro-F1

All feature ablations are evaluated on identical test rows within each period. Test class supports are included in the JSON artifact.

| Test snapshot period | Test rows | Persistence | Static RFM XGBoost | Temporal RFM XGBoost | + basket | Full 16 features |
|---|---:|---:|---:|---:|---:|---:|
| 2010-12–2011-02 | 12,455 | 0.7799 | 0.6586 | 0.6717 | 0.6706 | 0.6615 |
| 2011-04–2011-06 | 12,633 | 0.7342 | 0.6669 | 0.6693 | 0.6712 | 0.6706 |
| 2011-07–2011-11 | 20,646 | 0.6686 | 0.6623 | 0.6727 | 0.6721 | 0.6829 |

The full feature set exceeds static-RFM XGBoost on the latest split, but not on the two earlier splits. Persistence exceeds every learned ablation on the first two splits and is below the full feature model on the latest split. The results therefore do **not** support a claim that the full temporal feature set is robustly superior across time. On the latest window, the incremental 16-feature result is macro-F1 0.6829; temporal RFM alone is 0.6727 and the 12-feature temporal-RFM-plus-basket condition is 0.6721.

### 90-day inactivity-proxy results

| Test snapshot period | Test rows | Positive rate | Static RFM XGBoost macro-F1 | Temporal RFM macro-F1 | + basket macro-F1 | Full macro-F1 | Full ROC-AUC | Full PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2010-10–2010-11 | 8,207 | 0.6163 | 0.6351 | 0.6474 | 0.6457 | 0.6558 | 0.7469 | 0.7943 |
| 2011-02–2011-03 | 8,608 | 0.6276 | 0.6745 | 0.6797 | 0.6779 | 0.6746 | 0.7746 | 0.8235 |
| 2011-07–2011-08 | 8,672 | 0.5367 | 0.6964 | 0.6896 | 0.6890 | 0.6881 | 0.7758 | 0.7626 |

The full feature set has higher macro-F1 than static RFM XGBoost in the first two windows, but lower macro-F1 in the latest window. Its PR-AUC is high in the first two windows but lower than those values in the latest window. Performance differences are period-dependent; the experiment does not establish robust improvement.

### Inactivity precision, recall, confusion matrices and class counts

The order below is **no inactivity**, then **90-day inactivity**. Confusion-matrix rows are actual class and columns are predicted class. The full-feature model results are:

| Test period | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| 2010-10–2010-11 | No inactivity | 0.5418 | 0.7269 | 0.6208 | 3,149 |
|  | Inactivity | 0.7840 | 0.6172 | 0.6907 | 5,058 |
| 2011-02–2011-03 | No inactivity | 0.5508 | 0.7520 | 0.6359 | 3,206 |
|  | Inactivity | 0.8121 | 0.6361 | 0.7134 | 5,402 |
| 2011-07–2011-08 | No inactivity | 0.7932 | 0.4973 | 0.6113 | 4,018 |
|  | Inactivity | 0.6717 | 0.8881 | 0.7649 | 4,654 |

Corresponding full-feature confusion matrices are `[[2289, 860], [1936, 3122]]`, `[[2411, 795], [1966, 3436]]`, and `[[1998, 2020], [521, 4133]]`. The latest period has substantially different class balance from the earlier periods, reinforcing the need to report per-period class metrics rather than a single aggregate.

For next-segment prediction, the test class supports are:

| Test period | At Risk | Champions | Loyal | Needs Attention | Potential Loyalists |
|---|---:|---:|---:|---:|---:|
| 2010-12–2011-02 | 2,715 | 2,820 | 1,787 | 2,187 | 2,946 |
| 2011-04–2011-06 | 2,748 | 2,809 | 1,925 | 2,264 | 2,887 |
| 2011-07–2011-11 | 4,549 | 4,621 | 3,199 | 3,495 | 4,782 |

Per-class precision/recall/F1 and confusion matrices for every model and split, along with train/test class counts, Brier scores and calibration bins, are recorded in the generated experiment JSON.

## Lookback-window comparison

These comparisons use the latest eligible periods, with 3-/6-/12-month feature windows and the combined 3/6/12 window. The target labels and evaluation rows remain fixed.

| Feature lookback | Next-segment macro-F1 (20,646 rows) | Inactivity macro-F1 (8,672 rows) | Inactivity ROC-AUC | Inactivity PR-AUC |
|---|---:|---:|---:|---:|
| 3 months | 0.3581 | 0.6388 | 0.7173 | 0.6867 |
| 6 months | 0.4836 | 0.6680 | 0.7490 | 0.7203 |
| 12 months | 0.6714 | 0.6986 | 0.7750 | 0.7600 |
| Combined 3/6/12 months plus basket features | 0.6721 | 0.6890 | 0.7767 | 0.7635 |

On this latest period, 12-month inputs outperform shorter single windows on macro-F1 for both tasks. Combining multiple windows slightly raises inactivity ranking metrics over the 12-month-only condition, while its macro-F1 is lower. This is one dataset and one latest period, not a general window-selection result.

On the latest purged split, the separate static-RFM logistic comparator scored next-segment macro-F1 **0.6345** and inactivity macro-F1 **0.7014** (positive-class F1 **0.7456**, ROC-AUC **0.7627**, PR-AUC **0.7392**). Its churn macro-F1 exceeds the full-feature XGBoost score, while its positive-class recall and ranking metrics are lower. This illustrates why conclusions depend on the metric and why the estimator-matched ablations are reported separately.

## Split, leakage and sample audit

| Target test period | Training candidates before purge | Incomplete/unlabeled train rows | Rows purged for target overlap | Training rows after purge | Test rows |
|---|---:|---:|---:|---:|---:|
| Next segment, 2010-12–2011-02 | 32,425 | 0 | 4,155 | 28,270 | 12,455 |
| Next segment, 2011-04–2011-06 | 49,098 | 0 | 4,218 | 44,880 | 12,633 |
| Next segment, 2011-07–2011-11 | 61,731 | 0 | 4,218 | 57,513 | 20,646 |
| Inactivity, 2010-10–2010-11 | 24,329 | 0 | 10,044 | 14,285 | 8,207 |
| Inactivity, 2011-02–2011-03 | 41,004 | 0 | 12,734 | 28,270 | 8,608 |
| Inactivity, 2011-07–2011-08 | 62,608 | 0 | 12,996 | 49,612 | 8,672 |

The earlier saved evaluation split did not purge labels crossing the test boundary. The new results are the controlled, purged comparisons. The previous inactivity artifact evaluated 8,368 rows because the old training pipeline first restricted samples to those with a next-segment label; this experiment uses all rows eligible for the inactivity target, giving 8,672 rows on the latest window. The two inactivity populations differ, so their scores should not be compared as if only the model changed.

Snapshot features are computed from data before each cutoff, and a direct audit found no duplicated customer-month rows. The purge removes future outcome information from training relative to each test origin. A remaining limitation is population dependence: segment labels are computed from relative monthly quintiles, and the same customers occur in earlier training and later testing.

## Historical result retained for reference

The pre-existing `outputs/model_metrics.json` and the first version of this report recorded a non-purged holdout: next-segment static logistic macro-F1 0.6366 versus temporal XGBoost 0.6865; inactivity positive-class F1 0.7357 versus 0.7594. That comparison changes both estimator and feature set and its train labels can cross into the holdout boundary. It is retained as the application's existing evaluation artifact, not treated as the controlled ablation result. **The experiment runner did not overwrite models, predictions, SHAP artifacts, or the existing model metrics.**

## Recommendation evaluation protocol (not run)

No LLM API key was configured for this run, so no LLM recommendations or comparative scores were generated. To make a future comparison reproducible:

1. Freeze a versioned set of customer profiles and evidence fields; generate both recommendations from identical evidence and a fixed prompt/model/temperature configuration.
2. Select 60 customers by a fixed random seed, stratified by observed RFM segment and available risk-evidence level. Store failed calls and exclusions.
3. Remove provider identifiers, randomize output order, and have two independent raters score each output. Adjudicate disagreements without seeing provider identity.
4. Score four dimensions from 1–5: **grounding** (unsupported claims → every claim traceable to evidence), **relevance** (unrelated → directly addresses observed customer state), **specificity** (generic → tailored to supplied facts), and **actionability** (no feasible step → concrete, measurable next action).
5. Report per-dimension distributions, paired differences, rater agreement, and API failure rates. Do not infer retention lift or business impact without a controlled intervention and observed outcomes.

This is a proposed protocol only; there are no LLM evaluation results in this report.

## Conclusion and limitations

The purged, estimator-matched experiments show that temporal and behavioural feature gains vary across targets and time. Full current features are not uniformly better than static RFM; persistence is a particularly strong next-segment reference on earlier periods. The 12-month lookback dominates single shorter lookbacks on the latest split, but multi-window results do not establish general superiority. These findings support continued evaluation, not novelty or business-impact claims.

The dataset covers one retailer and ends in 2011. It contains no treatment-response data, so the retention advisor cannot be evaluated for causal lift. Segment names are relative heuristics, inactivity is a proxy, raw probabilities are not post-hoc calibrated, and no configured LLM comparison was run. Cross-dataset validation, external temporal validation, recommendation rating, and prospective business testing remain future work.

## Reproduction

From the repository root, run the non-destructive experiment runner with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend backend/.venv/bin/python -m app.ml.research_experiments
```

Each run creates a new `outputs/research_experiments/run_<UTC timestamp>/ablation_results.json`. The runner reads the configured workbook and writes no model artifacts or production predictions. The output records the dataset SHA-256, feature groups, exact test periods, class distributions, per-class metrics, confusion matrices, calibration diagnostics, purge counts, and limitations.
