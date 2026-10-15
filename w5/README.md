# Beauty & wellness customer return prediction

This is a reproducible predictive-modeling submission for estimating whether a
beauty or wellness customer returns within 60 days of a snapshot. Because no
Week 1 dataset or business problem was present in the workspace, the default
run uses seeded **synthetic illustrative data**. Its results are not company
evidence; the generated Word report states this limitation.

## Run

```powershell
python -m pip install -r requirements.txt
python beauty_wellness_modeling.py --verbose
```

The run creates `outputs/predictive_modeling_strategy.docx`, a model comparison,
held-out metrics, test predictions, feature importances, ROC/importance plots,
the synthetic CSV, and a fitted model bundle.

## Supply real data

```powershell
python beauty_wellness_modeling.py --input path\to\customer_returns.csv --output-dir outputs
```

CSV must include these columns:

* Numeric: `visits_last_90d`, `avg_ticket_usd`, `days_since_last_visit`,
  `discount_share` (0–1), and `satisfaction_score` (1–5).
* Categorical: `membership` and `service_category`.
* Binary target: `returned_within_60d` (0 or 1).

Rows should represent customers at a consistent snapshot, and predictors must
be measured before the 60-day outcome window. The script rejects missing values
and invalid targets instead of silently imputing or re-labeling them. Do not
include direct identifiers or information recorded after the prediction date.

## Modeling and evaluation

The script compares standardized logistic regression with a one-hot encoded
random forest using five-fold stratified cross-validation on the training set.
It selects the model by mean CV ROC-AUC, selects an F1 threshold from training
out-of-fold probabilities, and reports ROC-AUC, average precision, precision,
recall, F1, accuracy, specificity, and Brier score on the held-out split.
For repeated customer snapshots, use customer-grouped and time-based validation
before deployment; random splitting alone would be optimistic.

The fitted bundle is an example, not deployment approval. Review calibration,
subgroup performance, privacy/consent, threshold economics, and temporal drift
before using scores for customer outreach.
