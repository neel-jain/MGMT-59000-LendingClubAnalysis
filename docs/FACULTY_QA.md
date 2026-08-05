# Faculty Q&A

Anticipated questions with prepared, honest answers. Where a real
number depends on the genuine dataset (not yet loaded in this
environment), the answer explains the *methodology* rather than
inventing a number.

---

**Q: Why Logistic Regression?**
It's the interpretable baseline every later model must beat — its
coefficients convert directly to odds ratios a credit committee or
regulator can read and defend without a SHAP layer. It's retained in
production alongside XGBoost specifically for that transparency, not
discarded once a stronger model was found.

**Q: Why Random Forest?**
It captures nonlinear relationships and feature interactions (e.g. "high
DTI matters more when income is also low") without hand-engineering
them, serving as a cross-check against both the linear baseline and the
boosted production model, with its own impurity + permutation
importance views.

**Q: Why XGBoost?**
Sequential boosting typically yields the strongest raw discrimination
on structured/tabular data, which is why it's the production scorer —
selected by test ROC-AUC among the three, not assumed in advance.

**Q: How was overfitting prevented?**
Three lines of defense: (1) a held-out test set never touched until
final evaluation; (2) Stratified 5-fold cross-validation during
hyperparameter search, so the chosen hyperparameters are validated
against data the final refit hasn't seen; (3) explicit regularization
in the search spaces themselves (e.g. XGBoost's `reg_alpha`/`reg_lambda`/
`gamma`, Random Forest's `max_depth`/`min_samples_leaf`). Learning
curves (Model Comparison page) additionally diagnose whether a
train-vs-validation gap suggests high variance.

**Q: How was data leakage prevented?**
The test set is carved out first, before any preprocessing statistic is
computed. Preprocessing (imputation medians, scaler mean/std, encoder
categories) lives INSIDE each model's scikit-learn `Pipeline`, so every
cross-validation fold refits its own statistics on only that fold's
training portion — a single preprocessor fit once on the full training
set (which is what Phase 1 originally produced for EDA use) would leak
each validation fold's own statistics into its training step if reused
across CV. The target variable itself was also defined to exclude loans
still in progress (`Current`, `Late`, `In Grace Period`), since including
them would use future/incomplete information.

**Q: How were thresholds selected?**
Two distinct threshold concepts exist, deliberately kept separate. The
model's binary DECISION threshold (Phase 3) is chosen per model to
minimize expected business cost, using a configurable relative cost
ratio between a false negative (a missed defaulter) and a false
positive (a wrongly-declined good borrower) — not the default 0.50. The
business RISK TIER boundaries (Low/Moderate/High/Very High Risk, Phase
4A) are a separate, JSON-backed configuration
(`reports/risk_threshold_config.json`) a credit-policy stakeholder can
edit directly, answering a broader question ("how do we communicate a
risk spectrum") than the binary accept/decline threshold does.

**Q: Why SHAP?**
SHAP has a rigorous game-theoretic foundation (Shapley values) and
provides EXACT (not approximate) explanations for both tree-based
models (`TreeExplainer`) and linear models (`LinearExplainer`) — one
consistent explainability framework across every model in the project,
at both the global (population) and local (individual borrower) level.

**Q: Why clustering/segmentation?**
Supervised learning answers "what is THIS borrower's risk?" but not
"what natural GROUPS of borrowers exist?" — a question relevant to
portfolio-level policy (marketing strategy, origination limits) that a
single per-borrower probability doesn't address. The segmentation
results are cross-validated against the supervised model's own
predictions (`compare_with_supervised_models()`), so it isn't an
unconnected side analysis.

**Q: How would this perform in production?**
Honestly: it's demo/portfolio-ready, not production-ready as-is. What's
already production-appropriate: the leakage-safe pipeline design, the
comprehensive automated test suite (265+ tests including edge cases and
integration tests), the configurable (not hardcoded) business rules, and
the measured performance characteristics (`PERFORMANCE_REPORT.md`).
What's explicitly NOT yet production-ready, stated plainly: no
authentication/authorization layer, no automated model-drift monitoring
or retraining cadence, and evaluation against a synthetic data fixture
rather than the full genuine dataset in this development environment.
`DEPLOYMENT.md` and `README.md`'s "Known Limitations" section cover this
in full.

**Q: Why did you build a segmentation model AND three supervised
models — isn't that redundant?**
They answer different questions with different assumptions (supervised:
learn from labeled outcomes to predict a specific target; unsupervised:
find structure without using the target at all) and are explicitly
cross-validated against each other rather than run independently — see
`SegmentationEngine.compare_with_supervised_models()`.

**Q: What was the hardest part of this project?**
Reasonable to answer honestly and specifically rather than generically
— e.g. designing the preprocessing-inside-`Pipeline` pattern correctly
to prevent leakage across every phase's cross-validation, or building
`SegmentationEngine.assign_segment()`'s nearest-centroid fallback so
every clustering algorithm (including ones without a native
`.predict()`) works uniformly through one interface.

**Q: What would you do differently if you started over?**
A defensible answer: design the `configurable_thresholds.py`/JSON
business-rule pattern from Phase 1 rather than introducing it in Phase
4A, since several earlier hardcoded constants (e.g. Phase 3's cost
ratio) would have benefited from the same hot-editable pattern from the
start.
