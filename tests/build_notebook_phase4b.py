"""
Generates notebooks/MGMT590_LendingClub_Segmentation_Phase4B.ipynb using nbformat.
Run once from the project root: python tests/build_notebook_phase4b.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# MGMT 590 — LendingClub Loan Default Risk (Indiana Borrowers)
## Phase 4B: Borrower Segmentation & Customer Profiling

**Course:** MGMT 59000, Summer 2026, Section DY2 — Purdue University

**Builds on:** Phase 1 (data pipeline), Phase 2 (EDA), Phase 3 (supervised
models), Phase 4A (`ExplainabilityEngine`, `RiskScoringEngine`,
configurable thresholds).

**Scope of this notebook:** develop a complete borrower-segmentation
framework using four new reusable Phase 4B modules --
`src/cluster_analysis.py`, `src/cluster_visualization.py`,
`src/segment_profiles.py`, `src/segmentation_engine.py` -- demonstrated
against the Phase 1 training split. Instead of predicting whether ONE
borrower will default, this phase identifies NATURAL borrower groups
with similar financial characteristics, complementing (not replacing)
Phase 3's supervised models.

**Not implemented here (explicitly deferred):** the Streamlit dashboard
(Phase 5).

> **Note on data:** as in Phases 1-4A, this notebook runs against
> whatever is currently at `data/splits/` and `models/`. If these still
> reflect the synthetic test fixture rather than the real ~37,515-row
> Indiana LendingClub extract, treat every specific segment name,
> profile number, and business claim below as illustrative of the
> *mechanism* only -- re-run Phases 1, 3, 4A, and this notebook once the
> genuine data is in place. Segment names in particular are assigned
> from the DATA (see Section 6), so on synthetic (effectively random)
> data they may not carry the same real-world meaning they would on
> genuine borrower data.
""")

code("""import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, utils
from src import cluster_analysis as ca
from src import cluster_visualization as cv
from src import segment_profiles as sprof
from src.segmentation_engine import SegmentationEngine
from src.risk_scoring import RiskScoringEngine

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
""")

code("""X_train, X_val, X_test, y_train, y_val, y_test = utils.load_splits()
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Production model: {config.PRODUCTION_MODEL_KEY}")
""")

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================
md("""## 1. Data Preparation for Clustering

### Which features drive cluster membership, and why

Clustering uses **numeric + ordinal features only**
(`config.CLUSTERING_NUMERIC_FEATURES` + `config.CLUSTERING_ORDINAL_FEATURES`
-- 14 numeric financial characteristics plus `grade`), deliberately
**excluding** the one-hot categorical columns (`home_ownership`,
`purpose`, `verification_status`, `term`, `application_type`,
`initial_list_status`) that drive the SUPERVISED models in Phase 3.

**Why excluded from clustering distance:** one-hot dummy columns are
binary (0/1); under Euclidean distance, several correlated dummies from
the same categorical variable can collectively dominate the distance
calculation over genuinely continuous financial signals like income or
DTI, producing clusters that mostly reproduce a categorical variable's
own categories rather than revealing new financial-behavior groupings.

**Why still valuable:** every excluded categorical column is fully used
downstream, in PROFILING each segment (Section 7) -- "what does Segment
2 typically look like" still reports typical home ownership, loan
purpose, etc. -- just not as an input to *defining* segment membership.

### Feature scaling
`StandardScaler` (z-score standardization), not `MinMaxScaler`: every
feature must contribute comparably to Euclidean distance regardless of
its native units (dollars vs. years vs. a percentage) -- without
scaling, `annual_inc` (tens of thousands) would dominate `dti` (a
percentage) in any distance-based clustering algorithm.

### Categorical encoding
`grade` is ordinal-encoded (A < B < ... < G, consistent with Phase 1's
encoding) then scaled alongside the numeric features, since it carries a
genuine ordering relevant to distance. Other categoricals are one-hot in
Phase 3's supervised pipeline but are excluded here entirely (see above).

### Outlier treatment
IQR-based winsorization (clip, not drop) at a wide 3x IQR multiplier --
K-Means and hierarchical clustering both use Euclidean distance, where a
single extreme value can pull a centroid or merge decision noticeably. A
wider multiplier than the classical 1.5x boxplot rule is used
deliberately, since 1.5x IQR would flag a large share of financial data
(income, revolving balance) as "outliers" when it is simply
legitimately right-skewed borrower variation. Clipping (not dropping)
means every borrower keeps a segment assignment.

### Dimensionality reduction
Evaluated separately in Section 2 below, purely for VISUALIZATION --
clustering itself operates on the full standardized feature space, not
a reduced-dimension projection (see Section 2 for why).
""")

code("""clipped_train = ca.clip_outliers(X_train)
n_clipped_features = sum(
    1 for col in config.CLUSTERING_NUMERIC_FEATURES
    if not np.allclose(clipped_train[col].to_numpy(), X_train[col].to_numpy(), equal_nan=True)
)
print(f"Clustering features: {len(config.CLUSTERING_NUMERIC_FEATURES)} numeric + {len(config.CLUSTERING_ORDINAL_FEATURES)} ordinal")
print(f"Features with at least one clipped outlier value: {n_clipped_features}")
""")

code("""clustering_preprocessor = ca.build_clustering_preprocessor()
X_clustering = clustering_preprocessor.fit_transform(clipped_train)
print(f"Clustering feature matrix shape: {X_clustering.shape}")
""")

# =============================================================================
# 2. DIMENSIONALITY REDUCTION
# =============================================================================
md("""## 2. Dimensionality Reduction: PCA vs. t-SNE vs. UMAP

All three are computed here for comparison, using a temporary K-Means(k=4)
labeling purely to color the scatter plots for visual comparison (the
engine's own fit, with the properly-selected k, happens in Section 4).

- **PCA**: a LINEAR projection preserving GLOBAL variance structure;
  deterministic; has a reusable `.transform()` for new data. Best when
  the axes themselves need stable, interpretable meaning across re-runs.
- **t-SNE**: preserves LOCAL neighborhood structure well but NOT global
  distances; no `.transform()` for new data (every run re-embeds
  jointly); best for visual cluster-separation sanity-checks only.
- **UMAP**: preserves both local AND some global structure better than
  t-SNE, and DOES support `.transform()` on new data.
""")

code("""quick_labels = ca.fit_kmeans(X_clustering, n_clusters=4).labels

pca_result = ca.fit_pca(X_clustering)
print(f"PCA: {pca_result.explained_variance_ratio.sum():.1%} of variance explained by 2 components")

sample_size = min(config.DIMENSIONALITY_REDUCTION_SAMPLE_SIZE, len(X_clustering))
rng = np.random.default_rng(config.RANDOM_STATE)
sample_idx = rng.choice(len(X_clustering), size=sample_size, replace=False)

tsne_result = ca.fit_tsne(X_clustering[sample_idx])
umap_result = ca.fit_umap(X_clustering[sample_idx])
print("t-SNE and UMAP fit on a", sample_size, "row sample for interactive speed.")
""")

code("""results = {"PCA": pca_result.coordinates[sample_idx], "t-SNE": tsne_result.coordinates}
if umap_result is not None:
    results["UMAP"] = umap_result.coordinates

fig = cv.plot_dimensionality_reduction_comparison(results, quick_labels[sample_idx])
plt.show()
""")

md("""**Comparison and recommendation:** PCA's 2 components typically explain
a modest share of total variance for this feature count (printed above)
-- expected, since borrower financial characteristics are genuinely
multi-dimensional rather than dominated by one or two axes. t-SNE/UMAP
often show visually tighter, more separated clusters than PCA precisely
BECAUSE they optimize for local separation rather than preserving true
distances -- useful for a compelling executive visual, but PCA's linear,
reusable projection is recommended as the PRIMARY method for this
project's ongoing visualizations (Section 5), since it can be
`.transform()`-ed consistently for any new borrower batch without
re-embedding the whole population, which t-SNE cannot do at all and
UMAP can do but at additional complexity. t-SNE/UMAP remain valuable
SECONDARY visual sanity-checks, shown here for comparison.
""")

# =============================================================================
# 3. OPTIMAL NUMBER OF CLUSTERS
# =============================================================================
md("""## 3. Determining the Optimal Number of Clusters

Four methods evaluated across k = 2 to 8:
- **Elbow method** (inertia): look for where adding another cluster
  stops reducing within-cluster variance much -- a visual judgment call.
- **Silhouette score**: higher is better (well-separated, cohesive clusters).
- **Calinski-Harabasz index**: higher is better (tends to favor MORE clusters).
- **Davies-Bouldin index**: LOWER is better (tends to favor FEWER, more compact clusters).
""")

code("""optimal_k_table = ca.evaluate_optimal_k(X_clustering, k_candidates=config.N_CLUSTERS_CANDIDATES)
optimal_k_table
""")

code("""recommended_k, explanation = ca.recommend_optimal_k(optimal_k_table)
print(explanation)
""")

code("""fig = cv.plot_optimal_k_analysis(optimal_k_table, recommended_k)
plt.show()
""")

md("""**Justification for the final cluster count:** rather than trusting
any single metric (each has known biases -- Calinski-Harabasz tends to
favor more clusters, Davies-Bouldin fewer), the recommendation above
averages each candidate k's RANK across silhouette, Calinski-Harabasz,
and Davies-Bouldin. The elbow method is reported alongside for visual
corroboration but excluded from the vote itself, since "the elbow" is a
visual judgment call without one unambiguous numeric rule.
`config.DEFAULT_N_CLUSTERS` reflects a business-driven choice of 4
segments as a reasonable, actionable number for underwriting policy and
marketing strategy -- `SegmentationEngine.fit(..., auto_select_k=True)`
is also available to defer entirely to the statistical recommendation
above.""")

# =============================================================================
# 4. CLUSTERING ALGORITHMS COMPARISON
# =============================================================================
md("""## 4. Clustering Algorithms Comparison

Four algorithms compared at the recommended k using the same validity
metrics as Section 3. Full advantages/disadvantages/business-
applicability/computational-considerations discussion lives in each
function's docstring in `src/cluster_analysis.py`
(`fit_kmeans`, `fit_agglomerative`, `fit_gaussian_mixture`, `fit_dbscan`).
""")

code("""algorithm_comparison_rows = []
for name, fit_fn in [("K-Means", ca.fit_kmeans), ("Agglomerative", ca.fit_agglomerative), ("Gaussian Mixture", ca.fit_gaussian_mixture)]:
    result = fit_fn(X_clustering, n_clusters=recommended_k)
    metrics = ca.evaluate_clustering(X_clustering, result.labels)
    algorithm_comparison_rows.append({"algorithm": name, "n_clusters_found": result.n_clusters, **metrics})

# DBSCAN doesn't take a target k -- included separately with a representative eps/min_samples.
dbscan_result = ca.fit_dbscan(X_clustering, eps=1.5, min_samples=10)
dbscan_metrics = ca.evaluate_clustering(X_clustering, dbscan_result.labels)
algorithm_comparison_rows.append({"algorithm": "DBSCAN", "n_clusters_found": dbscan_result.n_clusters, **dbscan_metrics})

pd.DataFrame(algorithm_comparison_rows)
""")

md("""**Business applicability summary:**
- **K-Means** (default): fast, produces convex/globular clusters that
  are easy to describe to a business audience ("borrowers near this
  centroid"); the right fit for reasonably continuous financial
  segments without highly irregular shapes.
- **Agglomerative Clustering**: useful as a CROSS-CHECK (does it agree
  with K-Means?) and for its dendrogram, but has no `.predict()` for new
  borrowers -- less suited as the deployed production algorithm.
- **Gaussian Mixture**: offers SOFT membership probabilities (a borrower
  can be 60% one segment, 40% another) -- a secondary lens for portfolio
  risk reporting when a hard boundary feels too rigid, but harder to
  explain to a non-technical audience than K-Means' simple
  nearest-centroid story.
- **DBSCAN**: typically produces one dominant cluster plus scattered
  noise on financial data that doesn't have well-separated density
  regions (borrower characteristics form a fairly continuous cloud) --
  the `n_clusters_found` above illustrates this limitation directly; not
  used as `SegmentationEngine`'s default algorithm for this reason.

**K-Means is used as `SegmentationEngine`'s default algorithm** for the
remainder of this notebook, consistent with `config.DEFAULT_CLUSTERING_ALGORITHM`.
""")

# =============================================================================
# 5. FIT THE SEGMENTATION ENGINE
# =============================================================================
md("""## 5. Fitting the Segmentation Engine

`SegmentationEngine` composes everything above (data preparation,
clustering, dimensionality reduction, profiling, business naming) into
one reusable, Streamlit-ready class -- mirroring `RiskScoringEngine` and
`ExplainabilityEngine`'s design from Phase 4A.""")

code("""engine = SegmentationEngine(n_clusters=recommended_k, algorithm="kmeans")
engine.fit(X_train, default_flags=y_train)
print(f"Fitted {engine.n_clusters} segments: {list(engine.fit_result.segment_names.values())}")
""")

# =============================================================================
# 6. CLUSTER VISUALIZATIONS
# =============================================================================
md("## 6. Cluster Visualizations")

code("""fig = engine.visualize_clusters(method="pca")
plt.show()
""")
md("""**Business interpretation:** each color is one borrower segment; the
axes are the two directions of greatest variance in the standardized
financial-characteristic feature space. Visually distinct, non-
overlapping colored regions indicate the segments are genuinely
separable financial profiles rather than an arbitrary partition.""")

code("""fig = engine.visualize_clusters(method="tsne")
plt.show()
""")

code("""profile_features = ["annual_inc", "dti", "loan_amnt", "int_rate", "emp_length_years", "revol_util"]
profile_df = X_train.copy()
profile_df["cluster"] = engine.fit_result.labels

fig = cv.plot_cluster_heatmap(profile_df, profile_features, segment_names=engine.fit_result.segment_names)
plt.show()
""")
md("""**Business interpretation:** red cells mean that segment is ABOVE the
overall-population average on that feature; blue means below. Reading
across one row gives that segment's full financial "signature" at a
glance.""")

code("""fig = cv.plot_parallel_coordinates(profile_df, profile_features, segment_names=engine.fit_result.segment_names)
plt.show()
""")

code("""fig = cv.plot_radar_chart(profile_df, profile_features, segment_names=engine.fit_result.segment_names)
plt.show()
""")

code("""fig = cv.plot_cluster_size_distribution(engine.fit_result.labels, segment_names=engine.fit_result.segment_names)
plt.show()
""")

code("""fig = cv.plot_feature_by_cluster(profile_df, "annual_inc", segment_names=engine.fit_result.segment_names, feature_label="Annual Income")
plt.show()
""")

code("""fig = cv.plot_feature_by_cluster(profile_df, "int_rate", segment_names=engine.fit_result.segment_names, feature_label="Interest Rate")
plt.show()
""")

code("""fig = cv.plot_feature_by_cluster(profile_df, "dti", segment_names=engine.fit_result.segment_names, feature_label="Debt-to-Income Ratio")
plt.show()
""")

code("""profile_df_with_default = profile_df.copy()
profile_df_with_default["default_flag"] = y_train.to_numpy()
fig = cv.plot_feature_by_cluster(profile_df_with_default, "default_flag", segment_names=engine.fit_result.segment_names, feature_label="Default Rate")
plt.show()
""")

code("""fig = cv.plot_feature_distribution_by_cluster(profile_df, "annual_inc", segment_names=engine.fit_result.segment_names, feature_label="Annual Income")
plt.show()
""")
md("""**Business interpretation:** the boxplot reveals within-segment
SPREAD that a bar-of-means chart hides -- two segments can have similar
average income but very different variability, which matters for how
tightly a lending policy can be tailored to that segment.""")

# =============================================================================
# 7. CLUSTER PROFILING
# =============================================================================
md("""## 7. Comprehensive Cluster Profiles

Every segment's typical income, DTI, loan amount, interest rate, grade,
employment length, home ownership, loan purpose, default rate, and
credit characteristics.""")

code("""engine.generate_cluster_profile()
""")

code("""for cluster_id in sorted(engine.fit_result.profiles.keys()):
    print(engine.describe_segment(cluster_id))
    print()
""")

# =============================================================================
# 8. BUSINESS LABELS
# =============================================================================
md("""## 8. Data-Driven Business Segment Names

Names are assigned FROM THE DATA (relative income/DTI/interest-rate/
default-rate z-scores across segments -- see
`segment_profiles.assign_segment_names`'s docstring for the exact
priority logic), never fixed in advance independent of the numbers.""")

code("""pd.Series(engine.fit_result.segment_names, name="segment_name").to_frame()
""")

# =============================================================================
# 9. BUSINESS RECOMMENDATIONS
# =============================================================================
md("""## 9. Business Recommendations by Segment

For every segment: primary risk level, lending recommendation, interest-
rate strategy, underwriting strategy, manual-review requirement,
marketing strategy, and portfolio-management notes.""")

code("""recommendations_table = pd.DataFrame([r.to_dict() for r in engine.recommend_business_actions().values()])
recommendations_table
""")

# =============================================================================
# 10. SEGMENT COMPARISON
# =============================================================================
md("""## 10. Segment Comparison Table

Income, interest rate, loan grade, DTI, employment length, default rate,
risk tier, and cluster size for every segment side-by-side.""")

code("""engine.compare_segments()
""")

md("""**How segments differ:** the table above is sorted by default rate
(riskiest first). Segments with similar interest rates but different
default rates suggest the CURRENT LendingClub-assigned grade/rate isn't
fully capturing the risk differentiation this clustering reveals --
directly relevant to Research Question 2 (are LendingClub grades
predictive?) and a candidate input for repricing policy.""")

# =============================================================================
# 11. RELATIONSHIP TO MACHINE LEARNING
# =============================================================================
md("""## 11. Relationship to the Supervised Machine Learning Models

Do high-risk clusters align with high PREDICTED default probabilities
from Phase 3's production model? Which clusters concentrate the most
actual defaults?""")

code("""ml_comparison = engine.compare_with_supervised_models()
ml_comparison[["segment_name", "n_borrowers", "mean_predicted_probability", "average_default_rate", "risk_tier"]]
""")

code("""fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ml_comparison["mean_predicted_probability"], ml_comparison["average_default_rate"], s=200,
           c=[cv._cluster_color(i) for i in range(len(ml_comparison))])
for _, row in ml_comparison.iterrows():
    ax.annotate(row["segment_name"], (row["mean_predicted_probability"], row["average_default_rate"]),
                textcoords="offset points", xytext=(8, 5), fontsize=9)
lims = [0, max(ml_comparison["mean_predicted_probability"].max(), ml_comparison["average_default_rate"].max()) * 1.15]
ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="Perfect agreement")
ax.set_xlabel("Mean Predicted Default Probability (Phase 3 production model)")
ax.set_ylabel("Actual Average Default Rate")
ax.set_title("Segmentation vs. Supervised Model Agreement", fontsize=13, fontweight="bold", loc="left")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
""")

md("""**Business interpretation:**
- **Do high-risk clusters align with high predicted default probabilities?**
  Segments falling near the diagonal reference line show strong
  agreement between the UNSUPERVISED clustering (based purely on
  financial-profile similarity) and the SUPERVISED model's learned risk
  ranking -- meaningful validation that both approaches are picking up
  the same underlying risk signal from different angles.
- **Which clusters contain the greatest concentration of defaults?**
  The segment(s) highest on both axes.
- **Can clustering explain patterns not captured by supervised learning?**
  A segment with a notably HIGHER actual default rate than its mean
  predicted probability would suggest the supervised model under-prices
  that specific financial-profile combination -- exactly the kind of
  gap segmentation is well-suited to surface, since it groups by overall
  profile similarity rather than a single learned probability.
- **How should Lending Club use both together?** The supervised model
  remains the primary basis for an INDIVIDUAL lending decision (it uses
  the full feature space and is validated against actual outcomes via
  Phase 3's metrics); segmentation adds a PORTFOLIO-level and POLICY-level
  lens -- setting origination limits, marketing strategy, and
  underwriting policy tiers by segment, informed by (and cross-checked
  against) the supervised model's risk ranking.
""")

# =============================================================================
# 12. RESEARCH QUESTION SUPPORT
# =============================================================================
md("""## 12. Research Question Support

**Which borrower segments represent the highest lending risk?**
Directly answered by Section 10's comparison table (sorted by default
rate) and Section 11's supervised-model cross-check -- the segment(s) at
the top of both rankings represent the highest lending risk, supported
by BOTH the segment's own historical default rate AND independent
agreement from the supervised production model's predicted probability.
This is a more robust conclusion than either signal alone, since the two
methods use different information (financial-profile similarity vs.
learned outcome prediction) and their agreement is itself evidence the
finding is not an artifact of one particular method.
""")

# =============================================================================
# 13. EXPORTABLE REPORTS
# =============================================================================
md("""## 13. Exportable Segment Reports

`SegmentationEngine.export_segment_summary()` returns an
`interpretation_utils.ExportableReport` -- `.to_markdown()` / `.to_json()`
output is ready for a future Streamlit `st.download_button()`.""")

code("""segment_report = engine.export_segment_summary()
print(segment_report.to_markdown()[:2500])
""")

# =============================================================================
# 14. SAVE ARTIFACTS
# =============================================================================
md("""## 14. Persisting Reusable Segmentation Artifacts

`SegmentationEngine.persist_segmentation_artifacts()` serializes the
fitted clustering model, clustering preprocessor, cluster centroids,
segment definitions (profile + recommendation per cluster), cluster
metadata, the segment-profile table, and the optimal-k analysis table --
everything a future Streamlit dashboard needs without recomputing
clustering from scratch.""")

code("""engine.persist_segmentation_artifacts()

print("Saved artifacts:")
print(f"  Clustering model:         {config.CLUSTERING_MODEL_PATH}")
print(f"  Clustering preprocessor:  {config.CLUSTERING_PREPROCESSOR_PATH}")
print(f"  Cluster centroids:        {config.CLUSTER_CENTROIDS_PATH}")
print(f"  Segment definitions:      {config.SEGMENT_DEFINITIONS_PATH}")
print(f"  Cluster metadata:         {config.CLUSTER_METADATA_PATH}")
print(f"  Segment profiles table:  {config.SEGMENT_PROFILES_PATH}")
print(f"  Optimal-k analysis:       {config.OPTIMAL_K_ANALYSIS_PATH}")
""")

# =============================================================================
# 15. PHASE TRANSITION
# =============================================================================
md("""## 15. Preparing for Phase 5 (Streamlit Dashboard)

### How borrower segmentation complements supervised learning

Phase 3's supervised models answer "what is THIS borrower's default
probability?" -- a precise, individually-validated prediction.
Segmentation answers a complementary question: "what natural GROUPS of
borrowers exist, and how do their overall financial profiles differ?"
Section 11 showed these two lenses agree directionally (high-predicted-
risk segments also show high actual default rates), which is itself
useful validation, while also giving Lending Club a portfolio/policy
tool (segment-level underwriting rules, marketing strategy, origination
limits) that a per-borrower probability alone doesn't provide.

### How `SegmentationEngine` will integrate into Streamlit

Every method returns either a `matplotlib.figure.Figure` (ready for
`st.pyplot(fig)`), a plain dataclass/DataFrame/string, or an
`interpretation_utils.ExportableReport` (ready for
`st.download_button()`). A borrower-detail page could call:

```python
engine = SegmentationEngine()  # cached via st.cache_resource; call .fit(X_train, y_train) once at startup
segment_name = engine.assign_segment(borrower_row).iloc[0]
st.metric("Borrower Segment", segment_name)
st.write(engine.describe_segment(cluster_id))
recommendation = engine.recommend_business_actions(cluster_id)
st.write(recommendation.lending_recommendation)
```

A portfolio-level page would call `engine.compare_segments()` and
`engine.visualize_clusters()` directly, and `engine.compare_with_supervised_models()`
to show the cross-validation view from Section 11.

### Public interfaces Phase 5 should use

| Module | Public interface |
|---|---|
| `src.cluster_analysis` | `build_clustering_preprocessor`, `clip_outliers`, `CLUSTERING_ALGORITHMS`, `evaluate_optimal_k`, `recommend_optimal_k` |
| `src.cluster_visualization` | `plot_dimensionality_reduction_scatter`, `plot_cluster_heatmap`, `plot_feature_by_cluster`, `plot_cluster_size_distribution` |
| `src.segment_profiles` | `SegmentProfile`, `SegmentRecommendation`, `build_segment_comparison_table` |
| `src.segmentation_engine` | `SegmentationEngine` (the primary interface -- fit once, then call any method above) |

### Explicitly deferred

- The Streamlit dashboard itself (Phase 5) — the architecture above is
  prepared for it; no dashboard code has been written in this phase.
- Integration testing, performance optimization, and deployment (Phase 6).
- Final code review, documentation, and presentation assets (Phase 7).
""")

nb["cells"] = cells
with open("notebooks/MGMT590_LendingClub_Segmentation_Phase4B.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Phase 4B notebook written with {len(cells)} cells.")
