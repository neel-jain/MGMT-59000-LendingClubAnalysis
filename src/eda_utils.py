"""
eda_utils.py
============
Phase 2 reusable module: exploratory data analysis, descriptive
statistics, business-oriented visualizations, and statistical testing
helpers for the MGMT 590 LendingClub Loan Default Risk capstone project.

This module is ADDITIVE to Phase 1 — it does not modify `config.py` or
`utils.py`. It imports from both and builds a higher-level analysis layer
on top of the cleaned dataset those modules already produce, so Phase 3
(modeling) and Phase 5 (Streamlit app) can also import these plotting and
testing helpers without duplicating logic.

Organized into sections:
    1. Plot styling / shared constants
    2. Dataset overview helpers
    3. Descriptive statistics
    4. Distribution & relationship visualizations
    5. Default-rate business visualizations
    6. Statistical testing (correlation, chi-square, t-test, ANOVA, CI, effect size)
    7. Feature-relationship / multicollinearity helpers

All plotting functions return the created `matplotlib.figure.Figure` so
notebooks can display, save, or further customize them, and follow a
consistent signature: business title, optional subtitle, labeled axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import config, labels, utils

logger = utils.get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. PLOT STYLING / SHARED CONSTANTS
# ---------------------------------------------------------------------------

# A single, consistent visual identity across every figure in the
# notebook — this is what separates an "executive-ready" deliverable from
# a pile of default-matplotlib charts.
PALETTE_SEQUENTIAL = "Blues_d"
PALETTE_DIVERGING = "RdYlGn_r"
COLOR_DEFAULT = "#C0392B"     # red — used consistently for "default" class
COLOR_PAID = "#2E86AB"        # blue — used consistently for "fully paid" class
FIGSIZE_STANDARD = (9, 5.5)
FIGSIZE_WIDE = (12, 5.5)
FIGSIZE_GRID = (14, 10)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 100


def _apply_titles(ax: plt.Axes, title: str, subtitle: Optional[str] = None) -> None:
    """
    Apply a consistent professional title/subtitle treatment to an Axes.

    Parameters
    ----------
    ax : plt.Axes
    title : str
        Main, bolded title.
    subtitle : str, optional
        Smaller descriptive subtitle rendered just below the title.
    """
    if subtitle:
        ax.set_title(f"{title}\n", fontsize=14, fontweight="bold", loc="left")
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=10, color="dimgray", style="italic", ha="left",
        )
    else:
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left")


# ---------------------------------------------------------------------------
# 2. DATASET OVERVIEW HELPERS
# ---------------------------------------------------------------------------


@dataclass
class DatasetOverview:
    """Container for the Section 1 executive dataset summary."""

    n_rows: int
    n_columns: int
    columns: List[str]
    dtypes: pd.Series
    missing_summary: pd.DataFrame
    duplicate_row_count: int
    target_balance: pd.Series


def build_dataset_overview(
    df: pd.DataFrame, target_column: str = config.TARGET_COLUMN
) -> DatasetOverview:
    """
    Build the executive dataset overview used to open the EDA notebook.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (output of Phase 1's `utils.clean_dataset`).
    target_column : str
        Name of the binary target column.

    Returns
    -------
    DatasetOverview
    """
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_summary = pd.DataFrame(
        {"missing_count": missing_counts, "missing_pct": missing_pct}
    )
    missing_summary = missing_summary[missing_summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )

    overview = DatasetOverview(
        n_rows=len(df),
        n_columns=df.shape[1],
        columns=list(df.columns),
        dtypes=df.dtypes,
        missing_summary=missing_summary,
        duplicate_row_count=int(df.duplicated().sum()),
        target_balance=df[target_column].value_counts(normalize=True).round(4),
    )
    logger.info(
        "Dataset overview built: %d rows, %d cols, %d duplicate rows, "
        "%d cols with missing values.",
        overview.n_rows, overview.n_columns, overview.duplicate_row_count,
        len(overview.missing_summary),
    )
    return overview


# Human-readable variable descriptions for the executive summary table.
# Centralized here (not re-derived from raw LendingClub documentation on
# every run) so Phase 5's Streamlit app can reuse the same glossary for
# in-app tooltips.
VARIABLE_DESCRIPTIONS: Dict[str, str] = {
    "loan_amnt": "Dollar amount of the loan requested by the borrower.",
    "term": "Number of monthly payments (36 or 60 months).",
    "int_rate": "Annual interest rate assigned to the loan (%).",
    "installment": "Fixed monthly payment owed by the borrower ($).",
    "grade": "LendingClub-assigned risk grade (A = lowest risk, G = highest).",
    "home_ownership": "Borrower's home ownership status (RENT/MORTGAGE/OWN/OTHER).",
    "annual_inc": "Self-reported annual income ($).",
    "verification_status": "Whether income was verified by LendingClub.",
    "purpose": "Borrower-stated reason for the loan.",
    "dti": "Debt-to-income ratio, excluding mortgage, based on reported income.",
    "delinq_2yrs": "Number of 30+ day delinquencies in the past 2 years.",
    "open_acc": "Number of open credit lines in the borrower's credit file.",
    "pub_rec": "Number of derogatory public records.",
    "revol_bal": "Total revolving credit balance ($).",
    "revol_util": "Revolving line utilization rate (%) — balance vs. total credit.",
    "total_acc": "Total number of credit lines currently in the credit file.",
    "initial_list_status": "Initial listing status of the loan (whole/fractional).",
    "application_type": "Individual vs. joint loan application.",
    "mort_acc": "Number of mortgage accounts.",
    "pub_rec_bankruptcies": "Number of public record bankruptcies.",
    "emp_length_years": "Borrower's employment length in years (parsed; 10 = 10+).",
    config.TARGET_COLUMN: "Binary target: 1 = Charged Off/Default, 0 = Fully Paid.",
}


def variable_description_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a variable-name / dtype / description reference table for the
    executive summary section.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: variable, dtype, description.
    """
    rows = [
        {
            "variable": col,
            "dtype": str(df[col].dtype),
            "description": VARIABLE_DESCRIPTIONS.get(col, "See project data dictionary."),
        }
        for col in df.columns
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------


def numeric_descriptive_stats(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Compute an extended descriptive-statistics table (beyond
    `DataFrame.describe()`), including variance, skewness, and kurtosis,
    for every numeric column supplied.

    Parameters
    ----------
    df : pd.DataFrame
    columns : Sequence[str]
        Numeric columns to summarize.

    Returns
    -------
    pd.DataFrame
        One row per variable with mean, median, std, variance, min, max,
        Q1, Q3, skewness, and excess kurtosis.
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found — skipping descriptive stats.", col)
            continue
        series = df[col].dropna()
        rows.append(
            {
                "variable": col,
                "mean": series.mean(),
                "median": series.median(),
                "std_dev": series.std(),
                "variance": series.var(),
                "min": series.min(),
                "q1_25pct": series.quantile(0.25),
                "q3_75pct": series.quantile(0.75),
                "max": series.max(),
                "skewness": stats.skew(series, nan_policy="omit"),
                "excess_kurtosis": stats.kurtosis(series, nan_policy="omit"),
                "n": series.count(),
            }
        )
    return pd.DataFrame(rows).set_index("variable").round(3)


def categorical_frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Build a frequency table (count + percentage) for a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
    column : str

    Returns
    -------
    pd.DataFrame
        Columns: count, percentage. Sorted descending by count.
    """
    counts = df[column].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "percentage": pct})


def interpret_skew_kurtosis(skewness: float, kurtosis: float) -> str:
    """
    Translate skewness/kurtosis values into a plain-language description,
    used to auto-generate interpretive text under descriptive tables.

    Parameters
    ----------
    skewness : float
    kurtosis : float
        Excess kurtosis (0 = normal-like tails).

    Returns
    -------
    str
        Short interpretive sentence.
    """
    if skewness > 1:
        skew_desc = "strongly right-skewed (a long tail of high values)"
    elif skewness > 0.5:
        skew_desc = "moderately right-skewed"
    elif skewness < -1:
        skew_desc = "strongly left-skewed (a long tail of low values)"
    elif skewness < -0.5:
        skew_desc = "moderately left-skewed"
    else:
        skew_desc = "roughly symmetric"

    if kurtosis > 1:
        kurt_desc = "heavier-tailed than normal (more outlier risk)"
    elif kurtosis < -1:
        kurt_desc = "lighter-tailed than normal (fewer extreme values)"
    else:
        kurt_desc = "close to normal-tailed"

    return f"Distribution is {skew_desc} and {kurt_desc}."


# ---------------------------------------------------------------------------
# 4. DISTRIBUTION & RELATIONSHIP VISUALIZATIONS
# ---------------------------------------------------------------------------


def plot_categorical_distribution(
    df: pd.DataFrame, column: str, title: str, subtitle: Optional[str] = None,
    order_by_count: bool = True, horizontal: bool = False,
) -> plt.Figure:
    """
    Bar chart of category frequencies with count labels.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Categorical column to plot.
    title, subtitle : str
        Professional title/subtitle text.
    order_by_count : bool
        Sort bars descending by frequency.
    horizontal : bool
        Render as a horizontal bar chart (useful for long category labels).

    Returns
    -------
    plt.Figure
    """
    counts = df[column].value_counts()
    if not order_by_count:
        counts = counts.sort_index()

    # Map raw category values to friendly labels before plotting so the
    # bars and their tick labels stay aligned.
    category_labels = [labels.category_label(column, v) for v in counts.index]
    column_label = labels.column_label(column)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    if horizontal:
        sns.barplot(y=category_labels, x=counts.values, ax=ax,
                    palette=PALETTE_SEQUENTIAL, hue=category_labels, legend=False)
        ax.set_xlabel("Number of Loans")
        ax.set_ylabel(column_label)
        for i, v in enumerate(counts.values):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9)
    else:
        sns.barplot(x=category_labels, y=counts.values, ax=ax,
                    palette=PALETTE_SEQUENTIAL, hue=category_labels, legend=False)
        ax.set_ylabel("Number of Loans")
        ax.set_xlabel(column_label)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_numeric_distribution(
    df: pd.DataFrame, column: str, title: str, subtitle: Optional[str] = None,
    bins: int = 40, show_kde: bool = True,
) -> plt.Figure:
    """
    Histogram + KDE overlay for a numeric column, annotated with mean and
    median reference lines.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    title, subtitle : str
    bins : int
    show_kde : bool
        Overlay a kernel density estimate.

    Returns
    -------
    plt.Figure
    """
    series = df[column].dropna()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sns.histplot(series, bins=bins, kde=show_kde, color=COLOR_PAID, ax=ax, edgecolor="white")
    ax.axvline(series.mean(), color=COLOR_DEFAULT, linestyle="--", linewidth=1.5,
               label=f"Mean = {series.mean():,.1f}")
    ax.axvline(series.median(), color="black", linestyle=":", linewidth=1.5,
               label=f"Median = {series.median():,.1f}")
    ax.set_xlabel(labels.column_label(column))
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_boxplot(
    df: pd.DataFrame, column: str, by: Optional[str] = None,
    title: str = "", subtitle: Optional[str] = None,
) -> plt.Figure:
    """Boxplot of a numeric column, optionally grouped by a categorical column."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    if by:
        sns.boxplot(data=df, x=by, y=column, ax=ax, palette=PALETTE_SEQUENTIAL,
                    hue=by, legend=False)
        ax.set_xlabel(labels.column_label(by))
        ax.set_xticklabels([labels.category_label(by, t.get_text()) for t in ax.get_xticklabels()])
        ax.tick_params(axis="x", rotation=30)
    else:
        sns.boxplot(data=df, y=column, ax=ax, color=COLOR_PAID)
    ax.set_ylabel(labels.column_label(column))
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_violin(
    df: pd.DataFrame, column: str, by: str,
    title: str = "", subtitle: Optional[str] = None,
) -> plt.Figure:
    """Violin plot of a numeric column split by a categorical/target column."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sns.violinplot(data=df, x=by, y=column, ax=ax, palette=PALETTE_DIVERGING,
                    hue=by, legend=False, cut=0)
    ax.set_xlabel(labels.column_label(by))
    ax.set_ylabel(labels.column_label(column))
    ax.set_xticklabels([labels.category_label(by, t.get_text()) for t in ax.get_xticklabels()])
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_scatter(
    df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
    title: str = "", subtitle: Optional[str] = None, sample_n: Optional[int] = 3000,
) -> plt.Figure:
    """
    Scatterplot of two numeric variables, optionally colored by a third
    (typically the target). Large datasets are subsampled for
    render-speed and readability (does not affect any statistical test —
    tests always use the full dataset).
    """
    plot_df = df if (sample_n is None or len(df) <= sample_n) else df.sample(
        sample_n, random_state=config.RANDOM_STATE
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    palette = {0: COLOR_PAID, 1: COLOR_DEFAULT} if hue == config.TARGET_COLUMN else None
    sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, ax=ax, alpha=0.5,
                     palette=palette, s=25)
    ax.set_xlabel(labels.column_label(x))
    ax.set_ylabel(labels.column_label(y))
    if hue:
        ax.legend(title=labels.column_label(hue), frameon=False)
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_hexbin(
    df: pd.DataFrame, x: str, y: str, title: str = "", subtitle: Optional[str] = None,
    gridsize: int = 30,
) -> plt.Figure:
    """Hexbin density plot — better than a scatterplot for large, overlapping numeric data."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    hb = ax.hexbin(df[x], df[y], gridsize=gridsize, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Loan Count")
    ax.set_xlabel(labels.column_label(x))
    ax.set_ylabel(labels.column_label(y))
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, columns: Sequence[str], title: str = "Correlation Matrix",
    subtitle: Optional[str] = None, method: str = "pearson",
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Correlation heatmap for a set of numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : Sequence[str]
    title, subtitle : str
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    (plt.Figure, pd.DataFrame)
        Figure and the underlying correlation matrix (for further reuse).
    """
    corr = df[list(columns)].corr(method=method)
    fig, ax = plt.subplots(figsize=(min(1.0 * len(columns) + 3, 14), min(0.8 * len(columns) + 3, 12)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    tick_labels = [labels.column_label(c) for c in corr.columns]
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=PALETTE_DIVERGING, center=0,
        square=True, linewidths=0.5, cbar_kws={"label": f"{method.title()} r"},
        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax,
    )
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig, corr


def plot_pairplot(
    df: pd.DataFrame, columns: Sequence[str], hue: Optional[str] = None,
    sample_n: Optional[int] = 1500,
) -> sns.axisgrid.PairGrid:
    """
    Seaborn pairplot of key numeric variables, optionally colored by the
    target. Subsampled for render speed on larger datasets.
    """
    plot_df = df if (sample_n is None or len(df) <= sample_n) else df.sample(
        sample_n, random_state=config.RANDOM_STATE
    )
    palette = {0: COLOR_PAID, 1: COLOR_DEFAULT} if hue == config.TARGET_COLUMN else None
    grid = sns.pairplot(
        plot_df[list(columns) + ([hue] if hue and hue not in columns else [])],
        hue=hue, palette=palette, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15},
    )
    grid.figure.suptitle(
        "Pairwise Relationships Among Key Numeric Variables", y=1.02,
        fontsize=14, fontweight="bold",
    )
    return grid


def plot_missing_value_heatmap(df: pd.DataFrame, title: str = "Missing Value Heatmap") -> plt.Figure:
    """Heatmap showing the location of missing values across rows/columns."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(df.isna(), cbar=False, cmap=["#EAECEE", COLOR_DEFAULT], ax=ax)
    ax.set_xticklabels([labels.column_label(c) for c in df.columns], rotation=90)
    ax.set_xlabel("Variable")
    ax.set_ylabel("Row Index")
    _apply_titles(ax, title, "Colored cells indicate a missing value at that row/column")
    fig.tight_layout()
    return fig


def plot_missing_value_bar(df: pd.DataFrame, title: str = "Missing Values by Column") -> plt.Figure:
    """Bar chart of missing-value counts/percentages per column (missingno-style summary)."""
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    if missing_pct.empty:
        ax.text(0.5, 0.5, "No missing values in this dataset.", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        ax.axis("off")
    else:
        sns.barplot(x=missing_pct.values, y=[labels.column_label(c) for c in missing_pct.index],
                    ax=ax, color=COLOR_DEFAULT)
        ax.set_xlabel("Missing (%)")
    _apply_titles(ax, title)
    fig.tight_layout()
    return fig


def plot_outlier_boxplots(
    df: pd.DataFrame, columns: Sequence[str], title: str = "Outlier Screen: Key Numeric Variables",
) -> plt.Figure:
    """
    Grid of standardized boxplots (z-scored) across multiple numeric
    columns to screen for outliers on a common scale in one figure.
    """
    n = len(columns)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(columns):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color=COLOR_PAID)
        axes[i].set_title(labels.column_label(col), fontsize=11, fontweight="bold")
        axes[i].set_ylabel("")
    for j in range(len(columns), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. DEFAULT-RATE BUSINESS VISUALIZATIONS
# ---------------------------------------------------------------------------


def default_rate_by_group(
    df: pd.DataFrame, group_column: str, target_column: str = config.TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Compute the default rate, loan count, and count of defaults for each
    level of a grouping column.

    Parameters
    ----------
    df : pd.DataFrame
    group_column : str
    target_column : str

    Returns
    -------
    pd.DataFrame
        Columns: loan_count, default_count, default_rate. Sorted by
        default_rate descending.
    """
    grouped = df.groupby(group_column, observed=True)[target_column].agg(
        loan_count="count", default_count="sum"
    )
    grouped["default_rate"] = (grouped["default_count"] / grouped["loan_count"]).round(4)
    return grouped.sort_values("default_rate", ascending=False)


def plot_default_rate_by_group(
    df: pd.DataFrame, group_column: str, title: str, subtitle: Optional[str] = None,
    target_column: str = config.TARGET_COLUMN, min_group_size: int = 5,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Bar chart of default rate by category, with the overall portfolio
    default rate drawn as a reference line and each bar labeled with its
    exact rate and underlying loan count.

    Parameters
    ----------
    df : pd.DataFrame
    group_column : str
    title, subtitle : str
    target_column : str
    min_group_size : int
        Groups with fewer than this many loans are dropped from the plot
        (too few observations for a reliable rate) but retained in the
        returned table.

    Returns
    -------
    (plt.Figure, pd.DataFrame)
    """
    summary = default_rate_by_group(df, group_column, target_column)
    if group_column == "grade":
        grade_order = config.ORDINAL_CATEGORY_ORDER[0]
        existing_order = [grade for grade in grade_order if grade in summary.index]
        summary = summary.reindex(existing_order)

    plot_summary = summary[summary["loan_count"] >= min_group_size]
    overall_rate = df[target_column].mean()

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bars = ax.bar(
        plot_summary.index.astype(str), plot_summary["default_rate"] * 100,
        color=COLOR_DEFAULT, alpha=0.85,
    )
    ax.axhline(overall_rate * 100, color="black", linestyle="--", linewidth=1.3,
               label=f"Portfolio Average = {overall_rate:.1%}")
    for bar, (_, row) in zip(bars, plot_summary.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{row['default_rate']:.1%}\n(n={int(row['loan_count']):,})",
            ha="center", va="bottom", fontsize=8.5,
        )
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel(labels.column_label(group_column))
    # Keep `summary` (returned to the caller) on raw group values; only the
    # displayed tick labels are humanized.
    ax.set_xticks(range(len(plot_summary)))
    ax.set_xticklabels(
        [labels.category_label(group_column, v) for v in plot_summary.index],
        rotation=45, ha="right",
    )
    ax.legend(frameon=False)
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig, summary


def bin_into_quartiles(
    df: pd.DataFrame, column: str, q: int = 4, labels: Optional[List[str]] = None
) -> pd.Series:
    """
    Bin a numeric column into quartiles (or arbitrary quantile count) for
    default-rate-by-band analysis. Uses `pd.qcut` with duplicate-safe
    handling for skewed distributions with many repeated values.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    q : int
        Number of quantile bins (4 = quartiles).
    labels : list[str], optional
        Custom bin labels; defaults to Q1..Qn (Q1 = lowest values).

    Returns
    -------
    pd.Series
        Categorical series of quantile-bin labels, same index as df.
    """
    if labels is None:
        labels = [f"Q{i + 1}" for i in range(q)]
    return pd.qcut(df[column], q=q, labels=labels, duplicates="drop")


def bin_into_bands(
    df: pd.DataFrame, column: str, bins: List[float], labels: List[str]
) -> pd.Series:
    """
    Bin a numeric column into fixed, business-defined bands (e.g.
    interest-rate bands, loan-amount bands) rather than quantiles, for
    interpretability in executive-facing charts.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    bins : list[float]
        Bin edges (len(labels) + 1 values).
    labels : list[str]

    Returns
    -------
    pd.Series
    """
    return pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)


# ---------------------------------------------------------------------------
# 6. STATISTICAL TESTING
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """
    Standardized statistical-test result container so every test in the
    notebook can be reported and interpreted with the same structure:
    null/alternative hypotheses, statistic, p-value, effect size, and a
    plain-language business interpretation.
    """

    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    effect_size_label: Optional[str]
    interpretation: str
    alpha: float = 0.05

    @property
    def is_significant(self) -> bool:
        return self.p_value < self.alpha

    def summary(self) -> str:
        sig = "statistically significant" if self.is_significant else "not statistically significant"
        effect_str = (
            f", effect size ({self.effect_size_label}) = {self.effect_size:.3f}"
            if self.effect_size is not None else ""
        )
        return (
            f"{self.test_name}: statistic = {self.statistic:.3f}, "
            f"p-value = {self.p_value:.4g}{effect_str} -> result is {sig} at alpha={self.alpha}.\n"
            f"H0: {self.null_hypothesis}\nH1: {self.alternative_hypothesis}\n"
            f"Business interpretation: {self.interpretation}"
        )


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """
    Compute Cohen's d effect size for two independent samples (pooled
    standard deviation). |d| ~ 0.2 small, ~0.5 medium, ~0.8 large.
    """
    a, b = group_a.dropna(), group_b.dropna()
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.var() + (n_b - 1) * b.var()) / (n_a + n_b - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Compute Cramer's V effect size for a chi-square test of association
    between two categorical variables. ~0.1 small, ~0.3 medium, ~0.5 large.
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.to_numpy().sum()
    r, k = contingency_table.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))


def run_independent_ttest(
    df: pd.DataFrame, numeric_column: str, group_column: str = config.TARGET_COLUMN,
    group_labels: Tuple = (0, 1),
) -> TestResult:
    """
    Independent-samples t-test comparing the mean of `numeric_column`
    between two groups of `group_column` (default: non-default vs.
    default borrowers), reporting Welch's t-test (does not assume equal
    variances — the safer default for real-world financial data).

    Parameters
    ----------
    df : pd.DataFrame
    numeric_column : str
    group_column : str
    group_labels : tuple
        The two group values to compare (order: (reference, comparison)).

    Returns
    -------
    TestResult
    """
    a = df.loc[df[group_column] == group_labels[0], numeric_column]
    b = df.loc[df[group_column] == group_labels[1], numeric_column]
    stat, p = stats.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
    d = cohens_d(a, b)

    interpretation = (
        f"Borrowers with {group_column}={group_labels[1]} have a mean "
        f"{numeric_column} of {b.mean():.2f} vs. {a.mean():.2f} for "
        f"{group_column}={group_labels[0]} "
        f"({'a meaningful' if abs(d) >= 0.2 else 'a negligible'} difference given the effect size)."
    )
    return TestResult(
        test_name=f"Welch's t-test: {numeric_column} by {group_column}",
        null_hypothesis=f"Mean {numeric_column} is equal across {group_column} groups.",
        alternative_hypothesis=f"Mean {numeric_column} differs across {group_column} groups.",
        statistic=stat, p_value=p, effect_size=d, effect_size_label="Cohen's d",
        interpretation=interpretation,
    )


def run_anova(
    df: pd.DataFrame, numeric_column: str, group_column: str,
) -> TestResult:
    """
    One-way ANOVA testing whether the mean of `numeric_column` differs
    across 3+ levels of a categorical `group_column` (e.g. loan grade),
    with eta-squared as the effect size.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_column : str
    group_column : str

    Returns
    -------
    TestResult
    """
    groups = [g.dropna().values for _, g in df.groupby(group_column, observed=True)[numeric_column]]
    groups = [g for g in groups if len(g) > 1]
    stat, p = stats.f_oneway(*groups)

    # Eta-squared: between-group SS / total SS
    grand_mean = df[numeric_column].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

    interpretation = (
        f"Mean {numeric_column} varies across {group_column} categories "
        f"({'meaningfully' if eta_sq >= 0.01 else 'only marginally'}, "
        f"eta-squared={eta_sq:.3f}), consistent with {group_column} carrying "
        f"real information about {numeric_column}."
    )
    return TestResult(
        test_name=f"One-way ANOVA: {numeric_column} by {group_column}",
        null_hypothesis=f"Mean {numeric_column} is equal across all {group_column} groups.",
        alternative_hypothesis=f"At least one {group_column} group has a different mean {numeric_column}.",
        statistic=stat, p_value=p, effect_size=eta_sq, effect_size_label="eta-squared",
        interpretation=interpretation,
    )


def run_chi_square_test(
    df: pd.DataFrame, column_a: str, column_b: str = config.TARGET_COLUMN,
) -> TestResult:
    """
    Chi-square test of independence between two categorical variables
    (typically a borrower characteristic vs. the binary default target),
    with Cramer's V as the effect size.

    Parameters
    ----------
    df : pd.DataFrame
    column_a : str
    column_b : str

    Returns
    -------
    TestResult
    """
    contingency = pd.crosstab(df[column_a], df[column_b])
    stat, p, dof, expected = stats.chi2_contingency(contingency)
    v = cramers_v(contingency)

    interpretation = (
        f"{column_a} and {column_b} show "
        f"{'a statistically detectable' if p < 0.05 else 'no statistically detectable'} "
        f"association (Cramer's V={v:.3f}, "
        f"{'weak' if v < 0.1 else 'moderate' if v < 0.3 else 'strong'} strength)."
    )
    return TestResult(
        test_name=f"Chi-square test: {column_a} vs. {column_b}",
        null_hypothesis=f"{column_a} and {column_b} are independent.",
        alternative_hypothesis=f"{column_a} and {column_b} are associated.",
        statistic=stat, p_value=p, effect_size=v, effect_size_label="Cramer's V",
        interpretation=interpretation,
    )


def pearson_and_spearman(
    df: pd.DataFrame, column_a: str, column_b: str
) -> Tuple[TestResult, TestResult]:
    """
    Compute both Pearson (linear) and Spearman (monotonic, rank-based)
    correlation between two numeric columns, each returned as a
    TestResult so p-values and interpretation follow the same reporting
    convention as the other tests.

    Parameters
    ----------
    df : pd.DataFrame
    column_a : str
    column_b : str

    Returns
    -------
    (TestResult, TestResult)
        (pearson_result, spearman_result)
    """
    paired = df[[column_a, column_b]].dropna()
    r_p, p_p = stats.pearsonr(paired[column_a], paired[column_b])
    r_s, p_s = stats.spearmanr(paired[column_a], paired[column_b])

    def _interpret(r: float) -> str:
        strength = (
            "strong" if abs(r) >= 0.5 else "moderate" if abs(r) >= 0.3 else
            "weak" if abs(r) >= 0.1 else "negligible"
        )
        direction = "positive" if r > 0 else "negative"
        return f"{strength} {direction} relationship between {column_a} and {column_b} (r={r:.3f})."

    pearson_result = TestResult(
        test_name=f"Pearson correlation: {column_a} vs. {column_b}",
        null_hypothesis=f"No linear correlation between {column_a} and {column_b} (rho=0).",
        alternative_hypothesis=f"A linear correlation exists between {column_a} and {column_b}.",
        statistic=r_p, p_value=p_p, effect_size=r_p, effect_size_label="Pearson r",
        interpretation=_interpret(r_p),
    )
    spearman_result = TestResult(
        test_name=f"Spearman correlation: {column_a} vs. {column_b}",
        null_hypothesis=f"No monotonic correlation between {column_a} and {column_b}.",
        alternative_hypothesis=f"A monotonic correlation exists between {column_a} and {column_b}.",
        statistic=r_s, p_value=p_s, effect_size=r_s, effect_size_label="Spearman rho",
        interpretation=_interpret(r_s),
    )
    return pearson_result, spearman_result


def proportion_confidence_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Wilson-score confidence interval for a proportion (e.g. a subgroup
    default rate) — more reliable than the normal approximation for
    proportions near 0 or 1, which is common with imbalanced default data.

    Parameters
    ----------
    successes : int
        Number of positive events (e.g. defaults).
    n : int
        Total number of trials (e.g. loans in the group).
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denom
    return p_hat, max(0.0, center - half_width), min(1.0, center + half_width)


def default_rate_ci_by_group(
    df: pd.DataFrame, group_column: str, target_column: str = config.TARGET_COLUMN,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Default rate with a Wilson-score confidence interval for every level
    of a grouping column — lets business stakeholders see which subgroup
    differences are backed by enough sample size to trust.

    Returns
    -------
    pd.DataFrame
        Columns: loan_count, default_count, default_rate, ci_lower, ci_upper.
    """
    rows = []
    for level, group in df.groupby(group_column, observed=True):
        n = len(group)
        successes = int(group[target_column].sum())
        point, lo, hi = proportion_confidence_interval(successes, n, confidence)
        rows.append(
            {group_column: level, "loan_count": n, "default_count": successes,
             "default_rate": round(point, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4)}
        )
    return pd.DataFrame(rows).set_index(group_column).sort_values("default_rate", ascending=False)


# ---------------------------------------------------------------------------
# 7. FEATURE-RELATIONSHIP / MULTICOLLINEARITY HELPERS
# ---------------------------------------------------------------------------


def high_correlation_pairs(
    corr_matrix: pd.DataFrame, threshold: float = 0.6
) -> pd.DataFrame:
    """
    Identify variable pairs whose absolute correlation exceeds
    `threshold` — used to flag candidate multicollinearity before
    modeling.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square correlation matrix (e.g. from `plot_correlation_heatmap`).
    threshold : float

    Returns
    -------
    pd.DataFrame
        Columns: variable_1, variable_2, correlation. Sorted by |correlation| descending.
    """
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({"variable_1": cols[i], "variable_2": cols[j], "correlation": round(r, 3)})
    if not pairs:
        return pd.DataFrame(columns=["variable_1", "variable_2", "correlation"])
    return pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)


def variance_inflation_factors(df: pd.DataFrame, numeric_columns: Sequence[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors (VIF) for a set of numeric
    predictors — a more rigorous multicollinearity diagnostic than
    pairwise correlation alone, since it captures multivariate
    redundancy. VIF > 5 (some practitioners use 10) suggests a variable
    is highly explainable by the others and a candidate for removal or
    dimensionality reduction in Phase 3.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_columns : Sequence[str]

    Returns
    -------
    pd.DataFrame
        Columns: variable, VIF. Sorted descending by VIF.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[list(numeric_columns)].dropna()
    X = (X - X.mean()) / X.std()  # standardize for numerically stable VIF
    X.insert(0, "const", 1.0)

    vifs = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.values, i)
        vifs.append({"variable": col, "VIF": round(vif, 2)})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False).reset_index(drop=True)
