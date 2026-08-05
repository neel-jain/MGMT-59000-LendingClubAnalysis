"""
utils.py
========
Reusable, well-tested utility functions for the MGMT 590 LendingClub Loan
Default Risk capstone project.

Organized into sections:
    1. Logging setup
    2. Filesystem helpers
    3. Data ingestion
    4. Data validation
    5. Data cleaning / feature engineering
    6. Preprocessing pipeline construction
    7. Train / validation / test splitting
    8. Serialization helpers (joblib)

Every function is self-contained, type-hinted, and documented so later
phases (model training, evaluation, Streamlit app) can import and reuse
it directly, e.g.:

    from src.utils import load_raw_data, clean_dataset, build_preprocessing_pipeline

Design principle: functions here perform ONE clearly named task each and
return data rather than mutating global state, which keeps the pipeline
composable and testable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config

# ---------------------------------------------------------------------------
# 1. LOGGING SETUP
# ---------------------------------------------------------------------------


def get_logger(name: str = "lendingclub_pipeline") -> logging.Logger:
    """
    Create (or retrieve) a configured logger that writes to both the
    console and a persistent log file at ``config.PIPELINE_LOG_PATH``.

    Safe to call multiple times (e.g. once per module) — handlers are only
    attached once per logger name to avoid duplicate log lines.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured (e.g. called earlier in the same
        # process) — return as-is to avoid duplicate handlers/log lines.
        return logger

    logger.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — ensure the logs directory exists first.
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(config.PIPELINE_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 2. FILESYSTEM HELPERS
# ---------------------------------------------------------------------------


def ensure_directories(directories: Optional[List[Path]] = None) -> None:
    """
    Create every directory in ``directories`` (default:
    ``config.REQUIRED_DIRS``) if it does not already exist.

    Idempotent and safe to call at the start of any script or notebook.

    Parameters
    ----------
    directories : list[Path], optional
        Directories to create. Defaults to the full set of project
        directories declared in config.REQUIRED_DIRS.
    """
    dirs_to_create = directories if directories is not None else config.REQUIRED_DIRS
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Verified %d project directories exist.", len(dirs_to_create))


# ---------------------------------------------------------------------------
# 3. DATA INGESTION
# ---------------------------------------------------------------------------


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw LendingClub CSV extract from disk.

    Parameters
    ----------
    path : Path
        Location of the raw CSV file. Defaults to config.RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset exactly as read from disk.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``path``.
    ValueError
        If the file is empty or cannot be parsed as CSV.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. Place the LendingClub "
            f"Indiana extract at this path before running the pipeline."
        )

    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Raw data file at {path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Raw data file at {path} could not be parsed as CSV.") from exc

    if df.empty:
        raise ValueError(f"Raw data file at {path} contained zero rows.")

    logger.info("Loaded raw dataset from %s — shape=%s", path, df.shape)
    return df


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    path : Path
        Destination file path.
    index : bool
        Whether to write the DataFrame index to disk. Defaults to False.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info("Saved dataframe with shape %s to %s", df.shape, path)


def load_dataframe(path: Path) -> pd.DataFrame:
    """
    Load a previously saved CSV (cleaned data or a train/val/test split).

    Parameters
    ----------
    path : Path
        Source file path.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found at {path}.")
    df = pd.read_csv(path)
    logger.info("Loaded dataframe from %s — shape=%s", path, df.shape)
    return df


# ---------------------------------------------------------------------------
# 4. DATA VALIDATION
# ---------------------------------------------------------------------------


def validate_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """
    Run a battery of data-quality checks on the raw dataset and return a
    structured validation report. Does NOT mutate or clean the data — see
    `clean_dataset()` for that.

    Checks performed
    -----------------
    - Schema drift: expected columns from config.EXPECTED_RAW_COLUMNS that
      are missing from the dataframe.
    - Missing values: count and percentage per column.
    - Duplicate rows: exact full-row duplicates.
    - Data types: pandas dtype per column (useful for spotting numeric
      columns that were read in as strings, e.g. "int_rate").
    - Invalid / out-of-range values for a few known business rules
      (negative loan amounts, negative annual income, loan_status values
      outside the expected vocabulary, state values outside Indiana).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by `load_raw_data()`.

    Returns
    -------
    dict
        Validation report with keys: "missing_columns", "missing_values",
        "duplicate_row_count", "dtypes", "invalid_values", "n_rows",
        "n_columns".
    """
    report: Dict[str, object] = {}

    report["n_rows"] = len(df)
    report["n_columns"] = df.shape[1]

    # --- Schema drift ---
    missing_columns = [
        col for col in config.EXPECTED_RAW_COLUMNS if col not in df.columns
    ]
    report["missing_columns"] = missing_columns
    if missing_columns:
        logger.warning("Dataset is missing %d expected columns: %s",
                        len(missing_columns), missing_columns)

    # --- Missing values ---
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_report = pd.DataFrame(
        {"missing_count": missing_counts, "missing_pct": missing_pct}
    )
    missing_report = missing_report[missing_report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    report["missing_values"] = missing_report

    # --- Duplicate rows ---
    duplicate_count = int(df.duplicated(keep="first").sum())
    report["duplicate_row_count"] = duplicate_count

    # --- Data types ---
    report["dtypes"] = df.dtypes

    # --- Invalid / out-of-range business-rule checks ---
    invalid: Dict[str, int] = {}

    if "loan_amnt" in df.columns:
        invalid["negative_or_zero_loan_amnt"] = int((df["loan_amnt"] <= 0).sum())

    if "annual_inc" in df.columns:
        invalid["negative_annual_inc"] = int((df["annual_inc"] < 0).sum())

    if config.LOAN_STATUS_COLUMN in df.columns:
        valid_statuses = set(config.LOAN_STATUS_TARGET_MAP.keys())
        # Loans outside the valid final-resolution statuses aren't
        # "invalid" data per se (e.g. "Current" is a legitimate status)
        # but they ARE out of scope for the binary target and will be
        # dropped during cleaning; flagged here for transparency.
        out_of_scope = df[~df[config.LOAN_STATUS_COLUMN].isin(valid_statuses)]
        invalid["loan_status_out_of_scope_rows"] = int(len(out_of_scope))

    if config.STATE_COLUMN in df.columns:
        non_indiana = int((df[config.STATE_COLUMN] != config.TARGET_STATE).sum())
        invalid["rows_outside_target_state"] = non_indiana

    report["invalid_values"] = invalid

    logger.info(
        "Validation complete: %d rows, %d columns, %d duplicate rows, "
        "%d columns with missing values, %d missing expected columns.",
        report["n_rows"],
        report["n_columns"],
        duplicate_count,
        len(missing_report),
        len(missing_columns),
    )
    return report


# ---------------------------------------------------------------------------
# 5. DATA CLEANING / FEATURE ENGINEERING
# ---------------------------------------------------------------------------


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Deduplicated copy of the input.
    """
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate rows (%d -> %d).", removed, before, len(df))
    return df


def filter_to_target_state(
    df: pd.DataFrame, state_code: str = config.TARGET_STATE
) -> pd.DataFrame:
    """
    Restrict the dataset to a single state (Indiana, by default).

    Parameters
    ----------
    df : pd.DataFrame
    state_code : str
        Two-letter state abbreviation to keep.

    Returns
    -------
    pd.DataFrame
        Filtered copy containing only rows for ``state_code``.
    """
    if config.STATE_COLUMN not in df.columns:
        logger.warning(
            "Column '%s' not found — skipping state filter.", config.STATE_COLUMN
        )
        return df.copy()

    before = len(df)
    filtered = df[df[config.STATE_COLUMN] == state_code].copy().reset_index(drop=True)
    logger.info(
        "Filtered to state='%s': %d -> %d rows.", state_code, before, len(filtered)
    )
    return filtered


def clean_percentage_columns(
    df: pd.DataFrame, columns: List[str] = config.PERCENTAGE_COLUMNS
) -> pd.DataFrame:
    """
    Convert percentage-formatted string columns (e.g. "13.56%") to numeric
    floats (e.g. 13.56). Values that are already numeric are left
    unchanged; unparseable values become NaN (handled later by imputation).

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to convert. Defaults to config.PERCENTAGE_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the specified columns converted to float.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Percentage column '%s' not found — skipping.", col)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Handles object dtype, pandas StringDtype, and any other
            # non-numeric dtype uniformly by treating values as text.
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace("%", "", regex=False)
                .str.replace("nan", "", regex=False)
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info(
            "Converted '%s' to numeric (%d NaNs after conversion).",
            col,
            int(df[col].isna().sum()),
        )
    return df


def parse_emp_length(
    df: pd.DataFrame,
    raw_column: str = config.EMP_LENGTH_RAW_COLUMN,
    new_column: str = config.EMP_LENGTH_NUMERIC_COLUMN,
    mapping: Dict[str, float] = config.EMP_LENGTH_MAP,
) -> pd.DataFrame:
    """
    Parse the free-text employment-length column (e.g. "10+ years",
    "< 1 year") into a numeric column of years using ``mapping``.

    Unrecognized or missing values become NaN and are imputed later by the
    numeric branch of the preprocessing pipeline (median strategy).

    Parameters
    ----------
    df : pd.DataFrame
    raw_column : str
        Name of the raw text column.
    new_column : str
        Name of the engineered numeric column to create.
    mapping : dict[str, float]
        Lookup table from raw string to numeric years.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``new_column`` added.
    """
    df = df.copy()
    if raw_column not in df.columns:
        logger.warning("Employment length column '%s' not found.", raw_column)
        df[new_column] = np.nan
        return df

    df[new_column] = df[raw_column].astype(str).str.strip().map(mapping)
    n_unmapped = int(df[new_column].isna().sum())
    logger.info(
        "Parsed '%s' -> '%s' (%d unmapped/missing values -> NaN).",
        raw_column,
        new_column,
        n_unmapped,
    )
    return df


def create_target_variable(
    df: pd.DataFrame,
    status_column: str = config.LOAN_STATUS_COLUMN,
    target_column: str = config.TARGET_COLUMN,
    status_map: Dict[str, int] = config.LOAN_STATUS_TARGET_MAP,
) -> pd.DataFrame:
    """
    Create the binary default target and drop loans whose status is not a
    final resolution (i.e. not in ``status_map``).

    Mapping (per project spec):
        "Charged Off" -> 1, "Default" -> 1, "Fully Paid" -> 0
        Everything else (e.g. "Current", "Late (31-120 days)",
        "In Grace Period") is REMOVED — those loans have not reached a
        final outcome and including them would mislabel censored data.

    Parameters
    ----------
    df : pd.DataFrame
    status_column : str
        Raw loan status column.
    target_column : str
        Name of the binary target column to create.
    status_map : dict[str, int]
        Mapping from raw status string to {0, 1}.

    Returns
    -------
    pd.DataFrame
        Filtered copy of ``df`` containing only resolved loans, with
        ``target_column`` added as an int (0/1) column.

    Raises
    ------
    KeyError
        If ``status_column`` is not present in ``df``.
    """
    if status_column not in df.columns:
        raise KeyError(f"Column '{status_column}' required to build target not found.")

    before = len(df)
    df = df[df[status_column].isin(status_map.keys())].copy()
    df[target_column] = df[status_column].map(status_map).astype(int)
    logger.info(
        "Built target '%s' from '%s': kept %d/%d rows (%d dropped as "
        "non-final status). Class balance: %s",
        target_column,
        status_column,
        len(df),
        before,
        before - len(df),
        df[target_column].value_counts(normalize=True).round(3).to_dict(),
    )
    return df.reset_index(drop=True)


def drop_excluded_columns(
    df: pd.DataFrame, columns: List[str] = config.EXCLUDED_COLUMNS
) -> pd.DataFrame:
    """
    Drop identifier / free-text / leakage-prone / superseded columns that
    should never enter the modeling feature set.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to drop if present (missing ones are silently ignored so
        this function is safe to call on partially-cleaned frames).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the columns removed.
    """
    existing = [col for col in columns if col in df.columns]
    df = df.drop(columns=existing)
    logger.info("Dropped %d excluded columns: %s", len(existing), existing)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate the full Phase 1 cleaning sequence on a raw dataframe:

        1. Filter to Indiana borrowers
        2. Remove duplicate rows
        3. Convert percentage columns to numeric
        4. Parse employment length to numeric years
        5. Build the binary target variable (drops non-final loan statuses)
        6. Drop excluded / leakage-prone columns

    This is the single function later phases (and the notebook) should
    call to go from "raw ingested CSV" to "modeling-ready dataframe".
    Categorical ENCODING is intentionally handled separately inside the
    scikit-learn ColumnTransformer (see build_preprocessing_pipeline)
    rather than here, so that encoders are fit only on the training split
    and reused (not refit) on validation/test/live data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by `load_raw_data()`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset: one row per resolved Indiana loan, with the
        binary target column present and excluded columns removed.
    """
    logger.info("Beginning full cleaning pipeline on raw shape %s", df.shape)

    df = filter_to_target_state(df)
    df = remove_duplicate_rows(df)
    df = clean_percentage_columns(df)
    df = parse_emp_length(df)
    df = create_target_variable(df)
    df = drop_excluded_columns(df)

    logger.info("Cleaning pipeline complete. Final shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 6. PREPROCESSING PIPELINE CONSTRUCTION
# ---------------------------------------------------------------------------


def build_preprocessing_pipeline(
    numeric_features: List[str] = config.NUMERIC_FEATURES,
    onehot_features: List[str] = config.ONEHOT_CATEGORICAL_FEATURES,
    ordinal_features: List[str] = config.ORDINAL_CATEGORICAL_FEATURES,
    ordinal_categories: List[List[str]] = config.ORDINAL_CATEGORY_ORDER,
) -> ColumnTransformer:
    """
    Build (but do not fit) the reusable scikit-learn preprocessing
    pipeline for this project.

    Structure
    ---------
    ColumnTransformer with three branches:
        - "numeric": SimpleImputer(median) -> StandardScaler
        - "onehot_categorical": SimpleImputer(most_frequent) -> OneHotEncoder
        - "ordinal_categorical": SimpleImputer(most_frequent) -> OrdinalEncoder
          (uses the natural risk ordering supplied in
          config.ORDINAL_CATEGORY_ORDER, e.g. grade A < B < ... < G)

    The returned object is UNFITTED. Call `.fit()` on the training split
    only (see `split_data`), then `.transform()` on validation/test/live
    data — never refit on anything but the training set, to avoid data
    leakage.

    Parameters
    ----------
    numeric_features : list[str]
    onehot_features : list[str]
    ordinal_features : list[str]
    ordinal_categories : list[list[str]]
        Category order per ordinal feature, positionally aligned with
        ``ordinal_features``.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing pipeline ready for `.fit()` / `.transform()`.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    onehot_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("onehot_categorical", onehot_pipeline, onehot_features),
            ("ordinal_categorical", ordinal_pipeline, ordinal_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        "Built preprocessing ColumnTransformer: %d numeric, %d one-hot, "
        "%d ordinal features.",
        len(numeric_features),
        len(onehot_features),
        len(ordinal_features),
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieve human-readable feature names produced by a FITTED
    preprocessing ColumnTransformer (useful for feature importance /
    coefficient inspection in later phases).

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A preprocessor that has already been `.fit()`.

    Returns
    -------
    list[str]
        Output feature names in the same order as the transformed array's
        columns.
    """
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# 7. TRAIN / VALIDATION / TEST SPLITTING
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    test_size: float = config.TEST_SIZE,
    validation_size: float = config.VALIDATION_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform a stratified train / validation / test split while preventing
    data leakage.

    Leakage-prevention strategy
    ----------------------------
    1. The TEST set is carved out first and is never touched again until
       final model evaluation (Phase 2+). No preprocessing statistics
       (imputation medians, scaler means/std, encoder categories) are
       ever computed using test rows.
    2. The VALIDATION set is carved out of the remaining training pool
       (used for hyperparameter tuning / model comparison), so it is also
       excluded from the data the preprocessor is fit on.
    3. The preprocessing pipeline (see build_preprocessing_pipeline) must
       be `.fit()` on X_train ONLY; validation and test sets are only ever
       `.transform()`-ed downstream (done in Phase 2's train_models.py).
    4. Stratification on the binary target is used at both split points to
       preserve class balance across all three sets, since default is a
       minority class.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (output of `clean_dataset`), containing both
        features and the target column.
    target_column : str
        Name of the binary target column.
    test_size : float
        Fraction of the full dataset reserved for testing.
    validation_size : float
        Fraction of the full dataset reserved for validation (computed
        relative to the ORIGINAL dataset size, then converted internally
        to the correct fraction of the remaining train+val pool).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be < 1.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 1: carve out the test set from the full dataset.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Step 2: carve out validation from the remaining train_val pool.
    # Convert the "fraction of full dataset" validation_size into the
    # equivalent fraction of the remaining (train_val) pool.
    relative_val_size = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    logger.info(
        "Split complete — train=%d (%.1f%%), val=%d (%.1f%%), test=%d "
        "(%.1f%%). Default rate — train=%.3f, val=%.3f, test=%.3f",
        len(X_train), 100 * len(X_train) / len(df),
        len(X_val), 100 * len(X_val) / len(df),
        len(X_test), 100 * len(X_test) / len(df),
        y_train.mean(), y_val.mean(), y_test.mean(),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Persist the six train/validation/test artifacts to config.SPLITS_DIR
    using the standard filenames defined in config.py, so Phase 2's
    train_models.py can load them without re-running the split.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series
    """
    save_dataframe(X_train, config.X_TRAIN_PATH)
    save_dataframe(X_val, config.X_VAL_PATH)
    save_dataframe(X_test, config.X_TEST_PATH)
    save_dataframe(y_train.to_frame(), config.Y_TRAIN_PATH)
    save_dataframe(y_val.to_frame(), config.Y_VAL_PATH)
    save_dataframe(y_test.to_frame(), config.Y_TEST_PATH)
    logger.info("All six train/validation/test split artifacts saved.")


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load the six previously saved train/validation/test artifacts.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train = load_dataframe(config.X_TRAIN_PATH)
    X_val = load_dataframe(config.X_VAL_PATH)
    X_test = load_dataframe(config.X_TEST_PATH)
    y_train = load_dataframe(config.Y_TRAIN_PATH)[config.TARGET_COLUMN]
    y_val = load_dataframe(config.Y_VAL_PATH)[config.TARGET_COLUMN]
    y_test = load_dataframe(config.Y_TEST_PATH)[config.TARGET_COLUMN]
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 8. SERIALIZATION HELPERS
# ---------------------------------------------------------------------------


def save_object(obj: object, path: Path) -> None:
    """
    Serialize any Python object (fitted pipeline, fitted model, etc.) to
    disk using joblib, creating parent directories as needed.

    Parameters
    ----------
    obj : object
        Object to serialize (e.g. a fitted ColumnTransformer or estimator).
    path : Path
        Destination file path (conventionally ``*.joblib``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Serialized object to %s", path)


def load_object(path: Path) -> object:
    """
    Load a joblib-serialized object from disk.

    Parameters
    ----------
    path : Path
        Source file path.

    Returns
    -------
    object
        The deserialized Python object.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Serialized object not found at {path}.")
    obj = joblib.load(path)
    logger.info("Loaded serialized object from %s", path)
    return obj
