"""Model training utilities and bootstrap dataset generation.

Supports two dataset modes:

synthetic (default)
    A 5000-sample, 10-feature sklearn make_classification dataset.
    Useful for quick iteration - no downloads, deterministic across
    machines, drift is easy to inject because the distribution is known.

breast-cancer
    The Wisconsin Breast Cancer dataset from sklearn.datasets (569 samples,
    30 features, binary classification).  Ship-ready for demonstrating the
    monitor on a real, non-synthetic distribution.  No download required;
    bundled with scikit-learn.

Select at training time::

    python -m model_monitor.training.train                   # synthetic
    python -m model_monitor.training.train --dataset breast-cancer
    make train DATASET=breast-cancer

The chosen dataset is recorded in ``data/reference/feature_schema.json``
so the simulation loop can reload the correct feature names without needing
to know which dataset was used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from model_monitor.training.evaluation import validate_model

MODEL_PATH = Path("models/current.pkl")
TRAIN_DATA_STEM = Path("data/reference/train_population")
REF_PATH = Path("data/reference/reference_stats.json")
SCHEMA_PATH = Path("data/reference/feature_schema.json")

DatasetName = Literal["synthetic", "breast-cancer"]
_KNOWN_DATASETS: set[str] = {"synthetic", "breast-cancer"}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def make_dataset(
    n_samples: int = 5_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate a synthetic binary classification dataset.

    Uses ``sklearn.datasets.make_classification`` with a moderate class
    separation (1.2) and a mix of informative and redundant features.
    The resulting distribution is Gaussian-ish, which makes PSI drift
    injection straightforward: shift the mean of any feature column by a
    known number of standard deviations to produce a predictable PSI value.

    Args:
        n_samples:    Number of rows.  Default 5 000 is enough for stable
                      PSI estimates while keeping training time under 2 s.
        random_state: Seed for reproducibility.

    Returns:
        (DataFrame with features + ``label`` column, list of feature names)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        class_sep=1.2,
        random_state=random_state,
    )

    feature_names = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y

    return df, feature_names


def load_dataset(name: DatasetName) -> tuple[pd.DataFrame, list[str]]:
    """Load a named dataset and return a labelled DataFrame.

    Args:
        name: ``"synthetic"`` or ``"breast-cancer"``.

    Returns:
        (DataFrame with features + ``label`` column, list of feature names)

    Raises:
        ValueError: if ``name`` is not a recognised dataset.
    """
    if name not in _KNOWN_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Choices: {sorted(_KNOWN_DATASETS)}"
        )

    if name == "synthetic":
        return make_dataset()

    # breast-cancer: 569 samples, 30 features, binary (malignant=1, benign=0).
    # Bundled with scikit-learn - no download, no network requirement.
    # Feature ranges vary widely (e.g. radius_mean ≈ 6–28, area_worst ≈ 185–4254)
    # so PSI is more sensitive than on the synthetic Gaussian dataset.
    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
    # sklearn uses 0=malignant, 1=benign by convention; map to 1=positive
    # (malignant) so the rarer class is the positive case, which is the
    # standard medical-classification convention and gives a more interesting
    # F1 score to track.
    df["label"] = (bunch.target == 0).astype(int)
    return df, list(bunch.feature_names)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """Fit a RandomForestClassifier on a labelled DataFrame.

    Args:
        df: DataFrame with feature columns and a ``label`` column.

    Returns:
        Fitted RandomForestClassifier.

    Raises:
        ValueError: if ``df`` does not contain a ``label`` column.
    """
    if "label" not in df.columns:
        raise ValueError("Training DataFrame must contain 'label' column")

    X = df.drop(columns=["label"])
    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Reference stats
# ---------------------------------------------------------------------------


def compute_reference_stats(df: pd.DataFrame, psi_bins: int = 10) -> dict[str, dict]:
    """Compute per-feature reference statistics for drift monitoring.

    Stores bin edges derived from the training distribution so that
    production PSI can be computed against the *same* bins used at
    training time.  Re-deriving bin edges from production data would
    make the two histograms incomparable - this is the key property
    that makes PSI a valid drift signal rather than an arbitrary number.

    Args:
        df:       Training DataFrame (must contain a ``label`` column).
        psi_bins: Number of equal-frequency bins for PSI computation.
                  Must match the ``bins`` argument passed to
                  ``compute_psi`` at inference time.  Default is 10.

    Returns:
        Dict mapping each feature name to its reference statistics dict,
        including ``psi_bin_edges`` for use by ``DriftMonitor``.
    """
    stats: dict[str, dict[str, object]] = {}

    for col in df.columns:
        if col == "label":
            continue

        vals = np.asarray(df[col].to_numpy(), dtype=float)

        # Equal-frequency (percentile) bin edges - the same method used
        # in compute_psi.  Storing them here means inference never has
        # to recompute from the reference distribution, which may not be
        # available at serving time.
        percentiles = np.linspace(0, 100, psi_bins + 1)
        bin_edges = np.unique(np.percentile(vals, percentiles))

        stats[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "hist": np.histogram(vals, bins=20)[0].tolist(),
            # PSI bin edges stored at training time so production drift
            # is always measured against the same reference buckets.
            "psi_bin_edges": bin_edges.tolist(),
        }

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(dataset: DatasetName = "synthetic") -> None:
    """Train and promote the initial model, writing all artefacts to the CWD.

    Accepts an optional ``dataset`` argument so the function can be called
    programmatically from tests and the integration e2e suite without
    argparse consuming pytest's ``sys.argv``.  The CLI entry point passes
    the parsed argument through; tests call ``main()`` or
    ``main(dataset="breast-cancer")`` directly.

    Args:
        dataset: Which dataset to train on.  ``"synthetic"`` (default) or
                 ``"breast-cancer"``.
    """
    dataset_name: DatasetName = dataset

    print(f"Loading dataset: {dataset_name}...")
    df, feature_names = load_dataset(dataset_name)
    print(f"  {len(df)} rows, {len(feature_names)} features")

    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(feature_names, indent=2))

    train_df, val_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    print("Training initial model...")
    model = train_model(train_df)

    f1 = validate_model(model, val_df)
    print(f"Validation F1: {f1:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")

    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    REF_PATH.write_text(json.dumps(compute_reference_stats(train_df), indent=2))

    # Save the training population so the simulation loop can draw batches
    # from the same distribution - not a re-instantiation of the loader
    # which could produce a structurally different draw.
    X_train = train_df[feature_names].values
    y_train = train_df["label"].values
    TRAIN_DATA_STEM.parent.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_DATA_STEM.with_suffix(".npy"), X_train)
    np.save(Path(str(TRAIN_DATA_STEM) + ".labels.npy"), y_train)
    print(
        f"Saved training population → {TRAIN_DATA_STEM.with_suffix('.npy')} "
        f"({len(X_train)} rows)"
    )

    # Promote the initial model so active.json is written.
    # Without this, Predictor.reload() finds no active version and every
    # predict_batch call returns action="none" regardless of drift level.
    from model_monitor.storage.model_store import ModelStore

    store = ModelStore()
    store.save_candidate(model)
    store.promote_candidate({"baseline_f1": f1, "dataset": dataset_name})
    print(f"Promoted initial model (baseline F1: {f1:.4f})")

    print("Bootstrap training complete.")


def _cli() -> None:  # pragma: no cover
    """CLI entry point - parses sys.argv and delegates to ``main()``.

    Kept separate from ``main()`` so that tests and the e2e integration suite
    can call ``main()`` directly without argparse consuming pytest's argv.
    """
    import argparse as _ap

    parser = _ap.ArgumentParser(
        description="Train and promote the initial model_monitor model.",
        formatter_class=_ap.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m model_monitor.training.train\n"
            "  python -m model_monitor.training.train --dataset breast-cancer\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(_KNOWN_DATASETS),
        default="synthetic",
        help=(
            "Dataset to train on.  'synthetic' (default) uses make_classification; "
            "'breast-cancer' uses the bundled sklearn Wisconsin dataset."
        ),
    )
    args = parser.parse_args()
    main(dataset=args.dataset)


if __name__ == "__main__":  # pragma: no cover
    _cli()
