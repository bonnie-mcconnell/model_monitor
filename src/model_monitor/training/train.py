"""Model training utilities and bootstrap dataset generation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models/current.pkl")
REF_PATH = Path("data/reference/reference_stats.json")
SCHEMA_PATH = Path("data/reference/feature_schema.json")


def make_dataset(
    n_samples: int = 5_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
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


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
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


def compute_reference_stats(df: pd.DataFrame) -> dict[str, dict]:
    stats: dict[str, dict[str, object]] = {}

    for col in df.columns:
        if col == "label":
            continue

        vals = np.asarray(df[col].to_numpy(), dtype=float)

        stats[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "hist": np.histogram(vals, bins=20)[0].tolist(),
        }

    return stats


def main() -> None:  # pragma: no cover
    print("Generating dataset...")
    df, feature_names = make_dataset()

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

    from model_monitor.training.evaluation import validate_model
    f1 = validate_model(model, val_df)
    print(f"Validation F1: {f1:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")

    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    REF_PATH.write_text(json.dumps(compute_reference_stats(train_df), indent=2))

    print("Bootstrap training complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
