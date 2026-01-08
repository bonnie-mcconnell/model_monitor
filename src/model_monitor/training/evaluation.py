import pandas as pd
from sklearn.metrics import f1_score # type: ignore


def validate_model(model, df: pd.DataFrame) -> float:
    """
    Evaluate a trained model on a labeled dataset.

    Returns:
        F1 score as float.
    """
    X = df.drop(columns=["label"])
    y = df["label"]

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = set(expected) - set(X.columns)

        if missing:
            raise ValueError(
                f"Retrain data schema mismatch. Missing features: {sorted(missing)}"
            )

        X = X[expected]

    preds = model.predict(X)
    return float(f1_score(y, preds, zero_division=0))


