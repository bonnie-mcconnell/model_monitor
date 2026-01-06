import pandas as pd
from sklearn.metrics import f1_score


def validate_model(model, df: pd.DataFrame) -> float:
    """
    Evaluate a trained model on a labeled dataset.

    Returns:
        F1 score as float.
    """
    X = df.drop(columns=["label"])

    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]

    y = df["label"]

    preds = model.predict(X)
    return float(f1_score(y, preds, zero_division=0))
