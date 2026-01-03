import pandas as pd
from sklearn.metrics import f1_score


def validate_model(model, df: pd.DataFrame) -> float:
    X = df.drop(columns=["label"])
    y = df["label"]

    preds = model.predict(X)
    score = f1_score(y, preds, zero_division=0)

    return float(score)
