import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_perf(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_metric(df: pd.DataFrame, column: str, title: str):
    if column not in df.columns:
        return None

    fig, ax = plt.subplots()
    ax.plot(df["batch_id"], df[column], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Batch")
    ax.set_ylabel(column.capitalize())
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_entropy(df):
    return plot_metric(df, "entropy", "Prediction Entropy")


def plot_f1(df):
    return plot_metric(df, "f1", "F1 Score")


def plot_accuracy(df):
    return plot_metric(df, "accuracy", "Accuracy")


def plot_performance(path: Path):
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty or "batch_id" not in df.columns:
        return None

    fig, ax = plt.subplots()
    ax.plot(df["batch_id"], df["accuracy"], label="Accuracy")
    ax.plot(df["batch_id"], df["f1"], label="F1")
    ax.set_title("Model Performance Over Time")
    ax.set_xlabel("Batch")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
