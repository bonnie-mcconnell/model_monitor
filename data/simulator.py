from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd


def stream_batches(
    df: pd.DataFrame,
    *,
    batch_size: int = 32,
    include_labels: bool = True,
) -> Iterator[dict]:
    """
    Simple batch generator for experiments and unit tests.

    Not intended for production inference paths.
    """
    if "label" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'label' column")

    cursor = 0
    batch_id = 0

    while cursor < len(df):
        batch = df.iloc[cursor : cursor + batch_size].copy()
        cursor += batch_size

        y = batch.pop("label") if include_labels else None

        yield {
            "X": batch,
            "y": y,
            "batch_id": f"test_batch_{batch_id}",
        }

        batch_id += 1
