# Simulation Analysis

This notebook walks through a complete monitoring cycle using the built-in
drift simulation. Run `make sim` first to generate data, then open this with
Jupyter or VS Code to explore the results.

## Setup

```python
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DB = Path("data/metrics/metrics.db")
```

## Load monitoring records

```python
con = sqlite3.connect(DB)
df = pd.read_sql(
    "SELECT * FROM metrics ORDER BY timestamp",
    con,
    parse_dates={"timestamp": {"unit": "s"}},
)
con.close()

print(f"{len(df)} batches recorded")
df[["accuracy", "f1", "drift_score"]].describe().round(4)
```

## Trust score trajectory

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(df["timestamp"], df["accuracy"], label="accuracy", alpha=0.8)
axes[0].plot(df["timestamp"], df["f1"], label="F1", alpha=0.8)
axes[0].axhline(0.75, color="red", linestyle="--", alpha=0.4, label="floor")
axes[0].set_ylabel("Performance")
axes[0].legend()

axes[1].plot(df["timestamp"], df["drift_score"], color="orange", label="PSI drift")
axes[1].axhline(0.1, color="gray", linestyle="--", alpha=0.4, label="moderate")
axes[1].axhline(0.2, color="red", linestyle="--", alpha=0.4, label="severe")
axes[1].set_ylabel("Drift (PSI)")
axes[1].legend()

# Mark decision events
colors = {"retrain": "blue", "rollback": "red", "promote": "green", "reject": "purple"}
for _, row in df[df["action"] != "none"].iterrows():
    c = colors.get(row["action"], "gray")
    for ax in axes:
        ax.axvline(row["timestamp"], color=c, alpha=0.5, linewidth=1.5)

axes[2].plot(df["timestamp"], df["drift_score"].apply(
    lambda x: max(0.0, 1.0 - (x - 0.1) / 0.2) if 0.1 < x < 0.3
    else (1.0 if x <= 0.1 else 0.0)
), color="steelblue", label="approx trust score")
axes[2].set_ylabel("Trust score")
axes[2].set_ylim(0, 1.1)
axes[2].legend()

plt.tight_layout()
plt.savefig("notebooks/trust_score_trajectory.png", dpi=150)
plt.show()
```

## Decision audit

```python
decisions = df[df["action"] != "none"][["timestamp", "action", "reason"]].copy()
decisions["timestamp"] = decisions["timestamp"].dt.strftime("%H:%M:%S")
print(decisions.to_string(index=False))
```

## PSI drift distribution

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["drift_score"], bins=30, edgecolor="white", color="steelblue", alpha=0.8)
ax.axvline(0.1, color="orange", linestyle="--", label="moderate threshold")
ax.axvline(0.2, color="red", linestyle="--", label="severe threshold")
ax.set_xlabel("PSI drift score")
ax.set_ylabel("Batch count")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/drift_distribution.png", dpi=150)
plt.show()

pct_stable = (df["drift_score"] < 0.1).mean() * 100
pct_moderate = ((df["drift_score"] >= 0.1) & (df["drift_score"] < 0.2)).mean() * 100
pct_severe = (df["drift_score"] >= 0.2).mean() * 100
print(f"Stable:   {pct_stable:.1f}%")
print(f"Moderate: {pct_moderate:.1f}%")
print(f"Severe:   {pct_severe:.1f}%")
```
