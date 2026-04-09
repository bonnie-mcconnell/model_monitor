# Notebooks

## `drift_simulation.ipynb`

Self-contained walkthrough of the monitoring pipeline against a synthetic 80-batch
drift scenario. Runs without a live server - all simulation logic is inline.

**What it shows:**

| Panel | Signal | What to look for |
|-------|--------|-----------------|
| PSI drift | Population Stability Index per batch | Near-zero pre-drift, jumps ~100× at batch 40 |
| Trust score | Weighted composite of accuracy, F1, confidence, drift, latency | Drops immediately when PSI spikes - leading indicator |
| Decision events | Policy engine output | `reject` fires the first post-drift batch |

**To run:**

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/drift_simulation.ipynb
```

All outputs are pre-computed and embedded - you can read the notebook on GitHub
without running it.
