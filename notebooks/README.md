# Notebooks

Two notebooks covering different use cases.

## `drift_simulation.ipynb`

Self-contained walkthrough using synthetic data. No prior setup needed -  runs
without a server, without a trained model, without any prior `make` commands.
All outputs are pre-computed and embedded so you can read it on GitHub without
running anything.

**What it shows:** 80-batch PSI drift scenario with a 2σ mean shift at batch 40,
alongside the trust score response and decision engine output.

```bash
# To run interactively:
pip install -e ".[notebooks]"
jupyter notebook notebooks/drift_simulation.ipynb
make notebook   # shortcut
```

---

## `simulation_analysis.ipynb`

Reads **live data** from the SQLite database populated by `make sim`. Shows the
same three panels but using your actual recorded batches rather than synthetic data.

**Prerequisite:** run `make train && make sim` first.

```bash
make train
make sim
jupyter notebook notebooks/simulation_analysis.ipynb
```

| | `drift_simulation.ipynb` | `simulation_analysis.ipynb` |
|---|---|---|
| **Data source** | Synthetic, self-contained | Live `data/metrics/metrics.db` |
| **Requires prior setup** | No | Yes (`make train && make sim`) |
| **Pre-run outputs** | Yes -  readable on GitHub | No -  run locally |
| **Use case** | Understand the system | Explore your own simulation |
