# Notebooks

Three notebooks covering the monitoring system from different angles.

## `drift_simulation.ipynb`

Self-contained walkthrough using synthetic data. No prior setup needed -
runs without a server, without a trained model, without any prior `make`
commands. All outputs are pre-computed and embedded so you can read it on
GitHub without running anything.

**What it shows:** 80-batch PSI drift scenario with a 2σ mean shift at
batch 40, alongside the trust score response and decision engine output.

```bash
pip install -e ".[notebooks]"
jupyter notebook notebooks/drift_simulation.ipynb
make notebook   # shortcut
```

---

## `simulation_analysis.ipynb`

Reads **live data** from the SQLite database populated by `make sim`. Shows
the same three panels but using your actual recorded batches rather than
synthetic data.

**Prerequisite:** run `make train && make sim` first.

```bash
make train
make sim
jupyter notebook notebooks/simulation_analysis.ipynb
```

---

## `monitor_evaluation.ipynb`

Statistical characterisation of the PSI monitor's operating point. Answers
two questions that production deployment requires:

1. **False-positive rate** - how often does the monitor fire on stable data?
2. **Detection latency** - given a real shift, how many batches pass before
   the monitor catches it? How does latency vary with shift magnitude?

**Key results** (BATCH\_SIZE=200, reference N=5,000, threshold=0.2):

| Shift (σ) | Detection rate | Median latency |
|-----------|---------------|----------------|
| 0.25–0.5  | 0%            | -              |
| 0.75      | 40%           | 9 batches      |
| 1.0       | 100%          | 3 batches      |
| 1.5+      | 100%          | immediate      |

False-positive rate on 200 stable trials: **0%**.

The analysis uses the same `compute_psi` implementation, reference bin edges,
and PSI threshold as the production server - the numbers are directly
comparable to a live deployment.

No prior setup needed; all dependencies are bundled with scikit-learn.

```bash
jupyter notebook notebooks/monitor_evaluation.ipynb
```

---

| | `drift_simulation` | `simulation_analysis` | `monitor_evaluation` |
|---|---|---|---|
| **Data source** | Synthetic, self-contained | Live `data/metrics/metrics.db` | Synthetic, self-contained |
| **Requires prior setup** | No | Yes (`make train && make sim`) | No |
| **Pre-run outputs** | Yes | No | Yes |
| **Use case** | Understand the system | Explore your own simulation | Validate the monitor |
