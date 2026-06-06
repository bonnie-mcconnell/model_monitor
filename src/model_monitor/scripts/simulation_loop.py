"""
Drift simulation with live terminal output.

Drives 80 batches through the inference pipeline. Drift is injected at
batch 40 by shifting the feature distribution. The output shows each
decision alongside the trust score and drift level so the monitoring
response is visible in real time.

Run with:
    python -m model_monitor.scripts.simulation_loop
    make sim
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from model_monitor.config.settings import AppConfig, load_config
from model_monitor.inference.predict import Predictor
from model_monitor.monitoring.causal_drift import CausalDriftAttributor
from model_monitor.monitoring.conformal import ConformalMonitor
from model_monitor.monitoring.data_quality import DataQualityMonitor
from model_monitor.monitoring.output_drift import OutputDriftMonitor
from model_monitor.monitoring.raw_data_buffer import RawDataBuffer
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.monitoring.threshold_advisor import ThresholdAdvisor
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.model_store import ModelStore, get_active_version
from model_monitor.training.retrain_pipeline import RetrainPipeline

logger = logging.getLogger(__name__)


def _configure_sim_logging() -> None:
    """Route simulation logs to a file so the terminal table stays readable.

    All structured log events are still recorded for inspection; they just
    do not interleave with the batch-by-batch output table.  The file path
    is printed at startup so operators know where to look.

    Idempotent: calling this function multiple times in the same process
    does not attach duplicate handlers.  Any existing handlers on the
    ``model_monitor`` logger are replaced, not accumulated.
    """
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "simulation.log"

    # Scope to the model_monitor logger tree only.  Clearing the root logger
    # would silently drop handlers set by uvicorn, pytest, or any wrapper
    # script that configures logging before calling this function.
    mm_log = logging.getLogger("model_monitor")
    mm_log.setLevel(logging.INFO)
    # Remove any handlers already on this logger (e.g. from a previous run
    # in the same process) before adding the file handler.  This is what
    # makes the function idempotent: handler count stays at exactly 1.
    for _h in list(mm_log.handlers):
        mm_log.removeHandler(_h)
        _h.close()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    mm_log.addHandler(fh)
    # Prevent events from bubbling to the root logger (which may have a
    # StreamHandler attached by pytest or the calling shell).
    mm_log.propagate = False

    # Print the log path to stderr (not stdout) so it is visible but
    # does not pollute the captured table output.
    print(f"  logs → {log_file}", file=sys.stderr)


# Terminal colours - disabled automatically when not a tty
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _action_colour(action: str) -> str:
    if action in {"reject", "rollback"}:
        return _RED
    if action in {"retrain", "promote"}:
        return _YELLOW
    return _DIM


def _load_feature_names(base: Path | None = None) -> list[str]:
    path = (base or Path("data/reference")) / "feature_schema.json"
    with path.open() as f:
        result: list[str] = json.load(f)
        return result


def _load_reference_bin_edges(
    feature_names: list[str],
    base: Path | None = None,
) -> dict[int, np.ndarray]:
    path = (base or Path("data/reference")) / "reference_stats.json"
    if not path.exists():
        return {}
    with path.open() as f:
        stats: dict[str, dict] = json.load(f)
    edges: dict[int, np.ndarray] = {}
    for i, name in enumerate(feature_names):
        if name in stats and "psi_bin_edges" in stats[name]:
            edges[i] = np.asarray(stats[name]["psi_bin_edges"], dtype=float)
    return edges


def _load_reference_features(
    feature_names: list[str],
    base: Path | None = None,
) -> np.ndarray | None:
    """Load the training population saved by ``train.py`` for DriftMonitor.

    ``train.py`` saves the exact training feature matrix to
    ``data/reference/train_population.npy`` so the simulation can use the
    identical distribution - not a re-instantiation of ``make_classification``
    with different parameters, which produces structurally different data
    because sklearn's algorithm is not iid-sampling from a fixed population.

    Returns None if the file is not found (run ``make train`` first).
    """
    npy_path = (base or Path("data/reference")) / "train_population.npy"
    if not npy_path.exists():
        return None
    result: np.ndarray = np.load(npy_path)
    return result


def simulate_stream(
    config: AppConfig,
    *,
    n_batches: int = 80,
    batch_size: int = 500,
    drift_at_batch: int = 40,
    drift_magnitude: float = 2.0,
    sim_drift_window: int = 5,
    data_dir: Path | None = None,
) -> None:
    """Run the end-to-end simulation, wiring all production components together.

    This function is the closest analogue to ``main()`` in a production inference
    service: it constructs every component from scratch, wires them together, and
    drives data through them batch by batch.

    Retrain data flow
    -----------------
    Two buffers run in parallel:

    ``RetrainEvidenceBuffer`` accumulates aggregated monitoring signals
    (accuracy, F1, drift score, trust score) and determines *when* to retrain.
    It is lightweight and survives restarts when checkpointed.

    ``RawDataBuffer`` accumulates the actual labeled ``(X, y)`` pairs from each
    batch and provides *what to train on*.  When a retrain fires and the buffer
    holds enough rows, the pipeline trains on recent observed data - data that
    reflects the current (possibly drifted) distribution.  If the buffer is
    not yet full (only possible very early in the run), fresh synthetic data is
    used as a fallback with an explicit warning.

    This is the correct production data flow.  Training on monitoring-metric
    aggregates (the previous broken behaviour) was structurally impossible:
    ``train_model()`` requires feature columns and a label column, not
    accuracy/F1/drift/trust floats.
    """
    _configure_sim_logging()

    try:
        feature_names = _load_feature_names(data_dir)
    except FileNotFoundError:
        logger.warning(
            "feature_schema.json not found - run 'make train' first",
            extra={"data_dir": str(data_dir or "data/reference")},
        )
        return

    stored_bin_edges = _load_reference_bin_edges(feature_names, data_dir)
    reference_features = _load_reference_features(feature_names, data_dir)

    if reference_features is None:
        print(
            "\n  ERROR: data/reference/train_population.npy not found."
            "\n  Run 'make train' first to generate the training population."
            "\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load labels from the companion file.  Features are already in
    # reference_features (loaded above for the DriftMonitor).
    _lbl_path = (data_dir or Path("data/reference")) / "train_population.labels.npy"
    if not _lbl_path.exists():
        print(
            "\n  ERROR: data/reference/train_population.labels.npy not found."
            "\n  Run 'make train' first to generate the training population."
            "\n",
            file=sys.stderr,
        )
        sys.exit(1)

    _X_pop = reference_features  # same array, alias for clarity
    _y_pop = np.load(_lbl_path)
    # within 80 batches.  Production deployments use the configured window.
    if sim_drift_window > 0 and sim_drift_window != config.drift.window:
        from model_monitor.config.settings import DriftConfig

        sim_config = config.model_copy(
            update={
                "drift": DriftConfig(
                    psi_threshold=config.drift.psi_threshold,
                    window=sim_drift_window,
                )
            }
        )
    else:
        sim_config = config

    model_store = ModelStore()
    active_meta = model_store.get_active_metadata()
    f1_baseline: float | None = active_meta.get("metrics", {}).get("baseline_f1")

    # ── New monitoring stack ──────────────────────────────────────────────
    # OutputDriftMonitor: PSI on predicted probability vectors.
    # Reference probabilities come from the training population so the
    # same training-time bin-edge principle applies as for input PSI.
    _ref_model = None
    try:
        from model_monitor.storage.model_store import ModelStore as _MS

        _ref_model = _MS().load_current()
    except Exception:
        pass

    output_drift_monitor: OutputDriftMonitor | None = None
    if _ref_model is not None:
        try:
            _ref_probs = _ref_model.predict_proba(_X_pop)
            output_drift_monitor = OutputDriftMonitor(
                _ref_probs, window=sim_config.drift.window, threshold=0.10
            )
        except Exception:
            pass

    # DataQualityMonitor: null rate + range violations + schema consistency.
    # Feature bounds are set to ±4 std of the training distribution -
    # values outside this range almost certainly indicate upstream corruption.
    _feature_names = [f"f{i}" for i in range(reference_features.shape[1])]
    _means = reference_features.mean(axis=0)
    _stds = reference_features.std(axis=0) + 1e-9
    _bounds = {
        name: (float(_means[i] - 4 * _stds[i]), float(_means[i] + 4 * _stds[i]))
        for i, name in enumerate(_feature_names)
    }
    data_quality_monitor = DataQualityMonitor(
        feature_names=_feature_names,
        feature_bounds=_bounds,
        max_null_rate=0.05,
        max_oor_rate=0.02,
    )

    # ConformalMonitor: calibrate on first 20% of training population.
    conformal_monitor: ConformalMonitor | None = None
    if _ref_model is not None:
        try:
            _n_cal = max(50, len(_y_pop) // 5)
            _cal_probs = _ref_model.predict_proba(_X_pop[:_n_cal])
            _cal_labels = _y_pop[:_n_cal].astype(int)
            conformal_monitor = ConformalMonitor(alpha=0.10)
            conformal_monitor.calibrate(_cal_probs, _cal_labels)
        except Exception:
            conformal_monitor = None

    # CausalDriftAttributor: learn Granger-causal structure on the training
    # population so drifting features can later be classified as genuine_shift
    # vs pipeline_suspect vs correlated_follower.
    causal_attributor: CausalDriftAttributor | None = None
    try:
        causal_attributor = CausalDriftAttributor(
            feature_names=_feature_names,
            psi_threshold=config.drift.psi_threshold,
            alpha=0.05,
            max_lag=3,
        )
        causal_attributor.fit(reference_features)
    except Exception:
        causal_attributor = None

    # ThresholdAdvisor: records stable-period PSI and trust scores so it can
    # recommend calibrated warn thresholds for this specific deployment.
    # Recommendations are emitted after the simulation completes.
    threshold_advisor = ThresholdAdvisor(
        feature_names=_feature_names,
        alpha=0.05,
        min_batches=20,
    )

    predictor = Predictor(
        config=sim_config,
        reference_features=reference_features,
        stored_bin_edges=stored_bin_edges if stored_bin_edges else None,
        f1_baseline=f1_baseline,
        output_drift_monitor=output_drift_monitor,
        data_quality_monitor=data_quality_monitor,
        conformal_monitor=conformal_monitor,
        causal_drift_attributor=causal_attributor,
        threshold_advisor=threshold_advisor,
    )
    # Load model weights from disk before the first predict_batch call.
    # Without this, active_model raises RuntimeError because self.model is None.
    predictor.reload()

    # Evidence buffer: aggregated signals that trigger the retrain gate.
    # Uses evidence_window (default 3) - the minimum number of monitoring
    # summaries required before acting.  This is deliberately separate from
    # min_samples (raw data rows), which governs what to train on.
    retrain_evidence = RetrainEvidenceBuffer(min_samples=config.retrain.evidence_window)
    # Raw data buffer: labeled (X, y) pairs that provide the training dataset.
    raw_data_buffer = RawDataBuffer(max_rows=50_000)
    retrain_pipeline = RetrainPipeline(model_store=model_store)
    metrics_store = MetricsStore()

    print(f"\n{_BOLD}━━━  model_monitor simulation  ━━━{_RESET}")
    print(f"  {n_batches} batches · drift injected at batch {drift_at_batch}")
    _od_hdr = "out_psi" if output_drift_monitor is not None else ""
    _dq_hdr = "dq_scr" if True else ""
    _cp_hdr = "cp_cov" if conformal_monitor is not None else ""
    _extra = (
        f"  {_od_hdr:>7}  {_dq_hdr:>7}  {_cp_hdr:>7}"
        if output_drift_monitor is not None
        else ""
    )
    print(f"  {'batch':>6}  {'in_psi':>7}  {'trust':>7}{_extra}  {'action'}")
    print(
        f"  {'─' * 6}  {'─' * 7}  {'─' * 7}{'  ─' * 7 if output_drift_monitor else ''}  {'─' * 20}"
    )

    for step in range(n_batches):
        batch_id = f"batch_{step}_{uuid.uuid4().hex[:6]}"
        drifted = step >= drift_at_batch
        shift = drift_magnitude if drifted else 0.0

        # Draw a fresh slice from the pre-generated population for each batch.
        rng = np.random.default_rng(step)
        idx = rng.choice(len(_X_pop), size=batch_size, replace=False)
        X = _X_pop[idx] + shift
        X_df = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(_y_pop[idx])

        preds, confs, decision = predictor.predict_batch(
            X_df,
            y_true=y,
            batch_id=batch_id,
            candidate_exists=model_store.has_candidate(),
        )

        # Accumulate labeled data for retraining.  Must happen before the
        # retrain check so the buffer is populated when the trigger fires.
        raw_data_buffer.add_batch(X, y.to_numpy(), feature_names)

        drift_score = predictor.last_drift_score
        trust_score = predictor.last_trust_score

        if predictor.last_metric_record is not None:
            try:
                metrics_store.write(predictor.last_metric_record)
            except Exception:
                logger.warning("metrics_store_write_failed", exc_info=True)

        # Use sklearn f1_score - not accuracy - for the evidence buffer so
        # the retrain threshold operates on the same metric as the decision engine.
        batch_f1 = float(f1_score(y, preds, zero_division=0))

        retrain_evidence.add_summary(
            accuracy=float((preds == y).mean()),
            f1=batch_f1,
            drift_score=drift_score,
            trust_score=trust_score,
            timestamp=time.time(),
        )

        colour = _action_colour(decision.action)
        marker = " ← drift" if step == drift_at_batch else ""
        _rec = predictor.last_metric_record
        _od = (
            f"  {_rec['output_drift_score']:>7.3f}"
            if (_rec and _rec.get("output_drift_score") is not None)
            else ("  " + " " * 7 if output_drift_monitor else "")
        )
        _dq = (
            f"  {_rec['data_quality_score']:>7.3f}"
            if (_rec and _rec.get("data_quality_score") is not None)
            else "  " + " " * 7
        )
        _cp = (
            f"  {_rec['conformal_coverage']:>7.3f}"
            if (_rec and _rec.get("conformal_coverage") is not None)
            else ("  " + " " * 7 if conformal_monitor else "")
        )
        _extras = f"{_od}{_dq}{_cp}" if output_drift_monitor is not None else ""
        print(
            f"  {step:>6}  {drift_score:>7.3f}  {trust_score:>7.3f}{_extras}  "
            f"{colour}{decision.action:<12}{_RESET}{_DIM}{marker}{_RESET}"
        )

        logger.info(
            "batch_decision",
            extra={
                "batch_id": batch_id,
                "step": step,
                "drifted": drifted,
                "drift_score": round(drift_score, 4),
                "trust_score": round(trust_score, 4),
                "f1": round(batch_f1, 4),
                "action": decision.action,
                "model_version": get_active_version(),
            },
        )

        if decision.action == "rollback":
            target = decision.metadata.get("target_version")
            if target is None:
                # No prior version to roll back to - happens at cold start when
                # no model has been promoted yet.  Skip silently; the monitor
                # will re-evaluate on the next batch.
                logger.warning(
                    "rollback_skipped_no_prior_version", extra={"batch_id": batch_id}
                )
            else:
                model_store.rollback(version=str(target))
                predictor.reload()
                print(f"  {_YELLOW}↩  rolled back to {target}{_RESET}")

        if decision.action == "retrain" and retrain_evidence.ready():
            _evidence_df = retrain_evidence.consume()

            if raw_data_buffer.ready(min_samples=config.retrain.min_samples):
                retrain_df = raw_data_buffer.consume()
                logger.info(
                    "retrain_using_observed_data",
                    extra={"n_samples": len(retrain_df)},
                )
            else:
                # The raw data buffer has not yet accumulated min_samples rows.
                # This only occurs very early in a deployment before the first
                # retrain trigger fires.  Fall back to fresh synthetic data and
                # log a warning so operators are aware.
                logger.warning(
                    "retrain_buffer_insufficient_using_synthetic_fallback",
                    extra={
                        "raw_buffer_size": raw_data_buffer.size(),
                        "min_samples": config.retrain.min_samples,
                    },
                )
                from model_monitor.training.train import make_dataset

                retrain_df, _ = make_dataset(
                    n_samples=max(500, config.retrain.min_samples * 2),
                    random_state=step,
                )

            result = retrain_pipeline.run(
                retrain_df,
                min_f1_improvement=config.retrain.min_f1_gain,
            )
            if result.promotion.promoted and result.candidate_model is not None:
                # Persist to disk so model_store.list_versions() reflects the
                # promotion and predictor.reload() loads the new weights.
                model_store.save_candidate(result.candidate_model)
                new_version = model_store.promote_candidate(
                    metrics={
                        "baseline_f1": result.promotion.candidate_f1,
                        "candidate_f1": result.promotion.candidate_f1,
                        "current_f1": result.promotion.current_f1,
                        "improvement": result.promotion.improvement,
                        "n_samples": result.n_samples,
                    }
                )
                predictor.reload()
                logger.info(
                    "simulation_model_promoted",
                    extra={
                        "version": new_version,
                        "candidate_f1": result.promotion.candidate_f1,
                        "improvement": result.promotion.improvement,
                    },
                )
                print(
                    f"  {_GREEN}↑  new model promoted "
                    f"v{new_version} (F1 {result.promotion.candidate_f1:.3f}){_RESET}"
                )
            elif not result.promotion.promoted:
                print(
                    f"  {_DIM}↷  candidate rejected (insufficient improvement){_RESET}"
                )

        time.sleep(0.02)

    print(f"\n{_DIM}  simulation finished - {n_batches} batches{_RESET}\n")

    # ── Post-simulation: causal drift summary ─────────────────────────────
    # Find the most recent batch where causal attribution ran - it won't
    # always be the last batch (which may be a reject with no label data).
    if causal_attributor is not None:
        from model_monitor.monitoring.types import MetricRecord as _MR

        _causal_record: _MR | None = None
        if predictor.last_metric_record is not None:
            _last = predictor.last_metric_record
            if _last.get("causal_drift_report") is not None:
                _causal_record = _last
        if _causal_record is None:
            try:
                for _rec in metrics_store.tail(limit=20):
                    if _rec.get("causal_drift_report") is not None:
                        _causal_record = _rec
                        break
            except Exception:
                pass
        _causal_json = _causal_record.get("causal_drift_report") if _causal_record else None
        if _causal_json:
            import json as _json

            try:
                _report: dict[str, object]
                if isinstance(_causal_json, str):
                    _report = _json.loads(_causal_json)
                elif isinstance(_causal_json, dict):
                    # last_metric_record stores the raw dict; MetricsStore
                    # stores JSON - handle both paths.
                    _report = _causal_json
                else:
                    _report = {}  # unexpected type - skip silently
                _cause = str(_report.get("dominant_cause", "unknown"))
                _causal_rec = str(_report.get("recommendation", ""))
                _colour = _RED if _cause == "pipeline_failure" else (_YELLOW if _cause == "mixed" else _GREEN)
                print(f"{_BOLD}Causal Drift Attribution{_RESET}")
                print(f"  Dominant cause: {_colour}{_cause.upper()}{_RESET}")
                print(f"  Recommendation: {_causal_rec}")
                _raw_results = _report.get("feature_results")
                _results: list[dict[str, object]] = (
                    list(_raw_results)
                    if isinstance(_raw_results, list)
                    else []
                )
                if _results:
                    print(f"  {'Feature':<18}  {'PSI':>6}  {'Classification'}")
                    print(f"  {'─'*18}  {'─'*6}  {'─'*22}")
                    for _fr_d in _results[:8]:  # cap at 8 rows for readability
                        _cls = str(_fr_d.get("drift_class", "stable"))
                        _cls_colour = (
                            _RED if _cls == "pipeline_suspect"
                            else (_YELLOW if _cls == "correlated_follower"
                                  else (_DIM if _cls == "stable" else _GREEN))
                        )
                        _psi_val = _fr_d.get("psi", 0.0)
                        print(
                            f"  {str(_fr_d.get('feature_name','?')):<18}  "
                            f"{float(_psi_val):>6.3f}  "  # type: ignore[arg-type]
                            f"{_cls_colour}{_cls}{_RESET}"
                        )
                print()
            except Exception:
                pass

    # ── Post-simulation: threshold advisor recommendations ────────────────
    # ThresholdAdvisor has observed N batches of stable-period signals.
    # It recommends calibrated warn thresholds for PSI and trust score that
    # match the natural variance of this specific deployment - eliminating
    # alert fatigue from generic defaults.
    try:
        if threshold_advisor.is_ready:
            _ta_rec = threshold_advisor.recommend()
            print(f"{_BOLD}Threshold Advisor Recommendations{_RESET}")
            print(
                f"  {_DIM}(calibrated from {threshold_advisor.n_observations} "
                f"stable-period batches){_RESET}"
            )
            print(f"  Global PSI warn threshold : {_ta_rec.psi_warn_global:.4f}")
            print(f"  Trust score warn threshold: {_ta_rec.trust_warn:.4f}")
            if _ta_rec.feature_names and _ta_rec.psi_warn_per_feature:
                print(f"  {'Feature':<28}  {'PSI warn':>8}")
                print(f"  {'─'*28}  {'─'*8}")
                for _fname, _fthresh in zip(_ta_rec.feature_names, _ta_rec.psi_warn_per_feature):
                    print(f"  {_fname:<28}  {_fthresh:>8.4f}")
            if _ta_rec.notes:
                print("\n  Notes:")
                for _note in _ta_rec.notes:
                    print(f"    · {_note}")
            print()
        else:
            print(
                f"{_DIM}  ThresholdAdvisor: {threshold_advisor.n_observations}/"
                f"{threshold_advisor.min_batches} stable batches observed "
                f"(run more stable batches before drift injection for recommendations){_RESET}\n"
            )
    except Exception:
        pass


def main() -> None:  # pragma: no cover
    # load_config() with no arguments uses importlib.resources to resolve
    # the bundled YAML files - works correctly regardless of working directory.
    cfg = load_config()
    simulate_stream(config=cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
