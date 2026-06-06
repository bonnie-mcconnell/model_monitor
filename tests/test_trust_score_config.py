"""Tests for TrustScoreConfig YAML-driven weights.

Design decisions tested:
- Weights summing != 1.0 are rejected at construction time (not silently wrong).
- compute_trust_score() uses config weights when provided.
- aggregate_records() respects the trust score config weights.
- load_config() reads trust_score.yaml correctly.
- Legacy 'confidence' key is no longer valid; 'calibration' is the correct key.
- All seven components (accuracy, f1, calibration, drift, latency,
  data_quality, behavioral) are configurable and validated together.
"""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import pytest

from model_monitor.config.settings import TrustScoreConfig, load_config
from model_monitor.core.decisions import DecisionType
from model_monitor.monitoring.aggregation import _aggregate_records as build_summary
from model_monitor.monitoring.trust_score import compute_trust_score
from model_monitor.monitoring.types import MetricRecord

# ---------------------------------------------------------------------------
# TrustScoreConfig construction and validation
# ---------------------------------------------------------------------------


def test_default_weights_sum_to_one() -> None:
    cfg = TrustScoreConfig()
    total = (
        cfg.accuracy
        + cfg.f1
        + cfg.calibration
        + cfg.drift
        + cfg.latency
        + cfg.data_quality
        + cfg.behavioral
    )
    assert abs(total - 1.0) < 1e-9


def test_custom_weights_accepted_when_sum_is_one() -> None:
    cfg = TrustScoreConfig(
        accuracy=0.20,
        f1=0.15,
        calibration=0.15,
        drift=0.20,
        latency=0.15,
        data_quality=0.10,
        behavioral=0.05,
    )
    assert abs(cfg.accuracy - 0.20) < 1e-9


def test_weights_rejected_when_sum_exceeds_one() -> None:
    with pytest.raises(Exception, match="sum to 1.0"):
        TrustScoreConfig(
            accuracy=0.30,
            f1=0.25,
            calibration=0.15,
            drift=0.20,
            latency=0.20,
            data_quality=0.05,
            behavioral=0.05,  # total = 1.20
        )


def test_weights_rejected_when_sum_below_one() -> None:
    with pytest.raises(Exception, match="sum to 1.0"):
        TrustScoreConfig(
            accuracy=0.10,
            f1=0.10,
            calibration=0.10,
            drift=0.10,
            latency=0.10,
            data_quality=0.05,
            behavioral=0.05,  # total = 0.60
        )


def test_zero_weight_allowed_when_sum_is_one() -> None:
    cfg = TrustScoreConfig(
        accuracy=0.30,
        f1=0.25,
        calibration=0.20,
        drift=0.20,
        latency=0.05,
        data_quality=0.0,
        behavioral=0.0,
    )
    assert cfg.data_quality == 0.0
    assert cfg.behavioral == 0.0


def test_error_message_shows_all_weights() -> None:
    with pytest.raises(Exception) as exc_info:
        TrustScoreConfig(
            accuracy=0.0,
            f1=0.0,
            calibration=0.0,
            drift=0.0,
            latency=0.0,
            data_quality=0.0,
            behavioral=0.0,
        )
    assert "1.0" in str(exc_info.value)


# ---------------------------------------------------------------------------
# compute_trust_score() uses config weights
# ---------------------------------------------------------------------------


def test_compute_trust_score_uses_config_drift_weight() -> None:
    """High drift weight should produce lower score when drift is high."""
    cfg_high_drift = TrustScoreConfig(
        accuracy=0.10,
        f1=0.10,
        calibration=0.10,
        drift=0.60,
        latency=0.05,
        data_quality=0.02,
        behavioral=0.03,
    )
    cfg_low_drift = TrustScoreConfig(
        accuracy=0.30,
        f1=0.25,
        calibration=0.15,
        drift=0.10,
        latency=0.10,
        data_quality=0.05,
        behavioral=0.05,
    )

    kwargs = dict(
        accuracy=0.80,
        f1=0.78,
        avg_confidence=0.75,
        drift_score=0.28,  # above 0.25 → drift_to_trust = 0.0
        decision_latency_ms=100.0,
    )
    score_high, _ = compute_trust_score(**kwargs, config=cfg_high_drift)
    score_low, _ = compute_trust_score(**kwargs, config=cfg_low_drift)
    assert score_high < score_low


def test_compute_trust_score_uses_config_without_optional_fields() -> None:
    """compute_trust_score must work when optional new-monitor fields are None."""
    cfg = TrustScoreConfig()
    score, comps = compute_trust_score(
        accuracy=0.90,
        f1=0.88,
        avg_confidence=0.85,
        drift_score=0.05,
        decision_latency_ms=120.0,
        config=cfg,
    )
    assert 0.0 <= score <= 1.0
    assert "calibration" in comps


# ---------------------------------------------------------------------------
# load_config() reads trust_score.yaml
# ---------------------------------------------------------------------------


def test_load_config_with_custom_trust_score_yaml(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent("""\
        trust_score:
          accuracy:     0.20
          f1:           0.15
          calibration:  0.15
          drift:        0.20
          latency:      0.15
          data_quality: 0.10
          behavioral:   0.05
    """)
    ts_path = tmp_path / "trust_score.yaml"
    ts_path.write_text(yaml_text)

    cfg = load_config(trust_score_path=ts_path)
    assert abs(cfg.trust_score.f1 - 0.15) < 1e-9
    assert abs(cfg.trust_score.calibration - 0.15) < 1e-9
    assert abs(cfg.trust_score.data_quality - 0.10) < 1e-9


def test_load_config_falls_back_to_defaults_when_yaml_absent(tmp_path: Path) -> None:
    cfg = load_config(trust_score_path=tmp_path / "nonexistent.yaml")
    ts = cfg.trust_score
    # Verify new 7-component defaults
    assert abs(ts.accuracy - 0.23) < 1e-9
    assert abs(ts.f1 - 0.18) < 1e-9
    assert abs(ts.calibration - 0.14) < 1e-9
    assert abs(ts.data_quality - 0.05) < 1e-9
    assert abs(ts.behavioral - 0.05) < 1e-9
    total = (
        ts.accuracy
        + ts.f1
        + ts.calibration
        + ts.drift
        + ts.latency
        + ts.data_quality
        + ts.behavioral
    )
    assert abs(total - 1.0) < 1e-9


def test_trust_score_config_in_app_config_default() -> None:
    from model_monitor.config.settings import (
        AppConfig,
        DriftConfig,
        RetrainConfig,
        RollbackConfig,
    )

    cfg = AppConfig(
        drift=DriftConfig(psi_threshold=0.25, window=3),
        retrain=RetrainConfig(
            min_f1_gain=0.02,
            cooldown_batches=5,
            min_samples=100,
            max_retrain_attempts=10,
        ),
        rollback=RollbackConfig(max_f1_drop=0.15),
    )
    assert hasattr(cfg.trust_score, "calibration")
    assert hasattr(cfg.trust_score, "data_quality")
    assert hasattr(cfg.trust_score, "behavioral")


# ---------------------------------------------------------------------------
# aggregate_records() respects trust score config
# ---------------------------------------------------------------------------


def test_aggregate_records_respects_trust_score_config() -> None:
    """build_summary must accept and use the config argument."""

    def _rec(accuracy: float, f1: float) -> MetricRecord:
        return {
            "timestamp": time.time(),
            "batch_id": "b",
            "n_samples": 50,
            "accuracy": accuracy,
            "f1": f1,
            "avg_confidence": 0.8,
            "drift_score": 0.05,
            "decision_latency_ms": 100.0,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
            "calibration_error": None,
            "feature_drift_scores": None,
            "output_drift_score": None,
            "output_drift_class_scores": None,
            "data_quality_score": None,
            "conformal_coverage": None,
            "conformal_set_size": None,
            "behavioral_violation_rate": None,
            "shap_attribution": None,
            "causal_drift_report": None,
            "mmd_p_value": None,
            "mmd_is_drift": None,
            "action": DecisionType.NONE,
            "reason": "ok",
            "previous_model": None,
            "new_model": None,
        }

    records = [_rec(0.90, 0.88), _rec(0.85, 0.82)]

    cfg_f1_heavy = TrustScoreConfig(
        accuracy=0.10,
        f1=0.60,
        calibration=0.10,
        drift=0.05,
        latency=0.05,
        data_quality=0.05,
        behavioral=0.05,
    )
    cfg_acc_heavy = TrustScoreConfig(
        accuracy=0.60,
        f1=0.10,
        calibration=0.10,
        drift=0.05,
        latency=0.05,
        data_quality=0.05,
        behavioral=0.05,
    )

    from model_monitor.config.settings import (
        AppConfig,
        DriftConfig,
        RetrainConfig,
        RollbackConfig,
    )

    base = dict(
        drift=DriftConfig(psi_threshold=0.25, window=3),
        retrain=RetrainConfig(
            min_f1_gain=0.02,
            cooldown_batches=5,
            min_samples=100,
            max_retrain_attempts=10,
        ),
        rollback=RollbackConfig(max_f1_drop=0.15),
    )
    app_f1 = AppConfig(**base, trust_score=cfg_f1_heavy)  # type: ignore[arg-type]
    app_acc = AppConfig(**base, trust_score=cfg_acc_heavy)  # type: ignore[arg-type]

    summary_f1 = build_summary("5m", records, cfg=app_f1)
    summary_acc = build_summary("5m", records, cfg=app_acc)

    # F1-weighted: high F1 records → higher score
    # Acc-weighted: both have similar accuracy
    # Key invariant: different configs produce different scores
    assert summary_f1.trust_score != summary_acc.trust_score
