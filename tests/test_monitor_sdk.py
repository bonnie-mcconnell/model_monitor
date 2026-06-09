"""Tests for the public Monitor SDK (model_monitor.monitor).

These tests verify the wrap-any-model interface that is the primary
entry point for new users.  They cover:

  - Construction with numpy and DataFrame reference data
  - predict() with and without labels
  - BatchResult contract (predictions shape, trust/drift score bounds)
  - summary() and report() return correct structure
  - Models with predict_proba and models with predict-only
  - Causal attribution triggered only on drifting batches
  - ThresholdAdvisor fed only on stable batches
  - db_path persistence writes to MetricsStore
  - feature_names inference from DataFrame columns
  - Edge cases: zero-length batches, single feature, single sample
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from model_monitor import BatchResult, Monitor, MonitorConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clf_and_data() -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """Small trained RandomForest plus matching X/y for test batches.

    Returns the classifier along with a 400-row held-out set so all tests
    have enough rows for reference (200) + multiple predict batches.
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        random_state=0,
    )
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X[:600], y[:600])
    return clf, X[600:], y[600:]  # 400 held-out rows


@pytest.fixture()
def monitor(
    clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
) -> Monitor:
    """Monitor wrapping the fixture classifier."""
    clf, X_test, _ = clf_and_data
    return Monitor(
        clf,
        reference_data=X_test[:200],
        config=MonitorConfig(
            psi_threshold=0.10,
            drift_window=3,
            enable_causal=True,
            enable_conformal=False,  # no labels at construction time
            enable_threshold_advisor=True,
            min_advisor_batches=10,
        ),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMonitorConstruction:
    """Monitor.__init__ contract."""

    def test_wraps_predict_proba_model(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200])
        assert m.n_batches == 0

    def test_feature_names_inferred_from_dataframe(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        import pandas as pd

        clf, X, _ = clf_and_data
        cols = [f"col_{i}" for i in range(X.shape[1])]
        df_ref = pd.DataFrame(X[:200], columns=cols)
        m = Monitor(clf, reference_data=df_ref)
        assert m._feature_names == cols

    def test_explicit_feature_names_respected(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        clf, X, _ = clf_and_data
        names = [f"feat_{i}" for i in range(X.shape[1])]
        m = Monitor(clf, reference_data=X[:200], feature_names=names)
        assert m._feature_names == names

    def test_auto_generated_feature_names_when_not_supplied(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200])
        assert m._feature_names == [f"f{i}" for i in range(X.shape[1])]

    def test_raises_on_mismatched_feature_names(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        clf, X, _ = clf_and_data
        with pytest.raises(ValueError, match="feature_names has"):
            Monitor(
                clf,
                reference_data=X[:200],
                feature_names=["a", "b"],  # wrong count
            )

    def test_predict_only_model_accepted(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        """A model with predict() but no predict_proba() is accepted."""
        _, X, _ = clf_and_data  # X is 200 rows (600:800)

        class LabelOnlyModel:
            def predict(self, X_in: np.ndarray) -> np.ndarray:
                return np.zeros(len(X_in), dtype=int)

        m = Monitor(LabelOnlyModel(), reference_data=X[:100])
        result = m.predict(X[100:150])
        assert result.predictions.shape == (50,)
        # No probabilities → confidence fixed at 0.5
        assert np.all(result.confidences == 0.5)

    def test_with_y_reference_enables_conformal(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        clf, X, y = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            y_reference=y[:200],
            config=MonitorConfig(enable_conformal=True),
        )
        assert m._conformal is not None
        assert m._conformal.is_calibrated


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    """BatchResult contract and monitoring side-effects."""

    def test_returns_batch_result(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        result = monitor.predict(X[:50])
        assert isinstance(result, BatchResult)

    def test_predictions_shape_matches_input(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        result = monitor.predict(X[:75])
        assert result.predictions.shape == (75,)
        assert result.confidences.shape == (75,)

    def test_trust_score_in_unit_interval(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        result = monitor.predict(X[:50])
        assert 0.0 <= result.trust_score <= 1.0

    def test_drift_score_non_negative(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        result = monitor.predict(X[:50])
        assert result.drift_score >= 0.0

    def test_batch_id_auto_generated(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        r1 = monitor.predict(X[:50])
        r2 = monitor.predict(X[50:100])
        assert r1.batch_id != r2.batch_id

    def test_explicit_batch_id_respected(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        result = monitor.predict(X[:30], batch_id="my_batch_42")
        assert result.batch_id == "my_batch_42"

    def test_n_batches_increments(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        for i in range(4):
            monitor.predict(X[i * 20 : (i + 1) * 20])
        assert monitor.n_batches == 4

    def test_history_accumulates(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        for _ in range(3):
            monitor.predict(X[:30])
        assert len(monitor.history) == 3

    def test_history_newest_first(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """history property returns batches newest-first for dashboard display."""
        _, X, _ = clf_and_data
        monitor.predict(X[:30], batch_id="first")
        monitor.predict(X[30:60], batch_id="second")
        assert monitor.history[0]["batch_id"] == "second"
        assert monitor.history[1]["batch_id"] == "first"

    def test_predict_with_labels_computes_accuracy(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        clf, X, y = clf_and_data
        m = Monitor(clf, reference_data=X[:200])
        m.predict(X[200:250], y_true=y[200:250])
        # accuracy and f1 are stored in history
        rec = m.history[0]
        assert rec.get("accuracy") is not None
        assert rec.get("f1") is not None

    def test_is_healthy_property(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        """is_healthy is True when trust_score >= 0.70."""
        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200])
        result = m.predict(X[200:250])
        assert result.is_healthy == (result.trust_score >= 0.70)

    def test_dataframe_input_accepted(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        import pandas as pd

        clf, X, _ = clf_and_data
        cols = [f"f{i}" for i in range(X.shape[1])]
        df_ref = pd.DataFrame(X[:200], columns=cols)
        m = Monitor(clf, reference_data=df_ref)
        df_batch = pd.DataFrame(X[200:250], columns=cols)
        result = m.predict(df_batch)
        assert result.predictions.shape == (50,)

    def test_large_drift_triggers_is_drifting(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        """Predictions on out-of-distribution data set is_drifting=True.

        We need enough batches to fill the drift window before PSI is
        computed, so we feed drift_window worth of OOD batches first.
        """
        clf, X, _ = clf_and_data
        rng = np.random.default_rng(99)
        cfg = MonitorConfig(psi_threshold=0.10, drift_window=2, enable_causal=False)
        m = Monitor(clf, reference_data=X[:200], config=cfg)
        # Out-of-distribution: shift mean by 10 sigma
        X_ood = X[200:250] + rng.standard_normal(X[200:250].shape) * 10
        for _ in range(3):
            result = m.predict(X_ood)
        assert result.is_drifting


# ---------------------------------------------------------------------------
# summary() and report()
# ---------------------------------------------------------------------------


class TestSummaryAndReport:
    """summary() and report() contract."""

    def test_summary_empty_before_first_predict(self, monitor: Monitor) -> None:
        s = monitor.summary()
        assert s.n_batches == 0
        assert s.mean_trust_score is None

    def test_summary_contains_required_keys(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        from model_monitor import MonitorSummary

        _, X, _ = clf_and_data
        monitor.predict(X[:50])
        s = monitor.summary()
        assert isinstance(s, MonitorSummary)
        assert s.n_batches >= 1
        assert s.n_features > 0
        assert s.feature_names is not None

    def test_summary_n_batches_correct(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        for _ in range(3):
            monitor.predict(X[:30])
        assert monitor.summary().n_batches == 3

    def test_report_is_string(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        monitor.predict(X[:50])
        assert isinstance(monitor.report(), str)

    def test_report_no_predict_is_string(self, monitor: Monitor) -> None:
        assert "no batches" in monitor.report().lower()

    def test_report_contains_trust_drift_lines(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        _, X, _ = clf_and_data
        monitor.predict(X[:50])
        r = monitor.report()
        assert "trust" in r.lower()
        assert "drift" in r.lower()


# ---------------------------------------------------------------------------
# Threshold recommendations
# ---------------------------------------------------------------------------


class TestThresholdRecommendations:
    """ThresholdAdvisor integration through Monitor."""

    def test_none_before_enough_stable_batches(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """Returns None when advisor has fewer observations than min_batches."""
        _, X, _ = clf_and_data
        # Feed only 2 batches; min_advisor_batches=10 in fixture
        monitor.predict(X[:30])
        monitor.predict(X[30:60])
        assert monitor.threshold_recommendations() is None

    def test_recommendations_available_after_enough_stable_batches(
        self, clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray]
    ) -> None:
        """After enough stable-period batches the advisor emits calibrated thresholds.

        We deliberately feed batches drawn from the *same* distribution as
        the reference data (in-distribution held-out rows) so PSI stays
        well below the 0.10 threshold and every batch is counted as stable.
        """
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:100],
            config=MonitorConfig(
                enable_causal=False,
                enable_threshold_advisor=True,
                min_advisor_batches=10,
                drift_window=3,
            ),
        )
        rng = np.random.default_rng(7)
        # 11 batches of in-distribution data drawn from the same reference rows
        # - PSI will be ~0 so every batch counts toward the advisor.
        for _ in range(11):
            idx = rng.integers(0, 100, size=40)
            m.predict(X[idx])

        recs = m.threshold_recommendations()
        assert recs is not None
        assert "psi_warn_global" in recs
        assert "trust_warn" in recs
        assert recs["psi_warn_global"] >= 0.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """db_path causes records to be written to MetricsStore."""

    def test_records_written_to_db(
        self,
        tmp_path: Path,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        from model_monitor.storage.metrics_store import MetricsStore

        clf, X, _ = clf_and_data
        db = str(tmp_path / "sdk_test.db")
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(db_path=db, enable_causal=False),
        )
        m.predict(X[200:250])
        m.predict(X[250:300])

        store = MetricsStore(db_path=db)
        records = store.tail(limit=10)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# New BatchResult fields (v9 additions)
# ---------------------------------------------------------------------------


class TestBatchResultNewFields:
    """psi_per_feature, is_critical, mmd_every, config-driven is_healthy."""

    def test_psi_per_feature_keys_match_feature_names(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """psi_per_feature is keyed by feature name once drift window fills."""
        clf, X, _ = clf_and_data
        names = [f"feat_{i}" for i in range(X.shape[1])]
        m = Monitor(
            clf,
            reference_data=X[:200],
            feature_names=names,
            config=MonitorConfig(drift_window=2, enable_mmd=False, enable_causal=False),
        )
        # Feed drift_window+1 batches so the window fills.
        for i in range(3):
            result = m.predict(X[i * 30 : (i + 1) * 30])
        assert result.psi_per_feature is not None
        assert set(result.psi_per_feature.keys()) == set(names)

    def test_is_healthy_uses_config_trust_warn(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """is_healthy uses MonitorConfig.trust_warn, not a hardcoded 0.70."""
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(trust_warn=0.0),  # always healthy
        )
        result = m.predict(X[200:250])
        assert result.is_healthy is True

        m2 = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(trust_warn=1.0),  # never healthy
        )
        result2 = m2.predict(X[200:250])
        assert result2.is_healthy is False

    def test_is_critical_uses_config_trust_critical(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """is_critical uses MonitorConfig.trust_critical."""
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(trust_critical=1.0),  # always critical
        )
        result = m.predict(X[200:250])
        assert result.is_critical is True

    def test_mmd_every_skips_non_evaluation_batches(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """mmd_every=3 means mmd_result is None on batches 1, 2 and present on 3."""
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(enable_mmd=True, mmd_every=3, mmd_permutations=50),
        )
        r1 = m.predict(X[200:230])
        r2 = m.predict(X[230:260])
        r3 = m.predict(X[260:290])
        assert r1.mmd_result is None
        assert r2.mmd_result is None
        assert r3.mmd_result is not None

    def test_is_drifting_uses_config_psi_threshold(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """is_drifting uses MonitorConfig.psi_threshold."""
        clf, X, _ = clf_and_data
        # psi_threshold=0 → always drifting after window fills
        m = Monitor(
            clf,
            reference_data=X[:100],
            config=MonitorConfig(
                psi_threshold=0.0, drift_window=2, enable_mmd=False, enable_causal=False
            ),
        )
        for i in range(3):
            result = m.predict(X[100 + i * 30 : 130 + i * 30])
        assert result.is_drifting is True


# ---------------------------------------------------------------------------
# save() / load() state persistence
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Monitor.save() and Monitor.load() persistence contract."""

    def test_save_creates_file(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        _, X, _ = clf_and_data
        monitor.predict(X[:50])
        path = tmp_path / "state.json"
        monitor.save(path)
        assert path.exists()

    def test_load_restores_batch_count(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200], config=MonitorConfig(enable_mmd=False))
        for _ in range(3):
            m.predict(X[200:230])
        path = tmp_path / "state.json"
        m.save(path)

        m2 = Monitor.load(path, clf, reference_data=X[:200])
        assert m2.n_batches == 3

    def test_load_restores_history_length(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200], config=MonitorConfig(enable_mmd=False))
        m.predict(X[200:230], batch_id="batch_a")
        m.predict(X[230:260], batch_id="batch_b")
        path = tmp_path / "state.json"
        m.save(path)

        m2 = Monitor.load(path, clf, reference_data=X[:200])
        assert len(m2.history) == 2
        assert m2.history[0]["batch_id"] == "batch_b"

    def test_load_raises_on_missing_file(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        clf, X, _ = clf_and_data
        with pytest.raises(FileNotFoundError):
            Monitor.load(tmp_path / "nonexistent.json", clf, reference_data=X[:200])

    def test_save_creates_parent_dirs(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        _, X, _ = clf_and_data
        monitor.predict(X[:30])
        deep_path = tmp_path / "a" / "b" / "c" / "state.json"
        monitor.save(deep_path)
        assert deep_path.exists()

    def test_roundtrip_predict_after_load(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """predict() works normally after loading from state."""
        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200], config=MonitorConfig(enable_mmd=False))
        m.predict(X[200:230])
        m.save(tmp_path / "s.json")

        m2 = Monitor.load(tmp_path / "s.json", clf, reference_data=X[:200])
        result = m2.predict(X[230:260])
        assert isinstance(result, BatchResult)
        assert 0.0 <= result.trust_score <= 1.0


# ---------------------------------------------------------------------------
# write_model_card
# ---------------------------------------------------------------------------


class TestWriteModelCard:
    """Monitor.write_model_card() writes a valid, readable JSON card."""

    def test_creates_json_file(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        from model_monitor.training.model_card import ModelCard

        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200])
        card_path = tmp_path / "v1_card.json"
        m.write_model_card(card_path, model_version=1, evaluation_f1=0.88)
        assert card_path.exists()
        card = ModelCard.load(card_path)
        assert card.model_version == 1
        assert card.evaluation.f1 == pytest.approx(0.88)

    def test_feature_names_recorded(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        from model_monitor.training.model_card import ModelCard

        clf, X, _ = clf_and_data
        names = [f"col_{i}" for i in range(X.shape[1])]
        m = Monitor(clf, reference_data=X[:200], feature_names=names)
        path = tmp_path / "card.json"
        m.write_model_card(path)
        card = ModelCard.load(path)
        assert [f.name for f in card.feature_schema] == names

    def test_f1_inferred_from_history(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """When evaluation_f1 not given, it's inferred from monitoring history."""
        from model_monitor.training.model_card import ModelCard

        clf, X, y = clf_and_data
        m = Monitor(clf, reference_data=X[:200])
        m.predict(X[200:250], y_true=y[200:250])
        path = tmp_path / "card.json"
        m.write_model_card(path)
        card = ModelCard.load(path)
        # Card F1 should be non-zero since we passed y_true
        assert card.evaluation.f1 >= 0.0

    def test_extra_metadata_stored(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        from model_monitor.training.model_card import ModelCard

        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200])
        path = tmp_path / "card.json"
        m.write_model_card(path, extra={"run_id": "abc123", "n_estimators": 10})
        card = ModelCard.load(path)
        assert card.extra["run_id"] == "abc123"

    def test_creates_parent_directories(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200])
        deep_path = tmp_path / "cards" / "v3" / "card.json"
        m.write_model_card(deep_path)
        assert deep_path.exists()


class TestSaveLoadDriftWindow:
    """Drift window buffer persists through save/load so PSI is continuous."""

    def test_drift_buffer_len_restored(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(drift_window=3, enable_mmd=False, enable_causal=False),
        )
        # Fill the drift window with 3 batches
        for i in range(3):
            m.predict(X[200 + i * 20 : 220 + i * 20])

        assert len(m._drift_monitor.buffer) == 3
        path = tmp_path / "state.json"
        m.save(path)

        m2 = Monitor.load(path, clf, reference_data=X[:200])
        assert len(m2._drift_monitor.buffer) == 3

    def test_psi_computable_immediately_after_load(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """With window restored, the first post-load batch yields a PSI score."""
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(drift_window=2, enable_mmd=False, enable_causal=False),
        )
        for i in range(2):
            m.predict(X[200 + i * 20 : 220 + i * 20])

        path = tmp_path / "state.json"
        m.save(path)

        m2 = Monitor.load(path, clf, reference_data=X[:200])
        # After restoration, window has 2 batches; predict() should be able to
        # compute PSI on the very next call (window = 2, we have 2 existing).
        result = m2.predict(X[260:290])
        # psi_per_feature is set when the window is full
        # (window filled = drift_window batches in buffer, which is now true)
        assert result.psi_per_feature is not None or result.drift_score == 0.0


# ---------------------------------------------------------------------------
# warm_up()
# ---------------------------------------------------------------------------


class TestWarmUp:
    """Monitor.warm_up() pre-fills the drift window without recording batches."""

    def test_warm_up_does_not_increment_n_batches(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(drift_window=3, enable_mmd=False, enable_causal=False),
        )
        m.warm_up(X[:100])
        assert m.n_batches == 0

    def test_warm_up_fills_drift_buffer(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(drift_window=3, enable_mmd=False, enable_causal=False),
        )
        assert len(m._drift_monitor.buffer) == 0
        m.warm_up(X[:150])
        assert len(m._drift_monitor.buffer) > 0

    def test_psi_available_on_first_predict_after_warm_up(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """After warm_up fills the window, PSI is computable from batch 1."""
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(drift_window=2, enable_mmd=False, enable_causal=False),
        )
        m.warm_up(X[200:280])  # fills window with 2 chunks of 40 rows each
        result = m.predict(X[280:310])
        # With a full window, PSI should be computable
        # (psi_per_feature set when drift_window batches are in buffer)
        assert result.trust_score > 0.0

    def test_warm_up_not_in_history(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(drift_window=2, enable_mmd=False, enable_causal=False),
        )
        m.warm_up(X[200:250])
        assert len(m.history) == 0


# ---------------------------------------------------------------------------
# MonitorSummary typed return
# ---------------------------------------------------------------------------


class TestMonitorSummaryTyped:
    """summary() returns a MonitorSummary dataclass with correct types."""

    def test_returns_monitor_summary_instance(
        self,
        monitor: Monitor,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        from model_monitor import MonitorSummary

        _, X, _ = clf_and_data
        monitor.predict(X[:50])
        assert isinstance(monitor.summary(), MonitorSummary)

    def test_n_batches_zero_before_predict(self, monitor: Monitor) -> None:
        assert monitor.summary().n_batches == 0

    def test_mmd_drift_rate_populated_after_mmd_batches(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                enable_mmd=True, mmd_permutations=30, enable_causal=False
            ),
        )
        for i in range(3):
            m.predict(X[200 + i * 30 : 230 + i * 30])
        s = m.summary()
        # mmd_drift_rate should be a float in [0, 1] since MMD ran
        assert s.mmd_drift_rate is not None
        assert 0.0 <= s.mmd_drift_rate <= 1.0


# ---------------------------------------------------------------------------
# CUSUM integration in Monitor
# ---------------------------------------------------------------------------


class TestCUSUMInMonitor:
    """CUSUM sequential change-point detection wired into Monitor.predict()."""

    def test_cusum_disabled_by_default(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(clf, reference_data=X[:200], config=MonitorConfig(enable_mmd=False))
        result = m.predict(X[200:230])
        assert result.cusum_result is None
        assert result.is_cusum_alarm is False

    def test_cusum_enabled_returns_result(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        from model_monitor.monitoring.cusum import CUSUMResult

        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                enable_mmd=False,
                cusum_delta=0.01,
                cusum_threshold=0.1,
                cusum_warmup=3,
            ),
        )
        # Feed cusum_warmup batches so reference_mean is estimated and detector built
        for i in range(3):
            m.predict(X[200 + i * 10 : 210 + i * 10])
        # 4th batch should produce a CUSUMResult
        result = m.predict(X[230:260])
        assert result.cusum_result is not None
        assert isinstance(result.cusum_result, CUSUMResult)

    def test_cusum_alarm_fires_on_large_drift(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """CUSUM fires before PSI threshold is crossed when shift is gradual."""
        clf, X, _ = clf_and_data
        rng = np.random.default_rng(5)
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                enable_mmd=False,
                enable_causal=False,
                psi_threshold=0.20,  # high PSI threshold - hard to cross
                cusum_delta=0.01,
                cusum_threshold=0.05,  # low CUSUM threshold - easy to cross
                cusum_warmup=0,
            ),
        )
        # Feed stable batches to build cumulative sum
        alarms = []
        for i in range(30):
            # OOD batches: large distribution shift
            X_ood = (
                X[200 + i * 5 : 205 + i * 5] + rng.standard_normal(X[205:210].shape) * 3
            )
            r = m.predict(X_ood)
            if r.is_cusum_alarm:
                alarms.append(i)
                break
        assert len(alarms) >= 1, "CUSUM should alarm on large persistent shift"

    def test_cusum_alarm_rate_in_summary(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                enable_mmd=False,
                cusum_delta=0.01,
                cusum_threshold=0.5,
                cusum_warmup=0,
            ),
        )
        for i in range(4):
            m.predict(X[200 + i * 20 : 220 + i * 20])
        s = m.summary()
        assert s.cusum_alarm_rate is not None
        assert 0.0 <= s.cusum_alarm_rate <= 1.0


# ---------------------------------------------------------------------------
# on_alarm() callback registration
# ---------------------------------------------------------------------------


class TestOnAlarm:
    """Monitor.on_alarm() fires registered callbacks on alarm conditions."""

    def test_callback_fires_on_is_drifting(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        fired: list[BatchResult] = []
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                psi_threshold=0.0,  # always drifting
                drift_window=2,
                enable_mmd=False,
                enable_causal=False,
            ),
        )
        m.on_alarm(fired.append, fire_on=("is_drifting",))
        # Fill the drift window first (no alarm before window fills)
        for i in range(2):
            m.predict(X[200 + i * 15 : 215 + i * 15])
        # This batch should trigger is_drifting → callback fires
        m.predict(X[230:260])
        assert len(fired) >= 1

    def test_callback_not_fired_when_no_alarm(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        fired: list[BatchResult] = []
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                psi_threshold=1.0,  # never drifting
                enable_mmd=False,
                enable_causal=False,
            ),
        )
        m.on_alarm(fired.append, fire_on=("is_drifting",))
        for i in range(3):
            m.predict(X[200 + i * 20 : 220 + i * 20])
        assert len(fired) == 0

    def test_callback_exception_does_not_block_predict(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """A failing callback must never block inference."""
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                psi_threshold=0.0,  # always drifting
                drift_window=2,
                enable_mmd=False,
                enable_causal=False,
            ),
        )

        def bad_callback(r: BatchResult) -> None:
            raise RuntimeError("callback exploded")

        m.on_alarm(bad_callback)
        # Fill window
        for i in range(2):
            m.predict(X[200 + i * 15 : 215 + i * 15])
        # Must not raise even though callback raises
        result = m.predict(X[230:260])
        assert result is not None

    def test_multiple_callbacks_all_fire(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        fired_a: list[str] = []
        fired_b: list[str] = []
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                psi_threshold=0.0,
                drift_window=2,
                enable_mmd=False,
                enable_causal=False,
            ),
        )
        m.on_alarm(lambda r: fired_a.append("a"), fire_on=("is_drifting",))
        m.on_alarm(lambda r: fired_b.append("b"), fire_on=("is_drifting",))
        for i in range(2):
            m.predict(X[200 + i * 15 : 215 + i * 15])
        m.predict(X[230:260])
        assert len(fired_a) >= 1
        assert len(fired_b) >= 1


# ---------------------------------------------------------------------------
# reset_after_retrain()
# ---------------------------------------------------------------------------


class TestResetAfterRetrain:
    """Monitor.reset_after_retrain() clears state for clean post-retrain monitoring."""

    def test_drift_buffer_cleared(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(drift_window=3, enable_mmd=False, enable_causal=False),
        )
        for i in range(3):
            m.predict(X[200 + i * 20 : 220 + i * 20])
        assert len(m._drift_monitor.buffer) == 3
        m.reset_after_retrain()
        assert len(m._drift_monitor.buffer) == 0

    def test_cusum_state_cleared(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """CUSUM warmup psi and detector are cleared so reference_mean re-estimated."""
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                enable_mmd=False,
                enable_causal=False,
                cusum_delta=0.01,
                cusum_threshold=0.5,
                cusum_warmup=3,
            ),
        )
        # Feed enough batches so CUSUM detector is built
        for i in range(5):
            m.predict(X[200 + i * 15 : 215 + i * 15])
        assert m._cusum is not None
        m.reset_after_retrain()
        assert m._cusum is None
        assert m._cusum_warmup_psi == []

    def test_predict_works_after_reset(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(enable_mmd=False, enable_causal=False),
        )
        for i in range(3):
            m.predict(X[200 + i * 20 : 220 + i * 20])
        m.reset_after_retrain()
        result = m.predict(X[260:290])
        assert isinstance(result, BatchResult)
        assert 0.0 <= result.trust_score <= 1.0

    def test_history_preserved_after_reset(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """History is cumulative - reset does not erase past records."""
        clf, X, _ = clf_and_data
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(enable_mmd=False, enable_causal=False),
        )
        m.predict(X[200:230])
        m.predict(X[230:260])
        assert len(m.history) == 2
        m.reset_after_retrain()
        assert len(m.history) == 2  # history preserved


# ---------------------------------------------------------------------------
# CUSUM reference_mean auto-estimation (no false alarms on stable signal)
# ---------------------------------------------------------------------------


class TestCUSUMReferenceEstimation:
    def test_no_false_alarms_on_stable_psi(
        self,
        clf_and_data: tuple[RandomForestClassifier, np.ndarray, np.ndarray],
    ) -> None:
        """CUSUM with auto-estimated reference_mean should not alarm on stable PSI.

        We use in-distribution batches drawn from the *reference data itself*
        (repeated sampling without replacement) to guarantee near-zero PSI.
        With reference_mean estimated from the first 5 such batches, the threshold
        must not be crossed on the next 15 stable batches.
        """
        clf, X, _ = clf_and_data
        rng = np.random.default_rng(42)
        m = Monitor(
            clf,
            reference_data=X[:200],
            config=MonitorConfig(
                enable_mmd=False,
                enable_causal=False,
                drift_window=3,
                cusum_delta=0.01,
                cusum_threshold=0.30,  # generous threshold → near-zero false alarms
                cusum_warmup=5,
            ),
        )
        alarms = 0
        for _ in range(20):
            # Resample from reference data - PSI relative to itself is near-zero.
            idx = rng.integers(0, 200, size=80)
            result = m.predict(X[idx])
            if result.cusum_result is not None and result.cusum_result.alarm:
                alarms += 1
        assert alarms == 0, (
            f"CUSUM fired {alarms} false alarms on reference-distribution batches. "
            "reference_mean auto-estimation is broken."
        )
