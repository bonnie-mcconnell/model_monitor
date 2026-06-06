"""Crash-safe model lifecycle executor: promote, rollback, retrain."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.core.model_actions import ModelAction
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline, RetrainResult
from model_monitor.training.train import compute_reference_stats, make_dataset

if TYPE_CHECKING:
    from model_monitor.monitoring.raw_data_buffer import RawDataBuffer

log = logging.getLogger(__name__)

_REF_STATS_PATH = Path("data/reference/reference_stats.json")


class DefaultModelActionExecutor:
    """Executes model lifecycle actions in a crash-safe manner.

    Guarantees:
    - No partial promotions (atomic rename via ModelStore)
    - Failures are always recorded in DecisionStore
    - Side effects are isolated per action type

    Retrain data flow
    -----------------
    The executor accepts an optional ``raw_data_buffer`` - a rolling buffer of
    recent labeled ``(X, y)`` pairs accumulated from live inference batches.
    When a retrain fires and the buffer holds enough rows, the pipeline trains
    on that observed data - data that reflects the current, possibly drifted,
    input distribution.

    When the buffer is absent or not yet full, the executor falls back to
    ``make_dataset()`` and logs a WARNING.  This fallback is always transparent:
    every retrain has an observable training source in the logs.

    After a successful promotion, ``reference_stats.json`` is rewritten from
    the retrain dataset so future PSI scores compare against the new model's
    training distribution.  This is best-effort: a failure here is logged but
    never blocks the promotion.
    """

    def __init__(
        self,
        *,
        model_store: ModelStore,
        retrain_pipeline: RetrainPipeline,
        decision_store: DecisionStore,
        raw_data_buffer: RawDataBuffer | None = None,
        dry_run: bool = False,
    ) -> None:
        self.store = model_store
        self.retrain_pipeline = retrain_pipeline
        self.decision_store = decision_store
        self._raw_data_buffer = raw_data_buffer
        self.dry_run = dry_run

    def execute(
        self,
        *,
        action: ModelAction,
        context: Mapping[str, Any],
    ) -> str | None:
        recent = self.decision_store.tail(limit=20)

        if any(
            r.action == action.value
            and r.model_version == self.store.get_active_version()
            for r in recent
        ):
            return None

        try:
            return self._execute_internal(action=action, context=context)

        except Exception as exc:
            failed_decision = Decision(
                action=DecisionType.SYSTEM_ERROR,
                reason=f"executor failure: {type(exc).__name__}",
                metadata={},
            )
            self.decision_store.record(
                decision=failed_decision,
                model_version=self.store.get_active_version(),
            )
            raise

    def _execute_internal(
        self,
        *,
        action: ModelAction,
        context: Mapping[str, Any],
    ) -> str | None:

        if action in {ModelAction.NONE, ModelAction.REJECT}:
            return None

        if action == ModelAction.ROLLBACK:
            version = context.get("version")
            if not version:
                raise ValueError("Rollback requires target version")
            return None if self.dry_run else self.store.rollback(version)

        if action == ModelAction.PROMOTE:
            metrics = context.get("metrics", {})
            # Normalise: if caller passed f1 but not baseline_f1, backfill it.
            if "f1" in metrics and "baseline_f1" not in metrics:
                metrics = {**metrics, "baseline_f1": metrics["f1"]}
            return None if self.dry_run else self.store.promote_candidate(metrics)

        if action == ModelAction.RETRAIN:
            if self.dry_run:
                return None
            return self._handle_retrain(context)

        raise ValueError(f"Unsupported action: {action}")

    def _resolve_retrain_dataset(self, context: Mapping[str, Any]) -> pd.DataFrame:
        """Return the best available labeled dataset for retraining.

        Priority order:
        1. ``context["retrain_df"]`` - explicit DataFrame from the caller.
           Used by tests and the simulation loop, which manage their own buffer.
        2. ``raw_data_buffer`` when wired and ready - normal API-driven path.
        3. Synthetic fallback via ``make_dataset()`` - logged at WARNING.
           Only occurs when the buffer has not yet accumulated enough rows
           (typically only at cold start or in tests without a wired buffer).
        """
        explicit_df: pd.DataFrame | None = context.get("retrain_df")
        if explicit_df is not None and not explicit_df.empty:
            return explicit_df

        cfg_min: int = context.get("min_samples", 100)

        if self._raw_data_buffer is not None and self._raw_data_buffer.ready(cfg_min):
            df = self._raw_data_buffer.consume()
            log.info(
                "retrain_using_observed_data",
                extra={"n_samples": len(df)},
            )
            return df

        log.warning(
            "retrain_buffer_insufficient_using_synthetic_fallback",
            extra={
                "raw_buffer_size": (
                    self._raw_data_buffer.size() if self._raw_data_buffer else 0
                ),
                "min_samples": cfg_min,
            },
        )
        retrain_df, _ = make_dataset(
            n_samples=max(500, cfg_min * 2),
            random_state=int(time.time()) % 10_000,
        )
        return retrain_df

    def _handle_retrain(self, context: Mapping[str, Any]) -> str | None:
        min_gain: float = context.get("min_f1_improvement", 0.02)
        retrain_df = self._resolve_retrain_dataset(context)

        try:
            current_model = self.store.load_current()
        except FileNotFoundError:
            current_model = None

        result: RetrainResult = self.retrain_pipeline.run(
            retrain_df=retrain_df,
            current_model=current_model,
            min_f1_improvement=min_gain,
        )

        if result.candidate_model is None:
            return None

        self.store.save_candidate(result.candidate_model)

        if not result.promotion.promoted:
            log.info(
                "retrain_candidate_rejected",
                extra={
                    "candidate_f1": result.promotion.candidate_f1,
                    "current_f1": result.promotion.current_f1,
                    "reason": result.promotion.reason,
                },
            )
            return None

        version = self.store.promote_candidate(
            metrics={
                "baseline_f1": result.promotion.candidate_f1,
                "candidate_f1": result.promotion.candidate_f1,
                "current_f1": result.promotion.current_f1,
                "improvement": result.promotion.improvement,
                "n_samples": result.n_samples,
            }
        )

        # Write a model card capturing training provenance.  Best-effort:
        # a card write failure must never block a successful promotion.
        self._write_model_card(
            version=version,
            retrain_df=retrain_df,
            result=result,
        )

        # Update reference statistics so future PSI compares against the new
        # model's training distribution, not the original one from ``make train``.
        self._refresh_reference_stats(retrain_df)

        log.info(
            "retrain_promoted",
            extra={
                "version": version,
                "candidate_f1": result.promotion.candidate_f1,
                "improvement": result.promotion.improvement,
                "n_samples": result.n_samples,
            },
        )
        return version

    def _write_model_card(
        self,
        version: str | None,
        retrain_df: pd.DataFrame,
        result: RetrainResult,
    ) -> None:
        """Write a :class:`ModelCard` JSON file alongside the promoted model.

        Best-effort: any failure is logged and swallowed so a card write
        never blocks a successful promotion.

        Args:
            version:    Version string returned by :meth:`promote_candidate`.
            retrain_df: Full retraining dataframe (before train/val split).
            result:     :class:`RetrainResult` with promotion metrics.
        """
        try:
            import os

            from model_monitor.training.model_card import (
                ModelEvaluation,
                build_model_card,
            )

            ver_int = int(version) if version is not None and str(version).isdigit() else 0
            feature_cols = [c for c in retrain_df.columns if c != "target"]
            X_train = retrain_df[feature_cols]

            evaluation = ModelEvaluation(
                accuracy=float(result.promotion.candidate_f1),  # best proxy available
                f1=float(result.promotion.candidate_f1),
                f1_improvement=float(result.promotion.improvement),
                n_eval_samples=max(1, int(result.n_samples * 0.2)),
                bootstrap_ci_lower=(
                    result.promotion.bootstrap_ci.lower
                    if result.promotion.bootstrap_ci is not None
                    else None
                ),
                bootstrap_ci_upper=(
                    result.promotion.bootstrap_ci.upper
                    if result.promotion.bootstrap_ci is not None
                    else None
                ),
            )

            card = build_model_card(
                model_version=ver_int,
                X_train=X_train,
                feature_names=feature_cols,
                evaluation=evaluation,
                promotion_reason=result.promotion.reason,
            )

            base = os.environ.get("MODEL_STORE_DIR", "data/models")
            card_path = Path(base) / f"v{ver_int}_card.json"
            card.save(card_path)
            log.info(
                "model_card_written",
                extra={"version": ver_int, "path": str(card_path)},
            )
        except Exception as exc:
            log.warning("model_card_write_failed", extra={"error": str(exc)})

    def _refresh_reference_stats(self, train_df: pd.DataFrame) -> None:
        """Rewrite reference_stats.json from the retrain dataset after promotion.

        Best-effort: failures are logged but never re-raised so a stale
        reference file does not block a successful promotion.
        """
        try:
            stats = compute_reference_stats(train_df)
            _REF_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _REF_STATS_PATH.write_text(json.dumps(stats, indent=2))
            log.info("reference_stats_updated", extra={"path": str(_REF_STATS_PATH)})
        except Exception as exc:
            log.warning(
                "reference_stats_update_failed",
                exc_info=exc,
                extra={"path": str(_REF_STATS_PATH)},
            )
