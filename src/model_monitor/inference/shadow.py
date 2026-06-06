"""Shadow-mode predictor for safe candidate model evaluation.

Shadow mode is the standard production technique for evaluating a candidate
model against live traffic without serving its predictions.  The primary
model's output is always returned; the candidate runs in parallel and its
results are compared but discarded.

Why this matters
----------------
Lab metrics (holdout F1, validation accuracy) measure a candidate on a static
slice of historical data.  Shadow mode measures it on the *current live
distribution*, catching regressions that only appear after deployment - for
example, a model that overfits to a recent retraining window and degrades on
the tails of the production distribution.

Shadow metrics accumulated here are a direct input to the promotion decision:
if the candidate's shadow agreement rate, F1, and trust score are all better
than the primary's over the same traffic, promotion confidence is high.

Design
------
- ``ShadowPredictor`` wraps a primary ``Predictor`` and an optional candidate
  ``Predictor``.
- ``predict_batch`` always runs the primary and returns its output.
- When a candidate is present, it is run in the same call and its results
  are recorded to ``shadow_stats`` but never returned.
- The candidate never writes to any store, fires any alert, or changes model
  state.  Its side-effect surface is zero.
- ``shadow_stats`` is reset by ``reset_shadow_stats()`` and consumed by
  ``consume_shadow_stats()`` - the same consume-and-clear pattern as
  ``RawDataBuffer``.

Thread safety
-------------
Designed for single-threaded asyncio use.  The candidate ``Predictor`` runs
synchronously inside ``predict_batch``; there is no concurrency between the
two predictors.  If the candidate raises, the exception is logged and shadow
evaluation is skipped for that batch - the primary result is always returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model_monitor.core.decisions import Decision
from model_monitor.inference.predict import Predictor

log = logging.getLogger(__name__)


@dataclass
class ShadowBatchResult:
    """Metrics from a single shadow-mode batch comparison."""

    batch_id: str
    primary_f1: float
    candidate_f1: float
    agreement_rate: float  # fraction of predictions that match
    primary_trust: float
    candidate_trust: float
    n_samples: int


@dataclass
class ShadowStats:
    """Aggregated shadow statistics over multiple batches."""

    n_batches: int = 0
    total_samples: int = 0
    mean_agreement_rate: float = 0.0
    mean_primary_f1: float = 0.0
    mean_candidate_f1: float = 0.0
    mean_primary_trust: float = 0.0
    mean_candidate_trust: float = 0.0
    # Running totals used for incremental mean computation.
    _sum_agreement: float = field(default=0.0, repr=False)
    _sum_primary_f1: float = field(default=0.0, repr=False)
    _sum_candidate_f1: float = field(default=0.0, repr=False)
    _sum_primary_trust: float = field(default=0.0, repr=False)
    _sum_candidate_trust: float = field(default=0.0, repr=False)

    def update(self, result: ShadowBatchResult) -> None:
        """Incorporate one batch result into the running aggregates."""
        self.n_batches += 1
        self.total_samples += result.n_samples
        self._sum_agreement += result.agreement_rate
        self._sum_primary_f1 += result.primary_f1
        self._sum_candidate_f1 += result.candidate_f1
        self._sum_primary_trust += result.primary_trust
        self._sum_candidate_trust += result.candidate_trust

        n = self.n_batches
        self.mean_agreement_rate = self._sum_agreement / n
        self.mean_primary_f1 = self._sum_primary_f1 / n
        self.mean_candidate_f1 = self._sum_candidate_f1 / n
        self.mean_primary_trust = self._sum_primary_trust / n
        self.mean_candidate_trust = self._sum_candidate_trust / n

    @property
    def candidate_beats_primary(self) -> bool:
        """True when the candidate outperforms the primary on both F1 and trust."""
        return (
            self.mean_candidate_f1 > self.mean_primary_f1
            and self.mean_candidate_trust > self.mean_primary_trust
        )


class ShadowPredictor:
    """Wraps a primary Predictor and an optional candidate in shadow mode.

    The primary's predictions are always returned.  The candidate runs
    silently alongside, accumulating comparison metrics without influencing
    any production output.

    Args:
        primary:    The currently active production predictor.
        candidate:  An optional candidate predictor to evaluate.  Can be
                    swapped out at any time via ``set_candidate``.

    Example::

        shadow = ShadowPredictor(primary=active_predictor)
        shadow.set_candidate(candidate_predictor)

        preds, confs, decision = shadow.predict_batch(X_df, y_true=y)
        # preds/confs/decision are always from primary

        stats = shadow.consume_shadow_stats()
        if stats.candidate_beats_primary:
            model_store.promote_candidate(...)
    """

    def __init__(
        self,
        primary: Predictor,
        candidate: Predictor | None = None,
    ) -> None:
        self._primary = primary
        self._candidate = candidate
        self._stats = ShadowStats()

    # ------------------------------------------------------------------
    # Candidate management
    # ------------------------------------------------------------------

    def set_candidate(self, candidate: Predictor | None) -> None:
        """Set or clear the shadow candidate.  Resets accumulated stats."""
        self._candidate = candidate
        self._stats = ShadowStats()

    def has_candidate(self) -> bool:
        """Return True when a candidate is loaded."""
        return self._candidate is not None

    # ------------------------------------------------------------------
    # Prediction (primary path always returned)
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        X: pd.DataFrame,
        y_true: Any | None = None,
        batch_id: str = "",
    ) -> tuple[np.ndarray, np.ndarray, Decision]:
        """Run primary (and optionally candidate) inference on a batch.

        The return value is always the primary predictor's output.  The
        candidate runs in the same call when present; its results are
        recorded in ``shadow_stats`` but never returned or persisted.

        Args:
            X:        Feature matrix for the batch.
            y_true:   Ground-truth labels.  Required for F1 and trust
                      computation on both primary and candidate.
            batch_id: Caller-assigned identifier, forwarded to both predictors.

        Returns:
            (predictions, confidences, decision) from the primary predictor.
        """
        preds, confs, decision = self._primary.predict_batch(
            X, y_true=y_true, batch_id=batch_id
        )

        if self._candidate is not None and y_true is not None:
            self._run_shadow(
                X=X,
                y_true=y_true,
                batch_id=batch_id,
                primary_preds=preds,
            )

        return preds, confs, decision

    def _run_shadow(
        self,
        X: pd.DataFrame,
        y_true: Any,
        batch_id: str,
        primary_preds: np.ndarray,
    ) -> None:
        """Run the candidate and record comparison metrics.  Never raises."""
        # Caller guarantees self._candidate is not None; bind to a local so
        # mypy can narrow the type without suppression comments.
        candidate = self._candidate
        if candidate is None:
            # This should never happen - _run_shadow is only called from
            # predict_batch when self._candidate is not None.
            raise RuntimeError("_run_shadow called with no candidate set")

        try:
            cand_preds, _cand_confs, _cand_decision = candidate.predict_batch(
                X, y_true=y_true, batch_id=f"{batch_id}_shadow"
            )

            agreement = float(np.mean(cand_preds == primary_preds))

            primary_f1 = (
                self._primary.last_metric_record["f1"]
                if self._primary.last_metric_record
                else 0.0
            )
            candidate_f1 = (
                candidate.last_metric_record["f1"]
                if candidate.last_metric_record
                else 0.0
            )
            primary_trust = self._primary.last_trust_score
            candidate_trust = candidate.last_trust_score

            result = ShadowBatchResult(
                batch_id=batch_id,
                primary_f1=primary_f1,
                candidate_f1=candidate_f1,
                agreement_rate=agreement,
                primary_trust=primary_trust,
                candidate_trust=candidate_trust,
                n_samples=len(primary_preds),
            )
            self._stats.update(result)

            log.debug(
                "shadow_batch_complete",
                extra={
                    "batch_id": batch_id,
                    "agreement": round(agreement, 4),
                    "primary_f1": round(primary_f1, 4),
                    "candidate_f1": round(candidate_f1, 4),
                },
            )
        except Exception as exc:
            log.warning(
                "shadow_batch_failed",
                exc_info=exc,
                extra={"batch_id": batch_id},
            )

    # ------------------------------------------------------------------
    # Stats access
    # ------------------------------------------------------------------

    @property
    def shadow_stats(self) -> ShadowStats:
        """Read-only view of accumulated shadow statistics."""
        return self._stats

    def consume_shadow_stats(self) -> ShadowStats:
        """Return accumulated stats and reset the counters.

        Use after enough batches have been observed to make a promotion
        decision based on live-traffic performance.
        """
        stats = self._stats
        self._stats = ShadowStats()
        return stats

    def reset_shadow_stats(self) -> None:
        """Clear accumulated shadow statistics without returning them."""
        self._stats = ShadowStats()

    # ------------------------------------------------------------------
    # Passthrough to primary
    # ------------------------------------------------------------------

    @property
    def last_metric_record(self) -> Any:
        """Last metric record from the primary predictor."""
        return self._primary.last_metric_record

    @property
    def last_drift_score(self) -> float:
        """Last drift score from the primary predictor."""
        return self._primary.last_drift_score

    @property
    def last_trust_score(self) -> float:
        """Last trust score from the primary predictor."""
        return self._primary.last_trust_score

    def reload(self) -> None:
        """Reload both primary and candidate models from disk."""
        self._primary.reload()
        if self._candidate is not None:
            self._candidate.reload()
