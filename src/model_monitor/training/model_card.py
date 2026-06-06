"""Model card: training provenance attached to every promoted model version.

Every time a model is promoted, a :class:`ModelCard` is written alongside the
model weights.  It captures the information needed to answer the questions an
ML engineer actually asks in production:

  - What dataset trained this model?
  - When was it trained and by what pipeline version?
  - What was its evaluation performance at promotion time?
  - What features did it expect?
  - Why was it promoted (what triggered retraining)?

This is not a compliance artefact - it is an operational debugging tool.  When
a model starts behaving oddly two weeks after deployment, the card tells you
whether the training data composition, baseline metrics, or feature schema
changed between versions.

The card is stored as JSON alongside the model weights in ``ModelStore``, and
is exposed via the ``/dashboard/models/{version}/card`` API endpoint.

Design notes
------------
- ``training_data_hash`` is the SHA-256 of the training feature matrix
  serialised as float64 bytes - the same hash used for retrain deduplication
  in ``RetrainEvidenceBuffer``.  This makes it possible to confirm whether
  two model versions were trained on identical data.
- ``feature_schema`` records name, dtype, and optional value bounds for each
  feature.  A mismatch between the card and the current production schema is
  an early warning of a silent feature pipeline change.
- ``evaluation`` records the held-out metrics at promotion time, not training
  metrics.  Recording training metrics here would be misleading.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Schema entry for one feature at training time.

    Attributes:
        name:       Feature name as it appears in the model's input.
        dtype:      NumPy dtype string (e.g. ``"float64"``, ``"int32"``).
        min_value:  Observed minimum in the training set (optional).
        max_value:  Observed maximum in the training set (optional).
        null_rate:  Fraction of null / NaN values in training data.
    """

    name: str
    dtype: str
    min_value: float | None = None
    max_value: float | None = None
    null_rate: float = 0.0


# ---------------------------------------------------------------------------
# ModelEvaluation
# ---------------------------------------------------------------------------


@dataclass
class ModelEvaluation:
    """Held-out evaluation metrics at the time of promotion.

    All metrics are computed on the held-out validation split, never on
    training data.

    Attributes:
        accuracy:           Accuracy score in [0, 1].
        f1:                 Macro-averaged F1 score.
        f1_improvement:     Improvement over the previous model's F1.
                            Positive means the new model is better.
        n_eval_samples:     Size of the held-out evaluation set.
        bootstrap_ci_lower: Lower bound of the bootstrap confidence interval
                            on the F1 improvement.  Promotion is blocked when
                            this is ≤ 0.
        bootstrap_ci_upper: Upper bound of the bootstrap confidence interval.
    """

    accuracy: float
    f1: float
    f1_improvement: float
    n_eval_samples: int
    bootstrap_ci_lower: float | None = None
    bootstrap_ci_upper: float | None = None


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------


@dataclass
class ModelCard:
    """Training provenance record written at every model promotion.

    Attributes:
        model_version:       Integer version number (matches ``ModelStore``).
        created_at:          Unix timestamp of promotion.
        created_at_iso:      ISO 8601 string for human readability.
        pipeline_version:    Optional semver string identifying the training
                             pipeline code version (set from env var
                             ``MODEL_MONITOR_VERSION`` if present).
        training_data_hash:  SHA-256 of the training feature matrix (float64
                             bytes).  Identical to the hash used for retrain
                             deduplication.
        n_training_samples:  Number of samples in the training set.
        feature_schema:      One :class:`FeatureSpec` per input feature.
        evaluation:          Held-out evaluation metrics at promotion.
        promotion_reason:    Human-readable reason string from the decision
                             engine (e.g. ``"f1_improvement=0.023"``).
        notes:               Optional free-text notes.  Use this for anything
                             that doesn't fit the structured fields.
        extra:               Arbitrary key-value metadata (hyperparameters,
                             experiment tracking IDs, etc.).
    """

    model_version: int
    created_at: float
    created_at_iso: str
    training_data_hash: str
    n_training_samples: int
    feature_schema: list[FeatureSpec]
    evaluation: ModelEvaluation
    promotion_reason: str
    pipeline_version: str = "unknown"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise the card to a JSON string."""
        return json.dumps(self._to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ModelCard:
        """Deserialise a card from a JSON string produced by :meth:`to_json`."""
        data = json.loads(text)
        return cls._from_dict(data)

    def save(self, path: Path | str) -> None:
        """Write the card to ``path`` as JSON.

        The directory is created if it does not exist.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> ModelCard:
        """Load a card from a JSON file written by :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def summary_dict(self) -> dict[str, object]:
        """Return a flat dict suitable for API responses and Streamlit display."""
        return {
            "model_version": self.model_version,
            "created_at": self.created_at,
            "created_at_iso": self.created_at_iso,
            "pipeline_version": self.pipeline_version,
            "training_data_hash": self.training_data_hash,
            "n_training_samples": self.n_training_samples,
            "n_features": len(self.feature_schema),
            "feature_names": [f.name for f in self.feature_schema],
            "accuracy": self.evaluation.accuracy,
            "f1": self.evaluation.f1,
            "f1_improvement": self.evaluation.f1_improvement,
            "bootstrap_ci_lower": self.evaluation.bootstrap_ci_lower,
            "bootstrap_ci_upper": self.evaluation.bootstrap_ci_upper,
            "n_eval_samples": self.evaluation.n_eval_samples,
            "promotion_reason": self.promotion_reason,
            "notes": self.notes,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict converts dataclass fields recursively - feature_schema becomes
        # a list of dicts, evaluation becomes a dict.  That's exactly what we want.
        return d

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ModelCard:
        features = [FeatureSpec(**f) for f in data.get("feature_schema", [])]
        evaluation = ModelEvaluation(**data["evaluation"])
        return cls(
            model_version=data["model_version"],
            created_at=data["created_at"],
            created_at_iso=data["created_at_iso"],
            training_data_hash=data["training_data_hash"],
            n_training_samples=data["n_training_samples"],
            feature_schema=features,
            evaluation=evaluation,
            promotion_reason=data.get("promotion_reason", ""),
            pipeline_version=data.get("pipeline_version", "unknown"),
            notes=data.get("notes", ""),
            extra=data.get("extra", {}),
        )


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def hash_training_data(X: object) -> str:
    """Compute SHA-256 of training features as float64 bytes.

    Uses the same hashing strategy as ``RetrainEvidenceBuffer`` for
    consistency - ``df.to_numpy(dtype=np.float64).tobytes()``.

    Args:
        X: Training feature matrix.  Accepted types: ``np.ndarray``,
           ``pd.DataFrame``, or any object with a ``to_numpy()`` method.

    Returns:
        64-character lowercase hex digest.
    """
    import numpy as np

    if hasattr(X, "to_numpy"):
        arr = getattr(X, "to_numpy")(dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def feature_schema_from_array(
    X: object,
    feature_names: list[str] | None = None,
) -> list[FeatureSpec]:
    """Build a list of :class:`FeatureSpec` from a training array.

    Args:
        X:             Feature matrix (numpy array or DataFrame).
        feature_names: Column names.  Inferred from DataFrame columns when
                       ``None``.  Falls back to ``f0, f1, …`` for arrays.

    Returns:
        One :class:`FeatureSpec` per feature.
    """
    import numpy as np

    try:
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            names = feature_names or list(X.columns)
            arr = X.to_numpy()
            dtypes = [str(X[c].dtype) for c in X.columns]
        else:
            arr = np.asarray(X)
            n = arr.shape[1] if arr.ndim == 2 else 1
            names = feature_names or [f"f{i}" for i in range(n)]
            dtypes = [str(arr.dtype)] * len(names)
    except ImportError:
        arr = np.asarray(X)
        n = arr.shape[1] if arr.ndim == 2 else 1
        names = feature_names or [f"f{i}" for i in range(n)]
        dtypes = [str(arr.dtype)] * len(names)

    arr_2d = arr.reshape(len(arr), -1) if arr.ndim != 2 else arr
    specs = []
    for i, (name, dtype) in enumerate(zip(names, dtypes)):
        col = arr_2d[:, i].astype(float)
        null_mask = ~np.isfinite(col)
        specs.append(
            FeatureSpec(
                name=name,
                dtype=dtype,
                min_value=float(np.nanmin(col)) if not null_mask.all() else None,
                max_value=float(np.nanmax(col)) if not null_mask.all() else None,
                null_rate=float(null_mask.mean()),
            )
        )
    return specs


def build_model_card(
    *,
    model_version: int,
    X_train: object,
    feature_names: list[str] | None = None,
    evaluation: ModelEvaluation,
    promotion_reason: str = "",
    pipeline_version: str | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> ModelCard:
    """Convenience constructor - builds a full :class:`ModelCard` in one call.

    Args:
        model_version:    Integer version number.
        X_train:          Training feature matrix (for hash and schema).
        feature_names:    Optional column names.
        evaluation:       Held-out evaluation metrics.
        promotion_reason: Human-readable reason string.
        pipeline_version: Semver string; reads ``MODEL_MONITOR_VERSION`` env
                          var when ``None``.
        notes:            Free-text notes.
        extra:            Arbitrary metadata dict.

    Returns:
        A fully populated :class:`ModelCard`.
    """
    import os

    if pipeline_version is None:
        pipeline_version = os.environ.get("MODEL_MONITOR_VERSION", "unknown")

    import numpy as np

    _arr = np.asarray(X_train)
    n_samples = int(_arr.shape[0]) if _arr.ndim >= 1 else 0
    now = time.time()

    return ModelCard(
        model_version=model_version,
        created_at=now,
        created_at_iso=_iso(now),
        training_data_hash=hash_training_data(X_train),
        n_training_samples=n_samples,
        feature_schema=feature_schema_from_array(X_train, feature_names),
        evaluation=evaluation,
        promotion_reason=promotion_reason,
        pipeline_version=pipeline_version,
        notes=notes,
        extra=extra or {},
    )


def _iso(ts: float) -> str:
    """Convert a Unix timestamp to an ISO 8601 string (UTC)."""
    import datetime

    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()
