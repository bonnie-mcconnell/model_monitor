from __future__ import annotations

import numpy as np
import pandas as pd

from model_monitor.config.settings import load_config
from model_monitor.contracts.behavioral.evaluators import JsonValidityEvaluator
from model_monitor.contracts.behavioral.policy import StrictBehaviorPolicy
from model_monitor.contracts.behavioral.runner import BehavioralContractRunner
from model_monitor.contracts.contract import Contract
from model_monitor.contracts.guarantee import Guarantee, Severity
from model_monitor.contracts.registry import EvaluatorRegistry
from model_monitor.inference.predict import Predictor


class DummyModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), 2), 0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictor(**kwargs: object) -> Predictor:
    cfg = load_config()
    p = Predictor(config=cfg, f1_baseline=0.85, **kwargs)  # type: ignore[arg-type]
    p.model = DummyModel()
    return p


def _make_runner(*, evaluator: JsonValidityEvaluator | None = None) -> BehavioralContractRunner:
    """Minimal runner with a single JSON-validity guarantee."""
    ev = evaluator or JsonValidityEvaluator()
    contract = Contract(
        contract_id="test_contract",
        version="1.0",
        scope="test",
        guarantees=(
            Guarantee(
                guarantee_id="valid_json",
                description="Output must be valid JSON",
                severity=Severity.CRITICAL,
                evaluator_id=ev.evaluator_id,
            ),
        ),
    )
    registry = EvaluatorRegistry()
    registry.register(ev)
    return BehavioralContractRunner(
        contract=contract,
        registry=registry,
        policy=StrictBehaviorPolicy(),
    )


_X = pd.DataFrame(np.random.rand(50, 3), columns=["a", "b", "c"])
_Y = pd.Series(np.random.randint(0, 2, size=50))


# ---------------------------------------------------------------------------
# Classical monitoring
# ---------------------------------------------------------------------------

def test_predictor_healthy_batch_returns_none_decision() -> None:
    predictor = _make_predictor()
    preds, confs, decision = predictor.predict_batch(
        _X, y_true=_Y, batch_id="batch_001"
    )
    assert len(preds) == 50
    assert len(confs) == 50
    assert decision.action == "none"


# ---------------------------------------------------------------------------
# Behavioral contract integration
# ---------------------------------------------------------------------------

def test_predict_batch_runs_behavioral_evaluation_when_runner_provided() -> None:
    """
    When behavioral_output is provided alongside a runner, predict_batch
    must populate last_behavioral_record after the call.
    """
    predictor = _make_predictor(behavioral_runner=_make_runner())

    assert predictor.last_behavioral_record is None

    predictor.predict_batch(
        _X,
        batch_id="b1",
        behavioral_output='{"key": "value"}',
    )

    assert predictor.last_behavioral_record is not None


def test_predict_batch_accepts_output_without_runner_configured() -> None:
    """
    Passing behavioral_output when no runner is configured must not raise -
    the argument is silently ignored.
    """
    predictor = _make_predictor()  # no behavioral_runner

    predictor.predict_batch(
        _X,
        batch_id="b2",
        behavioral_output='{"key": "value"}',
    )

    assert predictor.last_behavioral_record is None


def test_predict_batch_skips_behavioral_when_output_is_none() -> None:
    """
    When behavioral_output is None the runner must not be called even if
    one is configured - last_behavioral_record stays None.
    """
    predictor = _make_predictor(behavioral_runner=_make_runner())

    predictor.predict_batch(_X, batch_id="b3")  # no behavioral_output

    assert predictor.last_behavioral_record is None


def test_behavioral_evaluation_records_block_for_invalid_json() -> None:
    """
    An output that fails the JSON-validity guarantee must produce a
    BLOCK outcome in last_behavioral_record.
    """
    from model_monitor.contracts.outcome import DecisionOutcome

    predictor = _make_predictor(behavioral_runner=_make_runner())

    predictor.predict_batch(
        _X,
        batch_id="b4",
        behavioral_output="this is not json at all",
    )

    assert predictor.last_behavioral_record is not None
    assert predictor.last_behavioral_record.outcome == DecisionOutcome.BLOCK


def test_behavioral_evaluation_records_accept_for_valid_output() -> None:
    """
    A well-formed output must produce an ACCEPT outcome.
    """
    from model_monitor.contracts.outcome import DecisionOutcome

    predictor = _make_predictor(behavioral_runner=_make_runner())

    predictor.predict_batch(
        _X,
        batch_id="b5",
        behavioral_output='{"ticket_id": "T-1", "response": "Hello"}',
    )

    assert predictor.last_behavioral_record is not None
    assert predictor.last_behavioral_record.outcome == DecisionOutcome.ACCEPT


def test_behavioral_evaluation_does_not_block_inference_on_exception() -> None:
    """
    If the behavioral runner raises unexpectedly, predict_batch must still
    return a valid (predictions, confidences, decision) tuple - the
    inference path must never be blocked by monitoring failures.
    """
    class _ExplodingEvaluator(JsonValidityEvaluator):
        def evaluate(self, *, output: str) -> object:  # type: ignore[override]
            raise RuntimeError("simulated evaluator crash")

    runner = _make_runner(evaluator=_ExplodingEvaluator())
    predictor = _make_predictor(behavioral_runner=runner)

    # Must not raise - exception is caught and logged inside _run_behavioral_evaluation
    preds, confs, decision = predictor.predict_batch(
        _X,
        batch_id="b6",
        behavioral_output="any output",
    )

    assert len(preds) == 50
    assert predictor.last_behavioral_record is None  # evaluation failed, no record

