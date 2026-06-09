"""Tests for the CUSUM sequential change-point detector (monitoring.cusum).

Tests cover:
  - Construction validation
  - Warmup period suppresses alarms
  - Upward shift detected within bounded delay
  - Downward shift detected
  - No false alarm on stable signal (approximate ARL₀ check)
  - Auto-reset after alarm; continued detection works
  - Manual reset prevents stale cumulative sums firing
  - direction parameter gates which alarms fire
  - Result fields are consistent
"""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.monitoring.cusum import CUSUMDetector

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestCUSUMConstruction:
    def test_raises_on_non_positive_delta(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            CUSUMDetector(reference_mean=0.05, delta=0.0, threshold=4.0)

    def test_raises_on_non_positive_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            CUSUMDetector(reference_mean=0.05, delta=0.01, threshold=0.0)

    def test_raises_on_negative_warmup(self) -> None:
        with pytest.raises(ValueError, match="warmup_batches"):
            CUSUMDetector(
                reference_mean=0.05, delta=0.01, threshold=4.0, warmup_batches=-1
            )

    def test_raises_on_invalid_direction(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            CUSUMDetector(
                reference_mean=0.05,
                delta=0.01,
                threshold=4.0,
                direction="sideways",  # type: ignore[arg-type]
            )

    def test_initial_state(self) -> None:
        d = CUSUMDetector(reference_mean=0.05, delta=0.01, threshold=4.0)
        assert d.s_pos == 0.0
        assert d.s_neg == 0.0
        assert d.n_batches == 0


# ---------------------------------------------------------------------------
# Warmup suppresses alarms
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_no_alarm_during_warmup(self) -> None:
        """Large shifts during warmup must not trigger alarms."""
        d = CUSUMDetector(
            reference_mean=0.0, delta=0.01, threshold=0.1, warmup_batches=5
        )
        for _ in range(5):
            result = d.update(10.0)  # absurdly large shift
            assert not result.alarm, "alarm fired during warmup"

    def test_alarm_fires_after_warmup(self) -> None:
        """Detector fires once warmup is complete."""
        d = CUSUMDetector(
            reference_mean=0.0, delta=0.01, threshold=0.1, warmup_batches=2
        )
        d.update(10.0)
        d.update(10.0)
        result = d.update(10.0)  # batch 3 - past warmup
        assert result.alarm


# ---------------------------------------------------------------------------
# Upward shift detection
# ---------------------------------------------------------------------------


class TestUpwardShift:
    def test_detects_sustained_upward_shift(self) -> None:
        """A sustained shift of 5× the allowance must be detected within 80 batches."""
        rng = np.random.default_rng(0)
        ref_mean = 0.03
        delta = 0.02
        # threshold = 5 * delta so expected detection delay ~ threshold / (shift - delta)
        # shift = 5*delta, delay ~ threshold/(4*delta) = 5*delta/(4*delta) = 1.25 batches
        threshold = 5 * delta  # 0.10

        d = CUSUMDetector(
            reference_mean=ref_mean,
            delta=delta,
            threshold=threshold,
            direction="up",
            warmup_batches=3,
        )

        # Stable phase: feed 10 stable batches
        for _ in range(10):
            d.update(float(rng.normal(ref_mean, 0.005)))

        # Drift phase: shift mean to ref_mean + 5*delta
        alarms = []
        for i in range(80):
            result = d.update(float(rng.normal(ref_mean + 5 * delta, 0.005)))
            if result.alarm:
                alarms.append(i)
                break

        assert len(alarms) >= 1, "CUSUM failed to detect a 5× allowance upward shift"

    def test_change_point_is_set_on_alarm(self) -> None:
        """change_point is a non-negative integer when an alarm fires."""
        rng = np.random.default_rng(7)
        d = CUSUMDetector(
            reference_mean=0.0, delta=0.01, threshold=0.5, warmup_batches=2
        )
        for _ in range(3):
            d.update(float(rng.normal(0.0, 0.001)))

        fired = None
        for _ in range(50):
            result = d.update(0.5)  # large shift
            if result.alarm:
                fired = result
                break

        assert fired is not None
        assert fired.change_point is not None
        assert fired.change_point >= 0


# ---------------------------------------------------------------------------
# Downward shift detection
# ---------------------------------------------------------------------------


class TestDownwardShift:
    def test_detects_sustained_downward_shift(self) -> None:
        """A downward shift (e.g. trust score drop) is detected."""
        rng = np.random.default_rng(3)
        ref_mean = 0.90
        delta = 0.02
        threshold = 3.0

        d = CUSUMDetector(
            reference_mean=ref_mean,
            delta=delta,
            threshold=threshold,
            direction="down",
            warmup_batches=2,
        )

        for _ in range(5):
            d.update(float(rng.normal(ref_mean, 0.005)))

        fired = None
        for _ in range(40):
            result = d.update(float(rng.normal(ref_mean - 5 * delta, 0.005)))
            if result.alarm:
                fired = result
                break

        assert fired is not None, "CUSUM failed to detect downward shift in trust score"
        assert fired.alarm_direction == "down"

    def test_direction_up_does_not_fire_on_downward_shift(self) -> None:
        """direction='up' must ignore downward shifts."""
        d = CUSUMDetector(
            reference_mean=0.9,
            delta=0.01,
            threshold=0.5,
            direction="up",
            warmup_batches=0,
        )
        for _ in range(30):
            result = d.update(0.0)  # large downward shift
        assert not result.alarm

    def test_direction_down_does_not_fire_on_upward_shift(self) -> None:
        """direction='down' must ignore upward shifts."""
        d = CUSUMDetector(
            reference_mean=0.0,
            delta=0.01,
            threshold=0.5,
            direction="down",
            warmup_batches=0,
        )
        for _ in range(30):
            result = d.update(10.0)  # large upward shift
        assert not result.alarm


# ---------------------------------------------------------------------------
# Auto-reset and continued detection
# ---------------------------------------------------------------------------


class TestAutoReset:
    def test_cusum_resets_after_alarm_and_continues_detecting(self) -> None:
        """After an alarm the detector resets and can detect the next change."""
        d = CUSUMDetector(
            reference_mean=0.0, delta=0.01, threshold=0.3, warmup_batches=0
        )

        # First shift: trigger alarm
        first_alarm = None
        for _ in range(50):
            result = d.update(1.0)
            if result.alarm:
                first_alarm = result
                break
        assert first_alarm is not None, "first alarm not fired"
        assert d.s_pos == 0.0, "s_pos not reset after alarm"

        # Stable phase after reset
        for _ in range(3):
            d.update(0.0)

        # Second shift: should trigger another alarm
        second_alarm = None
        for _ in range(50):
            result = d.update(1.0)
            if result.alarm:
                second_alarm = result
                break
        assert second_alarm is not None, "second alarm not fired after reset"

    def test_manual_reset_zeroes_cumulative_sums(self) -> None:
        d = CUSUMDetector(
            reference_mean=0.0, delta=0.01, threshold=10.0, warmup_batches=0
        )
        for _ in range(5):
            d.update(1.0)
        assert d.s_pos > 0
        d.reset()
        assert d.s_pos == 0.0
        assert d.s_neg == 0.0


# ---------------------------------------------------------------------------
# False alarm rate (approximate ARL₀)
# ---------------------------------------------------------------------------


class TestFalseAlarmRate:
    """Under the null (no change), alarms should be infrequent.

    The theoretical average run length (ARL₀) for Page–Hinkley CUSUM with
    threshold h and allowance δ is approximately 2h/δ².  We check that over
    200 stable batches we get zero or very few alarms.
    """

    def test_low_false_alarm_rate_on_stable_signal(self) -> None:
        rng = np.random.default_rng(99)
        ref_mean = 0.05
        std = 0.01
        delta = 0.5 * std  # allowance = 0.5σ
        threshold = 20 * std  # high threshold → low false alarm rate

        d = CUSUMDetector(
            reference_mean=ref_mean,
            delta=delta,
            threshold=threshold,
            warmup_batches=5,
        )

        false_alarms = 0
        for _ in range(200):
            result = d.update(float(rng.normal(ref_mean, std)))
            if result.alarm:
                false_alarms += 1

        assert false_alarms <= 2, (
            f"Too many false alarms on stable signal: {false_alarms}. "
            "CUSUM may be miscalibrated."
        )


# ---------------------------------------------------------------------------
# Result fields consistency
# ---------------------------------------------------------------------------


class TestResultFields:
    def test_s_pos_non_negative(self) -> None:
        d = CUSUMDetector(
            reference_mean=0.5, delta=0.1, threshold=5.0, warmup_batches=0
        )
        for v in [-10.0, -5.0, 0.0, 0.5, 1.0]:
            result = d.update(v)
            assert result.s_pos >= 0.0

    def test_s_neg_non_negative(self) -> None:
        d = CUSUMDetector(
            reference_mean=0.5, delta=0.1, threshold=5.0, warmup_batches=0
        )
        for v in [10.0, 5.0, 0.5, 0.0, -0.5]:
            result = d.update(v)
            assert result.s_neg >= 0.0

    def test_n_batches_increments(self) -> None:
        d = CUSUMDetector(reference_mean=0.0, delta=0.01, threshold=5.0)
        for i in range(1, 6):
            d.update(0.0)
            assert d.n_batches == i

    def test_statistic_echoed_in_result(self) -> None:
        d = CUSUMDetector(reference_mean=0.0, delta=0.01, threshold=5.0)
        result = d.update(3.14)
        assert result.statistic == pytest.approx(3.14)

    def test_alarm_false_means_no_direction_no_changepoint(self) -> None:
        d = CUSUMDetector(
            reference_mean=0.0, delta=0.01, threshold=1000.0, warmup_batches=0
        )
        result = d.update(0.001)
        assert not result.alarm
        assert result.alarm_direction is None
        assert result.change_point is None
