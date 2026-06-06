"""CUSUM sequential change-point detection for ML monitoring.

Batch tests (PSI, MMD) ask "did the distribution change over this window?"
CUSUM asks "at what exact batch did the distribution change?"  It accumulates
evidence across batches and raises an alarm the moment accumulated evidence
exceeds a threshold.

This makes CUSUM strictly more sensitive than batch tests on small samples:
a batch test needs the full window to be drifted, whereas CUSUM can detect a
change that started two batches ago before the window averages it out.

Algorithm
---------
We use the Page–Hinkley variant of CUSUM, which is suited for online monitoring
of scalar statistics (e.g. rolling PSI, trust score, model error).

Let x_t be the monitored statistic at batch t.  We maintain two cumulative sums:

    S_t^+ = max(0, S_{t-1}^+ + (x_t - μ₀ - δ))   ← upward drift detector
    S_t^- = max(0, S_{t-1}^- - (x_t - μ₀ + δ))   ← downward drift detector

Where:
  μ₀:  reference mean (estimated from pre-drift observations)
  δ:   allowance (half the smallest shift worth detecting)
  h:   decision threshold (raise alarm when S_t^+ or S_t^- > h)

An alarm fires when ``S_t^+ > h`` (upward change - PSI or error increasing) or
``S_t^- > h`` (downward change - trust score decreasing).

When an alarm fires:
  - The approximate change point is the batch where the cumulative sum reset to
    zero before beginning its terminal ascent.  This is the standard CUSUM
    change-point estimate.
  - The detector resets its cumulative sums so it can detect subsequent changes.

Statistical properties
----------------------
- Under the null hypothesis (no change), the expected run-length before a false
  alarm is approximately 2h/δ² (Roberts 1959).  Setting h = 4σ gives ~500 ARL₀.
- The expected detection delay after a shift of size Δ > δ is approximately
  h / (Δ - δ).  A shift of 2δ is detected in roughly h / δ batches.
- CUSUM is optimal in the minimax sense: for any false-alarm rate, no other
  sequential test has smaller worst-case detection delay.

References
----------
Page, E. S. (1954). Continuous inspection schemes. Biometrika, 41(1/2), 100–115.

Mouss, H., Mouss, D., Mouss, N., & Sari, L. (2004). Test of Page-Hinkley, an
  approach for fault detection in an agro-alimentary production system.
  IMACS–IEEE CSCC 2004.

Montgomery, D. C. (2009). Introduction to Statistical Quality Control (6th ed.).
  Wiley. Chapter 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CUSUMResult:
    """Result of one CUSUM update call.

    Attributes:
        alarm:          True when a change point has been detected.
        alarm_direction: ``"up"`` when the statistic increased above baseline,
                        ``"down"`` when it decreased, ``None`` when no alarm.
        s_pos:          Current upward cumulative sum S_t^+.
        s_neg:          Current downward cumulative sum S_t^-.
        change_point:   Estimated batch index of the change point.  Set when
                        ``alarm=True``; None otherwise.
        n_since_reset:  Batches since the last alarm (or since construction).
        statistic:      The monitored value at this batch.
    """

    alarm: bool
    alarm_direction: Literal["up", "down"] | None
    s_pos: float
    s_neg: float
    change_point: int | None
    n_since_reset: int
    statistic: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CUSUMDetector:
    """Page–Hinkley CUSUM sequential change-point detector.

    Monitors a scalar statistic (PSI, trust score, MAE, error rate…) across
    batches and raises an alarm at the exact batch where a sustained shift
    begins - not just when a batch-level test rejects.

    Usage::

        detector = CUSUMDetector(reference_mean=0.03, delta=0.02, threshold=4.0)

        for batch in production_batches:
            result = detector.update(batch.drift_score)
            if result.alarm:
                print(f"Change detected at batch {result.change_point}")
                print(f"Direction: {result.alarm_direction}")
                # detector auto-resets; continues monitoring

    Parameters
    ----------
    reference_mean:
        Expected value of the monitored statistic during stable operation.
        Set from the mean over a stable pre-deployment window.
        For PSI: typically 0.02–0.05.
        For trust score: typically 0.85–0.95.
    delta:
        Allowance - half the smallest shift worth detecting.  A shift of
        ``2 * delta`` from ``reference_mean`` will be detected with expected
        delay ``threshold / delta``.  Smaller delta = more sensitive but slower
        to detect large shifts; larger delta = less sensitive to noise.
        Rule of thumb: ``delta = 0.5 * (shift_to_detect - reference_mean)``.
    threshold:
        Decision threshold ``h``.  Alarm fires when ``S_t^+ > h`` or
        ``S_t^- > h``.  Higher threshold = fewer false alarms but longer
        detection delay.  Rule of thumb: ``h = 4 * reference_std``.
    direction:
        ``"both"`` (default): detect both increases and decreases.
        ``"up"``: only detect increases (useful for PSI, error rate).
        ``"down"``: only detect decreases (useful for trust score, F1).
    warmup_batches:
        Number of batches to observe before the detector starts firing alarms.
        Prevents false alarms from initial transients.  Default: 5.

    Notes
    -----
    The detector resets its cumulative sums automatically after an alarm so it
    can detect subsequent change points.  ``change_point`` in the result gives
    the batch index (0-based, relative to construction time) of the estimated
    change point.
    """

    def __init__(
        self,
        *,
        reference_mean: float,
        delta: float,
        threshold: float,
        direction: Literal["both", "up", "down"] = "both",
        warmup_batches: int = 5,
    ) -> None:
        if delta <= 0:
            raise ValueError(f"delta must be positive; got {delta}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive; got {threshold}")
        if warmup_batches < 0:
            raise ValueError(f"warmup_batches must be >= 0; got {warmup_batches}")
        if direction not in {"both", "up", "down"}:
            raise ValueError(
                f"direction must be 'both', 'up', or 'down'; got {direction!r}"
            )

        self.reference_mean = reference_mean
        self.delta = delta
        self.threshold = threshold
        self.direction = direction
        self.warmup_batches = warmup_batches

        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._n: int = 0                  # total batches seen
        self._n_since_reset: int = 0      # batches since last alarm / reset
        self._last_reset_at: int = 0      # batch index of last reset

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def update(self, value: float) -> CUSUMResult:
        """Record one batch observation and check for a change point.

        Args:
            value: The monitored statistic for this batch (e.g. drift_score,
                   trust_score, MAE).

        Returns:
            :class:`CUSUMResult` with alarm status and current cumulative sums.
        """
        self._n += 1
        self._n_since_reset += 1

        # Page–Hinkley update equations (Page 1954).
        #
        # Upward detector S_t^+: accumulates evidence that x_t > mu0 + delta.
        #   S_t^+ = max(0, S_{t-1}^+ + (x_t - mu0 - delta))
        #   Fires when S_t^+ > h.
        #
        # Downward detector S_t^-: accumulates evidence that x_t < mu0 - delta.
        #   S_t^- = max(0, S_{t-1}^- - (x_t - mu0 + delta))
        #   Fires when S_t^- > h.
        #   Note: S^- grows when (x_t - mu0 + delta) < 0, i.e. x_t < mu0 - delta.
        new_s_pos = max(0.0, self._s_pos + (value - self.reference_mean - self.delta))
        new_s_neg = max(0.0, self._s_neg - (value - self.reference_mean + self.delta))

        self._s_pos = new_s_pos
        self._s_neg = new_s_neg

        # No alarms during warmup.
        if self._n_since_reset <= self.warmup_batches:
            return CUSUMResult(
                alarm=False,
                alarm_direction=None,
                s_pos=self._s_pos,
                s_neg=self._s_neg,
                change_point=None,
                n_since_reset=self._n_since_reset,
                statistic=value,
            )

        # Check for alarm.
        alarm = False
        alarm_direction: Literal["up", "down"] | None = None
        change_point: int | None = None

        up_alarm = (
            self.direction in {"both", "up"} and self._s_pos > self.threshold
        )
        down_alarm = (
            self.direction in {"both", "down"} and self._s_neg > self.threshold
        )

        if up_alarm or down_alarm:
            alarm = True
            alarm_direction = "up" if up_alarm else "down"
            # Estimate change point: the batch where S last reset to zero,
            # i.e. the start of the current ascending run.
            change_point = self._n - self._n_since_reset + self._last_reset_at
            self._reset()

        return CUSUMResult(
            alarm=alarm,
            alarm_direction=alarm_direction,
            s_pos=self._s_pos,
            s_neg=self._s_neg,
            change_point=change_point,
            n_since_reset=self._n_since_reset,
            statistic=value,
        )

    def reset(self) -> None:
        """Manually reset the cumulative sums (e.g. after a model retrain).

        Call this when a retraining event is expected to resolve the drift:
        resetting prevents the old cumulative sum from immediately triggering
        a false alarm on the post-retrain stable batches.
        """
        self._reset()

    @property
    def s_pos(self) -> float:
        """Current upward cumulative sum S_t^+."""
        return self._s_pos

    @property
    def s_neg(self) -> float:
        """Current downward cumulative sum S_t^-."""
        return self._s_neg

    @property
    def n_batches(self) -> int:
        """Total batches seen since construction."""
        return self._n

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _reset(self) -> None:
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._last_reset_at = self._n
        self._n_since_reset = 0
