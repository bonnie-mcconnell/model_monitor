"""model_monitor - production ML monitoring for any sklearn-compatible model.

Public API::

    from model_monitor import Monitor, MonitorConfig, BatchResult, MonitorSummary

    monitor = Monitor(clf, reference_data=X_train, feature_names=cols)

    # Batch inference (offline / bulk scoring):
    result = monitor.predict(X_batch, y_true=y_batch)
    print(result.trust_score, result.drift_score, result.is_joint_drifting)
    print(result.is_cusum_alarm)   # True when CUSUM detects a change point

    # Per-request inference (REST endpoints, event streams):
    pred = monitor.predict_one(row)        # returns immediately
    monitor.flush()                        # explicit flush at shutdown

    print(monitor.report())

For standalone detectors::

    from model_monitor.monitoring.mmd import MMDDriftDetector
    from model_monitor.monitoring.cusum import CUSUMDetector
"""

from __future__ import annotations

from model_monitor.monitor import BatchResult, Monitor, MonitorConfig, MonitorSummary

__all__ = ["BatchResult", "Monitor", "MonitorConfig", "MonitorSummary"]
