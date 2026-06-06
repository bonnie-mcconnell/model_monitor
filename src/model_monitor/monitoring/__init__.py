"""Monitoring primitives: drift detection, trust scoring, alerting.

Each sub-module is independently usable. The Monitor SDK in monitor.py
composes them; individual components can also be used standalone.
"""

from __future__ import annotations
