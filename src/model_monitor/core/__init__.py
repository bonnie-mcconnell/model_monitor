"""Decision engine, executor, and model lifecycle management.

The decision engine is a pure function - no I/O, fully replayable.
Side-effects (retrain, promote, rollback) live in DecisionExecutor.
"""

from __future__ import annotations
