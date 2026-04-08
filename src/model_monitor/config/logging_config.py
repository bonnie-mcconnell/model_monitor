"""Structured logging setup for the application."""
import logging
import sys


def setup_logging(
    *,
    level: int = logging.INFO,
    logger_name: str | None = None,
) -> None:
    """
    Configure structured logging for the application.

    Safe to call multiple times.
    """
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return  # Prevent duplicate handlers

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
