"""Configuration loading, validation, and settings dataclasses.

All application-level configuration is typed here. Runtime behaviour is
controlled through YAML files under config/ and environment variables;
this package provides the Python interface to both.
"""

from __future__ import annotations
