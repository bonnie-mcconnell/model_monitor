# Security Policy

## Supported versions

| Branch | Status |
|--------|--------|
| `behavior-monitoring` | ✅ Active development |
| `main` | ✅ Maintained |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please email a description of the issue to the repository owner via the
contact information on the
[GitHub profile](https://github.com/bonnie-mcconnell). Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any suggested fix if you have one

You will receive a response within 72 hours. If the issue is confirmed,
a fix will be prioritised and a patch released before any public disclosure.

## Scope

The primary security-relevant surface in this project is the ingest API
(`POST /metrics/ingest` on the `behavior-monitoring` branch):

- The `X-API-Key` header is checked against the `MONITOR_API_KEY`
  environment variable. The endpoint returns 503 if the variable is unset,
  preventing accidental public exposure.
- Metric fields are range-validated by Pydantic before being written to
  the database. Values outside [0, 1] for rates and scores are rejected
  with 422.
- The SQLite database is local by default. If you expose the FastAPI server
  to a network, ensure `MONITOR_API_KEY` is set and the server is behind a
  reverse proxy with TLS.

## Dependencies

This project uses `pip-audit` or `safety` is recommended for scanning
dependencies. All direct dependencies are pinned to compatible ranges in
`pyproject.toml`.
