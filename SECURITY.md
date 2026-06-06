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

The primary security-relevant surfaces in this project:

**Ingest API** (`POST /metrics/ingest`):
- The `X-API-Key` header is checked against the `MONITOR_API_KEY`
  environment variable at request time (not at startup), so the key can be
  rotated without restarting. The endpoint returns 503 if the variable is
  unset, preventing accidental open ingestion.
- Metric fields are range-validated by Pydantic before being written to
  the database. Values outside `[0, 1]` for rates and scores are rejected
  with HTTP 422.

**Dashboard API** (`GET /dashboard/*`, `POST /dashboard/*`):
- When the `MONITOR_DASHBOARD_KEY` environment variable is set, all
  `/dashboard/*` routes require a matching `X-Api-Key` header.  An incorrect
  or missing key returns HTTP 401.
- When the variable is unset the dashboard is unauthenticated - suitable for
  local development and deployments protected at the network layer (reverse
  proxy with Nginx/Cloudflare Access, VPN, etc.).
- The same header name (`X-Api-Key`) and rotation semantics apply as for
  `MONITOR_API_KEY` on the ingest endpoint.  Both keys can be rotated without
  restarting the server.

**SQLite database** (`data/metrics/metrics.db`):
- Local by default. WAL mode is enabled for concurrent access but there is
  no encryption. If the database contains sensitive inference data, encrypt
  at rest via filesystem-level encryption.
- If you expose the FastAPI server to a network, ensure `MONITOR_API_KEY`
  is set and the server is behind a reverse proxy with TLS.

**Prometheus endpoint** (`GET /metrics`):
- Unauthenticated. Exposes operational metrics (trust score, F1, drift,
  decision counts). Restrict at the network layer in production.

## Dependencies

Running `pip-audit` or `safety` against this project's dependencies is
recommended for scanning for known vulnerabilities. All direct dependencies
are pinned to compatible ranges in `pyproject.toml`.
