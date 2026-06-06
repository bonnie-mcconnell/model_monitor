# Webhook integration examples

`WebhookAlerter` posts a small JSON payload to any HTTP endpoint when the
trust score crosses a threshold.  The payload schema is stable - field names
and types will not change without a major version bump.

## Payload schema

```json
{
  "window":      "5m",
  "trust_score": 0.52,
  "severity":    "critical",
  "ts":          1713967200.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `window` | string | Aggregation window that triggered the alert: `"5m"`, `"1h"`, or `"24h"` |
| `trust_score` | float | Trust score that triggered the alert, in `[0, 1]` |
| `severity` | string | `"warning"` (trust < 0.7) or `"critical"` (trust < 0.5) |
| `ts` | float | Unix timestamp (UTC) when the alert fired |

## Slack incoming webhooks

```python
from model_monitor.monitoring.alerting import WebhookAlerter

alerter = WebhookAlerter(
    url="https://hooks.slack.com/services/T.../B.../...",
    severity_filter="critical",   # only page on critical; suppress warnings
)
```

Slack expects the payload wrapped in a `text` field.  Add a thin adapter:

```python
import requests

def slack_post(url, json, timeout):
    msg = (
        f":rotating_light: *model-monitor* | "
        f"window={json['window']} | "
        f"trust={json['trust_score']:.3f} | "
        f"severity={json['severity']}"
    )
    return requests.post(url, json={"text": msg}, timeout=timeout)

alerter = WebhookAlerter(url="https://hooks.slack.com/...", _post=slack_post)
```

## PagerDuty Events API v2

```python
import requests, time

def pagerduty_post(url, json, timeout):
    payload = {
        "routing_key": "<YOUR_INTEGRATION_KEY>",
        "event_action": "trigger",
        "dedup_key": f"model-monitor-{json['window']}",
        "payload": {
            "summary": f"Trust score {json['trust_score']:.3f} ({json['severity']})",
            "source": "model-monitor",
            "severity": "critical" if json["severity"] == "critical" else "warning",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(json["ts"])),
            "custom_details": json,
        },
    }
    return requests.post(
        "https://events.pagerduty.com/v2/enqueue",
        json=payload,
        timeout=timeout,
    )

alerter = WebhookAlerter(url="https://events.pagerduty.com/v2/enqueue",
                         _post=pagerduty_post)
```

## Generic HTTP endpoint

```python
alerter = WebhookAlerter(
    url="https://alerts.internal/model-monitor",
    timeout_s=5.0,
    severity_filter=None,   # receive all alerts
)
```

## Wire into the aggregation loop

```python
from model_monitor.monitoring.alerting import check_alerts
from model_monitor.monitoring.alerting import WebhookAlerter

alerter = WebhookAlerter(url="https://hooks.slack.com/...")

# Inside your aggregation callback:
check_alerts(
    window="5m",
    summary={"trust_score": current_trust_score},
    alerter=alerter,
)
```
