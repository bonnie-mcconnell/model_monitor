# TODO import??
DECISION_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "trust_score": {"type": "number"},
        "f1_drop": {"type": "number"},
        "baseline_f1": {"type": "number"},
        "current_f1": {"type": "number"},
        "drift_score": {"type": "number"},
        "cooldown_batches": {"type": "integer"},
    },
    "additionalProperties": False,
}
