"""Carry execution settings through selection, local metrics and packages."""

from .common import digest
from .cache_compat import FIELDS, normalized


def inference_profiles(meta):
    stored = meta.get("inference_provenance")
    if stored is not None:
        return stored["profiles"]
    return [{k: normalized(k, meta.get(k)) for k in FIELDS}]


def selection_provenance(metadata, receipts):
    profiles = {digest(p): p for m in metadata for p in inference_profiles(m)}
    return {
        "schema": "inference_provenance_v1",
        "profiles": [profiles[k] for k in sorted(profiles)],
        "candidate_profiles": {
            m["candidate_id"]: digest(inference_profiles(m)[0]) for m in metadata
        },
        "source_caches": receipts,
    }


class Inventory:
    def __init__(self):
        self.conditions = {}
        self.evidence = {}

    def add(self, condition, meta):
        bucket = self.conditions.setdefault(condition, {})
        for p in inference_profiles(meta):
            h = digest(p)
            row = bucket.setdefault(h, {"settings": p, "map_appearances": 0})
            row["map_appearances"] += 1
        for src in meta.get("inference_provenance", {}).get("source_caches", []):
            if src.get("evidence"):
                self.evidence[digest(src["evidence"])] = src["evidence"]

    def summary(self):
        profiles = [r["settings"] for b in self.conditions.values() for r in b.values()]

        def different(key):
            return len({digest(p[key]) for p in profiles if p.get(key) is not None}) > 1

        dtype = different("precision")
        scope = different("precision_scope")
        return {
            "schema": "inference_inventory_v1",
            "conditions": {
                c: [b[k] for k in sorted(b)] for c, b in sorted(self.conditions.items())
            },
            "mixed_precision": dtype or scope,
            "mixed_precision_dtype": dtype,
            "mixed_precision_scope": scope,
            "heterogeneous_hardware": different("GPU") or different("batch"),
            "incomplete_provenance": any(
                any(p.get(k) is None for k in FIELDS) for p in profiles
            ),
            "legacy_evidence": [self.evidence[k] for k in sorted(self.evidence)],
        }
