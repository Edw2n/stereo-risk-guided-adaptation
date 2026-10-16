"""Read-only cache compatibility. Legacy gaps require explicit hashed evidence."""

import json
from pathlib import Path
import h5py
import numpy as np
from .common import read, sha, digest

FIELDS = (
    "checkpoint_sha256",
    "upstream_commit",
    "overlap",
    "precision",
    "precision_scope",
    "GPU",
    "batch",
)


def normalized(key, value):
    return value.lower() if key == "precision" and isinstance(value, str) else value


def load_index(folder, ids):
    folder = Path(folder).resolve()
    path = folder / "CACHE_INDEX.json"
    mapping = read(path) if path.exists() else {c: str(folder) for c in ids}
    if set(mapping) != set(ids):
        raise ValueError("cache index must contain exactly the frozen candidate IDs")
    entries = {}
    for cid, raw in mapping.items():
        entry = (
            {"root": raw, "layout": "native_v1"} if isinstance(raw, str) else dict(raw)
        )
        root = Path(entry["root"]).expanduser()
        entry["root"] = str((root if root.is_absolute() else folder / root).resolve())
        if entry.get("layout", "native_v1") not in ("native_v1", "legacy_v1"):
            raise ValueError("unknown cache layout")
        if entry.get("layout") == "legacy_v1":
            e = entry.get("evidence")
            if not e or not e.get("sha256"):
                raise ValueError(
                    "legacy cache requires a hash-pinned evidence attestation"
                )
            p = Path(e["path"])
            p = p if p.is_absolute() else folder / p
            if sha(p) != e["sha256"]:
                raise ValueError("legacy evidence hash mismatch")
            att = read(p)
            if att.get("schema") != "legacy_cache_attestation_v1":
                raise ValueError("unknown legacy attestation schema")
            if not {"config_sha256", "checkpoint_sha256", "condition"} <= set(
                att["cache_match"]
            ):
                raise ValueError(
                    "attestation must bind cache config, checkpoint and condition"
                )
            if not all(
                k in att["inference"] and att["inference"][k] is not None
                for k in FIELDS
            ):
                raise ValueError("incomplete attested inference settings")
            if not att.get("sources") or not att.get("rationale"):
                raise ValueError(
                    "legacy attestation needs evidence sources and rationale"
                )
            sources = []
            for src in att["sources"]:
                sp = Path(src["path"])
                sp = sp if sp.is_absolute() else p.parent / sp
                if sha(sp) != src["sha256"]:
                    raise ValueError("legacy source evidence hash mismatch")
                sources.append({"path": str(sp.resolve()), "sha256": src["sha256"]})
            entry["attestation"] = att
            entry["evidence_receipt"] = {
                "path": str(p.resolve()),
                "sha256": e["sha256"],
                "sources": sources,
                "rationale": att["rationale"],
                "status": "explicit execution-record attestation; never inferred from prediction values",
            }
        entries[cid] = entry
    return entries


def read_candidate(entry, row, cid, signals, shape=(2, 1080, 1920)):
    root = Path(entry["root"])
    if entry.get("layout", "native_v1") == "legacy_v1":
        path = (
            root
            / "cache"
            / row["condition"]
            / cid
            / row["scene"]
            / f"{row['frame']}.h5"
        )
    else:
        path = (
            root
            / "cache"
            / row["condition"]
            / row["scene"]
            / f"{row['frame']}_{cid}.h5"
        )
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["provenance_json"])
        for k, v in {
            "pair_id": row["pair_id"],
            "condition": row["condition"],
            "candidate_id": cid,
        }.items():
            if meta.get(k) != v:
                raise ValueError("cache pair/candidate mismatch: " + k)
        original_metadata_sha256 = digest(meta)
        arrays = {}
        array_hashes = {}
        for k in ["disparity", *signals]:
            ds = f[k]
            x = ds[:]
            if x.dtype != np.float32 or x.shape != shape or not np.isfinite(x).all():
                raise ValueError("cache shape/dtype/nonfinite: " + k)
            from screening.official_foundation_io import arr_sha

            if ds.attrs.get("array_sha256") != arr_sha(x):
                raise ValueError("cache array content hash mismatch: " + k)
            arrays[k] = x
            array_hashes[k] = ds.attrs["array_sha256"]
    filled = []
    if entry.get("layout") == "legacy_v1":
        att = entry["attestation"]
        for k, v in att["cache_match"].items():
            if meta.get(k) != v:
                raise ValueError("legacy evidence does not match cache: " + k)
        for k in FIELDS:
            attested = normalized(k, att["inference"][k])
            existing = meta.get(
                k, meta.get("effective_reference_batch_size") if k == "batch" else None
            )
            if existing is not None and normalized(k, existing) != attested:
                raise ValueError("legacy attestation contradicts cache: " + k)
            if existing is None:
                filled.append(k)
            meta[k] = attested
    for k in FIELDS:
        if k not in meta or meta[k] is None:
            raise ValueError("cache missing inference provenance: " + k)
        meta[k] = normalized(k, meta[k])
    receipt = {
        "path": str(path),
        "layout": entry.get("layout", "native_v1"),
        "supplemented_fields": filled,
        "original_metadata_sha256": original_metadata_sha256,
        "array_sha256": array_hashes,
    }
    if "evidence_receipt" in entry:
        receipt["evidence"] = entry["evidence_receipt"]
    return arrays, meta, receipt
