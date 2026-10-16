"""Verify the complete distributed source tree without importing torch."""

import hashlib
from pathlib import Path
from .common import ROOT, read, sha


def verify_vendor(root=ROOT):
    root = Path(root)
    m = read(root / "provenance/dinov2_vendor.json")
    tree = root / "vendor/dinov2"
    expected = {x["path"]: x["sha256"] for x in m["files"]}
    actual = {
        str(p.relative_to(tree)): sha(p)
        for p in tree.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    if actual != expected:
        raise ValueError(
            f"vendor inventory/hash mismatch: missing={sorted(expected.keys()-actual.keys())}, "
            f"extra={sorted(actual.keys()-expected.keys())}, "
            f"changed={[k for k in expected.keys() & actual.keys() if expected[k] != actual[k]]}"
        )
    h = hashlib.sha256()
    for path, value in sorted(actual.items()):
        h.update(path.encode() + b"\0" + value.encode() + b"\n")
    if h.hexdigest() != m["tree_sha256"]:
        raise ValueError("vendor tree hash mismatch")
    return {"files": len(actual), "tree_sha256": h.hexdigest(), "status": "PASS"}
