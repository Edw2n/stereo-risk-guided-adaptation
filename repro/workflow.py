"""CPU-only indexing, prediction layout and archived-score reporting."""
import csv
from pathlib import Path

from .common import ROOT, bind, csvwrite, journal, output, read, sha, write
from screening.official_foundation_io import CONDITIONS, observations, disp_path


def index(a):
    ids = {c["id"] for c in read(ROOT / "configs/challenge/bank8.json")["candidates"]}
    mapping, cohort = {}, None
    sources = {}
    for source in a.runs:
        source = source.resolve()
        spec = read(source / "RUN.json")["spec"]
        if spec.get("task") != "challenge-infer" or spec.get("preset") != "croco-07-fp16" or spec["condition"] != a.condition:
            raise ValueError("Index requires matching CroCo 0.7 candidate runs")
        cid = spec["candidate"]["id"]
        if cid not in ids or cid in mapping:
            raise ValueError("Duplicate or unknown candidate")
        rows = journal(source / "observations.jsonl")
        pairs = {r["pair_id"] for r in rows}
        if len(rows) != len(pairs) or pairs != set(spec["pairs"]) or not pairs:
            raise ValueError("Candidate run is incomplete")
        if cohort is not None and cohort != pairs:
            raise ValueError("Candidate cohorts differ")
        cohort = pairs
        for row in observations([a.condition]):
            if row["pair_id"] in pairs:
                p = source / "cache" / a.condition / row["scene"] / f'{row["frame"]}_{cid}.h5'
                if not p.is_file():
                    raise FileNotFoundError(p)
        mapping[cid] = str(source)
        sources[cid] = {"run": sha(source / "RUN.json"), "journal": sha(source / "observations.jsonl")}
    if set(mapping) != ids:
        raise ValueError("Exactly all eight frozen candidates are required")
    out = output(a.run_id)
    bind(out, {"task": "cache-index", "condition": a.condition, "sources": sources})
    write(out / "CACHE_INDEX.json", mapping)
    write(out / "COMPLETE.json", {"pairs": len(cohort), "candidates": len(ids), "inference": False})
    print(out)


def layout(a):
    """Link complete condition folders; no prediction copying or inference."""
    mapping = {}
    for root in a.sources:
        root = root.resolve()
        for condition in CONDITIONS:
            source = root / condition
            if not source.exists():
                continue
            if condition in mapping:
                raise ValueError("Duplicate condition: " + condition)
            expected = {disp_path(root, r, eye) for r in observations([condition]) for eye in ("left", "right")}
            actual = set(source.rglob("*.dsp5"))
            if expected != actual or any(not p.is_file() for p in expected):
                raise ValueError("Incomplete or extra prediction inventory: " + condition)
            # Read only the small metadata attribute. Array checks occur in delta/select.
            import h5py, json
            for path in expected:
                with h5py.File(path, "r") as f:
                    meta = json.loads(f.attrs["provenance_json"])
                if meta.get("method") != a.method or meta.get("condition") != condition:
                    raise ValueError("Mixed method/condition layout")
            mapping[condition] = str(source)
    if not mapping:
        raise ValueError("No condition folders found")
    out = output(a.run_id)
    bind(out, {"task": "layout", "method": a.method, "sources": mapping})
    target = out / "predictions"
    target.mkdir(exist_ok=True)
    for condition, source in mapping.items():
        link = target / condition
        if link.is_symlink() and link.resolve() == Path(source):
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(link)
        link.symlink_to(source, target_is_directory=True)
    write(out / "COMPLETE.json", {"method": a.method, "conditions": sorted(mapping), "inference": False})
    print(target)


def report(a):
    out = output(a.run_id)
    evidence = ROOT / "evidence/challenge"
    receipts = [evidence / d / "SERVER_RESULT.json" for d in
                ("croco_private_robust_A_20261001", "croco_tile_global_robust_B_20261003")]
    native = evidence / "croco_o07_tile_global_receive_20261002/condition_task_vs_identity_delta.csv"
    bind(out, {"task": "challenge-report", "sources": {str(p.relative_to(evidence)): sha(p) for p in [*receipts, native]}})
    rows, conditions = [], []
    for path in receipts:
        q = read(path)
        e0, delta = q["clean"]["Abs"], q["robust"]["delta_abs"]
        rows.append({"method": q["method"], "entry_id": q["entry_id"], "domain": "archived official server evaluation",
                     "clean_Abs": e0, "clean_1px": q["clean"]["1px"], "clean_D1": q["clean"]["D1"],
                     "delta_Abs": delta, "E0_plus_delta": e0 + delta,
                     "RbS_from_displayed_metrics": 0.5 * e0 / 3.4545 + 0.5 * (e0 + delta) / 18.4505,
                     "source_url": q["source_url"], "snapshot_utc": q["fetched_utc"]})
        conditions.extend({"method": q["method"], **c} for c in q["conditions"])
    csvwrite(out / "official_scores.csv", rows)
    csvwrite(out / "official_conditions.csv", conditions)
    with native.open(newline="") as f:
        native_rows = list(csv.DictReader(f))
    csvwrite(out / "native_disagreement.csv", native_rows)
    write(out / "COMPLETE.json", {"inference": False, "uploaded": False,
        "official_scores": "Archived server values; no local test ground truth or current ranking claim.",
        "native_disagreement": "Per-condition native prediction disagreement; distinct from subsampled server results.",
        "identity_clean": "No official overlap-0.7 Identity clean score; no Identity RbS is inferred."})
    print(out)


def run(a):
    {"index": index, "layout": layout, "report": report}[a.action](a)
