"""Freeze one common Fixed-best using development journals only; no inference."""

import argparse
from pathlib import Path

from .common import bind, csvwrite, digest, journal, read, sha, write
from .revision import load_plan
from sensor_v4_metrics import native_values


def rank_shared(dev):
    """Equal model weight after pooling valid pixels within each model."""
    models = sorted(dev)
    if len(models) != 2 or any(not dev[m] for m in models):
        raise ValueError("two complete model development cohorts required")
    keys = [{(r["pair_id"], r["condition"]) for r in dev[m]} for m in models]
    if keys[0] != keys[1] or any(len(keys[i]) != len(dev[m]) for i, m in enumerate(models)):
        raise ValueError("model cohorts differ or contain duplicates")
    ids = sorted(dev[models[0]][0]["candidate_statistics"])
    for rows in dev.values():
        for row in rows:
            if row["split"] != "dev":
                raise ValueError("selection accepts development observations only")
            if sorted(row["candidate_statistics"]) != ids:
                raise ValueError("candidate set mismatch")
    result = []
    for cid in ids:
        record = {"candidate_id": cid}
        scores = []
        for model in models:
            totals = {
                k: sum(r["candidate_statistics"][cid]["full"][k] for r in dev[model])
                for k in ("n", "abs_sum", "px1_count", "d1_count")
            }
            values = native_values(totals)
            scores.append(values)
            record[model + "_valid_pixels"] = totals["n"]
            for metric in ("D1", "Abs", "1px"):
                record[model + "_" + metric] = values[metric]
        for metric in ("D1", "Abs", "1px"):
            record["model_mean_" + metric] = sum(s[metric] for s in scores) / len(models)
        result.append(record)
    return sorted(result, key=lambda r: (r["model_mean_D1"], r["model_mean_Abs"], r["model_mean_1px"], r["candidate_id"]))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-tag", required=True)
    args = p.parse_args()
    out, plan, cfg, bank, freeze = load_plan(args.run_id)
    if Path(args.output_tag).name != args.output_tag or args.output_tag in (".", ".."):
        raise ValueError("plain output tag required")
    target = out / args.output_tag
    target.mkdir(exist_ok=False)
    sources, dev = [], {}
    expected = {(r["pair_id"], r["incoming_condition"]) for r in plan["dev_assignment"]["records"]}
    for model in ("raft", "acv"):
        folder = out / "fixed-dev" / model
        if read(folder / "COMPLETE.json")["observations"] != 328:
            raise ValueError("328 completed observations required")
        source = folder / "observations.jsonl"
        dev[model] = journal(source)
        if len(dev[model]) != 328 or {(r["pair_id"], r["condition"]) for r in dev[model]} != expected:
            raise ValueError("development assignment mismatch")
        sources.append({"model": model, "path": str(source), "sha256": sha(source)})
    spec = {
        "scope": "one shared candidate across both models, all conditions, all assignments",
        "model_weights": {"raft": 0.5, "acv": 0.5},
        "selection": "mean of model-wise pooled dev D1; tie mean Abs, mean 1px, lexical ID",
        "observations_per_model": 328,
        "sequences": 25,
        "heldout_used": False,
        "plan_sha256": digest(plan),
        "inputs": sources,
        "code_sha256": sha(Path(__file__)),
        "authorization": "User requested a single Fixed-best shared by both models, 2026-10-14",
    }
    bind(target, spec)
    ranking = rank_shared(dev)
    csvwrite(target / "dev_candidate_ranking.csv", ranking)
    result = {**spec, "candidate_id": ranking[0]["candidate_id"], "top_two": ranking[:2],
              "top_two_D1_gap_pp": ranking[1]["model_mean_D1"] - ranking[0]["model_mean_D1"]}
    write(target / "SHARED_FIXED_BEST.json", result)
    write(target / "COMPLETE.json", {"GPU_inference": False, "heldout_read": False, "candidate_id": result["candidate_id"]})
    print(target / "SHARED_FIXED_BEST.json", flush=True)
    print(result["candidate_id"], result["top_two"], flush=True)


if __name__ == "__main__":
    main()
