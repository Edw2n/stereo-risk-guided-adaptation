"""Report completed camera-ready stages only; never starts inference."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .common import ROOT, bind, csvwrite, digest, journal, read, sha, write
from .revision import load_plan, consensus_scores, choose_consensus, CONDITIONS
from .revision_offline import source_rows, summarize
from sensor_v4_metrics import bootstrap_weights, ci, table1_summary, native_values


def full_row(original, method, cid=None, stats=None):
    if stats is None:
        stats = original["candidate_statistics"][cid]["full"]
    return {
        "model": original["model"],
        "pair_id": original["pair_id"],
        "scene": original["scene"],
        "condition": original["condition"],
        "assignment_seed": original["assignment_seed"],
        "proposal_seed": 0,
        "candidate_count": 31,
        "method": method,
        "candidate_id": cid,
        "stats": stats,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--research-root", type=Path)
    p.add_argument("--evidence", type=Path, default=ROOT / "evidence/camera_ready")
    p.add_argument("--output-tag", required=True)
    p.add_argument("--shared-fixed-best", type=Path)
    a = p.parse_args()
    out, plan, cfg, bank, freeze = load_plan(a.run_id)
    if Path(a.output_tag).name != a.output_tag or a.output_tag in (".", ".."):
        raise ValueError("plain tag")
    target = out / "reports" / a.output_tag
    target.mkdir(parents=True, exist_ok=False)
    shared = read(a.shared_fixed_best) if a.shared_fixed_best else None
    fixed_method = "Fixed-best (shared)" if shared else "Fixed-best"
    if shared:
        if (shared["heldout_used"] or shared["observations_per_model"] != 328
                or shared["plan_sha256"] != digest(plan)
                or shared["model_weights"] != {"raft": 0.5, "acv": 0.5}
                or shared["candidate_id"] not in {c["id"] for c in bank["candidates"]}):
            raise ValueError("invalid shared development freeze")
        for source in shared["inputs"]:
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("shared dev source changed")
    old, sources = source_rows(a.evidence, a.research_root)
    if shared:
        sources.append({"path": str(a.shared_fixed_best), "sha256": sha(a.shared_fixed_best)})
    consensus_file = out / "offline/consensus/choices.jsonl"
    consensus_complete = consensus_file.parent / "COMPLETE.json"
    consensus_rows = journal(consensus_file) if consensus_complete.exists() else []
    consensus = {
        (r["model"], r["assignment_seed"], r["pair_id"]): r["candidate_id"]
        for r in consensus_rows
    }
    if len(consensus) != len(consensus_rows):
        raise ValueError("duplicate consensus observations")
    if consensus_rows:
        sources.extend(
            {"path": str(f), "sha256": sha(f)}
            for f in (consensus_file, consensus_complete)
        )
    available = {}
    fixed = {}
    denoise = {}
    clean = {}
    for m in ("raft", "acv"):
        available[m] = {
            s: (out / s / m / "COMPLETE.json").exists() for s in plan["stages"]
        }
        if available[m]["fixed-dev"]:
            f = out / "fixed-dev" / m / "FIXED_BEST.json"
            fixed[m] = read(f)
            if fixed[m]["heldout_used"] or fixed[m]["observations"] != 328:
                raise ValueError("dev freeze invalid")
            sources.append({"path": str(f), "sha256": sha(f)})
            if shared:
                fixed[m] = {"candidate_id": shared["candidate_id"], "heldout_used": False,
                            "observations": 328, "scope": shared["scope"],
                            "shared_freeze_sha256": sha(a.shared_fixed_best)}
        for stage, store in [("denoise", denoise), ("clean", clean)]:
            if available[m][stage]:
                f = out / stage / m / "observations.jsonl"
                store[m] = journal(f)
                sources.append({"path": str(f), "sha256": sha(f)})
    bind(
        target,
        {
            "plan_sha256": digest(plan),
            "inputs": sources,
            "available": available,
            "code_sha256": sha(Path(__file__)),
        },
    )
    records = []
    for row in old:
        m = row["model"]
        for method in row["methods"]:
            records.append(
                full_row(row, method["method"], method["candidate_id"], method["full"])
            )
        key = m, row["assignment_seed"], row["pair_id"]
        if key in consensus:
            records.append(full_row(row, "Consensus-Global", consensus[key]))
        if m in fixed:
            records.append(full_row(row, fixed_method, fixed[m]["candidate_id"]))
        if m in denoise:
            rr = next(
                r
                for r in denoise[m]
                if r["pair_id"] == row["pair_id"] and r["condition"] == row["condition"]
            )
            records.append(
                full_row(row, "Gaussian-Denoise", stats=rr["methods"][0]["full"])
            )
    # Incomplete consensus is not silently reported on fewer observations.
    for m in ("raft", "acv"):
        n = sum(r["method"] == "Consensus-Global" and r["model"] == m for r in records)
        if n not in (0, 516):
            records = [
                r
                for r in records
                if not (r["method"] == "Consensus-Global" and r["model"] == m)
            ]
    summary = summarize(records, target)
    # Paired incremental benefit over each baseline, not subtraction of CI endpoints.
    paired = []
    scenes = sorted({r["scene"] for r in records})
    weights = bootstrap_weights(scenes, 1000, 1741)
    for model in ("raft", "acv"):
        methods = sorted({r["method"] for r in records if r["model"] == model})
        for baseline in (
            "Identity",
            "Random",
            "Consensus-Global",
            fixed_method,
            "Gaussian-Denoise",
        ):
            if baseline not in methods:
                continue
            all_points, all_boots = [], []
            for assignment in (101, 202, 303):
                pp, bb = [], []
                for condition in CONDITIONS:
                    values, boot = {}, {}
                    for method in (baseline, "Signal-Global"):
                        rr = [
                            {"scene": r["scene"], "full": r["stats"]}
                            for r in records
                            if r["model"] == model
                            and r["assignment_seed"] == assignment
                            and r["condition"] == condition
                            and r["method"] == method
                        ]
                        values[method], boot[method] = table1_summary(
                            rr, scenes, weights, "full"
                        )
                    pp.append(
                        [
                            values[baseline][k] - values["Signal-Global"][k]
                            for k in ("Abs", "1px", "D1")
                        ]
                    )
                    bb.append(boot[baseline] - boot["Signal-Global"])
                all_points.append(np.vstack([pp, np.mean(pp, axis=0)]))
                all_boots.append(np.stack(bb + [np.mean(bb, axis=0)]))
            point = np.mean(all_points, axis=0)
            bs = np.mean(all_boots, axis=0)
            for i, condition in enumerate(CONDITIONS + ["six_corruption_macro"]):
                row = {
                    "model": model,
                    "baseline": baseline,
                    "condition": condition,
                    "positive_means": "Signal-Global has lower task error",
                }
                for j, metric in enumerate(("Abs", "1px", "D1")):
                    row[metric + "_advantage"] = float(point[i, j])
                    row[metric + "_ci_low"], row[metric + "_ci_high"] = ci(bs[i, :, j])
                paired.append(row)
    csvwrite(target / "signal_vs_baselines.csv", paired)
    clean_summary = []
    clean_choices = []
    for m, rows in clean.items():
        if len(rows) != 172 or len({r["pair_id"] for r in rows}) != 172:
            raise ValueError("incomplete clean")
        scenes = sorted({r["scene"] for r in rows})
        weights = bootstrap_weights(scenes, 1000, 1741)
        chosen = {}
        for r in rows:
            mm = {
                v["method"]: v["full"] for v in r["methods"] if v["method"] != "Clean"
            }
            if m in fixed:
                mm[fixed_method] = r["candidate_statistics"][fixed[m]["candidate_id"]][
                    "full"
                ]
            q = r["pool"]
            with h5py.File(q["path"], "r") as f:
                g = f[q["group"]]
                ids = json.loads(g.attrs["candidate_ids"])
                scores = consensus_scores(g["disparity"][:], ids)
            cid = choose_consensus(
                scores, r["pair_id"], "clean", m, plan["consensus"]["tie_seed"]
            )
            mm["Consensus-Global"] = r["candidate_statistics"][cid]["full"]
            if m in denoise:
                rr = next(
                    x
                    for x in denoise[m]
                    if x["pair_id"] == r["pair_id"] and x["condition"] == "clean"
                )
                mm["Gaussian-Denoise"] = rr["methods"][0]["full"]
            chosen[r["pair_id"]] = mm
            clean_choices.append(
                {
                    "model": m,
                    "pair_id": r["pair_id"],
                    "condition": "clean",
                    "consensus_candidate": cid,
                    "fixed_best_candidate": fixed.get(m, {}).get("candidate_id"),
                    "GT_used_by_selectors": False,
                }
            )
        points, boots = {}, {}
        for method in next(iter(chosen.values())):
            rr = [
                {"scene": r["scene"], "full": chosen[r["pair_id"]][method]}
                for r in rows
            ]
            points[method], boots[method] = table1_summary(rr, scenes, weights, "full")
        for method, q in points.items():
            row = {
                "model": m,
                "condition": "clean",
                "method": method,
                "observations": 172,
                "assignments": 1,
            }
            for j, k in enumerate(("Abs", "1px", "D1")):
                row[k] = q[k]
                row[k + "_ci_low"], row[k + "_ci_high"] = ci(boots[method][:, j])
                row[k + "_gain"] = points["Identity"][k] - q[k]
                row[k + "_gain_ci_low"], row[k + "_gain_ci_high"] = ci(
                    boots["Identity"][:, j] - boots[method][:, j]
                )
                row[k + "_fraction_observations_worse"] = float(
                    np.mean(
                        [
                            native_values(mm[method])[k]
                            > native_values(mm["Identity"])[k]
                            for mm in chosen.values()
                        ]
                    )
                )
            clean_summary.append(row)
    csvwrite(target / "clean_preservation.csv", clean_summary)
    csvwrite(target / "clean_choices.csv", clean_choices)
    write(target / "FIXED_BESTS.json", fixed)
    write(
        target / "COMPLETE.json",
        {
            "available": available,
            "GPU_inference": False,
            "clean_repeated_across_assignments": False,
            "historical_outputs_modified": False,
            "frozen_signal_or_bank_changed": False,
            "fixed_best_scope": "shared across models" if shared else "model-specific",
        },
    )
    print(target, flush=True)


if __name__ == "__main__":
    main()
