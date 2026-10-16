"""Read-only native-metric candidate-budget and consensus experiments."""

import argparse
import collections
import json
import time
from pathlib import Path

import h5py
import numpy as np

from .revision import (
    load_plan,
    subset_scores,
    consensus_scores,
    choose_consensus,
    CONDITIONS,
)
from .common import ROOT, append, bind, csvwrite, digest, journal, read, sha, write
from sensor_v4_metrics import bootstrap_weights, ci, native_values
from broad_sweep_v4 import select_scores
from sensor_v4_signals import select_without_gt, oracle_diagnostic

METRICS = ["Abs", "1px", "D1"]
KEYS = ["n", "abs_sum", "px1_count", "d1_count"]


def source_rows(evidence, research_root=None):
    out = []
    sources = []
    for s in (101, 202, 303):
        for m in ("raft", "acv"):
            p = (
                (
                    research_root
                    / "results"
                    / f"generic_joint_v1_heldout_seed{s}_20260926"
                    / m
                    / "observations.jsonl"
                )
                if research_root
                else evidence / "adaptation" / f"seed{s}" / f"{m}.jsonl"
            )
            rr = journal(p)
            if len(rr) != 172 or len({r["pair_id"] for r in rr}) != 172:
                raise ValueError("incomplete evidence")
            for r in rr:
                if r["model"] != m or r["assignment_seed"] != s:
                    raise ValueError("metadata mismatch")
            out.extend(rr)
            sources.append({"path": str(p), "sha256": sha(p)})
    return out, sources


def summarize(records, folder):
    """Pair sequence weights across methods and assignments; no pixel bootstrap."""
    scenes = sorted({r["scene"] for r in records})
    weights = bootstrap_weights(scenes, 1000, 1741)
    groups = collections.defaultdict(list)
    for r in records:
        groups[
            r["model"],
            r["proposal_seed"],
            r["candidate_count"],
            r["assignment_seed"],
            r["method"],
            r["condition"],
        ].append(r)
    vals = {}
    boots = {}
    for key, rr in groups.items():
        totals = np.array(
            [
                [sum(r["stats"][k] for r in rr if r["scene"] == s) for k in KEYS]
                for s in scenes
            ],
            np.float64,
        )

        def values(t):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.stack(
                    [
                        t[..., 1] / t[..., 0],
                        100 * t[..., 2] / t[..., 0],
                        100 * t[..., 3] / t[..., 0],
                    ],
                    axis=-1,
                )

        vals[key] = values(totals.sum(0))
        boots[key] = values(weights @ totals)
    prefixes = sorted({k[:-1] for k in vals})
    for prefix in prefixes:
        vals[(*prefix, "six_corruption_macro")] = np.mean(
            [vals[(*prefix, c)] for c in CONDITIONS], axis=0
        )
        boots[(*prefix, "six_corruption_macro")] = np.mean(
            [boots[(*prefix, c)] for c in CONDITIONS], axis=0
        )
    combinations = sorted({(k[0], k[1], k[2], k[4], k[5]) for k in vals})
    result = []
    for model, proposal, n, method, condition in combinations:
        points = np.stack(
            [vals[model, proposal, n, s, method, condition] for s in (101, 202, 303)]
        )
        bs = np.mean(
            [boots[model, proposal, n, s, method, condition] for s in (101, 202, 303)],
            axis=0,
        )
        row = {
            "model": model,
            "proposal_seed": proposal,
            "candidate_count": n,
            "method": method,
            "condition": condition,
        }
        for i, k in enumerate(METRICS):
            row[k] = float(points[:, i].mean())
            row[k + "_assignment_sd"] = float(points[:, i].std(ddof=1))
            row[k + "_ci_low"], row[k + "_ci_high"] = ci(bs[:, i])
        gain = np.stack(
            [
                vals[model, proposal, n, s, "Identity", condition]
                - vals[model, proposal, n, s, method, condition]
                for s in (101, 202, 303)
            ]
        )
        bg = np.mean(
            [
                boots[model, proposal, n, s, "Identity", condition]
                - boots[model, proposal, n, s, method, condition]
                for s in (101, 202, 303)
            ],
            axis=0,
        )
        for i, k in enumerate(METRICS):
            row[k + "_gain"] = float(gain[:, i].mean())
            row[k + "_gain_ci_low"], row[k + "_gain_ci_high"] = ci(bg[:, i])
        if all(
            (model, proposal, n, s, "Global Oracle", condition) in vals
            for s in (101, 202, 303)
        ):
            room = np.array(
                [
                    vals[model, proposal, n, s, "Identity", condition][2]
                    - vals[model, proposal, n, s, "Global Oracle", condition][2]
                    for s in (101, 202, 303)
                ]
            )
            broom = np.stack(
                [
                    boots[model, proposal, n, s, "Identity", condition][:, 2]
                    - boots[model, proposal, n, s, "Global Oracle", condition][:, 2]
                    for s in (101, 202, 303)
                ]
            )
            row["D1_global_oracle_headroom"] = float(room.mean())
            (
                row["D1_global_oracle_headroom_ci_low"],
                row["D1_global_oracle_headroom_ci_high"],
            ) = ci(broom.mean(0))
            if method in ("Signal-Global", "Consensus-Global"):
                ratios = np.divide(
                    gain[:, 2], room, out=np.full(3, np.nan), where=room > 1e-12
                )
                row["D1_global_recovery"] = (
                    float(ratios.mean()) if np.isfinite(ratios).all() else None
                )
                bgs = np.stack(
                    [
                        boots[model, proposal, n, s, "Identity", condition][:, 2]
                        - boots[model, proposal, n, s, method, condition][:, 2]
                        for s in (101, 202, 303)
                    ]
                )
                br = np.divide(
                    bgs, broom, out=np.full_like(bgs, np.nan), where=broom > 1e-12
                )
                row["D1_global_recovery_ci_low"], row["D1_global_recovery_ci_high"] = (
                    ci(br.mean(0))
                )
        if method == "Signal-Pixel Fusion":
            room = np.array(
                [
                    vals[model, proposal, n, s, "Identity", condition][2]
                    - vals[model, proposal, n, s, "Pixel Oracle", condition][2]
                    for s in (101, 202, 303)
                ]
            )
            broom = np.stack(
                [
                    boots[model, proposal, n, s, "Identity", condition][:, 2]
                    - boots[model, proposal, n, s, "Pixel Oracle", condition][:, 2]
                    for s in (101, 202, 303)
                ]
            )
            bgs = np.stack(
                [
                    boots[model, proposal, n, s, "Identity", condition][:, 2]
                    - boots[model, proposal, n, s, method, condition][:, 2]
                    for s in (101, 202, 303)
                ]
            )
            ratios = np.divide(
                gain[:, 2], room, out=np.full(3, np.nan), where=room > 1e-12
            )
            row["D1_pixel_oracle_headroom"] = float(room.mean())
            row["D1_pixel_recovery"] = (
                float(ratios.mean()) if np.isfinite(ratios).all() else None
            )
            br = np.divide(
                bgs, broom, out=np.full_like(bgs, np.nan), where=broom > 1e-12
            )
            row["D1_pixel_recovery_ci_low"], row["D1_pixel_recovery_ci_high"] = ci(
                br.mean(0)
            )
        result.append(row)
    csvwrite(folder / "native_metrics.csv", result)
    # Keep bank sampling variation separate from assignment variation and cluster CIs.
    variation = []
    for model in ("raft", "acv"):
        for n in sorted({r["candidate_count"] for r in result}):
            for method in sorted({r["method"] for r in result}):
                rr = [
                    r
                    for r in result
                    if r["model"] == model
                    and r["candidate_count"] == n
                    and r["method"] == method
                    and r["condition"] == "six_corruption_macro"
                ]
                if len(rr) < 2:
                    continue
                v = {
                    "model": model,
                    "candidate_count": n,
                    "method": method,
                    "proposal_seeds": len(rr),
                }
                for k in [
                    "Abs",
                    "1px",
                    "D1",
                    "D1_gain",
                    "D1_global_oracle_headroom",
                    "D1_global_recovery",
                ]:
                    x = [r[k] for r in rr if r.get(k) is not None]
                    v[k + "_mean"] = float(np.mean(x)) if x else None
                    v[k + "_proposal_sd"] = (
                        float(np.std(x, ddof=1)) if len(x) > 1 else None
                    )
                variation.append(v)
    csvwrite(folder / "proposal_variation.csv", variation)
    frequencies = collections.Counter(
        (
            r["model"],
            r["proposal_seed"],
            r["candidate_count"],
            r["method"],
            r["condition"],
            r.get("candidate_id"),
        )
        for r in records
        if r.get("candidate_id")
    )
    csvwrite(
        folder / "selection_frequency.csv",
        [
            dict(
                zip(
                    [
                        "model",
                        "proposal_seed",
                        "candidate_count",
                        "method",
                        "condition",
                        "candidate_id",
                    ],
                    key,
                ),
                count=count,
            )
            for key, count in sorted(frequencies.items())
        ],
    )
    return result


def plot(result, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=False)
    for ax, model, label in zip(axes, ("raft", "acv"), ("RAFT-Stereo", "ACVNet")):
        for method in ("Identity", "Random", "Signal-Global", "Global Oracle"):
            rr = sorted(
                [
                    r
                    for r in result
                    if r["model"] == model
                    and r["proposal_seed"] == 9137
                    and r["method"] == method
                    and r["condition"] == "six_corruption_macro"
                ],
                key=lambda r: r["candidate_count"],
            )
            if rr:
                ax.plot(
                    [r["candidate_count"] for r in rr],
                    [r["D1"] for r in rr],
                    "-o",
                    label=method,
                )
        ax.set_title(label)
        ax.set_xlabel("Candidates including Identity")
        ax.set_ylabel("D1 (%) ↓")
        ax.set_xticks([1, 4, 8, 16, 31])
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "candidate_budget.png", dpi=180)
    fig.savefig(folder / "candidate_budget.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--evidence", type=Path, default=ROOT / "evidence/camera_ready")
    p.add_argument(
        "--stage", choices=["budget", "consensus", "conditions"], required=True
    )
    p.add_argument("--research-root", type=Path)
    p.add_argument("--output-tag")
    a = p.parse_args()
    out, plan, cfg, bank, freeze = load_plan(a.run_id)
    tag = a.output_tag or a.stage
    if Path(tag).name != tag or tag in (".", ".."):
        raise ValueError("plain output tag required")
    folder = out / "offline" / tag
    folder.mkdir(parents=True, exist_ok=True)
    rows, sources = source_rows(a.evidence, a.research_root)
    bind(
        folder,
        {
            "plan_sha256": digest(plan),
            "sources": sources,
            "stage": a.stage,
            "code_sha256": sha(Path(__file__)),
            "selector_code_sha256": sha(ROOT / "repro/revision.py"),
        },
    )
    if (folder / "COMPLETE.json").exists():
        print("already complete", folder)
        return
    start = time.monotonic()
    records = []
    native31_mismatches = 0
    done = journal(folder / "choices.jsonl") if a.stage == "consensus" else []
    cached = {(r["model"], r["assignment_seed"], r["pair_id"]): r for r in done}
    for i, row in enumerate(rows):
        model, pair, c, s = (
            row["model"],
            row["pair_id"],
            row["condition"],
            row["assignment_seed"],
        )
        rec = {
            "model": model,
            "pair_id": pair,
            "incoming_condition": c,
            "primary_signal": freeze["primary_signals"][model],
        }

        def add(method, cid, proposal, n, stats=None):
            records.append(
                {
                    "model": model,
                    "pair_id": pair,
                    "scene": row["scene"],
                    "condition": c,
                    "assignment_seed": s,
                    "proposal_seed": proposal,
                    "candidate_count": n,
                    "method": method,
                    "candidate_id": cid,
                    "stats": (
                        row["candidate_statistics"][cid]["full"]
                        if stats is None
                        else stats
                    ),
                }
            )

        if a.stage == "budget":
            for proposal, subsets in plan["subsets"].items():
                for n, subset in subsets.items():
                    ids = sorted(subset)
                    scores = subset_scores(row["pairwise_rank_counts"], ids)
                    picked = select_scores(scores, rec, cfg)["candidate_id"]
                    oracle = oracle_diagnostic(
                        {
                            cid: native_values(row["candidate_statistics"][cid]["full"])
                            for cid in ids
                        }
                    )
                    random = select_without_gt(ids, {}, "Random", None, rec, cfg)
                    if len(ids) == 31 and (
                        picked != row["picks"]["Signal-Global"]
                        or oracle != row["picks"]["Global Oracle"]
                        or random != row["picks"]["Random"]
                    ):
                        native31_mismatches += 1
                    for method, cid in [
                        ("Identity", "identity"),
                        ("Random", random),
                        ("Signal-Global", picked),
                        ("Global Oracle", oracle),
                    ]:
                        add(method, cid, int(proposal), int(n))
        elif a.stage == "consensus":
            key = (model, s, pair)
            if key not in cached:
                q = row["pool"]
                with h5py.File(q["path"], "r") as f:
                    g = f[q["group"]]
                    ids = json.loads(g.attrs["candidate_ids"])
                    # No risk, GT validity, D1, or error is read by the selector.
                    ds = g["disparity"][:]
                    if ds.shape != (31, 50000):
                        raise ValueError("wrong candidate pool")
                    scores = consensus_scores(ds, ids)
                    coord_hash = (
                        __import__("hashlib")
                        .sha256(g["coordinates"][:].tobytes())
                        .hexdigest()
                    )
                cid = choose_consensus(
                    scores, pair, c, model, plan["consensus"]["tie_seed"]
                )
                choice = {
                    "model": model,
                    "assignment_seed": s,
                    "pair_id": pair,
                    "condition": c,
                    "candidate_id": cid,
                    "scores": scores,
                    "coordinates_sha256": coord_hash,
                    "GT_used": False,
                }
                append(folder / "choices.jsonl", choice)
                cached[key] = choice
            cid = cached[key]["candidate_id"]
            add("Consensus-Global", cid, 0, 31)
            for method in ("Identity", "Signal-Global", "Global Oracle"):
                add(method, row["picks"][method], 0, 31)
        else:
            for m in row["methods"]:
                add(m["method"], m["candidate_id"], 0, 31, m["full"])
        if (i + 1) % 32 == 0:
            print(
                a.stage,
                i + 1,
                "/",
                len(rows),
                round(time.monotonic() - start, 1),
                flush=True,
            )
    if native31_mismatches:
        raise AssertionError(f"31-candidate regression mismatch: {native31_mismatches}")
    if a.stage == "budget":
        csvwrite(
            folder / "choices.csv",
            [{k: v for k, v in r.items() if k != "stats"} for r in records],
        )
    result = summarize(records, folder)
    if a.stage == "budget":
        plot(result, folder)
    write(
        folder / "COMPLETE.json",
        {
            "observations": len(rows),
            "new_inference": False,
            "native31_mismatches": native31_mismatches,
            "wall_seconds": time.monotonic() - start,
            "bootstrap": {"unit": "sequence", "replicates": 1000, "seed": 1741},
            "support": "full native metrics; consensus scoring uses 50k geometry coordinates",
            "no_signal_bank_reselection": True,
        },
    )
    print(folder, flush=True)


if __name__ == "__main__":
    main()
