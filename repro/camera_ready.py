"""Reproduce the final paper tables and plot previews from saved evidence."""
import csv
from pathlib import Path

import numpy as np

from .common import ROOT, bind, csvwrite, journal, output, read, sha, write
from .paper import CONDITIONS, METRICS, report_context, summarize_adaptation
from .revision import choose_consensus
from .revision_offline import source_rows, summarize
from .revision_report import full_row
from .revision_shared_fixed import rank_shared
from sensor_v4_metrics import bootstrap_weights, ci, table1_summary


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def paired_comparisons(records):
    """Use common sequence weights, not differences between marginal CI limits."""
    scenes = sorted({r["scene"] for r in records})
    weights = bootstrap_weights(scenes, 1000, 1741)
    rows = []
    for model in ("raft", "acv"):
        for baseline in ("Identity", "Random", "Fixed-best (shared)", "Consensus-Global", "Gaussian-Denoise"):
            points, boots = [], []
            for seed in (101, 202, 303):
                pp, bb = [], []
                for condition in CONDITIONS:
                    values, draws = {}, {}
                    for method in (baseline, "Signal-Global"):
                        rr = [{"scene": r["scene"], "full": r["stats"]} for r in records
                              if (r["model"], r["assignment_seed"], r["condition"], r["method"])
                              == (model, seed, condition, method)]
                        if not rr:
                            raise ValueError("Incomplete method cohort")
                        values[method], draws[method] = table1_summary(rr, scenes, weights, "full")
                    pp.append([values[baseline][m] - values["Signal-Global"][m] for m in METRICS])
                    bb.append(draws[baseline] - draws["Signal-Global"])
                points.append(np.vstack([pp, np.mean(pp, axis=0)]))
                boots.append(np.stack(bb + [np.mean(bb, axis=0)]))
            point, boot = np.mean(points, axis=0), np.mean(boots, axis=0)
            for i, condition in enumerate([*CONDITIONS, "six_corruption_macro"]):
                row = {"model": model, "baseline": baseline, "condition": condition,
                       "positive_means": "Signal-Global has lower task error"}
                for j, metric in enumerate(METRICS):
                    row[metric + "_advantage"] = float(point[i, j])
                    row[metric + "_ci_low"], row[metric + "_ci_high"] = ci(boot[i, :, j])
                rows.append(row)
    return rows


def clean_summary(evidence, fixed, denoise):
    choices = read_csv(evidence / "clean_choices.csv")
    by_pair = {(r["model"], r["pair_id"]): r for r in choices}
    if len(by_pair) != 344 or len(choices) != 344:
        raise ValueError("Expected 172 clean choices for each model")
    result = []
    for model in ("raft", "acv"):
        rows = journal(evidence / "clean" / f"{model}.jsonl")
        if len(rows) != 172 or len({r["pair_id"] for r in rows}) != 172:
            raise ValueError("Incomplete clean cohort")
        scenes = sorted({r["scene"] for r in rows})
        weights = bootstrap_weights(scenes, 1000, 1741)
        methods = {}
        for r in rows:
            if r["split"] != "test" or r["condition"] != "clean":
                raise ValueError("Unexpected clean evaluation domain")
            mm = {m["method"]: m["full"] for m in r["methods"] if m["method"] != "Clean"}
            choice = by_pair[model, r["pair_id"]]
            if choice["fixed_best_candidate"] != fixed:
                raise ValueError("Clean Fixed-best differs from the shared development choice")
            mm["Fixed-best (shared)"] = r["candidate_statistics"][fixed]["full"]
            mm["Consensus-Global"] = r["candidate_statistics"][choice["consensus_candidate"]]["full"]
            mm["Gaussian-Denoise"] = denoise[model, r["pair_id"], "clean"]
            for method, stats in mm.items():
                methods.setdefault(method, []).append({"scene": r["scene"], "full": stats})
        points, boots = {}, {}
        for method, rr in methods.items():
            points[method], boots[method] = table1_summary(rr, scenes, weights, "full")
        for method, values in points.items():
            row = {"model": model, "condition": "clean", "method": method,
                   "observations": 172, "assignments": 1}
            for j, metric in enumerate(METRICS):
                row[metric] = values[metric]
                row[metric + "_ci_low"], row[metric + "_ci_high"] = ci(boots[method][:, j])
                row[metric + "_gain"] = points["Identity"][metric] - values[metric]
                row[metric + "_gain_ci_low"], row[metric + "_gain_ci_high"] = ci(boots["Identity"][:, j] - boots[method][:, j])
            result.append(row)
    return result


def table(rows, target):
    lookup = {(r["model"], r["method"]): r for r in rows if r["condition"] == "six_corruption_macro"}
    methods = ["Clean", "Identity", "Random", "Fixed-best (shared)", "Signal-Global", "Signal-Pixel Fusion", "Global Oracle", "Pixel Oracle"]
    text = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
            r" & & Reference & \multicolumn{3}{c}{Baselines} & \multicolumn{2}{c}{Adaptation} & \multicolumn{2}{c}{Oracles} \\",
            r"Model & Metric $\downarrow$ & Source / Clean & Identity & Random & Fixed-best & Signal-Global & Signal-Pixel & Global & Pixel \\", r"\midrule"]
    for model, title in (("raft", "RAFT-Stereo"), ("acv", "ACVNet")):
        for metric, label in zip(METRICS, ("Abs (px)", r"1px (\%)", r"D1 (\%)")):
            text.append(" & ".join([title if metric == "Abs" else "", label + r" $\downarrow$",
                                    *[f"{lookup[model, m][metric]:.3f}" for m in methods]]) + r" \\")
    text += [r"\bottomrule", r"\end{tabular}"]
    target.write_text("\n".join(text) + "\n")


def run(a):
    evidence = a.evidence.resolve()
    context = getattr(a, "context_evidence", ROOT / "evidence/paper").resolve()
    context_paths = [
        "phase_a/raft.jsonl", "phase_a/acv.jsonl",
        "sensor_v4_figure_c_layers_dev_20260926_r2/figure_c_proxy_layers_compact.csv",
        "sensor_v4_all_proxy_response_dev_20260926/all_proxy_condition_response.csv",
        "sensor_v4_all_proxy_response_dev_20260926/condition_task_metrics.csv",
    ]
    missing = [str(context / name) for name in context_paths if not (context / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing paper report context: " + ", ".join(missing))
    sources = sorted([*evidence.rglob("*.jsonl"), evidence / "clean_choices.csv"])
    out = output(a.run_id)
    bind(out, {"task": "paper-report", "evidence": str(evidence),
               "sources": {str(p.relative_to(evidence)): sha(p) for p in sources},
               "context_evidence": str(context),
               "context_sources": {name: sha(context / name) for name in context_paths}})
    dev = {m: journal(evidence / "fixed-dev" / f"{m}.jsonl") for m in ("raft", "acv")}
    expected = {(r["pair_id"], r["incoming_condition"]) for r in
                read(ROOT / "configs/paper/assignments/dev328.json")["records"]}
    if any(len(rr) != 328 or {(r["pair_id"], r["condition"]) for r in rr} != expected for rr in dev.values()):
        raise ValueError("Incomplete fixed development assignment")
    ranking = rank_shared(dev)
    fixed = ranking[0]["candidate_id"]
    frozen = read(ROOT / "configs/revision/shared_fixed_best_20261014.json")
    if fixed != frozen["candidate_id"]:
        raise ValueError("Development evidence no longer reproduces the frozen shared candidate")
    csvwrite(out / "dev_candidate_ranking.csv", ranking)
    write(out / "shared_fixed_best.json", {"candidate_id": fixed, "selection": frozen["selection"], "heldout_used": False})
    denoise = {}
    for model in ("raft", "acv"):
        for r in journal(evidence / "denoise" / f"{model}.jsonl"):
            key = model, r["pair_id"], r["condition"]
            if key in denoise:
                raise ValueError("Duplicate denoising observation")
            denoise[key] = r["methods"][0]["full"]
    consensus = {}
    for r in journal(evidence / "consensus.jsonl"):
        key = r["model"], r["assignment_seed"], r["pair_id"]
        picked = choose_consensus(r["scores"], r["pair_id"], r["condition"], r["model"])
        if key in consensus or r["GT_used"] or picked != r["candidate_id"]:
            raise ValueError("Invalid or changed consensus selection")
        consensus[key] = picked
    old, _ = source_rows(evidence)
    if len(consensus) != len(old):
        raise ValueError("Incomplete consensus cohort")
    records = []
    for r in old:
        for m in r["methods"]:
            records.append(full_row(r, m["method"], m["candidate_id"], m["full"]))
        records.append(full_row(r, "Fixed-best (shared)", fixed))
        records.append(full_row(r, "Consensus-Global", consensus[r["model"], r["assignment_seed"], r["pair_id"]]))
        records.append(full_row(r, "Gaussian-Denoise", stats=denoise[r["model"], r["pair_id"], r["condition"]]))
    summary = summarize(records, out)
    csvwrite(out / "signal_vs_baselines.csv", paired_comparisons(records))
    csvwrite(out / "clean_preservation.csv", clean_summary(evidence, fixed, denoise))
    table(summary, out / "adaptation.tex")
    from .plots import adaptation_plot

    adaptation_plot(summary, out)
    report_context(context, out)
    # Keep derived headroom/recovery intervals without emitting an older adaptation table.
    _, d1_summary, _, per_seed_d1 = summarize_adaptation(evidence)
    for name, rows in (("three_seed_D1", d1_summary), ("per_seed_D1", per_seed_d1)):
        csvwrite(out / (name + ".csv"), rows)
        write(out / (name + ".json"), rows)
    write(out / "COMPLETE.json", {"inference": False, "observations": len(old), "assignment_seeds": [101, 202, 303],
        "shared_fixed_best": fixed,
        "adaptation_metrics": "native_metrics.csv",
        "proxy_previews": "Redrawn from saved summaries; pixel metrics require paper table2/samepixel caches.",
        "assignment_variation": "Standard deviation across three assignments; separate from sequence-cluster intervals.",
        "clean_consensus": "Native metrics reaggregated using saved candidate choices; reranking needs saved disparity pools.",
        "corrupt_consensus": "Choices recomputed from saved per-candidate scores; score extraction needs saved disparity pools.",
        "CI": "Paired sequence-cluster percentile intervals, 1000 draws, seed 1741"})
    print(out)
