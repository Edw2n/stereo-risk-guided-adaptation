import json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import (
    ROOT,
    read,
    write,
    csvwrite,
    sha,
    digest,
    journal,
    append,
    profile,
    output,
    bind,
    gpu_lock,
)
from sensor_v4_metrics import table1_summary, bootstrap_weights, ci, native_values
from report_generic_heldout_correction_v2 import METHODS, NAMES, quantities

METRICS = ["Abs", "1px", "D1"]
CONDITIONS = [
    "brightness",
    "contrast",
    "saturation",
    "gaussian_noise",
    "shot_noise",
    "speckle_noise",
]


def summarize_adaptation(evidence):
    seeds = [101, 202, 303]
    raw = {
        (s, m): journal(evidence / "adaptation" / f"seed{s}" / f"{m}.jsonl")
        for s in seeds
        for m in ["raft", "acv"]
    }
    scenes = sorted({r["scene"] for rows in raw.values() for r in rows})
    weights = bootstrap_weights(scenes, 1000, 1741)
    per = []
    derived = []
    boots = {}
    db = {}
    for (seed, model), rows in raw.items():
        if len(rows) != 172 or len({r["pair_id"] for r in rows}) != 172:
            raise ValueError("final report needs 172 unique pairs per seed/model")
        assignment = {
            r["pair_id"]: r["incoming_condition"]
            for r in read(ROOT / "configs/paper/assignments" / f"seed{seed}.json")[
                "records"
            ]
        }
        if any(
            r["split"] != "test" or r["condition"] != assignment[r["pair_id"]]
            for r in rows
        ):
            raise ValueError("assignment mismatch")
        vals = {}
        for method in METHODS:
            pp = []
            bb = []
            for cond in CONDITIONS:
                sub = [
                    {
                        "scene": r["scene"],
                        "full": next(
                            x["full"] for x in r["methods"] if x["method"] == method
                        ),
                    }
                    for r in rows
                    if r["condition"] == cond
                ]
                q, b = table1_summary(sub, scenes, weights, "full")
                pp.append([q[k] for k in METRICS])
                bb.append(b)
            vals[method] = np.mean(pp, axis=0)
            boots[seed, model, method] = np.mean(bb, axis=0)
            per.append(
                {
                    "seed": seed,
                    "model": model,
                    "method": method,
                    **dict(zip(METRICS, vals[method])),
                }
            )
        order = [
            "Identity",
            "Signal-Global",
            "Global Oracle",
            "Signal-Pixel Fusion",
            "Pixel Oracle",
        ]
        qq = quantities(*[vals[m][2] for m in order])
        bs = quantities(*[boots[seed, model, m][:, 2] for m in order])
        for name, value, b in zip(NAMES, qq, bs):
            db[seed, model, name] = b
            lo, hi = ci(b)
            derived.append(
                {
                    "seed": seed,
                    "model": model,
                    "quantity": name,
                    "value": float(value),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    summary = []
    ds = []
    for model in ["raft", "acv"]:
        for method in METHODS:
            rows = [r for r in per if r["model"] == model and r["method"] == method]
            q = {
                "model": model,
                "method": method,
                "condition": "six_corruption_macro",
                "seeds": 3,
            }
            for k, metric in enumerate(METRICS):
                x = np.array([r[metric] for r in rows])
                lo, hi = ci(
                    np.mean([boots[s, model, method][:, k] for s in seeds], axis=0)
                )
                q.update(
                    {
                        metric: float(x.mean()),
                        metric + "_seed_SD": float(x.std(ddof=1)),
                        metric + "_ci_low": lo,
                        metric + "_ci_high": hi,
                    }
                )
            summary.append(q)
        for name in NAMES:
            x = [
                r["value"]
                for r in derived
                if r["model"] == model and r["quantity"] == name
            ]
            lo, hi = ci(np.mean([db[s, model, name] for s in seeds], axis=0))
            ds.append(
                {
                    "model": model,
                    "quantity": name,
                    "mean": float(np.mean(x)),
                    "seed_SD": float(np.std(x, ddof=1)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "unit": "fraction" if "recovery" in name else "percentage points",
                    "seeds": 3,
                }
            )
    return summary, ds, per, derived


def report(a):
    from .camera_ready import run

    return run(a)


def report_context(evidence, out):
    """Development task metrics and previews from saved proxy summaries."""
    from .plots import evidence_plots, condition_response_plot

    evidence_plots(evidence, out)
    condition_response_plot(evidence, out)
    task = []
    for model in ["raft", "acv"]:
        rows = journal(evidence / "phase_a" / f"{model}.jsonl")
        scenes = sorted({r["scene"] for r in rows})
        weights = bootstrap_weights(scenes, 1000, 1741)
        cp = []
        cb = []
        for cond in ["clean", *CONDITIONS]:
            q, b = table1_summary(
                [r for r in rows if r["condition"] == cond], scenes, weights, "full"
            )
            task.append({"model": model, "condition": cond, **q})
            if cond != "clean":
                cp.append([q[k] for k in METRICS])
                cb.append(b)
        q = {"model": model, "condition": "six_corruption_macro"}
        for k, metric in enumerate(METRICS):
            lo, hi = ci(np.mean(cb, axis=0)[:, k])
            q.update(
                {
                    metric: float(np.mean(cp, axis=0)[k]),
                    metric + "_ci_low": lo,
                    metric + "_ci_high": hi,
                }
            )
        task.append(q)
    csvwrite(out / "table1_development.csv", task)


def table2(a):
    from table2_pool_v4 import load_samples
    from sensor_v4_metrics import risk_points, risk_cluster_summary

    cfg = read(ROOT / "configs/paper/phase_a.json")
    rows = journal(a.journal)
    rows = [r for r in rows if r["split"] == "dev" and r.get("table", "T12") == "T12"]
    if not rows:
        raise ValueError("no development Table 2 rows")
    out = output(a.run_id)
    bind(
        out,
        {
            "task": "table2",
            "journal_sha256": sha(a.journal),
            "pixels": a.pixels,
            "common_support": a.common_support,
            "bootstrap": a.bootstrap,
        },
    )
    scenes = sorted({r["scene"] for r in rows})
    weights = bootstrap_weights(scenes, a.bootstrap, 1741)
    results = []
    images = []
    macro = {}
    for model in sorted({r["model"] for r in rows}):
        for condition in ["clean", *CONDITIONS]:
            rr = [
                r for r in rows if r["model"] == model and r["condition"] == condition
            ]
            if not rr:
                continue
            pools = []
            si = []
            for r in rr:
                d = load_samples(r, a.pixels)
                if a.common_support:
                    if "common_support" not in d:
                        raise ValueError("missing GT-only secondary mask")
                    mask = d["common_support"]
                    d = {k: v[mask] for k, v in d.items()}
                names = [
                    k
                    for k in d
                    if k not in ["error", "d1", "pixel_index", "common_support"]
                ]
                pools.append(d)
                si.append(np.full(len(d["error"]), scenes.index(r["scene"])))
                for signal in names:
                    images.append(
                        {
                            "model": model,
                            "condition": condition,
                            "pair_id": r["pair_id"],
                            "scene": r["scene"],
                            "signal": signal,
                            **risk_points(d["d1"], d[signal], d["error"]),
                        }
                    )
            y = np.concatenate([d["d1"] for d in pools])
            error = np.concatenate([d["error"] for d in pools])
            scene_index = np.concatenate(si)
            for signal in names:
                risk = np.concatenate([d[signal] for d in pools])
                q, b = risk_cluster_summary(
                    y,
                    risk,
                    error,
                    scene_index,
                    scenes,
                    weights,
                    cfg["metrics"]["table2"],
                )
                results.append(
                    {"model": model, "condition": condition, "signal": signal, **q}
                )
                if condition != "clean":
                    macro.setdefault((model, signal), []).append((q, b))
            print(model, condition, "complete", flush=True)
    for (model, signal), values in macro.items():
        if len(values) != 6:
            continue
        row = {"model": model, "condition": "six_corruption_macro", "signal": signal}
        for metric in ["AUROC", "AUPR", "Spearman", "prevalence"]:
            row[metric] = float(np.mean([q[metric] for q, _ in values]))
        for j, metric in enumerate(["AUROC", "AUPR", "prevalence"]):
            row[metric + "_ci_low"], row[metric + "_ci_high"] = ci(
                np.mean([b[:, j] for _, b in values], axis=0)
            )
        results.append(row)
    csvwrite(out / "pooled_metrics.csv", results)
    csvwrite(out / "within_image_metrics.csv", images)
    write(
        out / "COMPLETE.json",
        {
            "inference": False,
            "signal_reselection": False,
            "support": "GT<192 secondary" if a.common_support else "primary",
            "pixels_per_pair": a.pixels,
        },
    )
    print(out)


def require_assets(records):
    from sensor_v4_common import ROOT as asset_root

    missing = [
        str(asset_root / a["path"])
        for r in records
        for a in r["assets"].values()
        if not (asset_root / a["path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} assets; run assets download-paper first. Example: {missing[0]}"
        )


def setup_inference(a, filename):
    cfg = read(ROOT / "configs/paper" / filename)
    out = output(a.run_id)
    cfg["storage_root"] = str(out / "artifacts")
    cfg["storage_guard_root"] = str(
        Path(profile().get("storage_guard_root", profile()["storage_root"])).resolve()
    )
    if a.batch:
        cfg["models"][a.model]["batch"] = a.batch
    if cfg["models"][a.model]["batch"] < 1:
        raise ValueError("batch must be positive")
    return cfg, out


def phase_a(a):
    from run_sensor_v4 import model_worker

    cfg, out = setup_inference(a, "phase_a.json")
    records = read(ROOT / "configs/paper/split500.json")["records"]
    records = [r for r in records if a.split == "all" or r["split"] == a.split]
    if a.limit:
        records = records[: a.limit]
    if a.conditions:
        if not set(a.conditions) <= set(cfg["conditions"]):
            raise ValueError("condition outside frozen protocol")
        cfg["conditions"] = a.conditions
    require_assets(records)
    bind(
        out,
        {"task": "phase-a", "config": cfg, "records": [r["pair_id"] for r in records]},
    )
    if (out / a.model / "COMPLETE").exists():
        print("already complete")
        return
    if (out / a.model).exists():
        raise FileExistsError(
            "partial Phase A run: use a new run ID; original cache is preserved"
        )
    with gpu_lock():
        model_worker(a.model, cfg, records, "A", out)


def adaptation(a):
    import torch
    from PIL import Image
    from sensor_v4_images import corrupt
    from generic_transform_v1 import transform
    import generic_heldout_v1 as evaluator
    from broad_sweep_v4 import compile_counts
    from sensor_v4_common import ROOT as KROOT

    cfg, out = setup_inference(a, "adaptation.json")
    cfg["heldout_run"]["assignment_seed"] = a.seed
    bank = read(ROOT / "configs/paper/bank.json")
    frozen = read(ROOT / "configs/paper/freeze.json")
    if digest(bank) != frozen["bank_sha256"]:
        raise ValueError("frozen bank hash mismatch")
    assignment = read(ROOT / "configs/paper/assignments" / f"seed{a.seed}.json")[
        "records"
    ]
    if a.limit:
        assignment = assignment[: a.limit]
    records = {
        r["pair_id"]: r for r in read(ROOT / "configs/paper/split500.json")["records"]
    }
    require_assets([records[r["pair_id"]] for r in assignment])
    bind(
        out,
        {
            "task": "adaptation",
            "seed": a.seed,
            "config": cfg,
            "bank_sha256": digest(bank),
            "freeze_sha256": digest(frozen),
            "assignment": assignment,
        },
    )
    folder = out / a.model
    folder.mkdir(exist_ok=True)
    j = folder / "observations.jsonl"
    done = {r["pair_id"] for r in journal(j)}
    todo = [r for r in assignment if r["pair_id"] not in done]
    if not todo:
        print("already complete")
        return
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mc = cfg["models"][a.model]
    checkpoint = KROOT / mc["checkpoint"]
    if sha(checkpoint) != mc["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    free, _ = torch.cuda.mem_get_info()
    required = 62.684 if a.model == "acv" and mc["batch"] == 4 else 12.522
    if free / 2**30 < required + 1:
        raise RuntimeError("insufficient free VRAM for validated batch")
    with gpu_lock():
        if a.model == "raft":
            from raft_adapter import RAFTAdapter

            net = RAFTAdapter(checkpoint, mc["iterations"], last_n=mc["last_n"])
        else:
            from acv_adapter import ACVAdapter

            net = ACVAdapter(checkpoint, mc["maxdisp"])
        so = compile_counts(cfg)
        evaluator.RUN = a.run_id
        signal = frozen["primary_signals"][a.model]
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

        def prep(row):
            r = records[row["pair_id"]]
            condition = row["incoming_condition"]
            clean = [
                np.asarray(Image.open(KROOT / r["assets"][kind]["path"]).convert("RGB"))
                for kind in ["frame_left", "frame_right"]
            ]
            incoming = [
                corrupt(rgb, condition, r, eye, cfg) for eye, rgb in enumerate(clean)
            ]
            rgb = [
                [transform(image, c) for c in bank["candidates"]] for image in incoming
            ]
            return r, condition, clean, rgb

        with ThreadPoolExecutor(max_workers=1) as cpu:
            future = cpu.submit(prep, todo[0])
            for i, row in enumerate(todo):
                r, condition, clean, rgb = future.result()
                if i + 1 < len(todo):
                    future = cpu.submit(prep, todo[i + 1])
                disp = {}
                risk = {}
                timing = {}
                hashes = {}
                for first in range(0, len(bank["candidates"]), mc["batch"]):
                    last = min(first + mc["batch"], len(bank["candidates"]))
                    pred, sigs, t = net.predict_batch(
                        rgb[0][first:last], rgb[1][first:last]
                    )
                    for k, c in enumerate(bank["candidates"][first:last]):
                        disp[c["id"]] = pred[k].copy()
                        risk[c["id"]] = sigs[signal][k].copy()
                        hashes[c["id"]] = {
                            kind: __import__("hashlib")
                            .sha256(rgb[eye][first + k].tobytes())
                            .hexdigest()
                            for eye, kind in enumerate(["frame_left", "frame_right"])
                        }
                    for k, v in t.items():
                        timing[k] = timing.get(k, 0) + v
                cp, _, _ = net.predict_batch([clean[0]], [clean[1]])
                bundle = {
                    "record": r,
                    "model": a.model,
                    "condition": condition,
                    "disparities": disp,
                    "risks": risk,
                    "clean": cp[0].copy(),
                    "input_sha256": hashes,
                    "RGB_cache_path": None,
                    "timing": timing,
                }
                result = evaluator.evaluate(
                    bundle,
                    cfg,
                    bank,
                    frozen,
                    {"heldout_pixel_fusion_approved": True},
                    out,
                    so,
                )
                result["assignment_seed"] = a.seed
                append(j, result)
                print(
                    a.model, a.seed, len(done) + i + 1, "/", len(assignment), flush=True
                )
        write(
            folder / "COMPLETE.json",
            {
                "observations": len(assignment),
                "wall_seconds": time.monotonic() - start,
                "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
                "GPU": torch.cuda.get_device_name(),
            },
        )


def run(a):
    if a.action == "qualitative":
        from .qualitative import run as render

        render(a)
        return
    if a.action == "samepixel":
        from .samepixel import run as analyze

        analyze(a)
        return
    {"report": report, "table2": table2, "phase-a": phase_a, "adaptation": adaptation}[
        a.action
    ](a)
