"""Camera-ready additions, isolated from frozen reproduction commands.

Explicit subcommands; no training, submission, or implicit inference in reports.
"""

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import (
    ROOT,
    append,
    bind,
    bootstrap,
    csvwrite,
    digest,
    gpu_lock,
    journal,
    output,
    read,
    sha,
    write,
)

bootstrap()
from sensor_v4_common import seed as stable_seed
from sensor_v4_metrics import native_maps, sufficient, native_values
from sensor_v4_rank_selection import selection_support

CONDITIONS = [
    "brightness",
    "contrast",
    "saturation",
    "gaussian_noise",
    "shot_noise",
    "speckle_noise",
]


def consensus_scores(disparities, ids):
    """No GT/mask input. Every fixed geometry coordinate contributes equally."""
    d = np.asarray(disparities)
    if d.ndim != 2 or d.shape[0] != len(ids) or not np.isfinite(d).all():
        raise ValueError("finite K x P disparities required")
    center = np.median(d, axis=0)
    values = np.abs(d.astype(np.float64) - center).mean(axis=1)
    return dict(zip(ids, map(float, values)))


def choose_consensus(scores, pair, condition, model, tie_seed=20261012):
    best = min(scores.values())
    tied = sorted(k for k, v in scores.items() if abs(v - best) <= 1e-12)
    return str(
        np.random.default_rng(
            stable_seed(tie_seed, "Consensus-Global", pair, condition, model)
        ).choice(tied)
    )


def subset_scores(counts, subset):
    ids = counts["candidate_ids"]
    ix = [ids.index(k) for k in subset]
    lo, eq = np.asarray(counts["less_counts"]), np.asarray(counts["equal_counts"])
    n = counts["pixels"]
    return {
        ids[i]: float((n + sum(lo[j, i] + 0.5 * eq[j, i] for j in ix if j != i)) / n)
        for i in ix
    }


def prepare(a):
    out = output(a.run_id)
    bank = read(ROOT / "configs/paper/bank.json")
    freeze = read(ROOT / "configs/paper/freeze.json")
    if digest(bank) != freeze["bank_sha256"]:
        raise ValueError("frozen bank mismatch")
    dev = read(a.dev_assignment)
    records = read(ROOT / "configs/paper/split500.json")["records"]
    dev_ids = {r["pair_id"] for r in records if r["split"] == "dev"}
    assert {r["pair_id"] for r in dev["records"]} == dev_ids and len(dev_ids) == 328
    assert {r["incoming_condition"] for r in dev["records"]} == set(CONDITIONS)
    assert dev["bank_sha256"] == digest(bank)
    nonidentity = sorted(c["id"] for c in bank["candidates"] if c["id"] != "identity")
    subsets = {}
    for s in (9137, 9138, 9139):
        ordered = list(np.random.default_rng(s).permutation(nonidentity))
        subsets[str(s)] = {
            str(n): ["identity"] + ordered[: n - 1] for n in (1, 4, 8, 16, 31)
        }
    spec = {
        "revision_id": "camera_ready_v1_20261012",
        "authorization": "Frozen additional-comparison protocol, 2026-10-12",
        "stages": ["fixed-dev", "clean", "denoise"],
        "models": ["raft", "acv"],
        "bank_sha256": digest(bank),
        "freeze_sha256": digest(freeze),
        "base_config_sha256": sha(ROOT / "configs/paper/adaptation.json"),
        "split_sha256": sha(ROOT / "configs/paper/split500.json"),
        "dev_assignment_sha256": digest(dev),
        "dev_assignment": dev,
        "subsets": subsets,
        "assignment_seeds": [101, 202, 303],
        "consensus": {
            "pool_pixels": 50000,
            "score": "mean absolute disparity disagreement from pixel-wise candidate median",
            "support": "fixed geometry x>=4; no GT-valid/error mask",
            "tie_seed": 20261012,
            "atol": 1e-12,
        },
        "denoiser": {
            "method": "Gaussian smoothing",
            "sigma": 1.0,
            "domain": "float32 sRGB/255",
            "kernel": "OpenCV (0,0), sigmaX=sigmaY=1",
            "border": "BORDER_REFLECT_101",
            "quantization": "final clip [0,1], np.rint(x*255), uint8",
            "selection": "one predeclared setting, not fitted to held-out or dev outcomes",
            "same_policy_on_clean_and_all_corruptions": True,
        },
        "fixed_best": {
            "selection": "pooled 328-dev D1, tie Abs then 1px",
            "heldout_used": False,
        },
        "precision": "fp32",
        "tf32": False,
        "model_processes": 1,
        "cpu": {"torch_threads": 2, "preparation_threads": 2, "metric_workers": 1},
        "storage": "configured storage_root; historical artifacts read-only",
        "optional_severity_or_new_models": False,
    }
    bind(out, spec)
    write(out / "PLAN.json", spec)
    print(out, flush=True)


def load_plan(run_id):
    out = output(run_id)
    plan = read(out / "PLAN.json")
    cfg, bank, freeze = [
        read(ROOT / "configs/paper" / f)
        for f in ("adaptation.json", "bank.json", "freeze.json")
    ]
    for value, key in (
        (digest(bank), "bank_sha256"),
        (digest(freeze), "freeze_sha256"),
        (sha(ROOT / "configs/paper/adaptation.json"), "base_config_sha256"),
        (sha(ROOT / "configs/paper/split500.json"), "split_sha256"),
    ):
        if value != plan[key]:
            raise ValueError("source/config mismatch: " + key)
    return out, plan, cfg, bank, freeze


def stage_records(plan, stage):
    records = {
        r["pair_id"]: r for r in read(ROOT / "configs/paper/split500.json")["records"]
    }
    if stage == "fixed-dev":
        assignments = [
            (r["pair_id"], r["incoming_condition"])
            for r in plan["dev_assignment"]["records"]
        ]
    elif stage == "clean":
        assignments = [(k, "clean") for k, r in records.items() if r["split"] == "test"]
    else:
        assignments = {(k, "clean") for k, r in records.items() if r["split"] == "test"}
        for s in plan["assignment_seeds"]:
            assignments.update(
                (r["pair_id"], r["incoming_condition"])
                for r in read(ROOT / f"configs/paper/assignments/seed{s}.json")[
                    "records"
                ]
            )
        assignments = sorted(assignments)
    return [(records[k], c) for k, c in assignments]


def native_stats(pred, gt, cfg):
    e, d, v, common = native_maps(pred, gt)
    return {
        "full": sufficient(e, d, v),
        "common_support": sufficient(e, d, common),
        "shared_signal_support": sufficient(
            e, d, v & selection_support(pred.shape, cfg)
        ),
    }


def evaluate_fixed(bundle, cfg, bank, folder):
    import h5py
    from sensor_v4_common import ROOT as KROOT
    from candidate_adequacy_v4 import geometry_pool

    r, condition = bundle["record"], bundle["condition"]
    if r["split"] != "dev":
        raise ValueError("Fixed-best development data only")
    with h5py.File(KROOT / r["assets"]["disp1_left"]["path"], "r") as f:
        gt = f["disparity"][:]
    ids = sorted(bundle["disparities"])
    stats = {c: native_stats(bundle["disparities"][c], gt, cfg) for c in ids}
    indices = geometry_pool(r["pair_id"], bundle["disparities"]["identity"].shape, cfg)
    pools = folder / "pool"
    pools.mkdir(exist_ok=True)
    pool = pools / (r["pair_id"].replace("/", "_") + "_" + condition + ".h5")
    temp = pool.with_suffix(".partial.h5")
    with h5py.File(temp, "w") as f:
        f.attrs["candidate_ids"] = json.dumps(ids)
        f.attrs["support"] = "fixed geometry x>=4, no GT selection"
        f.create_dataset("coordinates", data=indices)
        for name, source in [
            ("disparity", bundle["disparities"]),
            ("risk", bundle["risks"]),
        ]:
            f.create_dataset(
                name,
                data=np.stack([source[c].ravel()[indices] for c in ids]),
                compression="lzf",
            )
    os.replace(temp, pool)
    return {
        "model": bundle["model"],
        "pair_id": r["pair_id"],
        "scene": r["scene"],
        "split": "dev",
        "condition": condition,
        "candidate_statistics": stats,
        "input_sha256": bundle["input_sha256"],
        "pool": str(pool),
        "timing": bundle["timing"],
    }


def fixed_report(folder, plan):
    rows = journal(folder / "observations.jsonl")
    expected = {
        (r["pair_id"], r["incoming_condition"])
        for r in plan["dev_assignment"]["records"]
    }
    if {(r["pair_id"], r["condition"]) for r in rows} != expected or len(rows) != 328:
        raise ValueError("328 complete unique dev observations required")
    ids = sorted(rows[0]["candidate_statistics"])
    values = []
    for c in ids:
        for condition in [*CONDITIONS, "pooled_dev"]:
            rr = (
                rows
                if condition == "pooled_dev"
                else [r for r in rows if r["condition"] == condition]
            )
            st = {
                k: sum(r["candidate_statistics"][c]["full"][k] for r in rr)
                for k in ("n", "abs_sum", "px1_count", "d1_count")
            }
            values.append(
                {"candidate_id": c, "condition": condition, **native_values(st)}
            )
    rank = sorted(
        [v for v in values if v["condition"] == "pooled_dev"],
        key=lambda v: (v["D1"], v["Abs"], v["1px"], v["candidate_id"]),
    )
    csvwrite(folder / "fixed_best_candidates.csv", values)
    write(
        folder / "FIXED_BEST.json",
        {
            "candidate_id": rank[0]["candidate_id"],
            "top_two": rank[:2],
            "top_two_D1_gap_pp": rank[1]["D1"] - rank[0]["D1"],
            "selection": "pooled dev D1, Abs, 1px",
            "observations": 328,
            "sequences": 25,
            "heldout_used": False,
            "journal_sha256": sha(folder / "observations.jsonl"),
        },
    )


def infer(a):
    import cv2, h5py, torch
    from PIL import Image
    from sensor_v4_common import ROOT as KROOT
    from sensor_v4_images import corrupt, quantize
    from generic_transform_v1 import transform
    from broad_sweep_v4 import compile_counts
    from .paper import require_assets

    out, plan, cfg, bank, freeze = load_plan(a.run_id)
    tasks = stage_records(plan, a.stage)
    if a.limit:
        tasks = tasks[: a.limit]
    folder = out / a.stage / a.model
    folder.mkdir(parents=True, exist_ok=True)
    cfg["storage_root"] = str(out / "artifacts")
    from .common import profile

    cfg["storage_guard_root"] = str(
        Path(profile().get("storage_guard_root", profile()["storage_root"])).resolve()
    )
    # Recording/storage scheduling only; frozen numerical operators stay unchanged.
    cfg["broad_sweep"]["native_count_threads"] = 2
    cfg["heldout_run"]["qualitative_pair_ids"] = []
    cfg["models"][a.model]["batch"] = a.batch
    if a.batch not in ((1, 2) if a.model == "raft" else (1, 2, 4)):
        raise ValueError("only measured safe batch candidates")
    spec = {
        "plan_sha256": digest(plan),
        "stage": a.stage,
        "model": a.model,
        "batch": a.batch,
        "limit": a.limit,
        "config": cfg,
        "runner_sha256": sha(Path(__file__)),
        "tasks": [(r["pair_id"], c) for r, c in tasks],
    }
    signature = bind(folder, spec)
    if (folder / "COMPLETE.json").exists():
        print("already complete", folder, flush=True)
        return
    done = journal(folder / "observations.jsonl")
    if any(r["revision_signature"] != signature for r in done):
        raise ValueError("resume mismatch")
    keys = {(r["pair_id"], r["condition"]) for r in done}
    if len(keys) != len(done):
        raise ValueError("duplicate journal rows")
    todo = [(r, c) for r, c in tasks if (r["pair_id"], c) not in keys]
    require_assets([r for r, c in todo])
    if not todo:
        if a.stage == "fixed-dev" and not a.limit:
            fixed_report(folder, plan)
        write(
            folder / "COMPLETE.json",
            {"observations": len(done), "resumed_finalization": True},
        )
        return
    torch.set_num_threads(2)
    torch.manual_seed(20260925)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mc = cfg["models"][a.model]
    checkpoint = KROOT / mc["checkpoint"]
    if sha(checkpoint) != mc["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    required = (
        {1: 16.805, 2: 33.45, 4: 62.685} if a.model == "acv" else {1: 7.6, 2: 12.53}
    )[a.batch]
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < required + 2:
        raise RuntimeError("insufficient free GPU memory for this explicit batch")
    signal = freeze["primary_signals"][a.model]
    cv2.setNumThreads(1)
    start = time.monotonic()

    def prep(item):
        r, c = item
        t = time.monotonic()
        clean = [
            np.asarray(Image.open(KROOT / r["assets"][eye]["path"]).convert("RGB"))
            for eye in ("frame_left", "frame_right")
        ]
        if any(x.shape != (1080, 1920, 3) for x in clean):
            raise ValueError("native resolution changed")
        incoming = [corrupt(x, c, r, eye, cfg) for eye, x in enumerate(clean)]
        candidates = bank["candidates"]
        if a.stage == "denoise":
            rgb = [
                [
                    quantize(
                        cv2.GaussianBlur(
                            x.astype(np.float32) / 255,
                            (0, 0),
                            plan["denoiser"]["sigma"],
                            borderType=cv2.BORDER_REFLECT_101,
                        )
                    )
                ]
                for x in incoming
            ]
            ids = ["gaussian_sigma1"]
        else:
            with futures.ThreadPoolExecutor(
                max_workers=plan["cpu"]["preparation_threads"]
            ) as workers:
                pairs = list(
                    workers.map(
                        lambda cand: [transform(x, cand) for x in incoming], candidates
                    )
                )
            rgb = [[v[eye] for v in pairs] for eye in (0, 1)]
            ids = [v["id"] for v in candidates]
        hashes = {
            cid: [hashlib.sha256(rgb[eye][j].tobytes()).hexdigest() for eye in (0, 1)]
            for j, cid in enumerate(ids)
        }
        return r, c, rgb, ids, hashes, time.monotonic() - t

    def evaluate(bundle):
        t = time.monotonic()
        if a.stage == "fixed-dev":
            row = evaluate_fixed(bundle, cfg, bank, folder)
        elif a.stage == "clean":
            import generic_heldout_v1 as evaluator

            evaluator.RUN = a.run_id + "_clean"
            bundle["clean"] = bundle["disparities"]["identity"]
            bundle["RGB_cache_path"] = None
            row = evaluator.evaluate(
                bundle,
                cfg,
                bank,
                freeze,
                {"heldout_pixel_fusion_approved": True},
                folder,
                so,
            )
            row["assignment_seed"] = None
            row["scope"] = (
                "new clean-preservation experiment; not assigned-corruption study"
            )
        else:
            r = bundle["record"]
            with h5py.File(KROOT / r["assets"]["disp1_left"]["path"], "r") as f:
                gt = f["disparity"][:]
            row = {
                "model": a.model,
                "pair_id": r["pair_id"],
                "scene": r["scene"],
                "condition": bundle["condition"],
                "split": r["split"],
                "methods": [
                    {
                        "method": "Gaussian-Denoise",
                        **native_stats(
                            bundle["disparities"]["gaussian_sigma1"], gt, cfg
                        ),
                    }
                ],
                "input_sha256": bundle["input_sha256"],
                "timing": bundle["timing"],
            }
        row["revision_signature"] = signature
        row["timing"]["CPU_revision_evaluation_seconds"] = time.monotonic() - t
        return row

    with gpu_lock():
        if a.model == "raft":
            from raft_adapter import RAFTAdapter

            net = RAFTAdapter(checkpoint, mc["iterations"], last_n=mc["last_n"])
        else:
            from acv_adapter import ACVAdapter

            net = ACVAdapter(checkpoint, mc["maxdisp"])
        so = compile_counts(cfg) if a.stage == "clean" else None
        torch.cuda.reset_peak_memory_stats()
        write(
            folder / "STARTED.json",
            {
                "pid": os.getpid(),
                "batch": a.batch,
                "GPU": torch.cuda.get_device_name(),
                "free_before_GiB": free / 2**30,
                "checkpoint_sha256": sha(checkpoint),
                "precision": "fp32",
                "expected": len(tasks),
            },
        )

        def collect(pending):
            row = pending.result()
            append(folder / "observations.jsonl", row)
            done.append(row)
            elapsed = time.monotonic() - start
            write(
                folder / "STATUS.json",
                {
                    "stage": a.stage,
                    "model": a.model,
                    "completed": len(done),
                    "total": len(tasks),
                    "elapsed_seconds": elapsed,
                    "last_pair": row["pair_id"],
                    "last_condition": row["condition"],
                    "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
                },
            )
            print(
                a.stage,
                a.model,
                len(done),
                "/",
                len(tasks),
                round(elapsed, 1),
                flush=True,
            )

        with futures.ThreadPoolExecutor(
            max_workers=1
        ) as prep_pool, futures.ThreadPoolExecutor(max_workers=1) as metric_pool:
            nxt = prep_pool.submit(prep, todo[0])
            pending = None
            for i, item in enumerate(todo):
                r, c, rgb, ids, hashes, prep_seconds = nxt.result()
                if i + 1 < len(todo):
                    nxt = prep_pool.submit(prep, todo[i + 1])
                ds, rs, timing = {}, {}, {"preprocessing_seconds": prep_seconds}
                for first in range(0, len(ids), a.batch):
                    last = min(len(ids), first + a.batch)
                    pred, sigs, t = net.predict_batch(
                        rgb[0][first:last], rgb[1][first:last]
                    )
                    for j, cid in enumerate(ids[first:last]):
                        if (
                            pred[j].shape != (1080, 1920)
                            or not np.isfinite(pred[j]).all()
                        ):
                            raise ValueError("invalid disparity")
                        if not np.isfinite(sigs[signal][j, :, 4:]).all():
                            raise ValueError("invalid primary risk")
                        ds[cid] = pred[j].copy()
                        rs[cid] = sigs[signal][j].copy()
                    for k, v in t.items():
                        timing[k] = timing.get(k, 0) + v
                del rgb, pred, sigs
                if pending is not None:
                    collect(pending)
                pending = metric_pool.submit(
                    evaluate,
                    {
                        "record": r,
                        "condition": c,
                        "model": a.model,
                        "disparities": ds,
                        "risks": rs,
                        "timing": timing,
                        "input_sha256": hashes,
                    },
                )
            if pending is not None:
                collect(pending)
        if a.stage == "fixed-dev" and not a.limit:
            fixed_report(folder, plan)
        write(
            folder / "COMPLETE.json",
            {
                "observations": len(done),
                "wall_seconds": time.monotonic() - start,
                "predictions_this_attempt": len(todo)
                * (1 if a.stage == "denoise" else 31),
                "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
                "batch": a.batch,
                "checkpoint_sha256": mc["checkpoint_sha256"],
                "precision": "fp32",
            },
        )


def main():
    p = argparse.ArgumentParser(__doc__)
    s = p.add_subparsers(dest="action", required=True)
    q = s.add_parser("prepare")
    q.add_argument("--run-id", required=True)
    q.add_argument("--dev-assignment", type=Path, required=True)
    q = s.add_parser("infer")
    q.add_argument("--run-id", required=True)
    q.add_argument("--stage", choices=["fixed-dev", "clean", "denoise"], required=True)
    q.add_argument("--model", choices=["raft", "acv"], required=True)
    q.add_argument("--batch", type=int, required=True)
    q.add_argument("--limit", type=int)
    a = p.parse_args()
    try:
        {"prepare": prepare, "infer": infer}[a.action](a)
    except Exception as exc:
        if a.action == "infer":
            write(
                output(a.run_id) / a.stage / a.model / "ERROR.json",
                {"type": type(exc).__name__, "message": str(exc), "pid": os.getpid()},
            )
        raise


if __name__ == "__main__":
    main()
