import contextlib, json, os, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import h5py
from PIL import Image
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
from .cache_compat import load_index, read_candidate
from .inference_provenance import Inventory, selection_provenance
from screening.official_foundation_io import (
    observations,
    rgb_path,
    disp_path,
    write_dsp5,
    read_dsp5,
    native_output,
    CONDITIONS,
)
from screening.croco_native import (
    pack_reference_inputs,
    assemble_native,
    save_cache,
    verify_cache,
    INTERNAL_INDICES,
)


def cache_path(root, row, cid):
    return (
        Path(root)
        / "cache"
        / row["condition"]
        / row["scene"]
        / f'{row["frame"]}_{cid}.h5'
    )


def infer(a):
    import torch
    from generic_transform_v1 import transform

    p = profile()
    foundation = a.preset == "foundation"
    cfg = read(
        ROOT / "configs/challenge" / ("foundation.json" if foundation else "croco.json")
    )
    mc = cfg["model"]
    prefix = "foundation" if foundation else "croco"
    mc["upstream"]["path"] = p[prefix + "_upstream"]
    mc["checkpoint"]["path"] = p[prefix + "_checkpoint"]
    if foundation:
        mc["inference"]["cfg_yaml"]["path"] = p["foundation_cfg"]
        mc["dinov2_source"] = str(ROOT / "vendor/dinov2")
    batch = a.batch or (4 if foundation else 16)
    if batch < 2 or batch % 2:
        raise ValueError("paired eyes require an even reference-image batch >=2")
    overlap = 0.9 if "-09-" in a.preset else 0.7
    precision = "fp32" if a.preset.endswith("fp32") else "fp16"
    mc["inference"].update(
        batch_size=batch, reference_eye="both via swap+flip", precision=precision
    )
    if not foundation:
        mc["inference"]["tile_overlap"] = overlap
    if a.precision_scope == "whole-call" and (
        foundation or overlap != 0.7 or precision != "fp16"
    ):
        raise ValueError(
            "whole-call scope applies only to the CroCo overlap-0.7 FP16 preset"
        )
    bank = read(ROOT / "configs/challenge/bank8.json")
    candidates = {c["id"]: c for c in bank["candidates"]}
    if a.candidate not in candidates:
        raise ValueError("candidate is not in frozen bank")
    if (foundation or overlap == 0.9) and a.candidate != "identity":
        raise ValueError("baseline preset is Identity-only")
    rows = list(observations([a.condition]))
    rows = rows[: a.limit] if a.limit else rows
    out = output(a.run_id)
    spec = {
        "task": "challenge-infer",
        "preset": a.preset,
        "model": mc,
        "overlap": overlap,
        "precision_scope": a.precision_scope,
        "batch": batch,
        "inputs": str(a.inputs.resolve()),
        "condition": a.condition,
        "candidate": candidates[a.candidate],
        "pairs": [r["pair_id"] for r in rows],
    }
    fingerprint = bind(out, spec)
    j = out / "observations.jsonl"
    done = {r["pair_id"]: r for r in journal(j)}
    todo = []
    for r in rows:
        if r["pair_id"] in done:
            continue
        path = cache_path(out, r, a.candidate)
        if not foundation and overlap == 0.7 and path.exists():
            verify_cache(
                path,
                {
                    "execution_sha256": fingerprint,
                    "pair_id": r["pair_id"],
                    "candidate_id": a.candidate,
                },
            )
            append(j, {"pair_id": r["pair_id"], "cache": str(path), "recovered": True})
            continue
        todo.append(r)
    if not todo:
        print("already complete")
        return
    if sha(mc["checkpoint"]["path"]) != mc["checkpoint"]["sha256"]:
        raise ValueError("checkpoint mismatch")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=mc["upstream"]["path"], text=True
    ).strip()
    if revision != mc["upstream"]["commit"]:
        raise ValueError("upstream commit mismatch")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    start = time.monotonic()
    with gpu_lock():
        if foundation:
            from screening.batch_adapter import BatchAdapter

            net = BatchAdapter(mc)
        elif precision == "fp16" and a.precision_scope == "forward-only":
            from screening.croco_fp16_proxy import ProxyAdapter

            net = ProxyAdapter(mc, overlap=overlap)
        else:
            from screening.croco_proxy_extended_extract import ProxyAdapter

            net = ProxyAdapter(mc, overlap=overlap)
        torch.cuda.reset_peak_memory_stats()

        def prepare(group):
            prepared = []
            for r in group:
                begin = time.monotonic()
                left = np.asarray(
                    Image.open(rgb_path(a.inputs, r, "left")).convert("RGB")
                )
                right = np.asarray(
                    Image.open(rgb_path(a.inputs, r, "right")).convert("RGB")
                )
                if left.shape != (1080, 1920, 3) or right.shape != left.shape:
                    raise ValueError("native RGB required")
                prepared.append(
                    (
                        r,
                        left,
                        right,
                        transform(left, candidates[a.candidate]),
                        transform(right, candidates[a.candidate]),
                        time.monotonic() - begin,
                    )
                )
            return prepared

        groups = [todo[i : i + batch // 2] for i in range(0, len(todo), batch // 2)]
        with ThreadPoolExecutor(max_workers=1) as prep, ThreadPoolExecutor(
            max_workers=2
        ) as writer:
            ahead = prep.submit(prepare, groups[0])
            pending = []
            for gi, group in enumerate(groups):
                items = ahead.result()
                if gi + 1 < len(groups):
                    ahead = prep.submit(prepare, groups[gi + 1])
                left, right, pad = pack_reference_inputs(items, batch)
                scope = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if a.precision_scope == "whole-call"
                    else contextlib.nullcontext()
                )
                with scope:
                    if foundation:
                        pred, timing = net.predict_batch(left, right)
                        risk = None
                    elif overlap == 0.9:
                        pred, timing = net.predict_plain(left, right)
                        risk = None
                    else:
                        pred, risk, timing = net.predict_with_risks(left, right)

                def save(item, d, risks):
                    r, ol, orr, tl, tr, seconds = item
                    meta = {
                        "execution_sha256": fingerprint,
                        "checkpoint_sha256": mc["checkpoint"]["sha256"],
                        "upstream_commit": revision,
                        "pair_id": r["pair_id"],
                        "condition": a.condition,
                        "candidate_id": a.candidate,
                        "GPU": torch.cuda.get_device_name(),
                        "batch": batch,
                        "overlap": overlap,
                        "precision": precision,
                        "precision_scope": a.precision_scope,
                        "input_sha256": [
                            __import__("hashlib").sha256(x.tobytes()).hexdigest()
                            for x in [tl, tr]
                        ],
                    }
                    if risks is not None:
                        arrays = assemble_native(d, risks, ol, orr)
                        receipt = save_cache(
                            cache_path(out, r, a.candidate), arrays, meta
                        )
                    else:
                        receipt = []
                        for eye, i in [("left", 0), ("right", 1)]:
                            path = disp_path(out / "predictions", r, eye)
                            em = {**meta, "reference_eye": eye}
                            if path.exists():
                                read_dsp5(path, em)
                            else:
                                write_dsp5(
                                    path,
                                    native_output(d[i], eye).astype(np.float32),
                                    em,
                                )
                            receipt.append(str(path))
                    return {
                        "pair_id": r["pair_id"],
                        "condition": a.condition,
                        "candidate": a.candidate,
                        "receipt": receipt,
                        "preprocess_seconds": seconds,
                    }

                for i, item in enumerate(items):
                    pending.append(
                        writer.submit(
                            save,
                            item,
                            pred[2 * i : 2 * i + 2].copy(),
                            (
                                None
                                if risk is None
                                else risk[2 * i : 2 * i + 2, INTERNAL_INDICES].copy()
                            ),
                        )
                    )
                while len(pending) >= batch // 2:
                    row = pending.pop(0).result()
                    append(j, row)
                print(
                    a.condition,
                    a.candidate,
                    min((gi + 1) * batch // 2, len(todo)),
                    "/",
                    len(todo),
                    flush=True,
                )
            for f in pending:
                append(j, f.result())
    write(
        out / "COMPLETE.json",
        {
            "pairs": len(rows),
            "invalid": 0,
            "runtime_seconds": time.monotonic() - start,
            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
            "GPU": torch.cuda.get_device_name(),
            "precision": precision,
            "precision_scope": a.precision_scope,
        },
    )
    print(out)


def select(a):
    from screening.croco_tier8_evaluation import (
        select_arrays,
        RAW,
        METHOD_SCOPES,
        POLICY,
    )

    bank = read(ROOT / "configs/challenge/bank8.json")
    ids = bank["candidate_ids"]
    rows = list(observations([a.condition]))
    rows = rows[: a.limit] if a.limit else rows
    entries = load_index(a.cache, ids)
    out = output(a.run_id)
    bind(
        out,
        {
            "task": "challenge-select",
            "cache": str(a.cache.resolve()),
            "cache_index_sha256": digest(entries),
            "condition": a.condition,
            "methods": a.methods,
            "bank_sha256": digest(bank),
            "policy": POLICY,
            "pairs": [r["pair_id"] for r in rows],
        },
    )
    j = out / "selection.jsonl"
    done = {r["pair_id"] for r in journal(j)}
    for r in rows:
        if r["pair_id"] in done:
            continue
        arrays = []
        meta = []
        receipts = []
        for cid in ids:
            values, metadata, receipt = read_candidate(entries[cid], r, cid, RAW)
            arrays.append(values)
            meta.append(metadata)
            receipts.append(receipt)
        for key in [
            "checkpoint_sha256",
            "upstream_commit",
            "overlap",
            "precision",
            "precision_scope",
        ]:
            if len({m[key] for m in meta}) != 1:
                raise ValueError("mixed candidate configuration: " + key)
        inference = selection_provenance(meta, receipts)
        disp = np.stack([x["disparity"] for x in arrays])
        raw = {k: np.stack([x[k] for x in arrays]) for k in RAW}
        pred, info = select_arrays(disp, raw, r["pair_id"], r["condition"], ids)
        for method in METHOD_SCOPES[a.methods]:
            method_id = method.replace("/", "_")
            base = out / "predictions" / method_id
            for eye, i in [("left", 0), ("right", 1)]:
                path = disp_path(base, r, eye)
                m = {
                    "condition": r["condition"],
                    "pair_id": r["pair_id"],
                    "reference_eye": eye,
                    "method": method,
                    "policy_sha256": digest(POLICY),
                    "bank_sha256": digest(bank),
                    "checkpoint_sha256": meta[0]["checkpoint_sha256"],
                    "inference_provenance": inference,
                }
                if path.exists():
                    read_dsp5(path, m)
                else:
                    write_dsp5(path, pred[method][i], m)
        append(
            j,
            {
                "pair_id": r["pair_id"],
                "condition": r["condition"],
                "selection": info,
                "inference_provenance": inference,
            },
        )
        print(r["pair_id"], flush=True)
    write(
        out / "COMPLETE.json",
        {"pairs": len(rows), "methods": METHOD_SCOPES[a.methods], "GT_used": False},
    )
    print(out)


def delta(a):
    from screening.official_foundation_delta_metrics import (
        disagreement,
        summarize_condition,
        summarize_conditions,
    )

    conditions = a.conditions or CONDITIONS[1:]
    if "clean" in conditions:
        raise ValueError("clean is the reference, not a corruption")
    out = output(a.run_id)
    bind(
        out,
        {
            "task": "delta",
            "clean": str(a.clean.resolve()),
            "corrupt": str(a.corrupt.resolve()),
            "conditions": conditions,
            "limit": a.limit,
            "clean_abs": a.clean_abs,
        },
    )
    pair_rows = []
    condition_rows = []
    inventory = Inventory()
    for condition in conditions:
        rows = list(observations([condition]))
        rows = rows[: a.limit] if a.limit else rows
        rr = []
        for r in rows:
            for eye in ["left", "right"]:
                clean, cm = read_dsp5(
                    disp_path(a.clean, {**r, "condition": "clean"}, eye),
                    {
                        "pair_id": r["pair_id"],
                        "condition": "clean",
                        "reference_eye": eye,
                    },
                )
                corrupt, xm = read_dsp5(
                    disp_path(a.corrupt, r, eye),
                    {
                        "pair_id": r["pair_id"],
                        "condition": condition,
                        "reference_eye": eye,
                    },
                )
                for k in [
                    "method",
                    "checkpoint_sha256",
                    "policy_sha256",
                    "bank_sha256",
                ]:
                    if cm.get(k) != xm.get(k):
                        raise ValueError("clean/corrupt method identity mismatch: " + k)
                inventory.add("clean", cm)
                inventory.add(condition, xm)
                rr.append({**r, "eye": eye, **disagreement(clean, corrupt)})
        _, totals = summarize_condition(rr, condition)
        condition_rows += totals
        pair_rows += rr
        print(condition, "complete", flush=True)
    macro = summarize_conditions(condition_rows, conditions, a.clean_abs)
    provenance = inventory.summary()
    macro["inference_provenance"] = provenance
    csvwrite(out / "pairs.csv", pair_rows)
    csvwrite(out / "conditions.csv", condition_rows)
    write(out / "summary.json", macro)
    write(
        out / "DOMAIN.json",
        {
            "domain": "both eyes, full native finite prediction maps; no GT mask",
            "official_server_score": False,
            "inference_provenance": provenance,
            "note": "official tool applies its own subsampling/mask; native delta is a local estimate",
        },
    )
    print(out)


def package(a):
    expected_name = (
        "disp1_subsampling" if a.kind == "clean" else "disp1_robust_subsampling"
    )
    t = next(
        r
        for r in read(ROOT / "provenance/official_tools.json")["executables"]
        if r["name"] == expected_name
    )
    if sha(a.tool) != t["sha256"]:
        raise ValueError("official packaging executable checksum mismatch")
    conditions = ["clean"] if a.kind == "clean" else CONDITIONS
    inventory = Inventory()
    for r in observations(conditions):
        for eye in ["left", "right"]:
            if not disp_path(a.layout, r, eye).is_file():
                raise FileNotFoundError(disp_path(a.layout, r, eye))
            with h5py.File(disp_path(a.layout, r, eye), "r") as f:
                metadata = json.loads(f.attrs.get("provenance_json", "{}"))
            for key, expected in {
                "pair_id": r["pair_id"],
                "condition": r["condition"],
                "reference_eye": eye,
            }.items():
                if metadata.get(key) != expected:
                    raise ValueError("package pair/eye metadata mismatch: " + key)
            inventory.add(r["condition"], metadata)
    provenance = inventory.summary()
    out = output(a.run_id)
    bind(
        out,
        {
            "task": "package",
            "kind": a.kind,
            "layout": str(a.layout.resolve()),
            "tool_sha256": t["sha256"],
            "inference_provenance": provenance,
        },
    )
    target = out / (
        "disp1_submission.hdf5" if a.kind == "clean" else "disp1_robustness.hdf5"
    )
    if target.exists():
        raise FileExistsError(target)
    source = a.layout / "clean/test" if a.kind == "clean" else a.layout
    command = [str(a.tool.resolve()), str(source.resolve())]
    env = dict(os.environ)
    libs = profile().get("official_tool_library_paths", [])
    env["LD_LIBRARY_PATH"] = ":".join([*libs, env.get("LD_LIBRARY_PATH", "")])
    with (out / "tool.log").open("w") as log:
        subprocess.run(
            command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    write(
        out / "PACKAGE_READY.json",
        {
            "path": str(target),
            "sha256": sha(target),
            "bytes": target.stat().st_size,
            "command": command,
            "uploaded": False,
            "inference_provenance": provenance,
        },
    )
    print(target)


def run(a):
    {"infer": infer, "select": select, "delta": delta, "package": package}[a.action](a)
