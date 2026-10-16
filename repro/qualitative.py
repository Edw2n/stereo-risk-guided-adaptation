"""CPU preview from saved dense predictions; no model inference or new selection."""

import json
import numpy as np, h5py
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from .common import ROOT, read, write, csvwrite, journal, output, bind, sha
from sensor_v4_common import ROOT as ASSET_ROOT
from sensor_v4_images import corrupt
from generic_transform_v1 import transform
from sensor_v4_metrics import native_maps, native_values


def run(a):
    matches = [r for r in journal(a.journal) if r["pair_id"] == a.pair]
    if len(matches) != 1:
        raise ValueError("choose one recorded pair/condition")
    row = matches[0]
    path = row.get("dense_qualitative_path")
    if not path:
        raise ValueError("no saved dense maps for this pair; no inference is launched")
    out = output(a.run_id)
    bind(
        out,
        {
            "task": "qualitative",
            "journal_sha256": sha(a.journal),
            "pair": a.pair,
            "dense_path": path,
        },
    )
    cfg = read(ROOT / "configs/paper/adaptation.json")
    bank = read(ROOT / "configs/paper/bank.json")
    record = next(
        r
        for r in read(ROOT / "configs/paper/split500.json")["records"]
        if r["pair_id"] == a.pair
    )
    source = [
        np.asarray(
            Image.open(ASSET_ROOT / record["assets"][eye]["path"]).convert("RGB")
        )
        for eye in ["frame_left", "frame_right"]
    ]
    incoming = [
        corrupt(rgb, row["condition"], record, i, cfg) for i, rgb in enumerate(source)
    ]
    with h5py.File(ASSET_ROOT / record["assets"]["disp1_left"]["path"], "r") as f:
        gt = f["disparity"][:]
    with np.load(path, allow_pickle=False) as f:
        ids = list(f["candidate_ids"])
        disp = f["disparity"]
        clean = f["clean"]
        sf = f["signal_fusion"]
        po = f["pixel_oracle"]
        sm = f["signal_assignment"]
        om = f["oracle_assignment"]
    if ids != sorted(c["id"] for c in bank["candidates"]):
        raise ValueError("dense bank mismatch")
    definitions = {c["id"]: c for c in bank["candidates"]}
    pred = {c: disp[i] for i, c in enumerate(ids)}
    pred["clean"] = clean
    ms = {m["method"]: native_values(m["full"]) for m in row["methods"]}
    go = row["picks"]["Global Oracle"]
    sg = row["picks"]["Signal-Global"]
    extra = [c for c in ids if c not in ["identity", go, sg]][:2]
    columns = [
        ("Clean", "clean"),
        ("Incoming (" + row["condition"] + ")", "identity"),
        ("Global Oracle", go),
        ("Signal-Global", sg),
        *[(c, c) for c in extra],
    ]
    fig, axes = plt.subplots(4, len(columns), figsize=(15, 7), layout="constrained")
    for col, (name, cid) in enumerate(columns):
        rgb = (
            source
            if cid == "clean"
            else [transform(x, definitions[cid]) for x in incoming]
        )
        error, _, valid, _ = native_maps(pred[cid], gt)
        values = (
            ms["Clean"]
            if cid == "clean"
            else native_values(row["candidate_statistics"][cid]["full"])
        )
        for i, ax in enumerate(axes[:, col]):
            if i < 2:
                ax.imshow(rgb[i])
            elif i == 2:
                ax.imshow(pred[cid], cmap="turbo", vmin=0, vmax=192)
            else:
                ax.imshow(
                    np.ma.array(error, mask=~valid), cmap="magma", vmin=0, vmax=100
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    ["Left", "Right", "Disparity prediction", "Absolute error"][i]
                )
        axes[0, col].set_title(
            name + ("\n" + cid if name != cid and cid != "clean" else ""), fontsize=8
        )
        axes[3, col].set_xlabel(
            f"Abs {values['Abs']:.2f}; 1px {values['1px']:.2f}%; D1 {values['D1']:.2f}%",
            fontsize=6,
        )
    for ext in ["png", "pdf"]:
        fig.savefig(out / ("global." + ext), dpi=130, bbox_inches="tight")
    plt.close(fig)
    colors = np.vstack(
        [plt.get_cmap("tab20")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20))]
    )[: len(ids)]
    colors[0] = [0.5, 0.5, 0.5, 1]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    fractions = []
    valid = native_maps(sf, gt)[2]
    support = valid.copy()
    support[:, :4] = False
    for j, (name, prediction, assignment) in enumerate(
        [("Signal-Pixel Fusion", sf, sm), ("Pixel Oracle (GT)", po, om)]
    ):
        axes[0, j].imshow(
            np.ma.array(assignment, mask=~support),
            cmap=ListedColormap(colors),
            vmin=-0.5,
            vmax=len(ids) - 0.5,
            interpolation="nearest",
        )
        axes[1, j].imshow(prediction, cmap="turbo", vmin=0, vmax=192)
        error, _, v, _ = native_maps(prediction, gt)
        axes[2, j].imshow(np.ma.array(error, mask=~v), cmap="magma", vmin=0, vmax=100)
        axes[0, j].set_title(name)
        for i, ax in enumerate(axes[:, j]):
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(
                    ["Candidate assignment", "Disparity prediction", "Absolute error"][
                        i
                    ]
                )
    for i, cid in enumerate(ids):
        fractions.append(
            {
                "candidate_id": cid,
                "signal_pixel_fraction": float((sm[support] == i).mean()),
                "pixel_oracle_fraction": float((om[support] == i).mean()),
            }
        )
    fig.legend(
        handles=[Patch(color=colors[i], label=c) for i, c in enumerate(ids)],
        loc="lower center",
        ncol=8,
        fontsize=7,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    for ext in ["png", "pdf"]:
        fig.savefig(out / ("pixel." + ext), dpi=130, bbox_inches="tight")
    plt.close(fig)
    csvwrite(out / "assignment_fractions.csv", fractions)
    write(out / "metrics.json", ms)
    write(
        out / "PROVENANCE.json",
        {
            "pair_id": a.pair,
            "condition": row["condition"],
            "model": row["model"],
            "new_inference": False,
            "new_selection": False,
            "signal_or_bank_modified": False,
            "scales": {"disparity": [0, 192], "absolute_error": [0, 100]},
            "metrics_unclipped": True,
            "Pixel_Oracle": "native absolute-error argmin, evaluation-only",
            "pixel_fusion": "fused disparity predictions, not RGB mosaic inference",
            "assignment_agreement": float((sm[support] == om[support]).mean()),
        },
    )
    print(out)
