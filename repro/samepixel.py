"""Candidate-axis diagnostics from the 31-candidate analysis pool, CPU only."""

import json
from pathlib import Path
import numpy as np, h5py
from scipy.stats import rankdata
from candidate_concordance import concordance, compile_helper
from sensor_v4_metrics import bootstrap_weights, ci
from .common import output, bind, journal, sha, write, csvwrite

FIELDS = [
    "candidate_Spearman",
    "concordance",
    "D1_changing_fraction",
    "risk_constant_fraction",
    "error_constant_fraction",
]


def run(a):
    out = output(a.run_id)
    bind(out, {"task": "samepixel", "journal_sha256": sha(a.journal)})
    lib = compile_helper(out)
    rows = []
    for row in journal(a.journal):
        with h5py.File(row["pool"]["path"], "r") as f:
            g = f[row["pool"]["group"]]
            valid = g["valid"][:].astype(bool)
            risk = g["risk"][:, valid]
            error = g["absolute_error"][:, valid]
            d1 = g["D1"][:, valid]
        if (
            risk.shape[0] != 31
            or not np.isfinite(risk).all()
            or not np.isfinite(error).all()
        ):
            raise ValueError("candidate cache contract")
        x = rankdata(risk, axis=0)
        y = rankdata(error, axis=0)
        x -= x.mean(0)
        y -= y.mean(0)
        xx = (x * x).sum(0)
        yy = (y * y).sum(0)
        den = np.sqrt(xx * yy)
        rho = np.divide(
            (x * y).sum(0), den, out=np.full(risk.shape[1], np.nan), where=den > 0
        )
        rows.append(
            {
                "model": row["model"],
                "pair_id": row["pair_id"],
                "scene": row["scene"],
                "condition": row["condition"],
                "candidate_Spearman": float(np.nanmean(rho)),
                "concordance": float(np.nanmean(concordance(risk, error, lib))),
                "D1_changing_fraction": float((d1.any(0) & ~d1.all(0)).mean()),
                "risk_constant_fraction": float((xx == 0).mean()),
                "error_constant_fraction": float((yy == 0).mean()),
            }
        )
    scenes = sorted({r["scene"] for r in rows})
    w = bootstrap_weights(scenes, 1000, 1741)
    summary = []
    points = []
    boots = []
    conditions = [
        "brightness",
        "contrast",
        "saturation",
        "gaussian_noise",
        "shot_noise",
        "speckle_noise",
    ]
    for c in conditions:
        selected = [r for r in rows if r["condition"] == c]
        if not selected:
            raise ValueError("all six conditions required")
        x = np.array(
            [
                [sum(r[m] for r in selected if r["scene"] == s) for m in FIELDS]
                for s in scenes
            ]
        )
        n = np.array([sum(r["scene"] == s for r in selected) for s in scenes])
        bn = w @ n
        points.append(x.sum(0) / n.sum())
        boots.append(
            np.divide(
                w @ x,
                bn[:, None],
                out=np.full((1000, len(FIELDS)), np.nan),
                where=bn[:, None] > 0,
            )
        )
    for c, p, b in zip(
        conditions + ["six_corruption_macro"],
        points + [np.mean(points, axis=0)],
        boots + [np.mean(boots, axis=0)],
    ):
        r = {"condition": c}
        for i, m in enumerate(FIELDS):
            r[m] = float(p[i])
            r[m + "_ci_low"], r[m + "_ci_high"] = ci(b[:, i])
        summary.append(r)
    csvwrite(out / "per_image.csv", rows)
    csvwrite(out / "summary.csv", summary)
    write(out / "summary.json", summary)
    print(out)
