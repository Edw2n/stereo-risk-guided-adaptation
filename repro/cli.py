import argparse, json, os, sys
from pathlib import Path
from .common import ROOT, bootstrap, configure, profile


def parser():
    p = argparse.ArgumentParser(description="Frozen paper and challenge reproduction")
    sub = p.add_subparsers(dest="task", required=True)
    c = sub.add_parser("configure")
    c.add_argument("--storage-root", type=Path, help="root for environments, assets and outputs")
    sub.add_parser("doctor")
    a = sub.add_parser("assets")
    a.add_argument(
        "action",
        choices=[
            "list",
            "verify-vendor",
            "download-model",
            "clone-model",
            "download-paper",
            "download-official",
        ],
    )
    a.add_argument("--model", choices=["raft", "acv", "croco", "foundation"])
    a.add_argument("--condition", default="clean")
    a.add_argument("--limit", type=int)
    q = sub.add_parser("paper")
    s = q.add_subparsers(dest="action", required=True)
    r = s.add_parser("report", aliases=["camera-ready"],
                     help="reproduce final paper tables and plot previews on CPU")
    r.add_argument("--run-id", required=True)
    r.add_argument("--evidence", type=Path, default=ROOT / "evidence/camera_ready",
                   help="final adaptation, development, clean and baseline observations")
    r.add_argument("--context-evidence", type=Path, default=ROOT / "evidence/paper",
                   help="development task journals and saved proxy summary CSVs")
    r = s.add_parser("revision", add_help=False, help="additional inference and cached analyses")
    r.add_argument("component", choices=["prepare", "infer", "offline", "shared-fixed-best", "report"])
    r.add_argument("arguments", nargs=argparse.REMAINDER)
    for action in ["phase-a", "adaptation"]:
        r = s.add_parser(action)
        r.add_argument("--model", choices=["raft", "acv"], required=True)
        r.add_argument("--run-id", required=True)
        r.add_argument("--limit", type=int)
        r.add_argument(
            "--batch", type=int, help="explicit smoke/throughput override; recorded"
        )
        if action == "phase-a":
            r.add_argument("--split", choices=["dev", "test", "all"], default="all")
            r.add_argument("--conditions", nargs="+")
        else:
            r.add_argument("--seed", type=int, choices=[101, 202, 303], required=True)
    r = s.add_parser("qualitative")
    r.add_argument("--journal", type=Path, required=True)
    r.add_argument("--pair", required=True)
    r.add_argument("--run-id", required=True)
    r = s.add_parser("samepixel")
    r.add_argument("--journal", type=Path, required=True)
    r.add_argument("--run-id", required=True)
    r = s.add_parser("table2")
    r.add_argument("--journal", type=Path, required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument(
        "--pixels", type=int, choices=[5000, 10000, 20000, 40000, 50000], default=20000
    )
    r.add_argument("--common-support", action="store_true")
    r.add_argument("--bootstrap", type=int, default=1000)
    q = sub.add_parser("challenge")
    s = q.add_subparsers(dest="action", required=True)
    r = s.add_parser("report", help="summarize archived server scores and native disagreement")
    r.add_argument("--run-id", required=True)
    r = s.add_parser("index", help="index completed native candidate runs")
    r.add_argument("--runs", nargs="+", type=Path, required=True)
    r.add_argument("--condition", required=True)
    r.add_argument("--run-id", required=True)
    r = s.add_parser("layout", help="assemble one method's condition directories using symlinks")
    r.add_argument("--sources", nargs="+", type=Path, required=True)
    r.add_argument("--method", choices=["Identity", "tile_disparity_std_unweighted_global"], required=True)
    r.add_argument("--run-id", required=True)
    r = s.add_parser("infer")
    r.add_argument(
        "--preset",
        choices=["croco-09-fp32", "croco-09-fp16", "croco-07-fp16", "foundation"],
        required=True,
    )
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--condition", required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--candidate", default="identity")
    r.add_argument("--batch", type=int)
    r.add_argument("--limit", type=int)
    r.add_argument(
        "--precision-scope",
        choices=["forward-only", "whole-call"],
        default="forward-only",
    )
    r = s.add_parser("select")
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--condition", required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument(
        "--methods", choices=["tile-global", "global", "all"], default="tile-global"
    )
    r.add_argument("--limit", type=int)
    r = s.add_parser("delta")
    r.add_argument("--clean", type=Path, required=True)
    r.add_argument("--corrupt", type=Path, required=True)
    r.add_argument("--conditions", nargs="+")
    r.add_argument("--run-id", required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--clean-abs", type=float)
    r = s.add_parser("package")
    r.add_argument("--layout", type=Path, required=True)
    r.add_argument("--tool", type=Path, required=True)
    r.add_argument("--kind", choices=["clean", "robust"], required=True)
    r.add_argument("--run-id", required=True)
    sub.add_parser("test")
    return p


def package_version(name):
    from importlib.metadata import version, PackageNotFoundError

    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def main():
    a = parser().parse_args()
    if a.task == "configure":
        configure(a.storage_root)
        return
    if a.task == "assets" and a.action == "verify-vendor":
        from .vendor import verify_vendor

        print(json.dumps(verify_vendor(), indent=2))
        return
    if a.task == "assets" and a.action == "list":
        from .assets import run

        run(a)
        return
    env = "challenge_python" if a.task == "challenge" else "paper_python"
    if getattr(a, "preset", None) == "foundation":
        env = "foundation_python"
    config = profile()
    py = str(Path(config[env]).expanduser())
    if not Path(py).is_file():
        raise SystemExit(f"Configured Python is missing: {py}. See docs/SETUP.md.")
    if os.environ.get("ROCO_REPRO_ENV") != env:
        os.environ["ROCO_REPRO_ENV"] = env
        for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
            os.environ[k] = "1"
        os.execv(py, [py, str(ROOT / "reproduce.py"), *sys.argv[1:]])
    bootstrap()
    if a.task == "assets":
        from .assets import run

        run(a)
        return
    if a.task == "doctor":
        print(
            json.dumps(
                {
                    "python": sys.version,
                    "packages": {
                        k: package_version(k)
                        for k in ["torch", "numpy", "scipy", "h5py", "Pillow"]
                    },
                    "profile": profile(),
                },
                indent=2,
            )
        )
        return
    if a.task == "test":
        import unittest

        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        )
        raise SystemExit(not result.wasSuccessful())
    if a.task == "paper" and a.action == "revision":
        import runpy
        modules = {"prepare": "revision", "infer": "revision", "offline": "revision_offline",
                   "shared-fixed-best": "revision_shared_fixed", "report": "revision_report"}
        sys.argv = ["reproduce.py paper revision " + a.component]
        if a.component in ("prepare", "infer"):
            sys.argv.append(a.component)
        sys.argv.extend(a.arguments)
        runpy.run_module("repro." + modules[a.component], run_name="__main__")
        return
    if a.task == "paper" and a.action in ("report", "camera-ready"):
        from .camera_ready import run
    elif a.task == "challenge" and a.action in ("index", "layout", "report"):
        from .workflow import run
    elif a.task == "paper":
        from .paper import run
    else:
        from .challenge import run
    run(a)
