import contextlib, csv, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def csvwrite(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        w.writeheader()
        w.writerows(rows)


def journal(path):
    return (
        [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        if Path(path).exists()
        else []
    )


def append(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def profile():
    p = ROOT / "configs/local.json"
    if not p.exists():
        raise SystemExit("Run: python reproduce.py configure --storage-root /storage1/USER/stereo-risk")
    return read(p)


def bootstrap():
    for p in [ROOT / "kernels/paper/scripts", ROOT / "kernels/challenge"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def output(name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("run ID must be a single directory name")
    config = profile()
    base = Path(config["storage_root"]).expanduser().resolve()
    guard = Path(config.get("storage_guard_root", base)).expanduser().resolve()
    out = (base / name).resolve()
    if not base.is_relative_to(guard) or not out.is_relative_to(base):
        raise ValueError("output escapes configured storage root/guard")
    out.mkdir(parents=True, exist_ok=True)
    return out


def implementation_sha256():
    """Bind new runs to source bytes as well as numerical configuration."""
    files = [ROOT / "reproduce.py"]
    for directory in ("repro", "kernels"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".cpp")
                     and "__pycache__" not in p.parts)
    return digest({str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})


def bind(out, spec):
    implementation = implementation_sha256()
    p = Path(out) / "RUN.json"
    if p.exists():
        if read(p)["spec"] != spec or read(p).get("implementation_sha256") != implementation:
            raise ValueError("run/config/source mismatch; use a new run ID")
    else:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "uncommitted"
        write(
            p,
            {
                "spec": spec,
                "spec_sha256": digest(spec),
                "implementation_sha256": implementation,
                "code_commit": commit,
                "command": sys.argv,
                "python": sys.executable,
            },
        )
    return digest(spec)


@contextlib.contextmanager
def gpu_lock():
    import fcntl

    p = Path(profile()["storage_root"])
    p.mkdir(parents=True, exist_ok=True)
    with (p / "gpu.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def profile_for(root):
    """Generate portable local paths without inspecting historical directories."""
    root = Path(root).expanduser().resolve()
    return {
        "storage_root": str(root / "outputs"),
        "storage_guard_root": str(root),
        "paper_python": str(root / "envs/paper/bin/python"),
        "challenge_python": str(root / "envs/croco/bin/python"),
        "foundation_python": str(root / "envs/foundation/bin/python"),
        "paper_data": str(root / "data"),
        "paper_checkpoints": str(root / "checkpoints"),
        "paper_third_party": str(root / "upstream"),
        "croco_upstream": str(root / "upstream/croco"),
        "croco_checkpoint": str(root / "checkpoints/crocostereo_finetune_spring.pth"),
        "foundation_upstream": str(root / "upstream/foundation"),
        "foundation_checkpoint": str(root / "checkpoints/foundation/model_best_bp2.pth"),
        "foundation_cfg": str(root / "checkpoints/foundation/cfg.yaml"),
        "official_tool_library_paths": [str(root / "tools/lib")],
    }


def configure(storage_root=None):
    local = ROOT / "configs/local.json"
    if local.exists():
        p = read(local)
        if storage_root is not None:
            requested = profile_for(storage_root)
            if Path(p["storage_root"]).resolve() != Path(requested["storage_root"]):
                raise ValueError("Existing profile uses a different root; edit configs/local.json explicitly.")
    elif storage_root is not None:
        p = profile_for(storage_root)
    else:
        raise SystemExit("Use configure --storage-root /storage1/USER/stereo-risk, or create configs/local.json first.")
    for link, key in [("data", "paper_data"), ("checkpoints", "paper_checkpoints"),
                      ("third_party", "paper_third_party")]:
        target = ROOT / "kernels/paper" / link
        if target.is_symlink():
            if target.resolve() != Path(p[key]).expanduser().resolve():
                raise ValueError("Asset link differs: " + str(target))
        elif target.exists():
            raise FileExistsError(target)
        else:
            target.symlink_to(Path(p[key]).expanduser(), target_is_directory=True)
    write(local, p)
    print(local)
