import collections, hashlib, json, os, re, shutil, subprocess, urllib.request, zipfile, zlib
from pathlib import Path
from .common import ROOT, read, profile, sha, write


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=120) as r, temp.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    return temp


def run(a):
    models = read(ROOT / "configs/assets.json")
    if a.action == "list":
        print(json.dumps(models, indent=2))
        return
    p = profile()
    if a.action in ["clone-model", "download-model"]:
        if not a.model:
            raise ValueError("--model required")
        m = models[a.model]
        if a.action == "clone-model":
            dest = Path(p[m["source_key"]]) / m.get("source_subdir", "")
            if not dest.exists():
                subprocess.run(["git", "clone", m["repo"], str(dest)], check=True)
                subprocess.run(
                    ["git", "checkout", "--detach", m["commit"]], cwd=dest, check=True
                )
            got = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=dest, text=True
            ).strip()
            if got != m["commit"]:
                raise ValueError("existing upstream revision differs; not overwritten")
            print(dest)
            return
        dest = Path(p[m["checkpoint_key"]]) / m.get("checkpoint_subpath", "")
        if dest.exists():
            if sha(dest) != m["sha256"]:
                raise ValueError("existing checkpoint hash mismatch")
            print(dest)
            return
        if m.get("manual_download"):
            raise SystemExit(
                "Download the upstream model/config pair first; see docs/ASSETS.md"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if m.get("gdrive_id"):
            import gdown

            tmp = dest.with_name(dest.name + ".partial")
            gdown.download(id=m["gdrive_id"], output=str(tmp), quiet=False)
        else:
            tmp = download(m["url"], dest)
        if m.get("zip_member_suffix"):
            with zipfile.ZipFile(tmp) as z:
                members = [
                    n for n in z.namelist() if n.endswith(m["zip_member_suffix"])
                ]
                if len(members) != 1:
                    raise ValueError("checkpoint ZIP inventory")
                decoded = tmp.with_suffix(".decoded")
                decoded.write_bytes(z.read(members[0]))
            tmp.unlink()
            tmp = decoded
        # Some upstream links package the checkpoint inside a named archive.
        # A PyTorch ZIP checkpoint itself is retained when its SHA already matches.
        if (
            sha(tmp) != m["sha256"]
            and zipfile.is_zipfile(tmp)
            and not m.get("zip_member_suffix")
        ):
            with zipfile.ZipFile(tmp) as z:
                members = [n for n in z.namelist() if Path(n).name == dest.name]
                if len(members) == 1:
                    decoded = tmp.with_suffix(".decoded")
                    decoded.write_bytes(z.read(members[0]))
                    tmp.unlink()
                    tmp = decoded
        if sha(tmp) != m["sha256"]:
            raise ValueError("download checksum mismatch; partial preserved")
        os.replace(tmp, dest)
        print(dest)
        return
    if a.action == "download-paper":
        import sys

        # RangeReader depends on requests; use the paper environment.
        from download_subset import RangeReader

        manifest = read(ROOT / "configs/paper/split500.json")["records"]
        if a.limit:
            manifest = manifest[: a.limit]
        metadata = read(ROOT / "provenance/spring_data_metadata.json")
        fs = (
            metadata["data"]["latestVersion"]["files"]
            if "data" in metadata
            else metadata["files"]
        )
        sizes = {str(f["dataFile"]["id"]): f["dataFile"]["filesize"] for f in fs}
        groups = collections.defaultdict(list)
        for r in manifest:
            for asset in r["assets"].values():
                groups[asset["source_url"]].append(asset)
        for url, assets in groups.items():
            pending = []
            for x in assets:
                rel = Path(x["path"]).relative_to("data")
                dest = Path(p["paper_data"]) / rel
                if dest.exists():
                    if (
                        dest.stat().st_size != x["size"]
                        or zlib.crc32(dest.read_bytes()) != x["crc32"]
                    ):
                        raise ValueError("existing asset mismatch")
                else:
                    pending.append((x, dest))
            if not pending:
                continue
            with zipfile.ZipFile(RangeReader(url, sizes[url.rsplit("/", 1)[-1]])) as z:
                for x, dest in pending:
                    data = z.read(x["zip_member"])
                    if len(data) != x["size"] or zlib.crc32(data) != x["crc32"]:
                        raise ValueError("ZIP member checksum")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".partial")
                    tmp.write_bytes(data)
                    os.replace(tmp, dest)
            print(url, "completed", len(pending), flush=True)
        return
    from screening.official_foundation_io import CONDITIONS, SEQUENCES

    if a.condition not in CONDITIONS:
        raise ValueError(a.condition)
    base = Path(p["storage_root"]) / "official_data"
    for row in read(ROOT / "provenance/official_data_manifest.json")["archives"]:
        if row["condition"] != a.condition:
            continue
        dest = base / "archives" / row["filename"]
        path = dest if dest.exists() else download(row["url"], dest)
        h = hashlib.new(row["checksum"]["type"].lower().replace("-", ""))
        with path.open("rb") as f:
            for b in iter(lambda: f.read(4 << 20), b""):
                h.update(b)
        if (
            path.stat().st_size != row["bytes"]
            or h.hexdigest() != row["checksum"]["value"]
        ):
            raise ValueError("official archive checksum")
        if path != dest:
            os.replace(path, dest)
        with zipfile.ZipFile(dest) as z:
            for name in z.namelist():
                if not name.endswith(".png"):
                    continue
                m = re.search(
                    r"(?:^|/)test/(\d{4})/frame_(left|right)/frame_\2_(\d{4})\.png$",
                    name,
                )
                if not m:
                    raise ValueError("unexpected official PNG")
                scene, eye, frame = m.groups()
                if scene not in SEQUENCES or not 1 <= int(frame) <= SEQUENCES[scene]:
                    raise ValueError("official inventory")
                target = (
                    base
                    / "inputs"
                    / a.condition
                    / "test"
                    / scene
                    / ("frame_" + eye)
                    / f"frame_{eye}_{frame}.png"
                )
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_suffix(".partial")
                    tmp.write_bytes(z.read(name))
                    os.replace(tmp, target)
        print(dest, flush=True)
